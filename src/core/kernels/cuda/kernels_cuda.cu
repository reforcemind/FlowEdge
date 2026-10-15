#include "kernels/cuda/kernels_cuda_api.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

namespace fe::cuda_ops {
namespace {

std::atomic<std::uint64_t> g_mallocs{0};

bool g_census = false;

struct CensusRow
{
  const char* kind{};
  int a{}, b{}, c{}, d{}, e{}, f{};
  unsigned n{};
};

constexpr int kCensusCap = 64;
CensusRow g_census_rows[kCensusCap];
int g_census_used = 0;
unsigned g_census_dropped = 0;

void census_hit(const char* kind, int a, int b = 0, int c = 0, int d = 0, int e = 0, int f = 0)
{
  if (!g_census)
    return;
  for (int i = 0; i < g_census_used; ++i) {
    CensusRow& row = g_census_rows[i];
    if (row.kind == kind && row.a == a && row.b == b && row.c == c && row.d == d && row.e == e &&
        row.f == f) {
      ++row.n;
      return;
    }
  }
  if (g_census_used >= kCensusCap) {
    ++g_census_dropped;
    return;
  }
  g_census_rows[g_census_used++] = CensusRow{kind, a, b, c, d, e, f, 1u};
}

struct GrowBuffer
{
  void* ptr{};
  std::size_t cap{};

  void* ensure(std::size_t bytes) noexcept
  {
    if (bytes <= cap)
      return ptr;
    void* next = nullptr;
    if (cudaMalloc(&next, bytes) != cudaSuccess)
      return nullptr;
    g_mallocs.fetch_add(1, std::memory_order_relaxed);
    if (ptr != nullptr)
      cudaFree(ptr);
    ptr = next;
    cap = bytes;
    return ptr;
  }
};

thread_local GrowBuffer g_a;
thread_local GrowBuffer g_b;
thread_local GrowBuffer g_c;
thread_local GrowBuffer g_d;

[[nodiscard]] bool ready() noexcept
{
  static int cached = -1;
  if (cached < 0) {
    int count = 0;
    cached = (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
  }
  return cached == 1;
}

bool upload(GrowBuffer& buf, const void* src, std::size_t bytes) noexcept
{
  void* dst = buf.ensure(bytes);
  return dst != nullptr && cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool download(void* dst, const void* src, std::size_t bytes) noexcept
{
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

int grid(int n)
{
  return (n + 255) / 256;
}

// Short L keeps threadIdx.x on the time axis so stores coalesce; ty fills a 256-thread
// block. Split-K below is for large IC, where this 2D grid still serializes the reduction.
void dense_conv_grid(int output_length, int out_channels, dim3& block, dim3& grid_dim)
{
  int tx = 16;
  if (output_length <= 4)
    tx = 4;
  else if (output_length <= 8)
    tx = 8;
  const int ty = 256 / tx;
  block = dim3(static_cast<unsigned>(tx), static_cast<unsigned>(ty));
  grid_dim = dim3(static_cast<unsigned>((output_length + tx - 1) / tx),
                  static_cast<unsigned>((out_channels + ty - 1) / ty));
}

__global__ void silu_kernel(float* x, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float v = x[i];
  x[i] = v / (1.0f + expf(-v));
}

__global__ void mish_kernel(float* x, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float v = x[i];
  x[i] = v * tanhf(fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v))));
}

__global__ void softplus_kernel(float* x, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float v = x[i];
  x[i] = fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
}

__global__ void gate_silu_kernel(const float* a, const float* g, float* out, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float v = g[i];
  out[i] = a[i] * (v / (1.0f + expf(-v)));
}

__global__ void matmul_f32_kernel(const float* in, const float* w, float* out, int rows, int in_dim,
                                  int out_dim)
{
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (o >= out_dim || r >= rows)
    return;
  float acc = 0.0f;
  const float* wr = w + (static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim));
  const float* ir = in + (static_cast<std::size_t>(r) * static_cast<std::size_t>(in_dim));
  for (int k = 0; k < in_dim; ++k)
    acc += ir[k] * wr[k];
  out[(static_cast<std::size_t>(r) * static_cast<std::size_t>(out_dim)) + o] = acc;
}

// FiLM and timestep GEMMs are 1xK. The 16x16 launch leaves 15/16 threads idle on rows=1.
__global__ void matmul_f32_row_kernel(const float* in, const float* w, float* out, int in_dim,
                                      int out_dim)
{
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_dim)
    return;
  const float* wr = w + (static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim));
  float acc = 0.0f;
  if ((in_dim & 3) == 0) {
    const float4* in4 = reinterpret_cast<const float4*>(in);
    const float4* wr4 = reinterpret_cast<const float4*>(wr);
    const int n4 = in_dim >> 2;
    for (int k = 0; k < n4; ++k) {
      const float4 a = __ldg(in4 + k);
      const float4 b = __ldg(wr4 + k);
      acc += (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w);
    }
  } else {
    for (int k = 0; k < in_dim; ++k)
      acc += __ldg(in + k) * __ldg(wr + k);
  }
  out[o] = acc;
}

// One warp owns an output: adjacent lanes read adjacent weights in the existing
// row-major layout. No repacking or additional workspace is needed.
__global__ void matmul_f32_warp_kernel(const float* in, const float* w, float* out, int in_dim,
                                       int out_dim)
{
  const int lane = threadIdx.x & 31;
  const int o = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (o >= out_dim)
    return;
  const float* wr = w + static_cast<std::size_t>(o) * in_dim;
  float acc = 0.0F;
  for (int k = lane; k < in_dim; k += 32)
    acc += __ldg(in + k) * __ldg(wr + k);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  if (lane == 0)
    out[o] = acc;
}

__global__ void linear_f32_row_kernel(const float* in, const float* weight, const float* bias,
                                      float* out, int in_dim, int out_dim, int apply_mish)
{
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_dim)
    return;
  const float* const row = weight + (static_cast<std::size_t>(o) * in_dim);
  float acc = 0.0F;
  if ((in_dim & 3) == 0) {
    const float4* input4 = reinterpret_cast<const float4*>(in);
    const float4* row4 = reinterpret_cast<const float4*>(row);
    for (int index = 0; index < (in_dim >> 2); ++index) {
      const float4 a = __ldg(input4 + index);
      const float4 b = __ldg(row4 + index);
      acc += (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w);
    }
  } else {
    for (int index = 0; index < in_dim; ++index)
      acc += __ldg(in + index) * __ldg(row + index);
  }
  acc += __ldg(bias + o);
  if (apply_mish != 0)
    acc *= tanhf(fmaxf(acc, 0.0F) + log1pf(expf(-fabsf(acc))));
  out[o] = acc;
}

__global__ void linear_f32_warp_kernel(const float* in, const float* weight, const float* bias,
                                       float* out, int in_dim, int out_dim, int apply_mish)
{
  const int lane = threadIdx.x & 31;
  const int o = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (o >= out_dim)
    return;
  const float* const row = weight + static_cast<std::size_t>(o) * in_dim;
  float acc = 0.0F;
  for (int index = lane; index < in_dim; index += 32)
    acc += __ldg(in + index) * __ldg(row + index);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  if (lane == 0) {
    acc += __ldg(bias + o);
    if (apply_mish != 0)
      acc *= tanhf(fmaxf(acc, 0.0F) + log1pf(expf(-fabsf(acc))));
    out[o] = acc;
  }
}

__global__ void matmul_bf16_kernel(const float* in, const std::uint16_t* w, float* out, int rows,
                                   int in_dim, int out_dim)
{
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (o >= out_dim || r >= rows)
    return;
  float acc = 0.0f;
  const std::uint16_t* wr = w + (static_cast<std::size_t>(o) * static_cast<std::size_t>(in_dim));
  const float* ir = in + (static_cast<std::size_t>(r) * static_cast<std::size_t>(in_dim));
  for (int k = 0; k < in_dim; ++k) {
    const unsigned bits = static_cast<unsigned>(wr[k]) << 16u;
    acc += ir[k] * __int_as_float(static_cast<int>(bits));
  }
  out[(static_cast<std::size_t>(r) * static_cast<std::size_t>(out_dim)) + o] = acc;
}

__global__ void rmsnorm_kernel(const float* in, const float* weight, float* out, int rows, int dim)
{
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows)
    return;
  const float* ir = in + (static_cast<std::size_t>(r) * static_cast<std::size_t>(dim));
  float* orow = out + (static_cast<std::size_t>(r) * static_cast<std::size_t>(dim));
  float ss = 0.0f;
  for (int i = 0; i < dim; ++i)
    ss += ir[i] * ir[i];
  const float scale = 1.0f / sqrtf((ss / static_cast<float>(dim)) + 1.0e-5f);
  for (int i = 0; i < dim; ++i)
    orow[i] = ir[i] * scale * weight[i];
}

__global__ void conv1d_causal_kernel(const float* x, const float* weight, const float* bias,
                                     float* y, int channels, int length, int kernel)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  const float* xc = x + (static_cast<std::size_t>(c) * static_cast<std::size_t>(length));
  const float* wc = weight + (static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel));
  float* yc = y + (static_cast<std::size_t>(c) * static_cast<std::size_t>(length));
  const float bc = bias[c];
  for (int t = 0; t < length; ++t)
    yc[t] = bc;
  for (int k = 0; k < kernel; ++k) {
    const float wk = wc[k];
    const int start = (kernel - 1) - k;
    for (int t = start; t < length; ++t)
      yc[t] += wk * xc[t - start];
  }
}

__global__ void conv1d_step_kernel(const float* window, const float* weight, const float* bias,
                                   float* y, int channels, int kernel)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  const float* wc = window + (static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel));
  const float* kw = weight + (static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel));
  float acc = bias[c];
  for (int k = 0; k < kernel; ++k)
    acc += kw[k] * wc[k];
  y[c] = acc;
}

__global__ void add3_kernel(float* x, const float* y, const float* z, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += y[i] + z[i];
}

__global__ void add_scaled_kernel(float* x, const float* dx, float scale, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += scale * dx[i];
}

__global__ void scaled_sum_kernel(const float* x, const float* dx, float scale, float* out, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = x[i] + (scale * dx[i]);
}

__global__ void add_heun_kernel(float* x, const float* k1, const float* k2, float dt, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += 0.5f * dt * (k1[i] + k2[i]);
}

__global__ void add_rk4_kernel(float* x, const float* k1, const float* k2, const float* k3,
                               const float* k4, float dt, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += (dt / 6.0f) * (k1[i] + (2.0f * k2[i]) + (2.0f * k3[i]) + k4[i]);
}

__global__ void time_embed_kernel(float t, const float* freqs, float* sinu, int half)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half)
    return;
  sinu[i] = sinf(t * freqs[i]);
  sinu[half + i] = cosf(t * freqs[i]);
}

template<int Kernel, int Stride>
__global__ void conv1d_dense_fixed_kernel(const float* x, const float* weight, const float* bias,
                                          float* y, int in_channels, int out_channels,
                                          int input_length, int output_length, int padding)
{
  const int ot = blockIdx.x * blockDim.x + threadIdx.x;
  const int oc = blockIdx.y * blockDim.y + threadIdx.y;
  if (ot >= output_length || oc >= out_channels)
    return;
  float sum = bias[oc];
  const int origin = ot * Stride - padding;
  for (int ic = 0; ic < in_channels; ++ic) {
    const float* input =
        x + (static_cast<std::size_t>(ic) * static_cast<std::size_t>(input_length));
    const float* wr = weight + (static_cast<std::size_t>((oc * in_channels) + ic) *
                                static_cast<std::size_t>(Kernel));
#pragma unroll
    for (int k = 0; k < Kernel; ++k) {
      const int index = origin + k;
      if (static_cast<unsigned>(index) < static_cast<unsigned>(input_length))
        sum += input[index] * wr[k];
    }
  }
  y[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(output_length)) + ot] = sum;
}

// One block per output channel on L=4 (and L=8 when dispatched). Lanes split IC;
// K taps stay in registers. L=16 stays on the 2D launch; split-K lost there.
template<int Kernel, int Stride, int Length>
__global__ void conv1d_splitk_kernel(const float* x, const float* weight, const float* bias,
                                     float* y, int in_channels, int out_channels, int input_length,
                                     int padding)
{
  const int oc = blockIdx.x;
  if (oc >= out_channels)
    return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ic_stride = blockDim.x;
  __shared__ float partial[16 * 16];
  float acc[Length];
#pragma unroll
  for (int ot = 0; ot < Length; ++ot)
    acc[ot] = 0.0f;
  for (int ic = threadIdx.x; ic < in_channels; ic += ic_stride) {
    const float* input =
        x + (static_cast<std::size_t>(ic) * static_cast<std::size_t>(input_length));
    const float* wr = weight + (static_cast<std::size_t>((oc * in_channels) + ic) *
                                static_cast<std::size_t>(Kernel));
    float wk[Kernel];
#pragma unroll
    for (int k = 0; k < Kernel; ++k)
      wk[k] = __ldg(wr + k);
    if constexpr (Length == 4 || Length == 8) {
      if (input_length == Length) {
        float in[Length];
        if constexpr (Length == 4) {
          const float4 in4 = __ldg(reinterpret_cast<const float4*>(input));
          in[0] = in4.x;
          in[1] = in4.y;
          in[2] = in4.z;
          in[3] = in4.w;
        } else {
          const float4 in_lo = __ldg(reinterpret_cast<const float4*>(input));
          const float4 in_hi = __ldg(reinterpret_cast<const float4*>(input + 4));
          in[0] = in_lo.x;
          in[1] = in_lo.y;
          in[2] = in_lo.z;
          in[3] = in_lo.w;
          in[4] = in_hi.x;
          in[5] = in_hi.y;
          in[6] = in_hi.z;
          in[7] = in_hi.w;
        }
#pragma unroll
        for (int ot = 0; ot < Length; ++ot) {
          const int origin = ot * Stride - padding;
#pragma unroll
          for (int k = 0; k < Kernel; ++k) {
            const int index = origin + k;
            if (static_cast<unsigned>(index) < static_cast<unsigned>(Length))
              acc[ot] += in[index] * wk[k];
          }
        }
      } else {
#pragma unroll
        for (int ot = 0; ot < Length; ++ot) {
          const int origin = ot * Stride - padding;
#pragma unroll
          for (int k = 0; k < Kernel; ++k) {
            const int index = origin + k;
            if (static_cast<unsigned>(index) < static_cast<unsigned>(input_length))
              acc[ot] += __ldg(input + index) * wk[k];
          }
        }
      }
    } else {
#pragma unroll
      for (int ot = 0; ot < Length; ++ot) {
        const int origin = ot * Stride - padding;
#pragma unroll
        for (int k = 0; k < Kernel; ++k) {
          const int index = origin + k;
          if (static_cast<unsigned>(index) < static_cast<unsigned>(input_length))
            acc[ot] += __ldg(input + index) * wk[k];
        }
      }
    }
  }
  const int warps = ic_stride >> 5;
#pragma unroll
  for (int ot = 0; ot < Length; ++ot) {
    float v = acc[ot];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, offset);
    if (lane == 0)
      partial[(warp * Length) + ot] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const float shift = bias[oc];
#pragma unroll
    for (int ot = 0; ot < Length; ++ot) {
      float sum = shift;
      for (int w = 0; w < warps; ++w)
        sum += partial[(w * Length) + ot];
      y[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(Length)) + ot] = sum;
    }
  }
}

__global__ void conv1d_dense_kernel(const float* x, const float* weight, const float* bias,
                                    float* y, int in_channels, int out_channels, int input_length,
                                    int output_length, int kernel, int stride, int padding)
{
  const int ot = blockIdx.x * blockDim.x + threadIdx.x;
  const int oc = blockIdx.y * blockDim.y + threadIdx.y;
  if (ot >= output_length || oc >= out_channels)
    return;
  float sum = bias[oc];
  const int origin = ot * stride - padding;
  for (int ic = 0; ic < in_channels; ++ic) {
    const float* input =
        x + (static_cast<std::size_t>(ic) * static_cast<std::size_t>(input_length));
    const float* wr = weight + (static_cast<std::size_t>((oc * in_channels) + ic) *
                                static_cast<std::size_t>(kernel));
    for (int k = 0; k < kernel; ++k) {
      const int index = origin + k;
      if (static_cast<unsigned>(index) < static_cast<unsigned>(input_length))
        sum += input[index] * wr[k];
    }
  }
  y[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(output_length)) + ot] = sum;
}

__global__ void conv1d_f16_k5_l4_kernel(const float* x, const __half* weight, const float* bias,
                                        float* y, int in_channels, int out_channels)
{
  const int oc = blockIdx.x;
  if (oc >= out_channels)
    return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __shared__ float partial[16 * 4];
  float acc[4]{};
  for (int ic = threadIdx.x; ic < in_channels; ic += blockDim.x) {
    const float4 input4 = __ldg(reinterpret_cast<const float4*>(x + (ic * 4)));
    const float input[4] = {input4.x, input4.y, input4.z, input4.w};
    const __half* const row = weight + (((oc * in_channels) + ic) * 5);
    float kernel[5];
#pragma unroll
    for (int k = 0; k < 5; ++k)
      kernel[k] = __half2float(__ldg(row + k));
#pragma unroll
    for (int ot = 0; ot < 4; ++ot) {
#pragma unroll
      for (int k = 0; k < 5; ++k) {
        const int index = ot - 2 + k;
        if (static_cast<unsigned>(index) < 4u)
          acc[ot] += input[index] * kernel[k];
      }
    }
  }
#pragma unroll
  for (int ot = 0; ot < 4; ++ot) {
    float value = acc[ot];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0)
      partial[(warp * 4) + ot] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int ot = 0; ot < 4; ++ot) {
      float sum = bias[oc];
      for (int index = 0; index < (blockDim.x >> 5); ++index)
        sum += partial[(index * 4) + ot];
      y[(oc * 4) + ot] = sum;
    }
  }
}

template<int Kernel, int Stride>
__global__ void conv_transpose1d_fixed_kernel(const float* x, const float* weight,
                                              const float* bias, float* y, int in_channels,
                                              int out_channels, int input_length, int output_length,
                                              int padding, int k_major)
{
  const int ot = blockIdx.x * blockDim.x + threadIdx.x;
  const int oc = blockIdx.y * blockDim.y + threadIdx.y;
  if (ot >= output_length || oc >= out_channels)
    return;
  float sum = bias[oc];
  for (int ic = 0; ic < in_channels; ++ic) {
    const float* input =
        x + (static_cast<std::size_t>(ic) * static_cast<std::size_t>(input_length));
#pragma unroll
    for (int k = 0; k < Kernel; ++k) {
      const int shifted = ot + padding - k;
      if (shifted < 0 || (shifted % Stride) != 0)
        continue;
      const int it = shifted / Stride;
      if (it >= input_length)
        continue;
      const float wv = k_major ? weight[((((k * out_channels) + oc) * in_channels) + ic)]
                               : weight[((((ic * out_channels) + oc) * Kernel) + k)];
      sum += input[it] * wv;
    }
  }
  y[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(output_length)) + ot] = sum;
}

template<int Kernel, int Stride, int OutLength>
__global__ void conv_transpose1d_splitk_kernel(const float* x, const float* weight,
                                               const float* bias, float* y, int in_channels,
                                               int out_channels, int input_length, int padding,
                                               int k_major)
{
  const int oc = blockIdx.x;
  if (oc >= out_channels)
    return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ic_stride = blockDim.x;
  __shared__ float partial[16 * 16];
  float acc[OutLength];
#pragma unroll
  for (int ot = 0; ot < OutLength; ++ot)
    acc[ot] = 0.0f;
  for (int ic = threadIdx.x; ic < in_channels; ic += ic_stride) {
    const float* input =
        x + (static_cast<std::size_t>(ic) * static_cast<std::size_t>(input_length));
#pragma unroll
    for (int ot = 0; ot < OutLength; ++ot) {
#pragma unroll
      for (int k = 0; k < Kernel; ++k) {
        const int shifted = ot + padding - k;
        if (shifted < 0 || (shifted % Stride) != 0)
          continue;
        const int it = shifted / Stride;
        if (it >= input_length)
          continue;
        const float wv = k_major ? __ldg(weight + ((((k * out_channels) + oc) * in_channels) + ic))
                                 : __ldg(weight + ((((ic * out_channels) + oc) * Kernel) + k));
        acc[ot] += __ldg(input + it) * wv;
      }
    }
  }
  const int warps = ic_stride >> 5;
#pragma unroll
  for (int ot = 0; ot < OutLength; ++ot) {
    float v = acc[ot];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, offset);
    if (lane == 0)
      partial[(warp * OutLength) + ot] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const float shift = bias[oc];
#pragma unroll
    for (int ot = 0; ot < OutLength; ++ot) {
      float sum = shift;
      for (int w = 0; w < warps; ++w)
        sum += partial[(w * OutLength) + ot];
      y[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(OutLength)) + ot] = sum;
    }
  }
}

__global__ void conv_transpose1d_kernel(const float* x, const float* weight, const float* bias,
                                        float* y, int in_channels, int out_channels,
                                        int input_length, int output_length, int kernel, int stride,
                                        int padding, int k_major)
{
  const int ot = blockIdx.x * blockDim.x + threadIdx.x;
  const int oc = blockIdx.y * blockDim.y + threadIdx.y;
  if (ot >= output_length || oc >= out_channels)
    return;
  float sum = bias[oc];
  for (int ic = 0; ic < in_channels; ++ic) {
    const float* input =
        x + (static_cast<std::size_t>(ic) * static_cast<std::size_t>(input_length));
    for (int k = 0; k < kernel; ++k) {
      const int shifted = ot + padding - k;
      if (shifted < 0 || (shifted % stride) != 0)
        continue;
      const int it = shifted / stride;
      if (it >= input_length)
        continue;
      const float wv = k_major ? weight[((((k * out_channels) + oc) * in_channels) + ic)]
                               : weight[((((ic * out_channels) + oc) * kernel) + k)];
      sum += input[it] * wv;
    }
  }
  y[(static_cast<std::size_t>(oc) * static_cast<std::size_t>(output_length)) + ot] = sum;
}

// One block per group: PushT launches 8 groups with 1024 values each, so a single
// serial thread per group leaves the SM idle.
template<bool Mish = false>
__global__ void group_norm_kernel(float* x, const float* weight, const float* bias, int channels,
                                  int length, int groups, float epsilon,
                                  const float* modulation = nullptr,
                                  const float* residual = nullptr)
{
  const int group = static_cast<int>(blockIdx.x);
  if (group >= groups)
    return;
  const int channels_per_group = channels / groups;
  const int first = group * channels_per_group;
  const int group_values = channels_per_group * length;
  float* const block = x + (static_cast<std::size_t>(first) * static_cast<std::size_t>(length));
  double sum = 0.0;
  double square_sum = 0.0;
  for (int i = static_cast<int>(threadIdx.x); i < group_values; i += static_cast<int>(blockDim.x)) {
    const double value = static_cast<double>(block[i]);
    sum += value;
    square_sum += value * value;
  }
  __shared__ double s_sum[256];
  __shared__ double s_sq[256];
  s_sum[threadIdx.x] = sum;
  s_sq[threadIdx.x] = square_sum;
  __syncthreads();
  for (int off = static_cast<int>(blockDim.x) / 2; off > 0; off >>= 1) {
    if (static_cast<int>(threadIdx.x) < off) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + off];
      s_sq[threadIdx.x] += s_sq[threadIdx.x + off];
    }
    __syncthreads();
  }
  __shared__ float mean_f;
  __shared__ float inv;
  if (threadIdx.x == 0) {
    const double count = static_cast<double>(group_values);
    const double mean = s_sum[0] / count;
    const double variance = fmax(0.0, (s_sq[0] / count) - (mean * mean));
    inv = static_cast<float>(1.0 / sqrt(variance + static_cast<double>(epsilon)));
    mean_f = static_cast<float>(mean);
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < group_values; i += static_cast<int>(blockDim.x)) {
    const int local = i / length;
    const int channel = first + local;
    const float scale = inv * weight[channel];
    const float shift = bias[channel];
    float value = ((block[i] - mean_f) * scale) + shift;
    if constexpr (Mish) {
      value *= tanhf(fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value))));
      if (modulation != nullptr)
        value = modulation[channel] * value + modulation[channels + channel];
      if (residual != nullptr)
        value += residual[first * length + i];
    }
    block[i] = value;
  }
}

__global__ void film_kernel(float* x, const float* scale, const float* bias, int channels,
                            int length)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= length || c >= channels)
    return;
  const std::size_t i = (static_cast<std::size_t>(c) * static_cast<std::size_t>(length)) + t;
  x[i] = (scale[c] * x[i]) + bias[c];
}

__global__ void diffusion_timestep_kernel(float timestep, float* out, int half, float factor)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half)
    return;
  const float phase = timestep * expf(-factor * static_cast<float>(i));
  out[i] = sinf(phase);
  out[half + i] = cosf(phase);
}

__global__ void add_bias_kernel(float* x, const float* bias, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  x[i] += bias[i];
}

__global__ void add_inplace_kernel(float* x, const float* y, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += y[i];
}

__global__ void add_bias_rows_kernel(float* x, const float* bias, int rows, int dim)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = rows * dim;
  if (i >= n)
    return;
  x[i] += bias[i % dim];
}

__global__ void split_xz_kernel(const float* xz, float* x_cm, float* z, int length, int d_inner)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= length || c >= d_inner)
    return;
  const std::size_t row = static_cast<std::size_t>(t) * static_cast<std::size_t>(2 * d_inner);
  x_cm[(static_cast<std::size_t>(c) * static_cast<std::size_t>(length)) + t] = xz[row + c];
  z[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c] =
      xz[row + static_cast<std::size_t>(d_inner) + c];
}

__global__ void channel_to_seq_kernel(const float* x_cm, float* x_sm, int length, int d_inner)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= length || c >= d_inner)
    return;
  x_sm[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c] =
      x_cm[(static_cast<std::size_t>(c) * static_cast<std::size_t>(length)) + t];
}

__global__ void gather_prefix_kernel(const float* rows, float* out, int length, int row_stride,
                                     int width)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= length || k >= width)
    return;
  out[(static_cast<std::size_t>(t) * static_cast<std::size_t>(width)) + k] =
      rows[(static_cast<std::size_t>(t) * static_cast<std::size_t>(row_stride)) + k];
}

__global__ void conv_shift_push_kernel(float* window, const float* xz, float* z, int channels,
                                       int kernel)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  float* w = window + (static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel));
  for (int k = 0; k < kernel - 1; ++k)
    w[k] = w[k + 1];
  w[kernel - 1] = xz[c];
  z[c] = xz[channels + c];
}

__global__ void discretize_and_scan_kernel(const float* delta, const float* a_neg, const float* b,
                                           const float* u, const float* c_proj, const float* d_skip,
                                           float* h, float* y, int length, int d_inner, int d_state,
                                           int row_stride)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= d_inner)
    return;
  for (int t = 0; t < length; ++t) {
    const float dt = delta[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c];
    const float ut = u[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c];
    float yt = d_skip[c] * ut;
    const std::size_t brow = static_cast<std::size_t>(t) * static_cast<std::size_t>(row_stride);
    for (int n = 0; n < d_state; ++n) {
      const std::size_t hc = (static_cast<std::size_t>(n) * static_cast<std::size_t>(d_inner)) + c;
      const float da = expf(dt * a_neg[hc]);
      h[hc] = (da * h[hc]) + (dt * b[brow + n] * ut);
      yt += h[hc] * c_proj[brow + n];
    }
    y[(static_cast<std::size_t>(t) * static_cast<std::size_t>(d_inner)) + c] = yt;
  }
}

__global__ void layout_horizon_to_channel_kernel(const float* in, float* out, int horizon,
                                                 int action_dim)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= horizon || c >= action_dim)
    return;
  out[(static_cast<std::size_t>(c) * static_cast<std::size_t>(horizon)) + t] =
      in[(static_cast<std::size_t>(t) * static_cast<std::size_t>(action_dim)) + c];
}

__global__ void layout_channel_to_horizon_kernel(const float* in, float* out, int horizon,
                                                 int action_dim)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= horizon || c >= action_dim)
    return;
  out[(static_cast<std::size_t>(t) * static_cast<std::size_t>(action_dim)) + c] =
      in[(static_cast<std::size_t>(c) * static_cast<std::size_t>(horizon)) + t];
}

__global__ void ddim_update_kernel(float* x, const float* eps, int n, float sqrt_alpha_t,
                                   float sqrt_beta_t, float sqrt_alpha_prev,
                                   float sqrt_one_minus_prev, int clip, float clip_range)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float predicted_original = (x[i] - (sqrt_beta_t * eps[i])) / sqrt_alpha_t;
  if (clip)
    predicted_original = fminf(clip_range, fmaxf(-clip_range, predicted_original));
  x[i] = (sqrt_alpha_prev * predicted_original) + (sqrt_one_minus_prev * eps[i]);
}

__global__ void diffusion_unnormalize_kernel(const float* normalized, const float* action_min,
                                             const float* action_max, float* action,
                                             int action_start, int values, int action_dim)
{
  const int output_index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (output_index >= values)
    return;
  const int channel = output_index % action_dim;
  const int input_index = (action_start * action_dim) + output_index;
  action[output_index] =
      ((normalized[input_index] + 1.0F) * 0.5F * (action_max[channel] - action_min[channel])) +
      action_min[channel];
}

__global__ void ddpm_update_kernel(float* x, const float* eps, const float* noise, int n,
                                   float sqrt_alpha_t, float sqrt_beta_t,
                                   float original_coefficient, float sample_coefficient,
                                   float sqrt_variance, int clip, float clip_range)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float predicted_original = (x[i] - (sqrt_beta_t * eps[i])) / sqrt_alpha_t;
  if (clip)
    predicted_original = fminf(clip_range, fmaxf(-clip_range, predicted_original));
  float next = (original_coefficient * predicted_original) + (sample_coefficient * x[i]);
  if (noise != nullptr)
    next += sqrt_variance * noise[i];
  x[i] = next;
}

} // namespace

bool device_available() noexcept
{
  return ready();
}

void silu(float* x, std::size_t n) noexcept
{
  if (n == 0 || !ready() || !upload(g_a, x, n * sizeof(float)))
    return;
  silu_kernel<<<grid(static_cast<int>(n)), 256>>>(static_cast<float*>(g_a.ptr),
                                                  static_cast<int>(n));
  download(x, g_a.ptr, n * sizeof(float));
}

void mish(float* x, std::size_t n) noexcept
{
  if (n == 0 || !ready() || !upload(g_a, x, n * sizeof(float)))
    return;
  mish_kernel<<<grid(static_cast<int>(n)), 256>>>(static_cast<float*>(g_a.ptr),
                                                  static_cast<int>(n));
  download(x, g_a.ptr, n * sizeof(float));
}

void softplus(float* x, std::size_t n) noexcept
{
  if (n == 0 || !ready() || !upload(g_a, x, n * sizeof(float)))
    return;
  softplus_kernel<<<grid(static_cast<int>(n)), 256>>>(static_cast<float*>(g_a.ptr),
                                                      static_cast<int>(n));
  download(x, g_a.ptr, n * sizeof(float));
}

void gate_silu(const float* a, const float* g, float* out, std::size_t n) noexcept
{
  if (n == 0 || !ready() || !upload(g_a, a, n * sizeof(float)) ||
      !upload(g_b, g, n * sizeof(float)))
    return;
  void* dst = g_c.ensure(n * sizeof(float));
  if (dst == nullptr)
    return;
  gate_silu_kernel<<<grid(static_cast<int>(n)), 256>>>(static_cast<const float*>(g_a.ptr),
                                                       static_cast<const float*>(g_b.ptr),
                                                       static_cast<float*>(dst),
                                                       static_cast<int>(n));
  download(out, dst, n * sizeof(float));
}

void rmsnorm(const float* in, const float* weight, float* out, std::size_t rows,
             std::size_t dim) noexcept
{
  if (rows == 0 || dim == 0 || !ready())
    return;
  const std::size_t in_bytes = rows * dim * sizeof(float);
  if (!upload(g_a, in, in_bytes) || !upload(g_b, weight, dim * sizeof(float)))
    return;
  void* dst = g_c.ensure(in_bytes);
  if (dst == nullptr)
    return;
  rmsnorm_kernel<<<grid(static_cast<int>(rows)), 256>>>(static_cast<const float*>(g_a.ptr),
                                                        static_cast<const float*>(g_b.ptr),
                                                        static_cast<float*>(dst),
                                                        static_cast<int>(rows),
                                                        static_cast<int>(dim));
  download(out, dst, in_bytes);
}

void conv1d_causal(const float* x, const float* weight, const float* bias, float* y,
                   std::size_t channels, std::size_t length, std::size_t kernel) noexcept
{
  if (channels == 0 || length == 0 || kernel == 0 || !ready())
    return;
  if (!upload(g_a, x, channels * length * sizeof(float)) ||
      !upload(g_b, weight, channels * kernel * sizeof(float)) ||
      !upload(g_c, bias, channels * sizeof(float)))
    return;
  void* dst = g_d.ensure(channels * length * sizeof(float));
  if (dst == nullptr)
    return;
  conv1d_causal_kernel<<<grid(static_cast<int>(channels)), 256>>>(
      static_cast<const float*>(g_a.ptr), static_cast<const float*>(g_b.ptr),
      static_cast<const float*>(g_c.ptr), static_cast<float*>(dst), static_cast<int>(channels),
      static_cast<int>(length), static_cast<int>(kernel));
  download(y, dst, channels * length * sizeof(float));
}

void conv1d_step(const float* window, const float* weight, const float* bias, float* y,
                 std::size_t channels, std::size_t kernel) noexcept
{
  if (channels == 0 || kernel == 0 || !ready())
    return;
  if (!upload(g_a, window, channels * kernel * sizeof(float)) ||
      !upload(g_b, weight, channels * kernel * sizeof(float)) ||
      !upload(g_c, bias, channels * sizeof(float)))
    return;
  void* dst = g_d.ensure(channels * sizeof(float));
  if (dst == nullptr)
    return;
  conv1d_step_kernel<<<grid(static_cast<int>(channels)), 256>>>(static_cast<const float*>(g_a.ptr),
                                                                static_cast<const float*>(g_b.ptr),
                                                                static_cast<const float*>(g_c.ptr),
                                                                static_cast<float*>(dst),
                                                                static_cast<int>(channels),
                                                                static_cast<int>(kernel));
  download(y, dst, channels * sizeof(float));
}

void matmul_f32(const float* in, const float* w, float* out, std::size_t rows, std::size_t in_dim,
                std::size_t out_dim) noexcept
{
  if (rows == 0 || in_dim == 0 || out_dim == 0 || !ready())
    return;
  if (!upload(g_a, in, rows * in_dim * sizeof(float)) ||
      !upload(g_b, w, out_dim * in_dim * sizeof(float)))
    return;
  void* dst = g_c.ensure(rows * out_dim * sizeof(float));
  if (dst == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((out_dim + 15) / 16),
                static_cast<unsigned>((rows + 15) / 16));
  matmul_f32_kernel<<<grid_dim, block>>>(static_cast<const float*>(g_a.ptr),
                                         static_cast<const float*>(g_b.ptr),
                                         static_cast<float*>(dst), static_cast<int>(rows),
                                         static_cast<int>(in_dim), static_cast<int>(out_dim));
  download(out, dst, rows * out_dim * sizeof(float));
}

void matmul_bf16(const float* in, const std::uint16_t* w, float* out, std::size_t rows,
                 std::size_t in_dim, std::size_t out_dim) noexcept
{
  if (rows == 0 || in_dim == 0 || out_dim == 0 || !ready())
    return;
  if (!upload(g_a, in, rows * in_dim * sizeof(float)) ||
      !upload(g_b, w, out_dim * in_dim * sizeof(std::uint16_t)))
    return;
  void* dst = g_c.ensure(rows * out_dim * sizeof(float));
  if (dst == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((out_dim + 15) / 16),
                static_cast<unsigned>((rows + 15) / 16));
  matmul_bf16_kernel<<<grid_dim, block>>>(static_cast<const float*>(g_a.ptr),
                                          static_cast<const std::uint16_t*>(g_b.ptr),
                                          static_cast<float*>(dst), static_cast<int>(rows),
                                          static_cast<int>(in_dim), static_cast<int>(out_dim));
  download(out, dst, rows * out_dim * sizeof(float));
}

void discretize_and_scan(const float* delta, const float* a_neg, const float* b, const float* u,
                         const float* c_proj, const float* d_skip, float* h, float* y,
                         std::size_t length, std::size_t d_inner, std::size_t d_state,
                         bool reset_state, std::size_t row_stride) noexcept
{
  // Sequential in time. Host reference until a device-resident SSM lands.
  row_stride = row_stride == 0 ? d_state : row_stride;
  if (reset_state)
    std::fill(h, h + (d_inner * d_state), 0.0f);
  for (std::size_t t = 0; t < length; ++t) {
    const float* dt = delta + (t * d_inner);
    const float* ut = u + (t * d_inner);
    const float* bt = b + (t * row_stride);
    const float* ct = c_proj + (t * row_stride);
    float* yt = y + (t * d_inner);
    for (std::size_t c = 0; c < d_inner; ++c)
      yt[c] = d_skip[c] * ut[c];
    for (std::size_t n = 0; n < d_state; ++n) {
      float* hn = h + (n * d_inner);
      const float* an = a_neg + (n * d_inner);
      const float bn = bt[n];
      const float cn = ct[n];
      for (std::size_t c = 0; c < d_inner; ++c) {
        const float da = std::exp(dt[c] * an[c]);
        hn[c] = (da * hn[c]) + (dt[c] * bn * ut[c]);
        yt[c] += hn[c] * cn;
      }
    }
  }
}

std::uint64_t device_malloc_count() noexcept
{
  return g_mallocs.load(std::memory_order_relaxed);
}

void device_census_begin() noexcept
{
  g_census_used = 0;
  g_census_dropped = 0;
  g_census = true;
}

void device_census_end() noexcept
{
  g_census = false;
}

void device_census_print() noexcept
{
  CensusRow rows[kCensusCap];
  const int used = g_census_used;
  for (int i = 0; i < used; ++i)
    rows[i] = g_census_rows[i];
  std::sort(rows, rows + used, [](const CensusRow& lhs, const CensusRow& rhs) {
    if (lhs.n != rhs.n)
      return lhs.n > rhs.n;
    return std::strcmp(lhs.kind, rhs.kind) < 0;
  });
  unsigned total = 0;
  for (int i = 0; i < used; ++i)
    total += rows[i].n;
  std::printf("native DDIM launch census (1 sample)\n");
  for (int i = 0; i < used; ++i) {
    const CensusRow& row = rows[i];
    if (std::strcmp(row.kind, "conv") == 0)
      std::printf("  %-12s ic=%d oc=%d Lin=%d Lout=%d K=%d s=%d  n=%u\n", row.kind, row.a, row.b,
                  row.c, row.d, row.e, row.f, row.n);
    else if (std::strcmp(row.kind, "upsample") == 0)
      std::printf("  %-12s ic=%d oc=%d Lin=%d Lout=%d K=%d kmaj=%d  n=%u\n", row.kind, row.a, row.b,
                  row.c, row.d, row.e, row.f, row.n);
    else if (std::strcmp(row.kind, "group_norm") == 0)
      std::printf("  %-12s C=%d L=%d G=%d  n=%u\n", row.kind, row.a, row.b, row.c, row.n);
    else if (std::strcmp(row.kind, "gemm") == 0)
      std::printf("  %-12s rows=%d in=%d out=%d  n=%u\n", row.kind, row.a, row.b, row.c, row.n);
    else if (std::strcmp(row.kind, "film") == 0)
      std::printf("  %-12s C=%d L=%d  n=%u\n", row.kind, row.a, row.b, row.n);
    else
      std::printf("  %-12s a=%d b=%d c=%d  n=%u\n", row.kind, row.a, row.b, row.c, row.n);
  }
  std::printf("  total launches %u  unique %d  dropped %u\n", total, used, g_census_dropped);
}

void* device_alloc(std::size_t bytes) noexcept
{
  if (bytes == 0 || !ready())
    return nullptr;
  void* ptr = nullptr;
  if (cudaMalloc(&ptr, bytes) != cudaSuccess)
    return nullptr;
  g_mallocs.fetch_add(1, std::memory_order_relaxed);
  return ptr;
}

void device_free(void* ptr) noexcept
{
  if (ptr != nullptr)
    cudaFree(ptr);
}

bool device_copy_h2d(void* dst, const void* src, std::size_t bytes) noexcept
{
  return dst != nullptr && src != nullptr &&
         cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool device_copy_d2h(void* dst, const void* src, std::size_t bytes) noexcept
{
  return dst != nullptr && src != nullptr &&
         cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

void silu_device(float* x, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr)
    return;
  census_hit("silu", static_cast<int>(n));
  silu_kernel<<<grid(static_cast<int>(n)), 256>>>(x, static_cast<int>(n));
}

void matmul_f32_device(const float* in, const float* w, float* out, std::size_t rows,
                       std::size_t in_dim, std::size_t out_dim) noexcept
{
  if (rows == 0 || in_dim == 0 || out_dim == 0 || in == nullptr || w == nullptr || out == nullptr)
    return;
  census_hit("gemm", static_cast<int>(rows), static_cast<int>(in_dim), static_cast<int>(out_dim));
  if (rows == 1 && in_dim >= 128 && out_dim >= 128) {
    matmul_f32_warp_kernel<<<static_cast<unsigned>((out_dim + 7) / 8), 256>>>(
        in, w, out, static_cast<int>(in_dim), static_cast<int>(out_dim));
    return;
  }
  if (rows == 1) {
    matmul_f32_row_kernel<<<grid(static_cast<int>(out_dim)), 256>>>(in, w, out,
                                                                    static_cast<int>(in_dim),
                                                                    static_cast<int>(out_dim));
    return;
  }
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((out_dim + 15) / 16),
                static_cast<unsigned>((rows + 15) / 16));
  matmul_f32_kernel<<<grid_dim, block>>>(in, w, out, static_cast<int>(rows),
                                         static_cast<int>(in_dim), static_cast<int>(out_dim));
}

void linear_f32_device(const float* in, const float* weight, const float* bias, float* out,
                       std::size_t in_dim, std::size_t out_dim, bool mish) noexcept
{
  if (in_dim == 0 || out_dim == 0 || in == nullptr || weight == nullptr || bias == nullptr ||
      out == nullptr)
    return;
  census_hit(mish ? "linear_mish" : "linear", 1, static_cast<int>(in_dim),
             static_cast<int>(out_dim));
  if (in_dim >= 128 && out_dim >= 128) {
    linear_f32_warp_kernel<<<static_cast<unsigned>((out_dim + 7) / 8), 256>>>(
        in, weight, bias, out, static_cast<int>(in_dim), static_cast<int>(out_dim), mish ? 1 : 0);
    return;
  }
  linear_f32_row_kernel<<<grid(static_cast<int>(out_dim)), 256>>>(in, weight, bias, out,
                                                                  static_cast<int>(in_dim),
                                                                  static_cast<int>(out_dim),
                                                                  mish ? 1 : 0);
}

void add3_device(float* x, const float* y, const float* z, std::size_t n) noexcept
{
  if (n == 0)
    return;
  add3_kernel<<<grid(static_cast<int>(n)), 256>>>(x, y, z, static_cast<int>(n));
}

void add_scaled_device(float* x, const float* dx, float scale, std::size_t n) noexcept
{
  if (n == 0)
    return;
  census_hit("add_scaled", static_cast<int>(n));
  add_scaled_kernel<<<grid(static_cast<int>(n)), 256>>>(x, dx, scale, static_cast<int>(n));
}

void scaled_sum_device(const float* x, const float* dx, float scale, float* out,
                       std::size_t n) noexcept
{
  if (n == 0)
    return;
  scaled_sum_kernel<<<grid(static_cast<int>(n)), 256>>>(x, dx, scale, out, static_cast<int>(n));
}

void add_heun_device(float* x, const float* k1, const float* k2, float dt, std::size_t n) noexcept
{
  if (n == 0)
    return;
  add_heun_kernel<<<grid(static_cast<int>(n)), 256>>>(x, k1, k2, dt, static_cast<int>(n));
}

void add_rk4_device(float* x, const float* k1, const float* k2, const float* k3, const float* k4,
                    float dt, std::size_t n) noexcept
{
  if (n == 0)
    return;
  add_rk4_kernel<<<grid(static_cast<int>(n)), 256>>>(x, k1, k2, k3, k4, dt, static_cast<int>(n));
}

void time_embed_device(float t, const float* freqs, float* sinu, std::size_t half) noexcept
{
  if (half == 0)
    return;
  time_embed_kernel<<<grid(static_cast<int>(half)), 256>>>(t, freqs, sinu, static_cast<int>(half));
}

void conv1d(const float* x, const float* weight, const float* bias, float* y,
            std::size_t in_channels, std::size_t out_channels, std::size_t input_length,
            std::size_t output_length, std::size_t kernel, std::size_t stride,
            std::size_t padding) noexcept
{
  if (in_channels == 0 || out_channels == 0 || input_length == 0 || output_length == 0 ||
      kernel == 0 || stride == 0 || !ready())
    return;
  if (!upload(g_a, x, in_channels * input_length * sizeof(float)) ||
      !upload(g_b, weight, out_channels * in_channels * kernel * sizeof(float)) ||
      !upload(g_c, bias, out_channels * sizeof(float)))
    return;
  void* dst = g_d.ensure(out_channels * output_length * sizeof(float));
  if (dst == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((output_length + 15) / 16),
                static_cast<unsigned>((out_channels + 15) / 16));
  conv1d_dense_kernel<<<grid_dim, block>>>(static_cast<const float*>(g_a.ptr),
                                           static_cast<const float*>(g_b.ptr),
                                           static_cast<const float*>(g_c.ptr),
                                           static_cast<float*>(dst), static_cast<int>(in_channels),
                                           static_cast<int>(out_channels),
                                           static_cast<int>(input_length),
                                           static_cast<int>(output_length),
                                           static_cast<int>(kernel), static_cast<int>(stride),
                                           static_cast<int>(padding));
  download(y, dst, out_channels * output_length * sizeof(float));
}

void conv_transpose1d(const float* x, const float* weight, const float* bias, float* y,
                      std::size_t in_channels, std::size_t out_channels, std::size_t input_length,
                      std::size_t output_length, std::size_t kernel, std::size_t stride,
                      std::size_t padding, bool k_major_weights) noexcept
{
  if (in_channels == 0 || out_channels == 0 || input_length == 0 || output_length == 0 ||
      kernel == 0 || stride == 0 || !ready())
    return;
  if (!upload(g_a, x, in_channels * input_length * sizeof(float)) ||
      !upload(g_b, weight, in_channels * out_channels * kernel * sizeof(float)) ||
      !upload(g_c, bias, out_channels * sizeof(float)))
    return;
  void* dst = g_d.ensure(out_channels * output_length * sizeof(float));
  if (dst == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((output_length + 15) / 16),
                static_cast<unsigned>((out_channels + 15) / 16));
  conv_transpose1d_kernel<<<grid_dim, block>>>(static_cast<const float*>(g_a.ptr),
                                               static_cast<const float*>(g_b.ptr),
                                               static_cast<const float*>(g_c.ptr),
                                               static_cast<float*>(dst),
                                               static_cast<int>(in_channels),
                                               static_cast<int>(out_channels),
                                               static_cast<int>(input_length),
                                               static_cast<int>(output_length),
                                               static_cast<int>(kernel), static_cast<int>(stride),
                                               static_cast<int>(padding), k_major_weights ? 1 : 0);
  download(y, dst, out_channels * output_length * sizeof(float));
}

void group_norm(float* x, const float* weight, const float* bias, std::size_t channels,
                std::size_t length, std::size_t groups, float epsilon) noexcept
{
  if (channels == 0 || length == 0 || groups == 0 || channels % groups != 0 || !ready())
    return;
  if (!upload(g_a, x, channels * length * sizeof(float)) ||
      !upload(g_b, weight, channels * sizeof(float)) ||
      !upload(g_c, bias, channels * sizeof(float)))
    return;
  group_norm_kernel<><<<static_cast<unsigned>(groups), 256>>>(static_cast<float*>(g_a.ptr),
                                                              static_cast<const float*>(g_b.ptr),
                                                              static_cast<const float*>(g_c.ptr),
                                                              static_cast<int>(channels),
                                                              static_cast<int>(length),
                                                              static_cast<int>(groups), epsilon);
  download(x, g_a.ptr, channels * length * sizeof(float));
}

void film(float* x, const float* scale, const float* bias, std::size_t channels,
          std::size_t length) noexcept
{
  if (channels == 0 || length == 0 || !ready())
    return;
  if (!upload(g_a, x, channels * length * sizeof(float)) ||
      !upload(g_b, scale, channels * sizeof(float)) || !upload(g_c, bias, channels * sizeof(float)))
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((channels + 15) / 16));
  film_kernel<<<grid_dim, block>>>(static_cast<float*>(g_a.ptr), static_cast<const float*>(g_b.ptr),
                                   static_cast<const float*>(g_c.ptr), static_cast<int>(channels),
                                   static_cast<int>(length));
  download(x, g_a.ptr, channels * length * sizeof(float));
}

void diffusion_timestep_embedding(float timestep, float* out, std::size_t n) noexcept
{
  if (n == 0 || n % 2 != 0 || out == nullptr)
    return;
  const std::size_t half = n / 2;
  if (half == 1) {
    out[0] = std::sin(timestep);
    out[1] = std::cos(timestep);
    return;
  }
  if (!ready())
    return;
  void* dst = g_a.ensure(n * sizeof(float));
  if (dst == nullptr)
    return;
  const float factor = logf(10000.0f) / static_cast<float>(half - 1);
  diffusion_timestep_kernel<<<grid(static_cast<int>(half)), 256>>>(timestep,
                                                                   static_cast<float*>(dst),
                                                                   static_cast<int>(half), factor);
  download(out, dst, n * sizeof(float));
}

bool device_copy_d2d(void* dst, const void* src, std::size_t bytes) noexcept
{
  census_hit("d2d", static_cast<int>(bytes));
  return dst != nullptr && src != nullptr &&
         cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess;
}

bool device_synchronize() noexcept
{
  return cudaDeviceSynchronize() == cudaSuccess;
}

void mish_device(float* x, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr)
    return;
  census_hit("mish", static_cast<int>(n));
  mish_kernel<<<grid(static_cast<int>(n)), 256>>>(x, static_cast<int>(n));
}

void add_bias_device(float* x, const float* bias, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr || bias == nullptr)
    return;
  census_hit("add_bias", static_cast<int>(n));
  add_bias_kernel<<<grid(static_cast<int>(n)), 256>>>(x, bias, static_cast<int>(n));
}

void conv1d_device(const float* x, const float* weight, const float* bias, float* y,
                   std::size_t in_channels, std::size_t out_channels, std::size_t input_length,
                   std::size_t output_length, std::size_t kernel, std::size_t stride,
                   std::size_t padding) noexcept
{
  if (in_channels == 0 || out_channels == 0 || input_length == 0 || output_length == 0 ||
      kernel == 0 || stride == 0 || x == nullptr || weight == nullptr || bias == nullptr ||
      y == nullptr)
    return;
  census_hit("conv", static_cast<int>(in_channels), static_cast<int>(out_channels),
             static_cast<int>(input_length), static_cast<int>(output_length),
             static_cast<int>(kernel), static_cast<int>(stride));
  const int oc = static_cast<int>(out_channels);
  const int ic = static_cast<int>(in_channels);
  const int in_len = static_cast<int>(input_length);
  const int out_len = static_cast<int>(output_length);
  const int pad = static_cast<int>(padding);
  if (ic >= 32) {
    int workers = 32;
    if (ic >= 256)
      workers = 256;
    else if (ic >= 128)
      workers = 128;
    if (kernel == 5 && stride == 1 && output_length == 4) {
      conv1d_splitk_kernel<5, 1, 4><<<oc, workers>>>(x, weight, bias, y, ic, oc, in_len, pad);
      return;
    }
    if (kernel == 3 && stride == 2 && output_length == 4) {
      conv1d_splitk_kernel<3, 2, 4><<<oc, workers>>>(x, weight, bias, y, ic, oc, in_len, pad);
      return;
    }
    if (kernel == 1 && stride == 1 && output_length == 4) {
      conv1d_splitk_kernel<1, 1, 4><<<oc, workers>>>(x, weight, bias, y, ic, oc, in_len, pad);
      return;
    }
    if (kernel == 5 && stride == 1 && output_length == 8) {
      conv1d_splitk_kernel<5, 1, 8><<<oc, workers>>>(x, weight, bias, y, ic, oc, in_len, pad);
      return;
    }
    if (kernel == 1 && stride == 1 && output_length == 8) {
      conv1d_splitk_kernel<1, 1, 8><<<oc, workers>>>(x, weight, bias, y, ic, oc, in_len, pad);
      return;
    }
  }
  dim3 block;
  dim3 grid_dim;
  dense_conv_grid(out_len, oc, block, grid_dim);
  if (kernel == 5 && stride == 1)
    conv1d_dense_fixed_kernel<5, 1>
        <<<grid_dim, block>>>(x, weight, bias, y, ic, oc, in_len, out_len, pad);
  else if (kernel == 3 && stride == 2)
    conv1d_dense_fixed_kernel<3, 2>
        <<<grid_dim, block>>>(x, weight, bias, y, ic, oc, in_len, out_len, pad);
  else if (kernel == 1 && stride == 1)
    conv1d_dense_fixed_kernel<1, 1>
        <<<grid_dim, block>>>(x, weight, bias, y, ic, oc, in_len, out_len, pad);
  else
    conv1d_dense_kernel<<<grid_dim, block>>>(x, weight, bias, y, ic, oc, in_len, out_len,
                                             static_cast<int>(kernel), static_cast<int>(stride),
                                             pad);
}

void conv1d_f16_k5_l4_device(const float* x, const void* weight, const float* bias, float* y,
                             std::size_t in_channels, std::size_t out_channels) noexcept
{
  if (x == nullptr || weight == nullptr || bias == nullptr || y == nullptr || in_channels == 0 ||
      out_channels == 0)
    return;
  census_hit("conv_f16_k5_l4", static_cast<int>(in_channels), static_cast<int>(out_channels), 4, 4,
             5, 1);
  int workers = 32;
  if (in_channels >= 256)
    workers = 256;
  else if (in_channels >= 128)
    workers = 128;
  conv1d_f16_k5_l4_kernel<<<static_cast<int>(out_channels), workers>>>(
      x, static_cast<const __half*>(weight), bias, y, static_cast<int>(in_channels),
      static_cast<int>(out_channels));
}

void conv_transpose1d_device(const float* x, const float* weight, const float* bias, float* y,
                             std::size_t in_channels, std::size_t out_channels,
                             std::size_t input_length, std::size_t output_length,
                             std::size_t kernel, std::size_t stride, std::size_t padding,
                             bool k_major_weights) noexcept
{
  if (in_channels == 0 || out_channels == 0 || input_length == 0 || output_length == 0 ||
      kernel == 0 || stride == 0 || x == nullptr || weight == nullptr || bias == nullptr ||
      y == nullptr)
    return;
  census_hit("upsample", static_cast<int>(in_channels), static_cast<int>(out_channels),
             static_cast<int>(input_length), static_cast<int>(output_length),
             static_cast<int>(kernel), k_major_weights ? 1 : 0);
  const int oc = static_cast<int>(out_channels);
  const int ic = static_cast<int>(in_channels);
  if (k_major_weights && ic >= 32 && kernel == 4 && stride == 2 && output_length == 8) {
    int workers = 32;
    if (ic >= 256)
      workers = 256;
    else if (ic >= 128)
      workers = 128;
    conv_transpose1d_splitk_kernel<4, 2, 8>
        <<<oc, workers>>>(x, weight, bias, y, ic, oc, static_cast<int>(input_length),
                          static_cast<int>(padding), k_major_weights ? 1 : 0);
    return;
  }
  dim3 block;
  dim3 grid_dim;
  dense_conv_grid(static_cast<int>(output_length), oc, block, grid_dim);
  if (kernel == 4 && stride == 2)
    conv_transpose1d_fixed_kernel<4, 2>
        <<<grid_dim, block>>>(x, weight, bias, y, static_cast<int>(in_channels), oc,
                              static_cast<int>(input_length), static_cast<int>(output_length),
                              static_cast<int>(padding), k_major_weights ? 1 : 0);
  else
    conv_transpose1d_kernel<<<grid_dim, block>>>(x, weight, bias, y, static_cast<int>(in_channels),
                                                 oc, static_cast<int>(input_length),
                                                 static_cast<int>(output_length),
                                                 static_cast<int>(kernel), static_cast<int>(stride),
                                                 static_cast<int>(padding),
                                                 k_major_weights ? 1 : 0);
}

void group_norm_device(float* x, const float* weight, const float* bias, std::size_t channels,
                       std::size_t length, std::size_t groups, float epsilon) noexcept
{
  if (channels == 0 || length == 0 || groups == 0 || channels % groups != 0 || x == nullptr ||
      weight == nullptr || bias == nullptr)
    return;
  census_hit("group_norm", static_cast<int>(channels), static_cast<int>(length),
             static_cast<int>(groups));
  group_norm_kernel<><<<static_cast<unsigned>(groups), 256>>>(x, weight, bias,
                                                              static_cast<int>(channels),
                                                              static_cast<int>(length),
                                                              static_cast<int>(groups), epsilon);
}

void group_norm_mish_device(float* x, const float* weight, const float* bias, std::size_t channels,
                            std::size_t length, std::size_t groups, float epsilon,
                            const float* modulation, const float* residual) noexcept
{
  if (channels == 0 || length == 0 || groups == 0 || channels % groups != 0 || x == nullptr ||
      weight == nullptr || bias == nullptr)
    return;
  census_hit("norm_mish", static_cast<int>(channels), static_cast<int>(length),
             static_cast<int>(groups));
  group_norm_kernel<true>
      <<<static_cast<unsigned>(groups), 256>>>(x, weight, bias, static_cast<int>(channels),
                                               static_cast<int>(length), static_cast<int>(groups),
                                               epsilon, modulation, residual);
}

void film_device(float* x, const float* scale, const float* bias, std::size_t channels,
                 std::size_t length) noexcept
{
  if (channels == 0 || length == 0 || x == nullptr || scale == nullptr || bias == nullptr)
    return;
  census_hit("film", static_cast<int>(channels), static_cast<int>(length));
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((channels + 15) / 16));
  film_kernel<<<grid_dim, block>>>(x, scale, bias, static_cast<int>(channels),
                                   static_cast<int>(length));
}

void diffusion_timestep_device(float timestep, float* out, std::size_t n) noexcept
{
  if (n == 0 || n % 2 != 0 || out == nullptr)
    return;
  census_hit("timestep", static_cast<int>(n));
  const std::size_t half = n / 2;
  if (half == 1) {
    const float host[2] = {sinf(timestep), cosf(timestep)};
    static_cast<void>(device_copy_h2d(out, host, 2 * sizeof(float)));
    return;
  }
  const float factor = logf(10000.0f) / static_cast<float>(half - 1);
  diffusion_timestep_kernel<<<grid(static_cast<int>(half)), 256>>>(timestep, out,
                                                                   static_cast<int>(half), factor);
}

void layout_horizon_to_channel_device(const float* in, float* out, std::size_t horizon,
                                      std::size_t action_dim) noexcept
{
  if (horizon == 0 || action_dim == 0 || in == nullptr || out == nullptr)
    return;
  census_hit("layout_h2c", static_cast<int>(horizon), static_cast<int>(action_dim));
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((horizon + 15) / 16),
                static_cast<unsigned>((action_dim + 15) / 16));
  layout_horizon_to_channel_kernel<<<grid_dim, block>>>(in, out, static_cast<int>(horizon),
                                                        static_cast<int>(action_dim));
}

void layout_channel_to_horizon_device(const float* in, float* out, std::size_t horizon,
                                      std::size_t action_dim) noexcept
{
  if (horizon == 0 || action_dim == 0 || in == nullptr || out == nullptr)
    return;
  census_hit("layout_c2h", static_cast<int>(horizon), static_cast<int>(action_dim));
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((horizon + 15) / 16),
                static_cast<unsigned>((action_dim + 15) / 16));
  layout_channel_to_horizon_kernel<<<grid_dim, block>>>(in, out, static_cast<int>(horizon),
                                                        static_cast<int>(action_dim));
}

void ddim_update_device(float* x, const float* eps, std::size_t n, float sqrt_alpha_t,
                        float sqrt_beta_t, float sqrt_alpha_prev, float sqrt_one_minus_prev,
                        bool clip, float clip_range) noexcept
{
  if (n == 0 || x == nullptr || eps == nullptr)
    return;
  census_hit("ddim", static_cast<int>(n));
  ddim_update_kernel<<<grid(static_cast<int>(n)), 256>>>(x, eps, static_cast<int>(n), sqrt_alpha_t,
                                                         sqrt_beta_t, sqrt_alpha_prev,
                                                         sqrt_one_minus_prev, clip ? 1 : 0,
                                                         clip_range);
}

void ddpm_update_device(float* x, const float* eps, const float* noise, std::size_t n,
                        float sqrt_alpha_t, float sqrt_beta_t, float original_coefficient,
                        float sample_coefficient, float sqrt_variance, bool clip,
                        float clip_range) noexcept
{
  if (n == 0 || x == nullptr || eps == nullptr)
    return;
  ddpm_update_kernel<<<grid(static_cast<int>(n)), 256>>>(x, eps, noise, static_cast<int>(n),
                                                         sqrt_alpha_t, sqrt_beta_t,
                                                         original_coefficient, sample_coefficient,
                                                         sqrt_variance, clip ? 1 : 0, clip_range);
}

void diffusion_unnormalize_device(const float* normalized, const float* action_min,
                                  const float* action_max, float* action, std::size_t action_start,
                                  std::size_t action_steps, std::size_t action_dim) noexcept
{
  const std::size_t values = action_steps * action_dim;
  if (normalized == nullptr || action_min == nullptr || action_max == nullptr ||
      action == nullptr || values == 0 ||
      values > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      action_start > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      action_dim > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return;
  census_hit("unnormalize", static_cast<int>(action_steps), static_cast<int>(action_dim));
  diffusion_unnormalize_kernel<<<grid(static_cast<int>(values)), 256>>>(
      normalized, action_min, action_max, action, static_cast<int>(action_start),
      static_cast<int>(values), static_cast<int>(action_dim));
}

void rmsnorm_device(const float* in, const float* weight, float* out, std::size_t rows,
                    std::size_t dim) noexcept
{
  if (rows == 0 || dim == 0 || in == nullptr || weight == nullptr || out == nullptr)
    return;
  rmsnorm_kernel<<<grid(static_cast<int>(rows)), 256>>>(in, weight, out, static_cast<int>(rows),
                                                        static_cast<int>(dim));
}

void conv1d_causal_device(const float* x, const float* weight, const float* bias, float* y,
                          std::size_t channels, std::size_t length, std::size_t kernel) noexcept
{
  if (channels == 0 || length == 0 || kernel == 0 || x == nullptr || weight == nullptr ||
      bias == nullptr || y == nullptr)
    return;
  conv1d_causal_kernel<<<grid(static_cast<int>(channels)), 256>>>(x, weight, bias, y,
                                                                  static_cast<int>(channels),
                                                                  static_cast<int>(length),
                                                                  static_cast<int>(kernel));
}

void conv1d_step_device(const float* window, const float* weight, const float* bias, float* y,
                        std::size_t channels, std::size_t kernel) noexcept
{
  if (channels == 0 || kernel == 0 || window == nullptr || weight == nullptr || bias == nullptr ||
      y == nullptr)
    return;
  conv1d_step_kernel<<<grid(static_cast<int>(channels)), 256>>>(window, weight, bias, y,
                                                                static_cast<int>(channels),
                                                                static_cast<int>(kernel));
}

void softplus_device(float* x, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr)
    return;
  softplus_kernel<<<grid(static_cast<int>(n)), 256>>>(x, static_cast<int>(n));
}

void gate_silu_device(const float* a, const float* g, float* out, std::size_t n) noexcept
{
  if (n == 0 || a == nullptr || g == nullptr || out == nullptr)
    return;
  gate_silu_kernel<<<grid(static_cast<int>(n)), 256>>>(a, g, out, static_cast<int>(n));
}

void add_inplace_device(float* x, const float* y, std::size_t n) noexcept
{
  if (n == 0 || x == nullptr || y == nullptr)
    return;
  add_inplace_kernel<<<grid(static_cast<int>(n)), 256>>>(x, y, static_cast<int>(n));
}

void add_bias_rows_device(float* x, const float* bias, std::size_t rows, std::size_t dim) noexcept
{
  if (rows == 0 || dim == 0 || x == nullptr || bias == nullptr)
    return;
  add_bias_rows_kernel<<<grid(static_cast<int>(rows * dim)), 256>>>(x, bias, static_cast<int>(rows),
                                                                    static_cast<int>(dim));
}

void split_xz_device(const float* xz, float* x_cm, float* z, std::size_t length,
                     std::size_t d_inner) noexcept
{
  if (length == 0 || d_inner == 0 || xz == nullptr || x_cm == nullptr || z == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((d_inner + 15) / 16));
  split_xz_kernel<<<grid_dim, block>>>(xz, x_cm, z, static_cast<int>(length),
                                       static_cast<int>(d_inner));
}

void channel_to_seq_device(const float* x_cm, float* x_sm, std::size_t length,
                           std::size_t d_inner) noexcept
{
  if (length == 0 || d_inner == 0 || x_cm == nullptr || x_sm == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((d_inner + 15) / 16));
  channel_to_seq_kernel<<<grid_dim, block>>>(x_cm, x_sm, static_cast<int>(length),
                                             static_cast<int>(d_inner));
}

void gather_prefix_device(const float* rows, float* out, std::size_t length, std::size_t row_stride,
                          std::size_t width) noexcept
{
  if (length == 0 || width == 0 || rows == nullptr || out == nullptr)
    return;
  dim3 block(16, 16);
  dim3 grid_dim(static_cast<unsigned>((length + 15) / 16),
                static_cast<unsigned>((width + 15) / 16));
  gather_prefix_kernel<<<grid_dim, block>>>(rows, out, static_cast<int>(length),
                                            static_cast<int>(row_stride), static_cast<int>(width));
}

void conv_shift_push_device(float* window, const float* xz, float* z, std::size_t channels,
                            std::size_t kernel) noexcept
{
  if (channels == 0 || kernel == 0 || window == nullptr || xz == nullptr || z == nullptr)
    return;
  conv_shift_push_kernel<<<grid(static_cast<int>(channels)), 256>>>(window, xz, z,
                                                                    static_cast<int>(channels),
                                                                    static_cast<int>(kernel));
}

void discretize_and_scan_device(const float* delta, const float* a_neg, const float* b,
                                const float* u, const float* c_proj, const float* d_skip, float* h,
                                float* y, std::size_t length, std::size_t d_inner,
                                std::size_t d_state, bool reset_state,
                                std::size_t row_stride) noexcept
{
  if (length == 0 || d_inner == 0 || d_state == 0 || delta == nullptr || a_neg == nullptr ||
      b == nullptr || u == nullptr || c_proj == nullptr || d_skip == nullptr || h == nullptr ||
      y == nullptr)
    return;
  row_stride = row_stride == 0 ? d_state : row_stride;
  if (reset_state)
    static_cast<void>(cudaMemset(h, 0, d_inner * d_state * sizeof(float)));
  discretize_and_scan_kernel<<<grid(static_cast<int>(d_inner)), 256>>>(
      delta, a_neg, b, u, c_proj, d_skip, h, y, static_cast<int>(length), static_cast<int>(d_inner),
      static_cast<int>(d_state), static_cast<int>(row_stride));
}

} // namespace fe::cuda_ops
