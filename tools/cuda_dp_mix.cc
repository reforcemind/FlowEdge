#include "api/engine.h"
#include "kernels/cuda/kernels_cuda_api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace {

struct MixRow
{
  std::string name;
  double p50_us;
  int calls;
};

std::vector<MixRow> g_rows;

struct DeviceBuffer
{
  float* ptr{};

  explicit DeviceBuffer(std::size_t n, float base = 0.03F)
  {
    if (n == 0uz)
      return;
    std::vector<float> host(n);
    for (std::size_t i{0uz}; i < n; ++i)
      host[i] = base * static_cast<float>(static_cast<int>(i % 17uz) - 8);
    ptr = static_cast<float*>(fe::cuda_ops::device_alloc(n * sizeof(float)));
    if (ptr != nullptr && !fe::cuda_ops::device_copy_h2d(ptr, host.data(), n * sizeof(float))) {
      fe::cuda_ops::device_free(ptr);
      ptr = nullptr;
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() { fe::cuda_ops::device_free(ptr); }

  [[nodiscard]] bool ok() const noexcept { return ptr != nullptr; }
};

template<typename Fn> double median_us(Fn fn, int warmup, int iters)
{
  for (int i{0}; i < warmup; ++i)
    fn();
  if (cudaDeviceSynchronize() != cudaSuccess)
    return 0.0;
  cudaEvent_t start{};
  cudaEvent_t stop{};
  if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
    return 0.0;
  std::vector<double> samples(static_cast<std::size_t>(iters));
  for (int i{0}; i < iters; ++i) {
    cudaEventRecord(start);
    fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms{};
    cudaEventElapsedTime(&ms, start, stop);
    samples[static_cast<std::size_t>(i)] = static_cast<double>(ms) * 1000.0;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::ranges::nth_element(samples, samples.begin() + samples.size() / 2uz);
  return samples[samples.size() / 2uz];
}

void report(const char* name, double us, int calls_per_sample)
{
  g_rows.push_back(MixRow{name, us, calls_per_sample});
  const double ms = (us * static_cast<double>(calls_per_sample)) / 1000.0;
  std::printf("%-36s %8.1f us  x%-4d  %7.1f ms/sample\n", name, us, calls_per_sample, ms);
}

const char* json_kind(const std::string& name)
{
  if (name.rfind("native ", 0uz) == 0uz)
    return "native";
  if (name.rfind("gemm ", 0uz) == 0uz)
    return "gemm";
  if (name.rfind("upsample ", 0uz) == 0uz)
    return "upsample";
  if (name.rfind("group_norm", 0uz) == 0uz || name.rfind("mish", 0uz) == 0uz ||
      name.rfind("norm+mish", 0uz) == 0uz)
    return "elementwise";
  return "conv";
}

void write_json(const char* path, const char* checkpoint)
{
  FILE* const file = std::fopen(path, "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "could not write %s\n", path);
    return;
  }
  const auto escape = [](const char* text) {
    std::string out;
    for (const char* p{text}; *p != '\0'; ++p) {
      if (*p == '\\' || *p == '"')
        out.push_back('\\');
      out.push_back(*p);
    }
    return out;
  };
  std::fprintf(file, "{\n");
  std::fprintf(file, "  \"tool\": \"flowedge_cuda_dp_mix\",\n");
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
    std::fprintf(file, "  \"device\": \"%s\",\n", escape(properties.name).c_str());
    std::fprintf(file, "  \"memory_bus_bits\": %d,\n", properties.memoryBusWidth);
    std::fprintf(file, "  \"memory_clock_khz\": %d,\n", properties.memoryClockRate);
    std::fprintf(file, "  \"theoretical_memory_gb_s\": %.3f,\n",
                 2.0 * properties.memoryClockRate * properties.memoryBusWidth / 8.0 / 1e6);
  }
  if (checkpoint != nullptr) {
    const std::string escaped = escape(checkpoint);
    std::fprintf(file, "  \"checkpoint\": \"%s\",\n", escaped.c_str());
  } else {
    std::fprintf(file, "  \"checkpoint\": null,\n");
  }
  std::fprintf(file, "  \"rows\": [\n");
  for (std::size_t i{0uz}; i < g_rows.size(); ++i) {
    const MixRow& row = g_rows[i];
    const double estimated_ms = (row.p50_us * static_cast<double>(row.calls)) / 1000.0;
    std::fprintf(file,
                 "    {\"name\": \"%s\", \"p50_us\": %.3f, \"calls\": %d, \"estimated_ms\": %.3f, "
                 "\"kind\": \"%s\"}%s\n",
                 row.name.c_str(), row.p50_us, row.calls, estimated_ms, json_kind(row.name),
                 i + 1uz == g_rows.size() ? "" : ",");
  }
  std::fprintf(file, "  ]\n}\n");
  std::fclose(file);
  std::printf("wrote %s (%zu rows)\n", path, g_rows.size());
}

} // namespace

int main(int argc, char** argv)
{
  if (!fe::cuda_ops::device_available()) {
    std::fprintf(stderr, "CUDA device not available\n");
    return 2;
  }

  const char* checkpoint = nullptr;
  const char* json_path = nullptr;
  for (int i{1}; i < argc; ++i) {
    const char* const arg = argv[i];
    const std::size_t n = std::strlen(arg);
    if (n >= 5uz && std::strcmp(arg + (n - 5uz), ".json") == 0)
      json_path = arg;
    else
      checkpoint = arg;
  }

  std::printf("Diffusion Policy CUDA bottleneck mix (10 DDIM steps, batch=1)\n");
  std::printf("%-36s %11s  %-5s  %s\n", "kernel", "p50", "n", "estimated");

  auto conv = [](std::size_t in_channels, std::size_t out_channels, std::size_t length,
                 std::size_t kernel, std::size_t stride, std::size_t padding,
                 std::size_t output_length) {
    DeviceBuffer x{in_channels * length};
    DeviceBuffer w{out_channels * in_channels * kernel, 0.001F};
    DeviceBuffer b{out_channels};
    DeviceBuffer y{out_channels * output_length};
    if (!x.ok() || !w.ok() || !b.ok() || !y.ok())
      return 0.0;
    return median_us(
        [&] {
          fe::cuda_ops::conv1d_device(x.ptr, w.ptr, b.ptr, y.ptr, in_channels, out_channels, length,
                                      output_length, kernel, stride, padding);
        },
        4, 12);
  };

  report("conv 512 L16 K5", conv(512uz, 512uz, 16uz, 5uz, 1uz, 2uz, 16uz), 40);
  report("conv 512 L8 K5", conv(512uz, 512uz, 8uz, 5uz, 1uz, 2uz, 8uz), 30);
  report("conv 1024 L8 K5", conv(1024uz, 1024uz, 8uz, 5uz, 1uz, 2uz, 8uz), 30);
  report("conv 1024 L4 K5", conv(1024uz, 1024uz, 4uz, 5uz, 1uz, 2uz, 4uz), 30);
  report("conv 2048 L4 K5", conv(2048uz, 2048uz, 4uz, 5uz, 1uz, 2uz, 4uz), 70);
  report("conv 4096->1024 L4 K5", conv(4096uz, 1024uz, 4uz, 5uz, 1uz, 2uz, 4uz), 10);
  report("downsample 512 L16 K3", conv(512uz, 512uz, 16uz, 3uz, 2uz, 1uz, 8uz), 10);
  report("downsample 1024 L8 K3", conv(1024uz, 1024uz, 8uz, 3uz, 2uz, 1uz, 4uz), 10);

  auto upsample = [](std::size_t channels, std::size_t input_length, std::size_t output_length,
                     bool k_major) {
    constexpr std::size_t kernel = 4uz;
    DeviceBuffer x{channels * input_length};
    DeviceBuffer w{channels * channels * kernel, 0.001F};
    DeviceBuffer b{channels};
    DeviceBuffer y{channels * output_length};
    if (!x.ok() || !w.ok() || !b.ok() || !y.ok())
      return 0.0;
    return median_us(
        [&] {
          fe::cuda_ops::conv_transpose1d_device(x.ptr, w.ptr, b.ptr, y.ptr, channels, channels,
                                                input_length, output_length, kernel, 2uz, 1uz,
                                                k_major);
        },
        4, 8);
  };
  report("upsample 1024 L4->8 K4", upsample(1024uz, 4uz, 8uz, false), 10);
  report("upsample 512 L8->16 K4", upsample(512uz, 8uz, 16uz, false), 10);
  report("upsample 1024 k-major", upsample(1024uz, 4uz, 8uz, true), 10);
  report("upsample 512 k-major", upsample(512uz, 8uz, 16uz, true), 10);

  auto gnorm = [](std::size_t channels, std::size_t length) {
    DeviceBuffer x{channels * length};
    DeviceBuffer w{channels, 1.0F};
    DeviceBuffer b{channels, 0.01F};
    if (!x.ok() || !w.ok() || !b.ok())
      return 0.0;
    return median_us(
        [&] {
          fe::cuda_ops::group_norm_device(x.ptr, w.ptr, b.ptr, channels, length, 8uz, 1.0e-5F);
        },
        8, 40);
  };
  report("group_norm 2048 L4", gnorm(2048uz, 4uz), 80);
  report("group_norm 512 L16", gnorm(512uz, 16uz), 50);
  report("group_norm 1024 L8", gnorm(1024uz, 8uz), 40);
  for (bool fused : {false, true}) {
    DeviceBuffer x{8192uz}, w{2048uz, 1.0F}, b{2048uz, 0.01F}, residual{8192uz};
    if (!x.ok() || !w.ok() || !b.ok() || !residual.ok()) {
      std::fprintf(stderr, "norm/mish benchmark allocation failed\n");
      return 3;
    }
    report(fused ? "norm+mish+residual fused" : "norm+mish+residual separate",
           median_us(
               [&] {
                 if (fused)
                   fe::cuda_ops::group_norm_mish_device(x.ptr, w.ptr, b.ptr, 2048uz, 4uz, 8uz,
                                                        1.0e-5F, nullptr, residual.ptr);
                 else {
                   fe::cuda_ops::group_norm_device(x.ptr, w.ptr, b.ptr, 2048uz, 4uz, 8uz, 1.0e-5F);
                   fe::cuda_ops::mish_device(x.ptr, 8192uz);
                   fe::cuda_ops::add_scaled_device(x.ptr, residual.ptr, 1.0F, 8192uz);
                 }
               },
               8, 40),
           120);
  }
  report(
      "mish 8192",
      [&] {
        DeviceBuffer x{8192uz};
        if (!x.ok())
          return 0.0;
        return median_us([&] { fe::cuda_ops::mish_device(x.ptr, 8192uz); }, 8, 40);
      }(),
      170);
  auto gemm = [](std::size_t in_dim, std::size_t out_dim) {
    DeviceBuffer in{in_dim};
    DeviceBuffer w{out_dim * in_dim};
    DeviceBuffer out{out_dim};
    if (!in.ok() || !w.ok() || !out.ok())
      return 0.0;
    return median_us(
        [&] { fe::cuda_ops::matmul_f32_device(in.ptr, w.ptr, out.ptr, 1uz, in_dim, out_dim); }, 8,
        24);
  };
  report("gemm 1x260x1024", gemm(260uz, 1024uz), 40);
  report("gemm 1x260x2048", gemm(260uz, 2048uz), 40);
  report("gemm 1x260x4096", gemm(260uz, 4096uz), 40);
  report("gemm 1x128x512", gemm(128uz, 512uz), 10);
  report("gemm 1x512x128", gemm(512uz, 128uz), 10);

  if (checkpoint != nullptr) {
    fe_engine* const engine = fe_engine_load_with_threads(checkpoint, 0u);
    if (engine == nullptr) {
      std::fprintf(stderr, "load failed: %s\n", fe_engine_last_error());
      return 3;
    }
    if (!fe_engine_cuda_resident(engine)) {
      std::fprintf(stderr,
                   "CUDA benchmark requires device-resident weights; CPU fallback rejected\n");
      fe_engine_free(engine);
      return 4;
    }
    std::vector<float> condition(fe_engine_condition_dim(engine), 0.1F);
    std::vector<float> noise(fe_engine_action_horizon(engine) * fe_engine_action_dim(engine));
    for (std::size_t i{0uz}; i < noise.size(); ++i)
      noise[i] = std::sin(0.37F * static_cast<float>(i + 1uz));
    std::vector<float> action(noise.size());
    const double sample_us = median_us(
        [&] {
          if (fe_engine_sample_diffusion(engine, condition.data(), noise.data(), 10uz,
                                         FE_DIFFUSION_DDIM, 0u, action.data()) != 0)
            std::abort();
        },
        2, 6);
    report("native DDIM x10", sample_us, 1);
    fe::cuda_ops::device_census_begin();
    if (fe_engine_sample_diffusion(engine, condition.data(), noise.data(), 10uz, FE_DIFFUSION_DDIM,
                                   0u, action.data()) != 0)
      std::abort();
    fe::cuda_ops::device_census_end();
    fe::cuda_ops::device_census_print();
    fe_engine_free(engine);
  }

  if (json_path != nullptr)
    write_json(json_path, checkpoint);
  return 0;
}
