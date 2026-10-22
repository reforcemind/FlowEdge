#include "kernels/kernels.h"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

// Mamba-130M shapes.
constexpr std::size_t kSeq = 4uz;
constexpr std::size_t kDModel = 768uz;
constexpr std::size_t kDInner = 1536uz;
constexpr std::size_t kDState = 16uz;
constexpr std::size_t kDConv = 4uz;
constexpr std::size_t kDtRank = 48uz;
constexpr std::size_t kProjW = kDtRank + (2uz * kDState); // x_proj output width

std::vector<float> filled(std::size_t n, float base = 0.03F)
{
  std::vector<float> v(n);
  for (std::size_t i{0uz}; i < n; ++i)
    v[i] = base * static_cast<float>(static_cast<int>(i % 17uz) - 8); // small, mixed-sign
  return v;
}

using benchmark::Counter;

struct BenchmarkPool
{
  explicit BenchmarkPool(unsigned threads)
      : ring(8uz), sequence(8uz), workers(threads), pool(ring, sequence, workers, threads)
  {
  }

  std::vector<fe::Task> ring;
  std::vector<std::size_t> sequence;
  std::vector<std::jthread> workers;
  fe::ThreadPool pool;
};

// out[r,o] = Σ_i in[r,i]·w[o,i]
// FLOP/s and streamed-weight bytes/s.
void run_matmul(benchmark::State& state, std::size_t rows, std::size_t in_dim, std::size_t out_dim)
{
  const std::vector<float> in = filled(rows * in_dim);
  const std::vector<float> w = filled(out_dim * in_dim);
  std::vector<float> out(rows * out_dim);
  for (auto _ : state) {
    fe::matmul(in, w, out, rows, in_dim, out_dim);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  const double flops = 2.0 * static_cast<double>(rows * in_dim * out_dim);
  state.counters["FLOP/s"] = Counter(flops, Counter::kIsIterationInvariantRate);
  state.counters["weight-B/s"] = Counter(static_cast<double>(out_dim * in_dim * sizeof(float)),
                                         Counter::kIsIterationInvariantRate);
}

void BM_matmul_in_proj(benchmark::State& state)
{
  run_matmul(state, kSeq, kDModel, 2uz * kDInner);
}
void BM_matmul_out_proj(benchmark::State& state)
{
  run_matmul(state, kSeq, kDInner, kDModel);
}
void BM_matmul_x_proj(benchmark::State& state)
{
  run_matmul(state, kSeq, kDInner, kProjW);
}
void BM_matmul_dt_proj(benchmark::State& state)
{
  run_matmul(state, kSeq, kDtRank, kDInner);
}
void BM_matmul_dp_conv_l4(benchmark::State& state)
{
  run_matmul(state, 4uz, 2048uz * 5uz, 2048uz);
}
void BM_matmul_dp_conv_l16(benchmark::State& state)
{
  run_matmul(state, 16uz, 512uz * 5uz, 512uz);
}

void BM_silu(benchmark::State& state)
{
  std::vector<float> x = filled(kSeq * kDInner);
  for (auto _ : state) {
    fe::silu(x);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * x.size()));
}

void BM_softplus(benchmark::State& state)
{
  std::vector<float> x = filled(kSeq * kDInner);
  for (auto _ : state) {
    fe::softplus(x);
    benchmark::DoNotOptimize(x.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * x.size()));
}

void BM_gate_silu(benchmark::State& state)
{
  const std::vector<float> a = filled(kSeq * kDInner);
  const std::vector<float> g = filled(kSeq * kDInner, 0.02F);
  std::vector<float> out(kSeq * kDInner);
  for (auto _ : state) {
    fe::gate_silu(a, g, out);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * out.size()));
}

void BM_rmsnorm(benchmark::State& state)
{
  const std::vector<float> in = filled(kSeq * kDModel);
  const std::vector<float> w = filled(kDModel, 1.0F);
  std::vector<float> out(kSeq * kDModel);
  for (auto _ : state) {
    fe::rmsnorm(in, w, out, kSeq, kDModel);
    benchmark::DoNotOptimize(out.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * out.size()));
}

void BM_conv1d_causal(benchmark::State& state)
{
  const std::vector<float> x = filled(kDInner * kSeq);
  const std::vector<float> w = filled(kDInner * kDConv);
  const std::vector<float> b = filled(kDInner);
  std::vector<float> y(kDInner * kSeq);
  for (auto _ : state) {
    fe::conv1d_causal(x, w, b, y, kDInner, kSeq, kDConv);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * y.size()));
}

void run_dense_conv1d(benchmark::State& state, std::size_t channels, std::size_t input_length,
                      std::size_t output_length, std::size_t kernel, std::size_t stride,
                      std::size_t padding, fe::ThreadPool* pool = nullptr)
{
  const std::vector<float> x = filled(channels * input_length);
  const std::vector<float> w = filled(channels * channels * kernel, 0.001F);
  const std::vector<float> b = filled(channels);
  std::vector<float> y(channels * output_length);
  std::vector<float> workspace(
      fe::conv1d_workspace_floats(channels, channels, output_length, kernel));
  for (auto _ : state) {
    fe::conv1d(x, w, b, y, channels, channels, input_length, output_length, kernel, stride, padding,
               pool, workspace);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  const double macs = static_cast<double>(channels) * static_cast<double>(channels) *
                      static_cast<double>(kernel) * static_cast<double>(output_length);
  state.counters["MAC/s"] = Counter(macs, Counter::kIsIterationInvariantRate);
}

void BM_diffusion_conv_512_l16(benchmark::State& state)
{
  run_dense_conv1d(state, 512uz, 16uz, 16uz, 5uz, 1uz, 2uz);
}

void BM_diffusion_conv_1024_l8(benchmark::State& state)
{
  run_dense_conv1d(state, 1024uz, 8uz, 8uz, 5uz, 1uz, 2uz);
}

void BM_diffusion_conv_2048_l4(benchmark::State& state)
{
  run_dense_conv1d(state, 2048uz, 4uz, 4uz, 5uz, 1uz, 2uz);
}

void BM_diffusion_conv_512_l16_threads4(benchmark::State& state)
{
  BenchmarkPool pool{4u};
  run_dense_conv1d(state, 512uz, 16uz, 16uz, 5uz, 1uz, 2uz, &pool.pool);
}

void BM_diffusion_conv_1024_l8_threads4(benchmark::State& state)
{
  BenchmarkPool pool{4u};
  run_dense_conv1d(state, 1024uz, 8uz, 8uz, 5uz, 1uz, 2uz, &pool.pool);
}

void BM_diffusion_conv_2048_l4_threads4(benchmark::State& state)
{
  BenchmarkPool pool{4u};
  run_dense_conv1d(state, 2048uz, 4uz, 4uz, 5uz, 1uz, 2uz, &pool.pool);
}

void BM_diffusion_downsample_512_l16(benchmark::State& state)
{
  run_dense_conv1d(state, 512uz, 16uz, 8uz, 3uz, 2uz, 1uz);
}

void BM_diffusion_upsample_512_l8(benchmark::State& state)
{
  constexpr std::size_t channels = 512uz;
  constexpr std::size_t input_length = 8uz;
  constexpr std::size_t output_length = 16uz;
  constexpr std::size_t kernel = 4uz;
  const std::vector<float> x = filled(channels * input_length);
  const std::vector<float> w = filled(channels * channels * kernel, 0.001F);
  const std::vector<float> b = filled(channels);
  std::vector<float> y(channels * output_length);
  std::vector<float> workspace(
      fe::conv_transpose1d_workspace_floats(channels, channels, input_length, 4uz));
  for (auto _ : state) {
    fe::conv_transpose1d(x, w, b, y, channels, channels, input_length, output_length, kernel, 2uz,
                         1uz, nullptr, workspace);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  constexpr double macs = static_cast<double>(channels * channels * kernel * input_length);
  state.counters["MAC/s"] = Counter(macs, Counter::kIsIterationInvariantRate);
}

void BM_discretize_and_scan(benchmark::State& state)
{
  const std::vector<float> dt = filled(kSeq * kDInner, 0.01F);
  const std::vector<float> a_neg = filled(kDInner * kDState, -1.0F);
  const std::vector<float> b = filled(kSeq * kDState);
  const std::vector<float> u = filled(kSeq * kDInner);
  const std::vector<float> c = filled(kSeq * kDState);
  const std::vector<float> d = filled(kDInner);
  std::vector<float> h(kDState * kDInner);
  std::vector<float> y(kSeq * kDInner);
  for (auto _ : state) {
    fe::discretize_and_scan(dt, a_neg, b, u, c, d, h, y, kSeq, kDInner, kDState, true);
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * y.size()));
}

} // namespace

BENCHMARK(BM_matmul_in_proj)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_out_proj)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_x_proj)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_dt_proj)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_matmul_dp_conv_l4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_matmul_dp_conv_l16)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_silu)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_softplus)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_gate_silu)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_rmsnorm)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_conv1d_causal)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_diffusion_conv_512_l16)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_diffusion_conv_1024_l8)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_diffusion_conv_2048_l4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_diffusion_conv_512_l16_threads4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_diffusion_conv_1024_l8_threads4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_diffusion_conv_2048_l4_threads4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_diffusion_downsample_512_l16)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_diffusion_upsample_512_l8)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_discretize_and_scan)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
