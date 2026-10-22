#include "api/engine.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

struct MixRow
{
  std::string name;
  double p50_us;
  int calls;
};

std::vector<MixRow> g_rows;

std::vector<float> filled(std::size_t n, float base = 0.03F)
{
  std::vector<float> v(n);
  for (std::size_t i{0uz}; i < n; ++i)
    v[i] = base * static_cast<float>(static_cast<int>(i % 17uz) - 8);
  return v;
}

template<typename Fn> double median_us(Fn fn, int warmup, int iters)
{
  for (int i{0}; i < warmup; ++i)
    fn();
  std::vector<double> samples(static_cast<std::size_t>(iters));
  for (int i{0}; i < iters; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    samples[static_cast<std::size_t>(i)] =
        std::chrono::duration<double, std::micro>(end - begin).count();
  }
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
  if (name.rfind("group_norm", 0uz) == 0uz || name.rfind("mish", 0uz) == 0uz)
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
  std::fprintf(file, "  \"tool\": \"flowedge_diffusion_breakdown\",\n");
  std::fprintf(file, "  \"hardware_concurrency\": %u,\n", std::thread::hardware_concurrency());
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

  std::printf("Diffusion Policy CPU bottleneck mix (10 DDIM steps, batch=1)\n");
  std::printf("%-36s %11s  %-5s  %s\n", "kernel", "p50", "n", "estimated");

  auto conv = [](std::size_t channels, std::size_t length, std::size_t kernel, std::size_t stride,
                 std::size_t padding, std::size_t output_length) {
    const std::vector<float> x = filled(channels * length);
    const std::vector<float> w = filled(channels * channels * kernel, 0.001F);
    const std::vector<float> b = filled(channels);
    std::vector<float> y(channels * output_length);
    std::vector<float> workspace(
        fe::conv1d_workspace_floats(channels, channels, output_length, kernel));
    return median_us(
        [&] {
          fe::conv1d(x, w, b, y, channels, channels, length, output_length, kernel, stride, padding,
                     nullptr, workspace);
        },
        4, 12);
  };

  report("conv 512 L16 K5", conv(512uz, 16uz, 5uz, 1uz, 2uz, 16uz), 40);
  report("conv 512 L8 K5", conv(512uz, 8uz, 5uz, 1uz, 2uz, 8uz), 30);
  report("conv 1024 L8 K5", conv(1024uz, 8uz, 5uz, 1uz, 2uz, 8uz), 30);
  report("conv 1024 L4 K5", conv(1024uz, 4uz, 5uz, 1uz, 2uz, 4uz), 30);
  report("conv 2048 L4 K5", conv(2048uz, 4uz, 5uz, 1uz, 2uz, 4uz), 70);
  report(
      "conv 4096->1024 L4 K5",
      [&] {
        constexpr std::size_t in_c = 4096uz, out_c = 1024uz, length = 4uz, kernel = 5uz;
        const std::vector<float> x = filled(in_c * length);
        const std::vector<float> w = filled(out_c * in_c * kernel, 0.001F);
        const std::vector<float> b = filled(out_c);
        std::vector<float> y(out_c * length);
        std::vector<float> workspace(fe::conv1d_workspace_floats(in_c, out_c, length, kernel));
        return median_us(
            [&] {
              fe::conv1d(x, w, b, y, in_c, out_c, length, length, kernel, 1uz, 2uz, nullptr,
                         workspace);
            },
            4, 12);
      }(),
      10);
  report("downsample 512 L16 K3", conv(512uz, 16uz, 3uz, 2uz, 1uz, 8uz), 10);
  report("downsample 1024 L8 K3", conv(1024uz, 8uz, 3uz, 2uz, 1uz, 4uz), 10);

  auto upsample = [](std::size_t channels, std::size_t input_length, std::size_t output_length) {
    constexpr std::size_t kernel = 4uz;
    const std::vector<float> x = filled(channels * input_length);
    const std::vector<float> w = filled(channels * channels * kernel, 0.001F);
    const std::vector<float> b = filled(channels);
    std::vector<float> y(channels * output_length);
    std::vector<float> workspace(
        fe::conv_transpose1d_workspace_floats(channels, channels, input_length, kernel));
    return median_us(
        [&] {
          fe::conv_transpose1d(x, w, b, y, channels, channels, input_length, output_length, kernel,
                               2uz, 1uz, nullptr, workspace);
        },
        4, 8);
  };
  report("upsample 1024 L4->8 K4", upsample(1024uz, 4uz, 8uz), 10);
  report("upsample 512 L8->16 K4", upsample(512uz, 8uz, 16uz), 10);

  auto upsample_kmajor = [](std::size_t channels, std::size_t input_length,
                            std::size_t output_length) {
    constexpr std::size_t kernel = 4uz;
    const std::vector<float> x = filled(channels * input_length);
    const std::vector<float> w = filled(channels * channels * kernel, 0.001F);
    std::vector<float> packed(w.size());
    for (std::size_t ic{0uz}; ic < channels; ++ic)
      for (std::size_t oc{0uz}; oc < channels; ++oc)
        for (std::size_t k{0uz}; k < kernel; ++k)
          packed[(((k * channels) + oc) * channels) + ic] =
              w[(((ic * channels) + oc) * kernel) + k];
    const std::vector<float> b = filled(channels);
    std::vector<float> y(channels * output_length);
    std::vector<float> workspace(
        fe::conv_transpose1d_workspace_floats(channels, channels, input_length, kernel));
    return median_us(
        [&] {
          fe::conv_transpose1d(x, packed, b, y, channels, channels, input_length, output_length,
                               kernel, 2uz, 1uz, nullptr, workspace, true);
        },
        4, 8);
  };
  report("upsample 1024 k-major", upsample_kmajor(1024uz, 4uz, 8uz), 10);
  report("upsample 512 k-major", upsample_kmajor(512uz, 8uz, 16uz), 10);

  report(
      "group_norm 2048 L4",
      [&] {
        std::vector<float> x = filled(2048uz * 4uz);
        const std::vector<float> w = filled(2048uz, 1.0F);
        const std::vector<float> b = filled(2048uz, 0.01F);
        return median_us([&] { fe::group_norm(x, w, b, 2048uz, 4uz, 8uz); }, 8, 40);
      }(),
      80);
  report(
      "mish 8192",
      [&] {
        std::vector<float> x = filled(8192uz);
        return median_us([&] { fe::mish(x); }, 8, 40);
      }(),
      310);

  report(
      "gemm 4x10240x2048",
      [&] {
        const std::vector<float> in = filled(4uz * 10240uz);
        const std::vector<float> w = filled(2048uz * 10240uz);
        std::vector<float> out(4uz * 2048uz);
        return median_us([&] { fe::matmul(in, w, out, 4uz, 10240uz, 2048uz); }, 4, 12);
      }(),
      70);

  auto gemm_pool = [&](unsigned threads) {
    std::vector<fe::Task> ring(8uz);
    std::vector<std::size_t> sequence(8uz);
    std::vector<std::jthread> workers(threads);
    fe::ThreadPool pool{ring, sequence, workers, threads};
    const std::vector<float> in = filled(4uz * 10240uz);
    const std::vector<float> w = filled(2048uz * 10240uz);
    std::vector<float> out(4uz * 2048uz);
    return median_us([&] { fe::matmul(in, w, out, 4uz, 10240uz, 2048uz, &pool); }, 4, 8);
  };
  report("gemm 4x10240x2048 t=2", gemm_pool(2u), 70);
  report("gemm 4x10240x2048 t=4", gemm_pool(4u), 70);

  if (checkpoint != nullptr) {
    std::vector<float> condition;
    std::vector<float> noise;
    std::vector<float> action;
    const auto time_native = [&](unsigned threads) {
      fe_engine* engine = fe_engine_load_with_threads(checkpoint, threads);
      if (engine == nullptr) {
        std::fprintf(stderr, "load t=%u failed: %s\n", threads, fe_engine_last_error());
        return;
      }
      if (condition.empty()) {
        condition.assign(fe_engine_condition_dim(engine), 0.1F);
        noise.resize(fe_engine_action_horizon(engine) * fe_engine_action_dim(engine));
        for (std::size_t i{0uz}; i < noise.size(); ++i)
          noise[i] = std::sin(0.37F * static_cast<float>(i + 1uz));
        action.resize(noise.size());
      }
      const unsigned got = fe_engine_thread_count(engine);
      const double sample_us = median_us(
          [&] {
            if (fe_engine_sample_diffusion(engine, condition.data(), noise.data(), 10uz,
                                           FE_DIFFUSION_DDIM, 0u, action.data()) != 0)
              std::abort();
          },
          2, 6);
      char name[48];
      std::snprintf(name, sizeof(name), "native DDIM x10 t=%u (pool=%u)", threads, got);
      report(name, sample_us, 1);
      fe_engine_free(engine);
    };
    time_native(0u);
    time_native(2u);
    time_native(4u);
    time_native(6u);
  }

  if (json_path != nullptr)
    write_json(json_path, checkpoint);
  return 0;
}
