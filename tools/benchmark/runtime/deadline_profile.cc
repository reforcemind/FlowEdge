#include "deadline_profile.h"

#include "api/engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::atomic<std::size_t> g_allocations{0uz};

enum class BudgetMetric
{
  kP99,
  kP999,
  kMax,
};

struct Options
{
  std::string model{};
  std::string json_path{};
  std::string compare_path{};
  std::string affinity{};
  std::size_t steps{10uz};
  std::size_t iterations{100uz};
  std::size_t warmup{10uz};
  std::size_t period_us{};
  unsigned threads{};
  double max_regression{};
  BudgetMetric metric{BudgetMetric::kMax};
  int solver{FE_SOLVER_EULER};
  bool compare{false};
  bool allow_environment_mismatch{false};
};

struct Metadata
{
  std::string digest{};
  std::string architecture{};
  std::string precision{};
  std::string cpu{};
  std::string os{};
  std::string compiler{};
  std::string build_type{};
  std::string affinity{};
  std::string solver{};
  std::size_t nfe{};
};

struct Baseline
{
  double p99{};
  Metadata metadata{};
};

constexpr int kExitRuntime = 1;
constexpr int kExitConfig = 2;
constexpr int kExitDeadline = 3;
constexpr int kExitRegression = 4;
constexpr int kExitIncompatible = 5;

[[nodiscard]] bool parse_size(std::string_view text, std::size_t& value) noexcept
{
  if (text.empty())
    return false;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

[[nodiscard]] bool parse_unsigned(std::string_view text, unsigned& value) noexcept
{
  std::size_t parsed{};
  if (!parse_size(text, parsed) || parsed > static_cast<std::size_t>(UINT_MAX))
    return false;
  value = static_cast<unsigned>(parsed);
  return true;
}

[[nodiscard]] bool parse_double(std::string_view text, double& value) noexcept
{
  if (text.empty())
    return false;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size() && std::isfinite(value);
}

[[nodiscard]] const char* solver_name(int solver) noexcept
{
  switch (solver) {
  case FE_SOLVER_EULER:
    return "euler";
  case FE_SOLVER_HEUN:
    return "heun";
  case FE_SOLVER_RK4:
    return "rk4";
  default:
    return "unknown";
  }
}

[[nodiscard]] bool parse_solver(std::string_view text, int& solver) noexcept
{
  if (text == "euler")
    solver = FE_SOLVER_EULER;
  else if (text == "heun")
    solver = FE_SOLVER_HEUN;
  else if (text == "rk4")
    solver = FE_SOLVER_RK4;
  else
    return false;
  return true;
}

[[nodiscard]] const char* metric_name(BudgetMetric metric) noexcept
{
  switch (metric) {
  case BudgetMetric::kP99:
    return "p99";
  case BudgetMetric::kP999:
    return "p999";
  case BudgetMetric::kMax:
    return "max";
  }
  return "max";
}

[[nodiscard]] bool parse_metric(std::string_view text, BudgetMetric& metric) noexcept
{
  if (text == "p99")
    metric = BudgetMetric::kP99;
  else if (text == "p999")
    metric = BudgetMetric::kP999;
  else if (text == "max")
    metric = BudgetMetric::kMax;
  else
    return false;
  return true;
}

[[nodiscard]] double budget_value(const fe::benchmark::Statistics& stats,
                                  BudgetMetric metric) noexcept
{
  switch (metric) {
  case BudgetMetric::kP99:
    return stats.p99;
  case BudgetMetric::kP999:
    return stats.p999;
  case BudgetMetric::kMax:
    return stats.max;
  }
  return stats.max;
}

[[nodiscard]] std::string digest_string(const fe_model_digest& digest)
{
  static constexpr char hex[] = "0123456789abcdef";
  std::string result{"fingerprint:"};
  result.reserve(27uz);
  for (const std::uint8_t byte : digest.bytes) {
    result.push_back(hex[byte >> 4u]);
    result.push_back(hex[byte & 0x0fu]);
  }
  return result;
}

[[nodiscard]] std::string json_escape(std::string_view value)
{
  std::string result;
  result.reserve(value.size() + 2uz);
  result.push_back('"');
  for (const char character : value) {
    if (character == '"' || character == '\\')
      result.push_back('\\');
    result.push_back(character);
  }
  result.push_back('"');
  return result;
}

[[nodiscard]] std::string architecture_name(std::uint32_t architecture)
{
  switch (architecture) {
  case FE_ARCH_MAMBA:
    return "mamba";
  case FE_ARCH_FLOW_HEAD:
    return "flow_head";
  case FE_ARCH_MAMBA_FLOW:
    return "mamba_flow";
  case FE_ARCH_DIFFUSION_HEAD:
    return "diffusion_head";
  default:
    return "unknown";
  }
}

[[nodiscard]] std::string precision_name(std::uint32_t precision)
{
  switch (precision) {
  case FE_PRECISION_F32:
    return "f32";
  case FE_PRECISION_BF16:
    return "bf16";
  case FE_PRECISION_MIXED:
    return "mixed";
  default:
    return "unknown";
  }
}

[[nodiscard]] std::string os_name()
{
#if defined(_WIN32)
  return "windows";
#elif defined(__APPLE__)
  return "macos";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}

[[nodiscard]] std::string compiler_name()
{
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#elif defined(_MSC_VER)
  return "msvc";
#else
  return "unknown";
#endif
}

[[nodiscard]] std::string build_type()
{
#if defined(NDEBUG)
  return "release";
#else
  return "debug";
#endif
}

[[nodiscard]] bool consume_option(int& index, int argc, char** argv, std::string_view name,
                                  std::string_view& value)
{
  if (std::string_view{argv[index]} != name || index + 1 >= argc)
    return false;
  value = argv[++index];
  return true;
}

[[nodiscard]] bool parse_options(int argc, char** argv, Options& options)
{
  if (argc < 2)
    return false;
  options.model = argv[1];
  for (int index{2}; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    std::string_view value{};
    if (consume_option(index, argc, argv, "--solver", value)) {
      if (!parse_solver(value, options.solver))
        return false;
    } else if (consume_option(index, argc, argv, "--steps", value)) {
      if (!parse_size(value, options.steps))
        return false;
    } else if (consume_option(index, argc, argv, "--period-us", value)) {
      if (!parse_size(value, options.period_us))
        return false;
    } else if (consume_option(index, argc, argv, "--iterations", value)) {
      if (!parse_size(value, options.iterations))
        return false;
    } else if (consume_option(index, argc, argv, "--warmup", value)) {
      if (!parse_size(value, options.warmup))
        return false;
    } else if (consume_option(index, argc, argv, "--threads", value)) {
      if (!parse_unsigned(value, options.threads))
        return false;
    } else if (consume_option(index, argc, argv, "--budget-metric", value)) {
      if (!parse_metric(value, options.metric))
        return false;
    } else if (consume_option(index, argc, argv, "--json", value)) {
      options.json_path = value;
    } else if (consume_option(index, argc, argv, "--compare", value)) {
      options.compare = true;
      options.compare_path = value;
    } else if (consume_option(index, argc, argv, "--max-p99-regression", value)) {
      if (!parse_double(value, options.max_regression) || options.max_regression < 0.0)
        return false;
    } else if (consume_option(index, argc, argv, "--cpu-affinity", value)) {
      options.affinity = value;
    } else if (argument == "--allow-environment-mismatch") {
      options.allow_environment_mismatch = true;
    } else {
      return false;
    }
  }
  return !options.model.empty() && options.steps > 0uz && options.iterations > 0uz &&
         options.period_us > 0uz;
}

void print_usage()
{
  std::fputs("usage: flowedge-profile MODEL --period-us US [options]\n"
             "  --solver euler|heun|rk4       solver (default: euler)\n"
             "  --steps N                     solver steps (default: 10)\n"
             "  --iterations N                timed samples (default: 100)\n"
             "  --warmup N                    untimed warmup calls (default: 10)\n"
             "  --budget-metric p99|p999|max  deadline metric (default: max)\n"
             "  --compare FILE                compare p99 with a profile\n"
             "  --max-p99-regression P        allowed regression percentage\n"
             "  --allow-environment-mismatch allow incompatible baseline\n"
             "  --threads N                   worker threads (default: automatic)\n"
             "  --cpu-affinity LIST           record requested CPU list\n"
             "  --json FILE|-                 write versioned JSON\n",
             stderr);
}

[[nodiscard]] std::string find_json_string(std::string_view json, std::string_view key)
{
  const std::string needle = '"' + std::string{key} + '"';
  const std::size_t start = json.find(needle);
  if (start == std::string_view::npos)
    return {};
  const std::size_t quote = json.find('"', json.find(':', start) + 1uz);
  if (quote == std::string_view::npos)
    return {};
  const std::size_t end = json.find('"', quote + 1uz);
  return end == std::string_view::npos ? std::string{}
                                       : std::string{json.substr(quote + 1uz, end - quote - 1uz)};
}

[[nodiscard]] bool find_json_double(std::string_view json, std::string_view key, double& value)
{
  const std::string needle = '"' + std::string{key} + '"';
  const std::size_t start = json.find(needle);
  if (start == std::string_view::npos)
    return false;
  const std::size_t colon = json.find(':', start);
  if (colon == std::string_view::npos)
    return false;
  const std::size_t end = json.find_first_of(",}\n", colon + 1uz);
  return parse_double(json.substr(colon + 1uz, end - colon - 1uz), value);
}

[[nodiscard]] bool read_baseline(const std::string& path, Baseline& baseline)
{
  std::ifstream input(path);
  if (!input)
    return false;
  const std::string json{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  if (!find_json_double(json, "p99_us", baseline.p99) || baseline.p99 <= 0.0)
    return false;
  baseline.metadata.digest = find_json_string(json, "checkpoint_digest");
  baseline.metadata.architecture = find_json_string(json, "architecture");
  baseline.metadata.precision = find_json_string(json, "precision");
  baseline.metadata.cpu = find_json_string(json, "cpu");
  baseline.metadata.os = find_json_string(json, "os");
  baseline.metadata.compiler = find_json_string(json, "compiler");
  baseline.metadata.build_type = find_json_string(json, "build_type");
  baseline.metadata.solver = find_json_string(json, "solver");
  double nfe{};
  if (find_json_double(json, "nfe", nfe) && nfe >= 0.0)
    baseline.metadata.nfe = static_cast<std::size_t>(nfe);
  return !baseline.metadata.digest.empty();
}

[[nodiscard]] bool compatible(const Metadata& current, const Metadata& baseline) noexcept
{
  return current.digest == baseline.digest && current.architecture == baseline.architecture &&
         current.precision == baseline.precision && current.cpu == baseline.cpu &&
         current.os == baseline.os && current.compiler == baseline.compiler &&
         current.build_type == baseline.build_type && current.affinity == baseline.affinity &&
         current.solver == baseline.solver && current.nfe == baseline.nfe;
}

void write_metadata(std::ostream& output, const Metadata& metadata)
{
  output << "\"flowedge_commit\":"
         << json_escape(
#ifdef FLOWEDGE_GIT_COMMIT
                FLOWEDGE_GIT_COMMIT
#else
                "unknown"
#endif
                )
         << ",\"checkpoint_digest\":" << json_escape(metadata.digest)
         << ",\"runtime\":{\"solver\":" << json_escape(metadata.solver)
         << ",\"nfe\":" << metadata.nfe << ",\"precision\":" << json_escape(metadata.precision)
         << "},\"system\":{\"architecture\":" << json_escape(metadata.architecture)
         << ",\"cpu\":" << json_escape(metadata.cpu) << ",\"os\":" << json_escape(metadata.os)
         << ",\"compiler\":" << json_escape(metadata.compiler)
         << ",\"build_type\":" << json_escape(metadata.build_type)
         << ",\"cpu_affinity\":" << json_escape(metadata.affinity) << "}";
}

void write_json(std::ostream& output, const Metadata& metadata, const Options& options,
                const fe::benchmark::Statistics& stats, std::size_t allocations, double budget,
                bool budget_passed, bool regression_checked, double baseline_p99, double regression,
                bool regression_passed)
{
  output << std::fixed << std::setprecision(3) << "{\"schema_version\":1,";
  write_metadata(output, metadata);
  output << ",\"benchmark\":{\"iterations\":" << options.iterations
         << ",\"warmup_iterations\":" << options.warmup << ",\"allocations\":" << allocations
         << ",\"mean_us\":" << stats.mean << ",\"p50_us\":" << stats.p50
         << ",\"p95_us\":" << stats.p95 << ",\"p99_us\":" << stats.p99
         << ",\"p999_us\":" << stats.p999 << ",\"max_us\":" << stats.max
         << "},\"budget\":{\"period_us\":" << options.period_us
         << ",\"metric\":" << json_escape(metric_name(options.metric)) << ",\"value_us\":" << budget
         << ",\"passed\":" << (budget_passed ? "true" : "false") << "}";
  if (regression_checked) {
    output << ",\"regression\":{\"baseline_p99_us\":" << baseline_p99
           << ",\"current_p99_us\":" << stats.p99 << ",\"percent\":" << regression
           << ",\"limit_percent\":" << options.max_regression
           << ",\"passed\":" << (regression_passed ? "true" : "false") << "}";
  }
  output << "}\n";
}

} // namespace

void* operator new(std::size_t bytes)
{
  g_allocations.fetch_add(1uz, std::memory_order_relaxed);
  if (void* memory = std::malloc(bytes == 0uz ? 1uz : bytes))
    return memory;
  throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept
{
  std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept
{
  std::free(memory);
}
void* operator new[](std::size_t bytes)
{
  return ::operator new(bytes);
}
void operator delete[](void* memory) noexcept
{
  std::free(memory);
}
void operator delete[](void* memory, std::size_t) noexcept
{
  std::free(memory);
}

int main(int argc, char** argv)
{
  if (argc == 2 && std::string_view{argv[1]} == "--help") {
    print_usage();
    return 0;
  }
  Options options{};
  if (!parse_options(argc, argv, options)) {
    print_usage();
    return kExitConfig;
  }

  const auto engine = std::unique_ptr<fe_engine, decltype(&fe_engine_free)>{
      options.threads == 0u ? fe_engine_load(options.model.c_str())
                            : fe_engine_load_with_threads(options.model.c_str(), options.threads),
      fe_engine_free};
  if (!engine) {
    std::cerr << "flowedge-profile: " << fe_engine_last_error() << '\n';
    return kExitRuntime;
  }

  fe_model_metadata model{};
  if (fe_engine_model_metadata(engine.get(), &model) != FE_STATUS_OK || model.action_dim == 0uz ||
      model.condition_dim == 0uz) {
    std::cerr << "flowedge-profile: checkpoint has no compatible flow head\n";
    return kExitRuntime;
  }
  if (options.steps > (static_cast<std::size_t>(UINT64_MAX) / 4uz)) {
    std::cerr << "flowedge-profile: steps overflow NFE\n";
    return kExitConfig;
  }

  Metadata metadata{};
  metadata.digest = digest_string(model.model_digest);
  metadata.architecture = architecture_name(model.architecture);
  metadata.precision = precision_name(model.precision);
  metadata.cpu = std::getenv("FLOWEDGE_BENCH_CPU") != nullptr ? std::getenv("FLOWEDGE_BENCH_CPU")
                                                              : "unspecified";
  metadata.os = os_name();
  metadata.compiler = compiler_name();
  metadata.build_type = build_type();
  metadata.affinity = options.affinity.empty() ? "unspecified" : options.affinity;
  metadata.solver = solver_name(options.solver);
  metadata.nfe = options.steps * (options.solver == FE_SOLVER_RK4    ? 4uz
                                  : options.solver == FE_SOLVER_HEUN ? 2uz
                                                                     : 1uz);

  std::vector<float> condition(model.condition_dim, 0.0F);
  std::vector<float> noise(model.action_dim, 0.0F);
  std::vector<float> action(model.action_dim, 0.0F);
  for (std::size_t index{0uz}; index < noise.size(); ++index)
    noise[index] = static_cast<float>((index % 17uz) + 1uz) * 0.01F;

  for (std::size_t index{0uz}; index < options.warmup; ++index) {
    if (fe_engine_sample_condition(engine.get(), condition.data(), noise.data(), options.steps,
                                   options.solver, action.data()) != FE_STATUS_OK) {
      std::cerr << "flowedge-profile: " << fe_engine_last_error() << '\n';
      return kExitRuntime;
    }
  }

  std::vector<double> samples;
  samples.reserve(options.iterations);
  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  for (std::size_t index{0uz}; index < options.iterations; ++index) {
    const auto start = std::chrono::steady_clock::now();
    if (fe_engine_sample_condition(engine.get(), condition.data(), noise.data(), options.steps,
                                   options.solver, action.data()) != FE_STATUS_OK) {
      std::cerr << "flowedge-profile: " << fe_engine_last_error() << '\n';
      return kExitRuntime;
    }
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
  }
  const std::size_t allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;
  std::ranges::sort(samples);
  const fe::benchmark::Statistics stats = fe::benchmark::summarize(samples);
  const double selected = budget_value(stats, options.metric);
  const bool budget_passed = fe::benchmark::within_budget(options.period_us, selected);

  Baseline baseline{};
  bool regression_checked{false};
  bool regression_passed{true};
  double regression{};
  if (options.compare) {
    if (!read_baseline(options.compare_path, baseline)) {
      std::cerr << "flowedge-profile: invalid baseline profile\n";
      return kExitConfig;
    }
    if (!options.allow_environment_mismatch && !compatible(metadata, baseline.metadata)) {
      std::cerr << "flowedge-profile: baseline environment is incompatible\n";
      return kExitIncompatible;
    }
    regression_checked = true;
    regression = fe::benchmark::regression_percent(baseline.p99, stats.p99);
    regression_passed = regression <= options.max_regression;
  }

  if (options.json_path != "-") {
    std::cout << std::fixed << std::setprecision(3) << "Checkpoint     " << metadata.digest
              << "\nSolver         " << metadata.solver << "\nNFE            " << metadata.nfe
              << "\n\nmean           " << stats.mean << " us\np50            " << stats.p50
              << " us\np95            " << stats.p95 << " us\np99            " << stats.p99
              << " us\np999           " << stats.p999 << " us\nmax            " << stats.max
              << " us\n\nallocations    " << allocations << "\nbudget         " << options.period_us
              << " us\nmargin         " << (static_cast<double>(options.period_us) - selected)
              << " us\n\n"
              << (budget_passed && regression_passed ? "PASS" : "FAIL") << '\n';
    if (regression_checked)
      std::cout << "Baseline p99    " << baseline.p99 << " us\nRegression      " << regression
                << " %\nLimit           " << options.max_regression << " %\n";
  }

  if (!options.json_path.empty()) {
    std::ofstream file;
    std::ostream* output = &std::cout;
    if (options.json_path != "-") {
      file.open(options.json_path, std::ios::trunc);
      if (!file) {
        std::cerr << "flowedge-profile: cannot write JSON profile\n";
        return kExitRuntime;
      }
      output = &file;
    }
    write_json(*output, metadata, options, stats, allocations, selected, budget_passed,
               regression_checked, baseline.p99, regression, regression_passed);
  }
  if (!budget_passed)
    return kExitDeadline;
  return regression_passed ? 0 : kExitRegression;
}
