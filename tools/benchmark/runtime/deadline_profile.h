#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace fe::benchmark {

// Samples must be sorted in ascending order. Nearest-rank keeps small benchmark
// runs deterministic and never interpolates between two observations.
[[nodiscard]] inline double percentile(std::span<const double> sorted, double quantile) noexcept
{
  if (sorted.empty())
    return 0.0;
  const auto rank = static_cast<std::size_t>(std::ceil(quantile * sorted.size()));
  return sorted[std::min(rank == 0uz ? 0uz : rank - 1uz, sorted.size() - 1uz)];
}

struct Statistics
{
  double mean{};
  double p50{};
  double p95{};
  double p99{};
  double p999{};
  double max{};
};

[[nodiscard]] inline Statistics summarize(std::span<const double> sorted) noexcept
{
  Statistics result{};
  if (sorted.empty())
    return result;
  for (const double sample : sorted)
    result.mean += sample;
  result.mean /= static_cast<double>(sorted.size());
  result.p50 = percentile(sorted, 0.50);
  result.p95 = percentile(sorted, 0.95);
  result.p99 = percentile(sorted, 0.99);
  result.p999 = percentile(sorted, 0.999);
  result.max = sorted.back();
  return result;
}

[[nodiscard]] inline bool within_budget(double period_us, double metric) noexcept
{
  return metric <= period_us;
}

[[nodiscard]] inline double regression_percent(double baseline, double current) noexcept
{
  return baseline > 0.0 ? ((current - baseline) / baseline) * 100.0 : 0.0;
}

} // namespace fe::benchmark
