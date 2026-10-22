#include "tools/benchmark/runtime/deadline_profile.h"
#include "protocol/deadline_flow.h"

#include <array>
#include <gtest/gtest.h>

TEST(DeadlineProfile, UsesNearestRankPercentiles)
{
  constexpr std::array<double, 10> samples{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  const fe::benchmark::Statistics stats = fe::benchmark::summarize(samples);
  EXPECT_DOUBLE_EQ(stats.mean, 5.5);
  EXPECT_DOUBLE_EQ(stats.p50, 5.0);
  EXPECT_DOUBLE_EQ(stats.p95, 10.0);
  EXPECT_DOUBLE_EQ(stats.p99, 10.0);
  EXPECT_DOUBLE_EQ(stats.p999, 10.0);
  EXPECT_DOUBLE_EQ(stats.max, 10.0);
}

TEST(DeadlineProfile, HandlesEmptySamples)
{
  const fe::benchmark::Statistics stats = fe::benchmark::summarize(std::span<const double>{});
  EXPECT_DOUBLE_EQ(stats.mean, 0.0);
  EXPECT_DOUBLE_EQ(stats.max, 0.0);
}

TEST(DeadlineProfile, ComputesRegressionAndBudget)
{
  constexpr std::array<double, 4> samples{10.0, 20.0, 30.0, 40.0};
  const fe::benchmark::Statistics stats = fe::benchmark::summarize(samples);
  EXPECT_TRUE(fe::benchmark::within_budget(40.0, stats.max));
  EXPECT_FALSE(fe::benchmark::within_budget(39.0, stats.max));
  EXPECT_DOUBLE_EQ(fe::benchmark::regression_percent(100.0, 105.0), 5.0);
  EXPECT_DOUBLE_EQ(fe::benchmark::regression_percent(100.0, 95.0), -5.0);
}
TEST(DeadlineFlow, SelectsHighestQualityWithinBufferAndHardwareBudget)
{
  const std::array plans{fe::ExecutionPlan{1u, 4u, 0u, 0u, 0u, 1u, 20u, 100u, 0.6},
                         fe::ExecutionPlan{2u, 10u, 0u, 0u, 0u, 1u, 60u, 200u, 0.9}};
  auto selector = fe::DeadlineFlow::create(plans);
  ASSERT_TRUE(selector);
  fe::PlanContext context{.slack_ns = 100u, .buffered_actions = 3u, .period_ns = 30u};
  EXPECT_EQ(selector->select(context), 2u);
  context.buffered_actions = 1u;
  EXPECT_EQ(selector->select(context), 1u);
  context.minimum_quality = 0.8;
  EXPECT_FALSE(selector->select(context));
  context.buffered_actions = 3u;
  context.available_memory_bytes = 100u;
  EXPECT_FALSE(selector->select(context));
  context.available_memory_bytes = 200u;
  context.hardware_mask = 2u;
  EXPECT_FALSE(selector->select(context));
  context.hardware_mask = 1u;
  context.healthy = false;
  EXPECT_FALSE(selector->select(context));
}

TEST(DeadlineFlow, ObservedSlowdownsAndSwitchCostsChangeAdmission)
{
  const std::array plans{fe::ExecutionPlan{1u, 4u, 0u, 0u, 0u, 1u, 20u, 0u, 0.6},
                         fe::ExecutionPlan{2u, 10u, 0u, 0u, 0u, 1u, 60u, 0u, 0.9}};
  auto selector = fe::DeadlineFlow::create(plans);
  ASSERT_TRUE(selector);
  fe::PlanContext context{.slack_ns = 100u};
  ASSERT_TRUE(selector->observe(2u, 110u));
  EXPECT_EQ(selector->select(context), 1u);
  context.switch_ns = 90u;
  EXPECT_FALSE(selector->select(context));
  EXPECT_FALSE(selector->observe(99u, 1u));
  EXPECT_FALSE(selector->observe(1u, 0u));
}

TEST(DeadlineFlow, RejectsInvalidPortfoliosAndHandlesSaturatedBufferTime)
{
  EXPECT_FALSE(fe::DeadlineFlow::create({}));
  std::array plans{fe::ExecutionPlan{1u, 4u, 0u, 0u, 0u, 1u, 20u, 0u, 0.6}};
  auto selector = fe::DeadlineFlow::create(plans);
  ASSERT_TRUE(selector);
  fe::PlanContext context{.slack_ns = 100u,
                          .buffered_actions = std::numeric_limits<std::uint64_t>::max(),
                          .period_ns = 30u};
  EXPECT_EQ(selector->select(context), 1u);
  context.reserve_ns = 100u;
  EXPECT_FALSE(selector->select(context));
  plans[0].quality = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(fe::DeadlineFlow::create(plans));
}
