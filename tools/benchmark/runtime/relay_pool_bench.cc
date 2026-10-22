#include "relay/protocol/messages.h"
#include "relay/scheduler/edf_scheduler.h"
#include "relay/worker/head_worker_pool.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace fe::relay;

std::atomic<std::size_t> g_allocations{0uz};

[[nodiscard]] std::uint64_t monotonic_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

template<typename Integer>
[[nodiscard]] bool parse_integer(std::string_view text, Integer& value) noexcept
{
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

[[nodiscard]] double percentile(const std::vector<double>& sorted, double probability)
{
  const auto rank =
      static_cast<std::size_t>(std::ceil(probability * static_cast<double>(sorted.size())));
  return sorted[std::clamp(rank, 1uz, sorted.size()) - 1uz];
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
  const std::string model = argc > 1 ? argv[1] : "models/mamba_flow.safetensors";
  std::size_t iterations{1'000uz};
  std::size_t workers{2uz};
  unsigned threads{};
  if ((argc > 2 && (!parse_integer(std::string_view{argv[2]}, iterations) || iterations == 0uz)) ||
      (argc > 3 &&
       (!parse_integer(std::string_view{argv[3]}, workers) || workers == 0uz || workers > 8uz)) ||
      (argc > 4 && (!parse_integer(std::string_view{argv[4]}, threads) || threads > 8u))) {
    std::cerr << "Usage: flowedge_relay_pool_bench [model] [iterations] [workers 1..8] "
                 "[threads-per-worker 0..8]\n";
    return 2;
  }

  auto pool_result = HeadWorkerPool::open(model, workers, threads);
  if (!pool_result) {
    std::cerr << "Relay pool benchmark: " << pool_result.error() << '\n';
    return 1;
  }
  HeadWorkerPool pool = std::move(*pool_result);
  const fe_model_metadata metadata = pool.model_metadata();
  std::vector<float> condition(metadata.condition_dim, 0.125f);
  std::vector<float> noise(metadata.action_dim, -0.25f);
  std::vector<std::uint64_t> started_at(iterations + 1uz);
  std::vector<double> latencies_us{};
  latencies_us.reserve(iterations);
  EdfScheduler scheduler{64uz};
  ConditionMessage outbound{};

  std::size_t submitted{};
  std::size_t completed{};
  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  const auto benchmark_start = std::chrono::steady_clock::now();
  while (completed < iterations) {
    bool progressed{false};
    while (submitted < iterations && scheduler.size() < scheduler.capacity()) {
      const std::uint64_t sequence = submitted + 1uz;
      const std::uint64_t started = monotonic_ns();
      if (make_condition_message(outbound, metadata, sequence, 1u, started, 0u, 1u, 6uz,
                                 FE_SOLVER_HEUN, condition, noise) != ProtocolResult::kSuccess ||
          scheduler.submit(outbound) != SubmitResult::kAccepted) {
        std::cerr << "Relay pool benchmark: failed to enqueue request\n";
        return 1;
      }
      started_at[sequence] = started;
      ++submitted;
      progressed = true;
    }

    while (pool.has_idle()) {
      const ConditionMessage* const request = scheduler.pop(monotonic_ns());
      if (request == nullptr)
        break;
      if (!pool.try_dispatch(*request)) {
        std::cerr << "Relay pool benchmark: " << pool.last_error() << '\n';
        return 1;
      }
      progressed = true;
    }

    for (std::size_t received{0uz}; received < workers; ++received) {
      const ActionMessage* const action = pool.ready_action();
      if (action == nullptr)
        break;
      if (validate(*action) != ProtocolResult::kSuccess ||
          action->metadata.status != FE_ACTION_COMPLETE || action->envelope.sequence == 0u ||
          action->envelope.sequence > iterations) {
        std::cerr << "Relay pool benchmark: worker returned an invalid action\n";
        return 1;
      }
      latencies_us.push_back(
          static_cast<double>(monotonic_ns() - started_at[action->envelope.sequence]) / 1'000.0);
      pool.release_ready_action();
      ++completed;
      progressed = true;
    }

    if (!progressed)
      std::this_thread::yield();
  }
  const auto benchmark_end = std::chrono::steady_clock::now();
  const std::size_t hot_allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;

  std::ranges::sort(latencies_us);
  double sum{};
  for (const double value : latencies_us)
    sum += value;
  const double seconds = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
  std::cout << std::fixed << std::setprecision(3)
            << "FlowEdge Relay preallocated worker-pool scheduler\n"
            << "requests=" << iterations << " workers=" << workers
            << " core_threads_per_worker=" << threads << " steps=6 solver=heun\n"
            << "shared_weight_bytes=" << pool.shared_weight_bytes() << '\n'
            << "mean_us=" << (sum / static_cast<double>(iterations))
            << " p50_us=" << percentile(latencies_us, 0.50)
            << " p99_us=" << percentile(latencies_us, 0.99) << '\n'
            << "throughput_req_s=" << (static_cast<double>(iterations) / seconds)
            << " hot_allocations=" << hot_allocations << '\n';
  return 0;
}
