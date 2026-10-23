#include "relay/protocol/messages.h"
#include "relay/scheduler/edf_scheduler.h"
#include "relay/shared_memory/shared_memory_ring.h"
#include "relay/worker/head_worker.h"

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
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

[[nodiscard]] bool accepted(SubmitResult result) noexcept
{
  return result == SubmitResult::kAccepted || result == SubmitResult::kAcceptedAndEvicted;
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
  std::size_t iterations{500uz};
  if (argc > 2 && (!parse_integer(std::string_view{argv[2]}, iterations) || iterations == 0uz)) {
    std::cerr << "Usage: flowedge_relay_bench [model] [iterations] [threads 0..8]\n";
    return 2;
  }
  std::optional<unsigned> threads{};
  if (argc > 3) {
    unsigned value{};
    if (!parse_integer(std::string_view{argv[3]}, value) || value > 8u) {
      std::cerr << "Usage: flowedge_relay_bench [model] [iterations] [threads 0..8]\n";
      return 2;
    }
    threads = value;
  }

  auto worker_result = HeadWorker::open(model, threads);
  if (!worker_result) {
    std::cerr << "Relay benchmark: " << worker_result.error() << '\n';
    return 1;
  }
  HeadWorker worker = std::move(*worker_result);
  const fe_model_metadata metadata = worker.model_metadata();
  std::vector<float> condition(metadata.condition_dim, 0.125f);
  std::vector<float> noise(metadata.action_dim, -0.25f);

  const std::string suffix = std::to_string(monotonic_ns());
  const std::string condition_name = "flowedge-relay-bench-condition-" + suffix;
  const std::string action_name = "flowedge-relay-bench-action-" + suffix;
  auto condition_owner_result =
      SharedMemoryRing::create(condition_name, RingConfig{16u, sizeof(ConditionMessage)});
  auto action_owner_result =
      SharedMemoryRing::create(action_name, RingConfig{16u, sizeof(ActionMessage)});
  if (!condition_owner_result || !action_owner_result) {
    std::cerr << "Relay benchmark: failed to create shared-memory rings\n";
    return 1;
  }
  auto condition_relay_result = SharedMemoryRing::open(condition_name);
  auto action_consumer_result = SharedMemoryRing::open(action_name);
  if (!condition_relay_result || !action_consumer_result) {
    std::cerr << "Relay benchmark: failed to open shared-memory rings\n";
    return 1;
  }
  SharedMemoryRing condition_producer = std::move(*condition_owner_result);
  SharedMemoryRing action_producer = std::move(*action_owner_result);
  SharedMemoryRing condition_relay = std::move(*condition_relay_result);
  SharedMemoryRing action_consumer = std::move(*action_consumer_result);
  EdfScheduler scheduler{16uz};

  auto outbound = std::make_unique<ConditionMessage>();
  auto inbound = std::make_unique<ConditionMessage>();
  auto consumed = std::make_unique<ActionMessage>();
  ActionMessage action{};
  std::vector<double> latencies_us;
  latencies_us.reserve(iterations);

  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  const auto benchmark_start = std::chrono::steady_clock::now();
  for (std::size_t i{0uz}; i < iterations; ++i) {
    const std::uint64_t started = monotonic_ns();
    const std::uint64_t sequence = i + 1uz;
    if (make_condition_message(*outbound, metadata, sequence, 1u, started, started + 1'000'000'000u,
                               sequence, 6uz, FE_SOLVER_HEUN, condition,
                               noise) != ProtocolResult::kSuccess) {
      std::cerr << "Relay benchmark: failed to build request\n";
      return 1;
    }
    if (condition_producer.try_push(wire_bytes(*outbound)) != RingResult::kSuccess) {
      std::cerr << "Relay benchmark: condition ring rejected request\n";
      return 1;
    }
    std::size_t bytes{};
    if (condition_relay.try_pop(std::as_writable_bytes(std::span{inbound.get(), 1uz}), bytes) !=
            RingResult::kSuccess ||
        bytes != wire_size(*inbound)) {
      std::cerr << "Relay benchmark: condition ring failed round trip\n";
      return 1;
    }
    const SubmitResult submitted = scheduler.submit(*inbound);
    if (!accepted(submitted)) {
      std::cerr << "Relay benchmark: scheduler rejected request\n";
      return 1;
    }
    worker.cancel_before(scheduler.newest_generation());
    const ConditionMessage* scheduled = scheduler.pop(monotonic_ns());
    if (scheduled == nullptr || !worker.begin(*scheduled)) {
      std::cerr << "Relay benchmark: worker rejected request: " << worker.last_error() << '\n';
      return 1;
    }
    WorkerStep state{};
    do {
      state = worker.advance(action);
    } while (state == WorkerStep::kInProgress);
    if (state != WorkerStep::kComplete || action.metadata.status != FE_ACTION_COMPLETE) {
      std::cerr << "Relay benchmark: worker did not complete request\n";
      return 1;
    }
    if (action_producer.try_push(wire_bytes(action)) != RingResult::kSuccess ||
        action_consumer.try_pop(std::as_writable_bytes(std::span{consumed.get(), 1uz}), bytes) !=
            RingResult::kSuccess ||
        bytes != wire_size(*consumed) || consumed->envelope.sequence != sequence) {
      std::cerr << "Relay benchmark: action ring failed round trip\n";
      return 1;
    }
    latencies_us.push_back(static_cast<double>(monotonic_ns() - started) / 1'000.0);
  }
  const auto benchmark_end = std::chrono::steady_clock::now();
  const std::size_t hot_allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;

  std::sort(latencies_us.begin(), latencies_us.end());
  double sum{};
  for (const double value : latencies_us)
    sum += value;
  const double seconds = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
  const double p99_us = percentile(latencies_us, 0.99);
  const double p99_ns_per_nfe =
      (p99_us * 1'000.0) / static_cast<double>(required_nfe(6u, FE_SOLVER_HEUN));
  const auto& stats = scheduler.stats();
  std::cout << std::fixed << std::setprecision(3)
            << "FlowEdge Relay producer->EDF->head-worker->consumer\n"
            << "requests=" << iterations << " steps=6 solver=heun threads="
            << (threads ? std::to_string(*threads) : std::string{"auto"}) << '\n'
            << "wire_bytes(condition/action)=" << wire_size(*outbound) << '/'
            << wire_size(*consumed) << " slot_capacity=" << sizeof(ConditionMessage) << '/'
            << sizeof(ActionMessage) << '\n'
            << "mean_us=" << (sum / static_cast<double>(iterations))
            << " p50_us=" << percentile(latencies_us, 0.50) << " p99_us=" << p99_us
            << " p999_us=" << percentile(latencies_us, 0.999) << '\n'
            << "p99_ns_per_nfe=" << p99_ns_per_nfe << '\n'
            << "throughput_req_s=" << (static_cast<double>(iterations) / seconds)
            << " hot_allocations=" << hot_allocations << " accepted=" << stats.accepted
            << " stale=" << stats.stale << " unreachable=" << stats.unreachable
            << " expired=" << stats.expired << " evicted=" << stats.evicted
            << " full=" << stats.full << '\n';
  return 0;
}
