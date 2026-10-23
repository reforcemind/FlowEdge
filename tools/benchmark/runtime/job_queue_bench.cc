#include "relay/adapters/routed_adapters.h"
#include "relay/worker/job_worker_pool.h"

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <span>
#include <string_view>
#include <thread>

namespace {

using namespace fe::relay;

std::atomic<std::size_t>
    g_allocations{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

struct GateBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept
  {
    return request.empty();
  }

  [[nodiscard]] bool begin() noexcept { return true; }

  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept
  {
    if (budget == 0uz)
      return {};
    bool closed{false};
    while (!gate.load(std::memory_order_acquire))
      gate.wait(closed, std::memory_order_relaxed);
    return {.step = BackendStep::kComplete, .completed_work_units = 1uz};
  }

  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 0uz; }
  [[nodiscard]] bool save_state(std::span<std::byte> state) const noexcept { return state.empty(); }
  [[nodiscard]] bool load_state(std::span<const std::byte> state) noexcept { return state.empty(); }
  [[nodiscard]] std::span<const std::byte> result() const noexcept { return {}; }

  std::atomic_bool gate{false};
};

static_assert(RoutedBackend<GateBackend>);

[[nodiscard]] JobDescriptor descriptor() noexcept
{
  JobDescriptor result{};
  result.kind = JobKind::kIterative;
  result.model_digest[0] = 1u;
  result.state_schema = 0x71756575652d7631u; // "queue-v1"
  result.session_id = 1u;
  result.generation = 1u;
  result.deadline_ns = std::numeric_limits<std::uint64_t>::max();
  result.total_work_units = 1u;
  return result;
}

} // namespace

void* operator new(std::size_t bytes)
{
  g_allocations.fetch_add(1uz, std::memory_order_relaxed);
  if (void* memory = std::malloc(bytes == 0uz ? 1uz : bytes)) // NOLINT(*-no-malloc,*-owning-memory)
    return memory;
  throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept
{
  std::free(memory); // NOLINT(*-no-malloc,*-owning-memory)
}

void operator delete(void* memory, std::size_t) noexcept
{
  std::free(memory); // NOLINT(*-no-malloc,*-owning-memory)
}

void* operator new[](std::size_t bytes)
{
  return ::operator new(bytes);
}

void operator delete[](void* memory) noexcept
{
  std::free(memory); // NOLINT(*-no-malloc,*-owning-memory)
}

void operator delete[](void* memory, std::size_t) noexcept
{
  std::free(memory); // NOLINT(*-no-malloc,*-owning-memory)
}

int main(int argc, char** argv)
{
  std::size_t rounds{2'000uz};
  if (argc > 1) {
    const std::string_view input{argv[1]};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), rounds);
    if (error != std::errc{} || end != input.data() + input.size() || rounds == 0uz) {
      std::cerr << "Usage: flowedge_job_queue_bench [rounds]\n";
      return 2;
    }
  }

  constexpr std::size_t kQueueDepth = 32uz;
  constexpr std::size_t kJobsPerRound = kQueueDepth + 1uz;
  GateBackend backend{};
  const JobDescriptor job_descriptor = descriptor();
  std::array<JobAdapterRegistration, 1> registrations{};
  JobAdapterRegistry registry{registrations};
  if (!registry.add(make_routed_adapter(backend, job_route(job_descriptor), 0uz, 0uz)))
    return 1;
  registry.freeze();
  std::array<JobAdapterRegistry, 1> registries{registry};
  auto created = JobWorkerPool::create(registries, kQueueDepth, {}, 1uz, 1uz);
  if (!created)
    return 1;
  JobWorkerPool pool = std::move(*created);

  JobRequestMessage request{};
  if (make_job_request(request, 1u, job_descriptor, 1u, {}) != ProtocolResult::kSuccess)
    return 1;

  std::uint64_t checksum{};
  std::chrono::nanoseconds elapsed{};
  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  for (std::size_t round{}; round < rounds; ++round) {
    backend.gate.store(false, std::memory_order_release);
    for (std::size_t job{}; job < kJobsPerRound; ++job) {
      request.envelope.sequence = (round * kJobsPerRound) + job + 1uz;
      if (pool.submit(request, 1u) != JobSubmitResult::kAccepted)
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    backend.gate.store(true, std::memory_order_release);
    backend.gate.notify_one();
    for (std::size_t completed{}; completed < kJobsPerRound;) {
      const JobResultMessage* const result = pool.ready_result();
      if (result == nullptr) {
        std::this_thread::yield();
        continue;
      }
      checksum += result->envelope.sequence;
      pool.release_ready_result();
      ++completed;
    }
    elapsed += std::chrono::steady_clock::now() - start;
  }
  const std::size_t allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;
  const auto jobs = static_cast<double>(rounds * kJobsPerRound);
  const auto nanoseconds = static_cast<double>(elapsed.count());
  std::cout << "rounds=" << rounds << " queue_depth=" << kQueueDepth
            << " ns_per_queued_job=" << nanoseconds / jobs
            << " queued_jobs_per_second=" << (jobs * 1'000'000'000.0) / nanoseconds
            << " hot_path_allocations=" << allocations << " checksum=" << checksum << '\n';
  return allocations == 0uz && pool.release_session(job_descriptor.session_id) ? 0 : 1;
}
