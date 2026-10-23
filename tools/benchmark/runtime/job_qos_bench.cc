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
  result.state_schema = 0x716f732d7631u; // "qos-v1"
  result.session_id = 1u;
  result.generation = 1u;
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
  std::size_t iterations{1'000'000uz};
  if (argc > 1) {
    const std::string_view input{argv[1]};
    const auto [end, error] =
        std::from_chars(input.data(), input.data() + input.size(), iterations);
    if (error != std::errc{} || end != input.data() + input.size() || iterations == 0uz) {
      std::cerr << "Usage: flowedge_job_qos_bench [iterations]\n";
      return 2;
    }
  }

  GateBackend backend{};
  const JobDescriptor job_descriptor = descriptor();
  std::array<JobAdapterRegistration, 1> registrations{};
  JobAdapterRegistry registry{registrations};
  if (!registry.add(make_routed_adapter(backend, job_route(job_descriptor), 0uz, 0uz)))
    return 1;
  registry.freeze();
  std::array<JobAdapterRegistry, 1> registries{registry};
  auto created =
      JobWorkerPool::create(registries, 4uz, {}, 1uz, 1uz, WorkerPlacement::kNone, nullptr, 0uz,
                            JobQosPolicy{.interactive_reserve_slots = 1uz,
                                         .critical_reserve_slots = 1uz});
  if (!created)
    return 1;
  JobWorkerPool pool = std::move(*created);

  JobRequestMessage request{};
  if (make_job_request(request, 1u, job_descriptor, 0u, {}, JobServiceClass::kBestEffort) !=
      ProtocolResult::kSuccess)
    return 1;
  for (std::uint64_t sequence{1u}; sequence <= 3u; ++sequence) {
    request.envelope.sequence = sequence;
    if (pool.submit(request) != JobSubmitResult::kAccepted)
      return 1;
  }

  JobResultMessage rejection{};
  request.envelope.sequence = 4u;
  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t iteration{}; iteration < iterations; ++iteration) {
    request.envelope.sequence = static_cast<std::uint64_t>(iteration) + 4u;
    if (pool.submit(request, 0u, &rejection) != JobSubmitResult::kQosCapacity ||
        job_result_code(rejection) != JobResultCode::kRejectedQos ||
        rejection.metadata.service_class != JobServiceClass::kBestEffort)
      return 1;
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const std::size_t allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;

  backend.gate.store(true, std::memory_order_release);
  backend.gate.notify_one();
  for (std::size_t completed{}; completed < 3uz;) {
    const JobResultMessage* result = pool.ready_result();
    if (result == nullptr) {
      std::this_thread::yield();
      continue;
    }
    pool.release_ready_result();
    ++completed;
  }
  if (!pool.release_session(job_descriptor.session_id))
    return 1;

  const double elapsed_ns =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  std::cout << "iterations=" << iterations
            << " ns_per_qos_rejection=" << elapsed_ns / static_cast<double>(iterations)
            << " qos_rejections=" << pool.qos_rejection_count()
            << " hot_path_allocations=" << allocations << '\n';
  return allocations == 0uz && pool.qos_rejection_count() == iterations ? 0 : 1;
}
