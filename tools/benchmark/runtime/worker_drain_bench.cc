#include "relay/adapters/routed_adapters.h"
#include "relay/jobs/state_codec.h"
#include "relay/worker/job_worker_pool.h"

#include <algorithm>
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
#include <vector>

namespace {

using namespace fe::relay;

std::atomic<std::size_t>
    g_allocations{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

struct DrainBackend
{
  [[nodiscard]] bool prepare(std::span<const std::byte> request) noexcept
  {
    return request.empty();
  }

  [[nodiscard]] bool begin() noexcept
  {
    completed = 0u;
    value = 1u;
    encode_result();
    return true;
  }

  [[nodiscard]] BackendAdvance advance(std::size_t budget) noexcept
  {
    const std::size_t count =
        static_cast<std::size_t>(std::min<std::uint64_t>(budget, kWorkUnits - completed));
    for (std::size_t index{}; index < count; ++index) {
      value = (value * 3u) + completed + 1u;
      ++completed;
      if (gate_first_boundary && completed == 1u) {
        boundary_reached.store(true, std::memory_order_release);
        boundary_reached.notify_one();
        bool closed{false};
        while (!release_boundary.load(std::memory_order_acquire))
          release_boundary.wait(closed, std::memory_order_relaxed);
      }
    }
    encode_result();
    return {.step = completed == kWorkUnits ? BackendStep::kComplete : BackendStep::kInProgress,
            .completed_work_units = count};
  }

  void cancel() noexcept {}
  [[nodiscard]] std::size_t state_bytes() const noexcept { return 16uz; }
  [[nodiscard]] bool save_state(std::span<std::byte> destination) const noexcept
  {
    StateWriter writer{destination};
    return writer.write(completed) && writer.write(value) && writer.remaining() == 0uz;
  }
  [[nodiscard]] bool load_state(std::span<const std::byte> source) noexcept
  {
    StateReader reader{source};
    const auto next_completed = reader.read<std::uint64_t>();
    const auto next_value = reader.read<std::uint64_t>();
    if (!next_completed || !next_value || *next_completed > kWorkUnits || reader.remaining() != 0uz)
      return false;
    completed = *next_completed;
    value = *next_value;
    encode_result();
    return true;
  }
  [[nodiscard]] std::span<const std::byte> result() const noexcept { return output; }

  void arm(bool enabled) noexcept
  {
    gate_first_boundary = enabled;
    boundary_reached.store(false, std::memory_order_relaxed);
    release_boundary.store(false, std::memory_order_relaxed);
  }

  void encode_result() noexcept
  {
    StateWriter writer{output};
    static_cast<void>(writer.write(value));
  }

  static constexpr std::uint64_t kWorkUnits = 8u;
  std::atomic_bool boundary_reached{false};
  std::atomic_bool release_boundary{false};
  bool gate_first_boundary{};
  std::uint64_t completed{};
  std::uint64_t value{};
  std::array<std::byte, sizeof(std::uint64_t)> output{};
};

static_assert(RoutedBackend<DrainBackend>);

[[nodiscard]] JobDescriptor descriptor() noexcept
{
  JobDescriptor result{};
  result.kind = JobKind::kStreaming;
  result.model_digest[0] = 1u;
  result.state_schema = 0x647261696e2d7631u; // "drain-v1"
  result.session_id = 1u;
  result.generation = 1u;
  result.total_work_units = DrainBackend::kWorkUnits;
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
  std::size_t iterations{10'000uz};
  if (argc > 1) {
    const std::string_view input{argv[1]};
    const auto [end, error] =
        std::from_chars(input.data(), input.data() + input.size(), iterations);
    if (error != std::errc{} || end != input.data() + input.size() || iterations == 0uz) {
      std::cerr << "Usage: flowedge_worker_drain_bench [iterations]\n";
      return 2;
    }
  }

  DrainBackend first{};
  DrainBackend second{};
  const JobDescriptor job_descriptor = descriptor();
  std::array<JobAdapterRegistration, 1> first_entries{};
  std::array<JobAdapterRegistration, 1> second_entries{};
  std::array<JobAdapterRegistry, 2> registries{JobAdapterRegistry{first_entries},
                                               JobAdapterRegistry{second_entries}};
  if (!registries[0].add(
          make_routed_adapter(first, job_route(job_descriptor), 0uz, sizeof(std::uint64_t))) ||
      !registries[1].add(
          make_routed_adapter(second, job_route(job_descriptor), 0uz, sizeof(std::uint64_t))))
    return 1;
  for (JobAdapterRegistry& registry : registries)
    registry.freeze();
  auto created =
      JobWorkerPool::create(registries, 1uz, {}, 1uz, 1uz, WorkerPlacement::kNone, nullptr, 256uz);
  if (!created)
    return 1;
  JobWorkerPool pool = std::move(*created);
  JobRequestMessage request{};
  if (make_job_request(request, 1u, job_descriptor, 0u, {}) != ProtocolResult::kSuccess)
    return 1;

  std::uint64_t checksum{};
  std::chrono::steady_clock::duration elapsed{};
  std::vector<std::uint64_t> samples(iterations);
  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  for (std::size_t iteration{}; iteration < iterations; ++iteration) {
    first.arm(true);
    second.arm(true);
    request.envelope.sequence = iteration + 1uz;
    if (pool.submit(request) != JobSubmitResult::kAccepted) {
      std::cerr << "submit failed at iteration " << iteration << '\n';
      return 1;
    }
    const std::size_t source_index = pool.worker_busy(0uz) ? 0uz : 1uz;
    DrainBackend& source = source_index == 0uz ? first : second;
    bool waiting{false};
    while (!source.boundary_reached.load(std::memory_order_acquire))
      source.boundary_reached.wait(waiting, std::memory_order_relaxed);
    const auto started = std::chrono::steady_clock::now();
    if (pool.request_worker_drain(source_index) != WorkerDrainResult::kStarted) {
      std::cerr << "drain request failed at iteration " << iteration << '\n';
      return 1;
    }
    source.release_boundary.store(true, std::memory_order_release);
    source.release_boundary.notify_one();

    const JobResultMessage* result{};
    while (result == nullptr) {
      result = pool.ready_result();
      if (result == nullptr)
        std::this_thread::yield();
    }
    if (validate(*result) != ProtocolResult::kSuccess ||
        job_result_code(*result) != JobResultCode::kComplete ||
        !pool.worker_drained(source_index)) {
      std::cerr << "migrated result failed at iteration " << iteration << '\n';
      std::cerr << "source=" << source_index
                << " protocol=" << static_cast<unsigned>(validate(*result))
                << " code=" << static_cast<unsigned>(job_result_code(*result))
                << " drained0=" << pool.worker_drained(0uz)
                << " drained1=" << pool.worker_drained(1uz) << '\n';
      return 1;
    }
    StateReader reader{result->payload_values()};
    const auto value = reader.read<std::uint64_t>();
    if (!value) {
      std::cerr << "result decode failed at iteration " << iteration << '\n';
      return 1;
    }
    const auto duration = std::chrono::steady_clock::now() - started;
    elapsed += duration;
    samples[iteration] = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
    checksum += *value + iteration;
    pool.release_ready_result();
    if (!pool.release_session(job_descriptor.session_id) || !pool.resume_worker(source_index)) {
      std::cerr << "drained lane release failed at iteration " << iteration << '\n';
      return 1;
    }
  }
  const std::size_t allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;
  std::ranges::sort(samples);
  const double nanoseconds =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  const std::uint64_t p50 = samples[(samples.size() - 1uz) / 2uz];
  const std::uint64_t p99 = samples[((samples.size() - 1uz) * 99uz) / 100uz];
  std::cout << "iterations=" << iterations
            << " drain_to_result_mean_ns=" << nanoseconds / static_cast<double>(iterations)
            << " drain_to_result_p50_ns=" << p50 << " drain_to_result_p99_ns=" << p99
            << " hot_path_allocations=" << allocations << " checksum=" << checksum << '\n';
  return allocations == 0uz ? 0 : 1;
}
