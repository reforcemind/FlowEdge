#include "relay/adapters/cooperative_adapters.h"
#include "relay/adapters/mamba_stream_adapter.h"
#include "relay/jobs/state_codec.h"

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace fe::relay;

std::atomic<std::size_t>
    g_allocations{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

[[nodiscard]] std::uint32_t result_bits(std::span<const std::byte> result) noexcept
{
  StateReader reader{result};
  const auto bits = reader.read<std::uint32_t>();
  return bits.value_or(0u);
}

[[nodiscard]] double ns_per_job(std::chrono::steady_clock::duration elapsed,
                                std::size_t iterations) noexcept
{
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()) /
         static_cast<double>(iterations);
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
  const std::string model = argc > 1 ? argv[1] : "models/mamba_flow.safetensors";
  std::size_t iterations{500uz};
  if (argc > 2) {
    const std::string_view input{argv[2]};
    const auto [end, error] =
        std::from_chars(input.data(), input.data() + input.size(), iterations);
    if (error != std::errc{} || end != input.data() + input.size() || iterations == 0uz) {
      std::cerr << "Usage: flowedge_mamba_stream_bench [model] [iterations]\n";
      return 2;
    }
  }

  constexpr std::array<std::int32_t, 4> tokens{1, 2, 3, 4};
  std::unique_ptr<fe_weights, decltype(&fe_weights_free)> weights{fe_weights_load(model.c_str()),
                                                                  fe_weights_free};
  if (!weights) {
    std::cerr << fe_engine_last_error() << '\n';
    return 1;
  }
  auto opened_direct = MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  auto opened_routed = MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  auto opened_source = MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  auto opened_destination = MambaStreamAdapter::open(weights.get(), tokens.size(), 0u);
  if (!opened_direct || !opened_routed || !opened_source || !opened_destination) {
    std::cerr << "Failed to open Mamba stream benchmark adapter\n";
    return 1;
  }
  MambaStreamAdapter direct = std::move(*opened_direct);
  MambaStreamAdapter routed_backend = std::move(*opened_routed);
  MambaStreamAdapter source = std::move(*opened_source);
  MambaStreamAdapter destination = std::move(*opened_destination);
  weights.reset();

  std::array<std::byte, tokens.size() * sizeof(std::int32_t)> request_storage{};
  const auto encoded = encode_mamba_stream_request(tokens, request_storage);
  if (!encoded)
    return 1;
  const auto payload = std::span{request_storage}.first(*encoded);
  const JobDescriptor descriptor = direct.make_descriptor(1u, 1u, tokens.size());
  if (!direct.prepare(payload) || !source.prepare(payload))
    return 1;

  std::array<JobAdapterRegistration, 1> registrations{};
  JobAdapterRegistry registry{registrations};
  if (!registry.add(routed_backend.registration()))
    return 1;
  registry.freeze();
  JobRequestMessage request{};
  if (make_job_request(request, 1u, descriptor, 0u, payload) != ProtocolResult::kSuccess)
    return 1;
  JobResultMessage result{};

  auto capsule_job = make_streaming_job(source, descriptor);
  if (!capsule_job)
    return 1;
  std::vector<std::byte> capsule(capsule_job->capsule_bytes());

  std::uint64_t direct_checksum{};
  std::uint64_t routed_checksum{};
  std::uint64_t migrated_checksum{};
  std::chrono::steady_clock::duration direct_elapsed{};
  std::chrono::steady_clock::duration routed_elapsed{};
  std::chrono::steady_clock::duration migrated_elapsed{};
  const auto run_direct = [&](std::size_t iteration) noexcept {
    const BackendAdvance completed =
        direct.begin() ? direct.advance(tokens.size()) : BackendAdvance{};
    if (completed.step != BackendStep::kComplete)
      return false;
    direct_checksum += result_bits(direct.result()) + iteration;
    return true;
  };
  const auto run_routed = [&](std::size_t iteration) noexcept {
    auto routed = registry.bind(request);
    if (!routed || !routed->start() || !routed->advance(tokens.size()) ||
        routed->write_result(result, 0u) != ProtocolResult::kSuccess)
      return false;
    routed_checksum += result_bits(result.payload_values()) + iteration;
    return true;
  };
  const auto run_migrated = [&](std::size_t iteration) noexcept {
    auto source_job = make_streaming_job(source, descriptor);
    if (!source_job || !source_job->start() || !source_job->advance(2uz))
      return false;
    const auto exported = source_job->export_capsule(capsule);
    auto destination_job = make_streaming_job(destination, descriptor);
    if (!exported || !destination_job ||
        !destination_job->restore_capsule(std::span{capsule}.first(*exported)) ||
        !destination_job->advance(tokens.size() - 2uz))
      return false;
    migrated_checksum += result_bits(destination.result()) + iteration;
    return true;
  };
  const auto measure = []<typename Operation>(std::chrono::steady_clock::duration& elapsed,
                                              const Operation& operation,
                                              std::size_t iteration) noexcept {
    const auto start = std::chrono::steady_clock::now();
    const bool succeeded = operation(iteration);
    elapsed += std::chrono::steady_clock::now() - start;
    return succeeded;
  };

  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  for (std::size_t iteration{}; iteration < iterations; ++iteration) {
    bool succeeded{};
    switch (iteration % 3uz) {
    case 0uz:
      succeeded = measure(direct_elapsed, run_direct, iteration) &&
                  measure(routed_elapsed, run_routed, iteration) &&
                  measure(migrated_elapsed, run_migrated, iteration);
      break;
    case 1uz:
      succeeded = measure(routed_elapsed, run_routed, iteration) &&
                  measure(migrated_elapsed, run_migrated, iteration) &&
                  measure(direct_elapsed, run_direct, iteration);
      break;
    default:
      succeeded = measure(migrated_elapsed, run_migrated, iteration) &&
                  measure(direct_elapsed, run_direct, iteration) &&
                  measure(routed_elapsed, run_routed, iteration);
      break;
    }
    if (!succeeded)
      return 1;
  }
  const std::size_t allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;

  const double direct_ns = ns_per_job(direct_elapsed, iterations);
  const double routed_ns = ns_per_job(routed_elapsed, iterations);
  const double migrated_ns = ns_per_job(migrated_elapsed, iterations);
  std::cout << "iterations=" << iterations << " tokens=" << tokens.size()
            << " direct_ns_per_job=" << direct_ns << " routed_ns_per_job=" << routed_ns
            << " route_overhead_ns_per_job=" << routed_ns - direct_ns
            << " migrate_finish_ns_per_job=" << migrated_ns << " capsule_bytes=" << capsule.size()
            << " hot_path_allocations=" << allocations
            << " checksum=" << (direct_checksum ^ routed_checksum ^ migrated_checksum) << '\n';
  return allocations == 0uz && direct_checksum == routed_checksum &&
                 direct_checksum == migrated_checksum
             ? 0
             : 1;
}
