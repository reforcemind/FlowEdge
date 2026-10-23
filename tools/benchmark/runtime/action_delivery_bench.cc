#include "relay/client/action_delivery.h"

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <ranges>
#include <span>
#include <string_view>

namespace {

using namespace fe::relay;

std::atomic<std::size_t>
    g_allocations{}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

[[nodiscard]] fe_model_metadata model_metadata() noexcept
{
  fe_model_metadata model{};
  model.struct_size = sizeof(model);
  model.protocol_version = FE_PROTOCOL_VERSION;
  model.architecture = FE_ARCH_FLOW_HEAD;
  model.precision = FE_PRECISION_F32;
  model.condition_dim = 4u;
  model.action_dim = 8u;
  model.model_digest.bytes[0] = 1u;
  return model;
}

[[nodiscard]] ActionMessage action_message(const fe_model_metadata& model,
                                           std::span<const float> values) noexcept
{
  ActionMessage action{};
  action.envelope.kind = MessageKind::kAction;
  action.envelope.sequence = 1u;
  action.envelope.session_id = 1u;
  action.metadata.struct_size = sizeof(fe_action_metadata);
  action.metadata.protocol_version = FE_PROTOCOL_VERSION;
  action.metadata.solver = 0u;
  action.metadata.status = FE_ACTION_COMPLETE;
  action.metadata.model_digest = model.model_digest;
  action.metadata.timestamp_ns = 1u;
  action.metadata.generation = 1u;
  action.metadata.condition_dim = model.condition_dim;
  action.metadata.action_dim = model.action_dim;
  std::ranges::copy(values, action.action.begin());
  action.envelope.struct_size = static_cast<std::uint32_t>(wire_size(action));
  return action;
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
      std::cerr << "Usage: flowedge_action_delivery_bench [iterations]\n";
      return 2;
    }
  }

  constexpr std::array lower{-2.0F, -2.0F};
  constexpr std::array upper{2.0F, 2.0F};
  constexpr std::array delta{2.0F, 2.0F};
  constexpr std::array old_values{0.0F, 0.0F, 0.2F, 0.1F, 0.4F, 0.2F, 0.6F, 0.3F};
  constexpr std::array new_values{0.1F, -0.1F, 0.3F, -0.2F, 0.5F, -0.3F, 0.7F, -0.4F};
  const fe_model_metadata model = model_metadata();
  auto created =
      ActionDeliveryGate::create(model,
                                 ActionDeliveryConfig{.control_dim = 2uz,
                                                      .overlap_steps = 2uz,
                                                      .step_period_ns = 1'000u,
                                                      .unsafe_policy = UnsafeActionPolicy::kReject},
                                 ActionSafetyLimits{.lower = lower,
                                                    .upper = upper,
                                                    .max_delta_per_step = delta});
  if (!created)
    return 1;
  ActionDeliveryGate gate = std::move(*created);
  ActionMessage first = action_message(model, old_values);
  ActionMessage replacement = action_message(model, new_values);
  replacement.envelope.sequence = 2u;
  replacement.metadata.generation = 2u;

  std::uint64_t checksum{};
  const std::size_t allocations_before = g_allocations.load(std::memory_order_relaxed);
  const auto direct_started = std::chrono::steady_clock::now();
  for (std::size_t iteration{}; iteration < iterations; ++iteration) {
    gate.reset();
    first.envelope.sequence = iteration + 1uz;
    first.metadata.generation = iteration + 1uz;
    if (gate.accept(first, 1u, 1u) != ActionChunkAcceptResult::kAccepted)
      return 1;
    for (std::uint64_t step{}; step < 4u; ++step) {
      const auto output = gate.next(1u + (step * 1'000u));
      if (!output)
        return 1;
      checksum += static_cast<std::uint64_t>((output->values[0] + 2.0F) * 100.0F);
    }
  }
  const auto direct_elapsed = std::chrono::steady_clock::now() - direct_started;

  const auto replace_started = std::chrono::steady_clock::now();
  for (std::size_t iteration{}; iteration < iterations; ++iteration) {
    gate.reset();
    first.envelope.sequence = 1u;
    first.metadata.generation = 1u;
    replacement.envelope.sequence = 2u;
    replacement.metadata.generation = 2u;
    if (gate.accept(first, 1u, 1u) != ActionChunkAcceptResult::kAccepted || !gate.next(1u) ||
        gate.accept(replacement, 1'001u, 1'001u) != ActionChunkAcceptResult::kReplaced) {
      return 1;
    }
    const auto output = gate.next(1'001u);
    if (!output || !output->blended)
      return 1;
    checksum += static_cast<std::uint64_t>((output->values[0] + 2.0F) * 100.0F);
  }
  const auto replace_elapsed = std::chrono::steady_clock::now() - replace_started;
  const std::size_t allocations =
      g_allocations.load(std::memory_order_relaxed) - allocations_before;
  const double direct_ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(direct_elapsed).count());
  const double replace_ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(replace_elapsed).count());
  std::cout << "iterations=" << iterations
            << " direct_accept_publish_ns_per_step=" << direct_ns / (iterations * 4.0)
            << " overlap_replace_publish_ns=" << replace_ns / static_cast<double>(iterations)
            << " hot_path_allocations=" << allocations << " checksum=" << checksum << '\n';
  return allocations == 0uz ? 0 : 1;
}
