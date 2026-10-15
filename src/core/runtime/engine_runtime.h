#pragma once

#include "arena/arena.h"
#include "arena/thread_pool.h"
#include "heads/diffusion/diffusion.h"
#include "heads/flow/flow.h"
#include "loader/safetensors.h"
#include "models/mamba/mamba.h"
#include "models/smolvla/smolvla_action_expert.h"
#include "models/transformer/transformer.h"
#include "protocol/model_identity.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

namespace fe {

struct FlowRequestMetadata
{
  std::uint64_t timestamp_ns{};
  std::uint64_t deadline_ns{};
  std::uint64_t generation{};
  std::size_t steps{};
  FlowHead::Method method{FlowHead::kEuler};
  bool generation_tracking{false};
};

// Owns the allocation-free policy runtime behind the C ABI.
class EngineRuntime
{
public:
  static constexpr unsigned kMaxPoolThreads = 8u;

  [[nodiscard]] static std::size_t required_slab_bytes(const ModelWeights& weights) noexcept;
  [[nodiscard]] static unsigned recommended_thread_count() noexcept;

  EngineRuntime(std::shared_ptr<const ModelWeights> weights, std::size_t slab_bytes,
                unsigned worker_threads, DiffusionPrecision diffusion_precision,
                const char*& error);
  EngineRuntime(std::shared_ptr<const ModelWeights> weights, std::size_t slab_bytes,
                unsigned worker_threads, const char*& error);
  ~EngineRuntime();

  EngineRuntime(const EngineRuntime&) = delete;
  EngineRuntime(EngineRuntime&&) = delete;
  EngineRuntime& operator=(const EngineRuntime&) = delete;
  EngineRuntime& operator=(EngineRuntime&&) = delete;

  [[nodiscard]] bool valid() const noexcept { return ready_; }
  [[nodiscard]] bool cuda_capable() const noexcept
  {
    return model_.valid() || flow_.valid() || diffusion_.valid();
  }
  [[nodiscard]] bool cuda_resident() const noexcept
  {
    return model_.cuda_resident() || flow_.cuda_resident() || diffusion_.cuda_resident();
  }
  [[nodiscard]] int diffusion_cuda_device() const noexcept
  {
#ifdef FLOWEDGE_CUDA
    return diffusion_.cuda_device_id();
#else
    return -1;
#endif
  }
  [[nodiscard]] bool has_compatible_flow_head() const noexcept;
  [[nodiscard]] std::size_t d_model() const noexcept;
  [[nodiscard]] std::size_t n_layers() const noexcept;
  [[nodiscard]] unsigned thread_count() const noexcept;
  [[nodiscard]] const ModelIdentity& identity() const noexcept { return identity_; }
  [[nodiscard]] const DeploymentProfile* deployment_profile() const noexcept
  {
    return weights_ ? weights_->deployment_profile() : nullptr;
  }
  [[nodiscard]] const FlowRequestMetadata& flow_request_metadata() const noexcept
  {
    return flow_request_;
  }
  [[nodiscard]] std::uint32_t flow_action_status() const noexcept;
  [[nodiscard]] std::uint64_t flow_remaining_nfe() const noexcept;
  [[nodiscard]] std::size_t action_dim() const noexcept;
  [[nodiscard]] std::size_t action_horizon() const noexcept;
  [[nodiscard]] std::size_t condition_dim() const noexcept;
  [[nodiscard]] const DiffusionConfig* diffusion_config() const noexcept
  {
    return diffusion_.valid() ? &diffusion_.config() : nullptr;
  }
  [[nodiscard]] DiffusionPrecision diffusion_precision() const noexcept
  {
    return diffusion_.precision();
  }
  [[nodiscard]] std::size_t decode_state_bytes() const noexcept;

  int run(const std::int32_t* tokens, std::size_t seq_len, float* out, const char*& error) noexcept;
  int run_embeddings(const float* embeddings, std::size_t seq_len, float* out,
                     const char*& error) noexcept;
  int run_embeddings_masked(const float* embeddings, const std::uint8_t* attention_mask,
                            std::size_t seq_len, float* out, const char*& error) noexcept;
  int smolvla_embed_suffix(const float* noisy_actions, std::size_t chunk_size, float timestep,
                           float* out, const char*& error) noexcept;
  int smolvla_run_expert(const float* suffix, std::size_t chunk_size, const float* prefix_keys,
                         const float* prefix_values, const std::uint8_t* prefix_mask,
                         std::size_t prefix_length, float* out, const char*& error) noexcept;
  int smolvla_project_actions(const float* hidden, std::size_t chunk_size, float* out,
                              const char*& error) noexcept;
  int smolvla_denoise(const float* noisy_actions, std::size_t chunk_size, float timestep,
                      const float* prefix_keys, const float* prefix_values,
                      const std::uint8_t* prefix_mask, std::size_t prefix_length, float* out,
                      const char*& error) noexcept;
  int smolvla_sample(const float* initial_noise, std::size_t chunk_size, std::size_t steps,
                     const float* prefix_keys, const float* prefix_values,
                     const std::uint8_t* prefix_mask, std::size_t prefix_length, float* out,
                     const char*& error) noexcept;
  int step(std::int32_t token, float* out, const char*& error) noexcept;
  void reset() noexcept;
  int sample(const std::int32_t* tokens, std::size_t seq_len, const float* noise, std::size_t steps,
             FlowHead::Method method, float* action, const char*& error) noexcept;
  int sample_condition(const float* condition, const float* noise, std::size_t steps,
                       FlowHead::Method method, float* action, const char*& error) noexcept;
  int sample_diffusion(const float* condition, const float* noise, std::size_t steps,
                       DiffusionHead::Scheduler scheduler, std::uint64_t seed, float* action,
                       const char*& error) noexcept;
  int sample_diffusion_device(const float* condition, const float* noise, std::size_t steps,
                              DiffusionHead::Scheduler scheduler, std::uint64_t seed,
                              std::size_t action_start, std::size_t action_steps, float* action,
                              const char*& error) noexcept;
  int denoise_diffusion(const float* condition, const float* normalized_sample, float timestep,
                        float* predicted_noise, const char*& error) noexcept;
  int flow_begin(const float* condition, const float* noise, std::size_t steps,
                 FlowHead::Method method, const char*& error) noexcept;
  int flow_begin_request(const float* condition, const float* noise,
                         const FlowRequestMetadata& metadata, const char*& error) noexcept;
  int flow_advance(std::size_t step_budget, float* action, std::size_t& steps_remaining,
                   const char*& error) noexcept;
  void cancel_before(std::uint64_t generation) noexcept;
  int export_decode_state(std::span<std::byte> destination, const char*& error) const noexcept;
  int import_decode_state(std::span<const std::byte> source, const char*& error) noexcept;

private:
  static constexpr std::size_t kThreadRingSlots = 128uz;

  int run_backbone(const std::int32_t* tokens, std::size_t seq_len, float* hidden,
                   const char*& error) noexcept;
  [[nodiscard]] bool has_backbone() const noexcept
  {
    return model_.valid() || transformer_.valid();
  }

  std::shared_ptr<const ModelWeights> weights_{};
  std::vector<std::byte> slab_;
  Arena arena_;
  Mamba model_;
  SmolVLAActionExpert smolvla_;
  Transformer transformer_;
  FlowHead flow_;
  DiffusionHead diffusion_;
  ModelIdentity identity_{};
  ThreadPool* pool_{};
  std::span<std::jthread> workers_{};
  std::span<float> decode_state_{};
  std::span<float> flow_workspace_{};
  std::span<float> diffusion_workspace_{};
  FlowHead::SamplerState flow_state_{};
  FlowRequestMetadata flow_request_{};
  std::atomic<std::uint64_t> latest_generation_{0u};
  bool ready_{};
};

} // namespace fe
