#pragma once

#include "arena/arena.h"
#include "heads/diffusion/diffusion_precision.h"
#include "loader/safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace fe {

class ThreadPool;
#ifdef FLOWEDGE_CUDA
class CudaDiffusionResident;
#endif

struct DiffusionConfig
{
  std::size_t action_dim{};
  std::size_t horizon{};
  std::size_t action_steps{};
  std::size_t observation_steps{};
  std::size_t condition_dim{};
  std::size_t stages{};
  std::size_t kernel{};
  std::size_t groups{};
  std::size_t timestep_dim{};
  std::size_t train_timesteps{};
  bool clip_sample{};
  float clip_sample_range{};
};

// Fixed-shape LeRobot ConditionalUnet1D action head. Inputs and outputs exposed
// by this class use [horizon][action_dim]; internal convolution buffers use
// [channels][horizon]. All inference storage is supplied by the caller.
class DiffusionHead
{
public:
  static constexpr std::size_t kMaxStages = 8uz;

  enum Scheduler : int
  {
    kDDIM = 0,
    kDDPM = 1,
  };

  DiffusionHead(std::span<const TensorView> weights, Arena& persistent,
                DiffusionPrecision precision = DiffusionPrecision::ExactF32) noexcept;
  ~DiffusionHead();
  DiffusionHead(const DiffusionHead&) = delete;
  DiffusionHead(DiffusionHead&&) = delete;
  DiffusionHead& operator=(const DiffusionHead&) = delete;
  DiffusionHead& operator=(DiffusionHead&&) = delete;

  [[nodiscard]] bool valid() const noexcept { return ok_; }
  [[nodiscard]] bool cuda_resident() const noexcept
  {
#ifdef FLOWEDGE_CUDA
    return cuda_ != nullptr;
#else
    return false;
#endif
  }
  [[nodiscard]] const DiffusionConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] DiffusionPrecision precision() const noexcept { return precision_; }
  [[nodiscard]] std::size_t sample_values() const noexcept
  {
    return cfg_.horizon * cfg_.action_dim;
  }
  [[nodiscard]] std::size_t sampler_workspace_size() const noexcept { return workspace_floats_; }
  [[nodiscard]] static std::size_t required_workspace_floats(
      std::span<const TensorView> weights) noexcept;
  [[nodiscard]] static std::size_t required_persistent_floats(
      std::span<const TensorView> weights) noexcept;

  void set_pool(ThreadPool* pool) noexcept { pool_ = pool; }

  // Predict epsilon for one scheduler timestep.
  [[nodiscard]] bool denoise(std::span<const float> condition,
                             std::span<const float> normalized_sample, float timestep,
                             std::span<float> workspace, std::span<float> predicted_noise) noexcept;

  // DDIM is deterministic for identical condition/noise. DDPM uses `seed` for
  // its per-step Gaussian noise and is deterministic for an identical seed.
  [[nodiscard]] bool sample(std::span<const float> condition, std::span<const float> initial_noise,
                            std::size_t inference_steps, Scheduler scheduler, std::uint64_t seed,
                            std::span<float> workspace, std::span<float> action) noexcept;
#ifdef FLOWEDGE_CUDA
  // CUDA-tensor entry point. Inputs and output are device pointers on cuda_device_id().
  // The output contains a compact action slice beginning at action_start.
  [[nodiscard]] bool sample_device(const float* condition, const float* initial_noise,
                                   std::size_t inference_steps, Scheduler scheduler,
                                   std::uint64_t seed, std::span<float> workspace, float* action,
                                   std::size_t action_start, std::size_t action_steps) noexcept;
  [[nodiscard]] int cuda_device_id() const noexcept;
#endif

private:
  struct ConvWeights
  {
    const float* weight{};
    const float* bias{};
    std::size_t in_channels{};
    std::size_t out_channels{};
    std::size_t kernel{};
    bool k_major{};
  };

  struct NormWeights
  {
    const float* weight{};
    const float* bias{};
  };

  struct LinearWeights
  {
    WeightView weight{};
    const float* bias{};
    std::size_t in_features{};
    std::size_t out_features{};
  };

  struct ResidualWeights
  {
    ConvWeights conv1{};
    NormWeights norm1{};
    LinearWeights film{};
    ConvWeights conv2{};
    NormWeights norm2{};
    ConvWeights residual{};
    bool identity_residual{};
  };

  struct DownStage
  {
    std::array<ResidualWeights, 2> residuals{};
    ConvWeights downsample{};
    bool identity_downsample{};
  };

  struct UpStage
  {
    std::array<ResidualWeights, 2> residuals{};
    ConvWeights upsample{};
  };

  [[nodiscard]] bool load_conv(std::span<const TensorView> weights, std::string_view prefix,
                               std::size_t out_channels, std::size_t in_channels,
                               std::size_t kernel, ConvWeights& result) noexcept;
  [[nodiscard]] bool load_linear(std::span<const TensorView> weights, std::string_view prefix,
                                 std::size_t out_features, std::size_t in_features,
                                 LinearWeights& result) noexcept;
  [[nodiscard]] bool load_norm(std::span<const TensorView> weights, std::string_view prefix,
                               std::size_t channels, NormWeights& result) noexcept;
  [[nodiscard]] bool load_residual(std::span<const TensorView> weights, std::string_view prefix,
                                   std::size_t in_channels, std::size_t out_channels,
                                   ResidualWeights& result) noexcept;
  [[nodiscard]] bool residual_forward(std::span<const float> input, std::size_t length,
                                      const ResidualWeights& weights,
                                      std::span<const float> condition_mish, Arena& arena,
                                      std::span<float> output) noexcept;
  [[nodiscard]] bool denoise_with_arena(std::span<const float> condition,
                                        std::span<const float> normalized_sample, float timestep,
                                        Arena& arena, std::span<float> predicted_noise) noexcept;

  DiffusionConfig cfg_{};
  DiffusionPrecision precision_{DiffusionPrecision::ExactF32};
  std::array<std::size_t, kMaxStages> dims_{};
  LinearWeights timestep_in_{};
  LinearWeights timestep_out_{};
  std::array<DownStage, kMaxStages> down_{};
  std::array<ResidualWeights, 2> middle_{};
  std::array<UpStage, kMaxStages - 1uz> up_{};
  ConvWeights final_conv_{};
  NormWeights final_norm_{};
  ConvWeights output_conv_{};
  const float* action_min_{};
  const float* action_max_{};
  const float* alphas_cumprod_{};
  std::size_t workspace_floats_{};
  ThreadPool* pool_{};
  bool ok_{};
#ifdef FLOWEDGE_CUDA
  CudaDiffusionResident* cuda_{};
#endif
};

} // namespace fe
