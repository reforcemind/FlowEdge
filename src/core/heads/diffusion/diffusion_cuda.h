#pragma once

#include "heads/diffusion/diffusion_precision.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace fe {

// Persistent device copies of dp.* weights and U-Net/DDIM scratch. Load
// allocates; denoise/sample must not cudaMalloc.
class CudaDiffusionResident
{
public:
  static constexpr std::size_t kMaxStages = 8;

  struct ConvHost
  {
    const float* weight{};
    const float* bias{};
    std::size_t in_channels{};
    std::size_t out_channels{};
    std::size_t kernel{};
    bool k_major{};
    bool approximate_candidate{};
  };

  struct NormHost
  {
    const float* weight{};
    const float* bias{};
    std::size_t channels{};
  };

  struct LinearHost
  {
    const float* weight{};
    const float* bias{};
    std::size_t in_features{};
    std::size_t out_features{};
  };

  struct ResidualHost
  {
    ConvHost conv1{};
    NormHost norm1{};
    LinearHost film{};
    ConvHost conv2{};
    NormHost norm2{};
    ConvHost residual{};
    bool identity_residual{};
  };

  struct DownHost
  {
    ResidualHost residuals[2]{};
    ConvHost downsample{};
    bool identity_downsample{};
  };

  struct UpHost
  {
    ResidualHost residuals[2]{};
    ConvHost upsample{};
  };

  struct LoadSpec
  {
    DiffusionPrecision precision{DiffusionPrecision::ExactF32};
    std::size_t action_dim{};
    std::size_t horizon{};
    std::size_t condition_dim{};
    std::size_t stages{};
    std::size_t groups{};
    std::size_t timestep_dim{};
    std::size_t workspace_floats{};
    std::array<std::size_t, kMaxStages> dims{};
    LinearHost timestep_in{};
    LinearHost timestep_out{};
    std::array<DownHost, kMaxStages> down{};
    std::array<ResidualHost, 2> middle{};
    std::array<UpHost, kMaxStages - 1> up{};
    ConvHost final_conv{};
    NormHost final_norm{};
    ConvHost output_conv{};
    const float* action_min{};
    const float* action_max{};
  };

  CudaDiffusionResident() = default;
  CudaDiffusionResident(const CudaDiffusionResident&) = delete;
  CudaDiffusionResident(CudaDiffusionResident&&) = delete;
  CudaDiffusionResident& operator=(const CudaDiffusionResident&) = delete;
  CudaDiffusionResident& operator=(CudaDiffusionResident&&) = delete;

  void release() noexcept;
  [[nodiscard]] bool load(const LoadSpec& spec) noexcept;
  [[nodiscard]] bool denoise(std::span<const float> condition,
                             std::span<const float> normalized_sample, float timestep,
                             std::span<float> predicted_noise) noexcept;
  [[nodiscard]] bool begin(std::span<const float> condition,
                           std::span<const float> normalized_sample) noexcept;
  [[nodiscard]] bool begin_device(const float* condition, const float* normalized_sample) noexcept;
  [[nodiscard]] bool denoise_current(float timestep) noexcept;
  void ddim_update(float sqrt_alpha_t, float sqrt_beta_t, float sqrt_alpha_prev,
                   float sqrt_one_minus_prev, bool clip, float clip_range) noexcept;
  [[nodiscard]] bool ddpm_update(float sqrt_alpha_t, float sqrt_beta_t, float original_coefficient,
                                 float sample_coefficient, float sqrt_variance, bool clip,
                                 float clip_range, const float* host_noise) noexcept;
  [[nodiscard]] bool copy_x(std::span<float> out) noexcept;
  [[nodiscard]] bool copy_action_device(float* out, std::size_t action_start,
                                        std::size_t action_steps) noexcept;
  [[nodiscard]] int device_id() const noexcept { return device_id_; }

private:
  struct Conv
  {
    float* weight{};
    void* half_weight{};
    float* bias{};
    std::size_t in_channels{};
    std::size_t out_channels{};
    std::size_t kernel{};
    bool k_major{};
  };

  struct Norm
  {
    float* weight{};
    float* bias{};
    std::size_t channels{};
  };

  struct Linear
  {
    float* weight{};
    float* bias{};
    std::size_t in_features{};
    std::size_t out_features{};
  };

  struct Residual
  {
    Conv conv1{};
    Norm norm1{};
    Linear film{};
    Conv conv2{};
    Norm norm2{};
    Conv residual{};
    bool identity_residual{};
  };

  struct Down
  {
    Residual residuals[2]{};
    Conv downsample{};
    bool identity_downsample{};
  };

  struct Up
  {
    Residual residuals[2]{};
    Conv upsample{};
  };

  struct Pack
  {
    float* base{};
    std::size_t cap{};
    std::size_t used{};
  };

  struct HalfPack
  {
    std::uint16_t* base{};
    std::size_t cap{};
    std::size_t used{};
  };

  struct Bump
  {
    float* base{};
    std::size_t cap{};
    std::size_t used{};
  };

  [[nodiscard]] static std::size_t align16(std::size_t n) noexcept;
  [[nodiscard]] static std::size_t weight_floats(const LoadSpec& spec) noexcept;
  [[nodiscard]] static std::size_t half_weight_values(const LoadSpec& spec) noexcept;
  [[nodiscard]] bool pack_put(Pack& pack, const float* host, std::size_t count,
                              float*& dst) noexcept;
  [[nodiscard]] bool upload_conv(Pack& pack, HalfPack& half_pack, const LoadSpec& spec,
                                 const ConvHost& src, Conv& dst) noexcept;
  [[nodiscard]] bool upload_norm(Pack& pack, const NormHost& src, Norm& dst) noexcept;
  [[nodiscard]] bool upload_linear(Pack& pack, const LinearHost& src, Linear& dst) noexcept;
  [[nodiscard]] bool upload_residual(Pack& pack, HalfPack& half_pack, const LoadSpec& spec,
                                     const ResidualHost& src, Residual& dst) noexcept;
  [[nodiscard]] float* bump_alloc(Bump& bump, std::size_t n) noexcept;
  [[nodiscard]] bool residual_forward(float* input, std::size_t length, const Residual& weights,
                                      float* condition_mish, Bump& bump, float* output) noexcept;
  [[nodiscard]] bool denoise_from_x(float timestep) noexcept;

  std::size_t action_dim_{};
  std::size_t horizon_{};
  std::size_t condition_dim_{};
  std::size_t stages_{};
  std::size_t groups_{};
  std::size_t timestep_dim_{};
  std::size_t values_{};
  std::array<std::size_t, kMaxStages> dims_{};
  Linear timestep_in_{};
  Linear timestep_out_{};
  std::array<Down, kMaxStages> down_{};
  std::array<Residual, 2> middle_{};
  std::array<Up, kMaxStages - 1> up_{};
  Conv final_conv_{};
  Norm final_norm_{};
  Conv output_conv_{};
  float* action_min_{};
  float* action_max_{};
  float* weights_{};
  std::uint16_t* half_weights_{};
  float* scratch_{};
  float* condition_{};
  float* x_{};
  float* predicted_{};
  float* gaussian_{};
  float* bump_{};
  std::size_t bump_cap_{};
  int device_id_{-1};
};

} // namespace fe
