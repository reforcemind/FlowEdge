#include "heads/diffusion/diffusion_cuda.h"
#include "kernels/cuda/kernels_cuda_api.h"

namespace fe {
namespace {

constexpr std::size_t kFinalNormGroups = 8;
constexpr float kGroupNormEps = 1.0e-5F;

} // namespace

std::size_t CudaDiffusionResident::align16(std::size_t n) noexcept
{
  return (n + 15) & ~std::size_t(15);
}

void CudaDiffusionResident::release() noexcept
{
  cuda_ops::device_free(weights_);
  cuda_ops::device_free(scratch_);
  weights_ = scratch_ = nullptr;
  condition_ = x_ = predicted_ = gaussian_ = bump_ = nullptr;
  action_min_ = action_max_ = nullptr;
  bump_cap_ = 0;
}

bool CudaDiffusionResident::pack_put(Pack& pack, const float* host, std::size_t count,
                                     float*& dst) noexcept
{
  pack.used = align16(pack.used);
  if (host == nullptr || count == 0 || pack.base == nullptr || pack.used + count > pack.cap)
    return false;
  dst = pack.base + pack.used;
  pack.used += count;
  return cuda_ops::device_copy_h2d(dst, host, count * sizeof(float));
}

bool CudaDiffusionResident::upload_conv(Pack& pack, const ConvHost& src, Conv& dst) noexcept
{
  dst.in_channels = src.in_channels;
  dst.out_channels = src.out_channels;
  dst.kernel = src.kernel;
  dst.k_major = src.k_major;
  dst.weight = nullptr;
  dst.bias = nullptr;
  if (src.weight == nullptr)
    return src.bias == nullptr;
  const std::size_t weight_n = src.out_channels * src.in_channels * src.kernel;
  return pack_put(pack, src.weight, weight_n, dst.weight) &&
         pack_put(pack, src.bias, src.out_channels, dst.bias);
}

bool CudaDiffusionResident::upload_norm(Pack& pack, const NormHost& src, Norm& dst) noexcept
{
  dst.channels = src.channels;
  return pack_put(pack, src.weight, src.channels, dst.weight) &&
         pack_put(pack, src.bias, src.channels, dst.bias);
}

bool CudaDiffusionResident::upload_linear(Pack& pack, const LinearHost& src, Linear& dst) noexcept
{
  dst.in_features = src.in_features;
  dst.out_features = src.out_features;
  const std::size_t weight_n = src.out_features * src.in_features;
  return pack_put(pack, src.weight, weight_n, dst.weight) &&
         pack_put(pack, src.bias, src.out_features, dst.bias);
}

bool CudaDiffusionResident::upload_residual(Pack& pack, const ResidualHost& src,
                                            Residual& dst) noexcept
{
  dst.identity_residual = src.identity_residual;
  if (!upload_conv(pack, src.conv1, dst.conv1) || !upload_norm(pack, src.norm1, dst.norm1) ||
      !upload_linear(pack, src.film, dst.film) || !upload_conv(pack, src.conv2, dst.conv2) ||
      !upload_norm(pack, src.norm2, dst.norm2))
    return false;
  return dst.identity_residual || upload_conv(pack, src.residual, dst.residual);
}

std::size_t CudaDiffusionResident::weight_floats(const LoadSpec& spec) noexcept
{
  std::size_t n = 0;
  const auto add = [&](std::size_t count) noexcept {
    n = align16(n);
    n += count;
  };
  const auto add_conv = [&](const ConvHost& conv, bool used) noexcept {
    if (!used || conv.weight == nullptr)
      return;
    add(conv.out_channels * conv.in_channels * conv.kernel);
    add(conv.out_channels);
  };
  const auto add_norm = [&](const NormHost& norm) noexcept {
    add(norm.channels);
    add(norm.channels);
  };
  const auto add_linear = [&](const LinearHost& linear) noexcept {
    add(linear.out_features * linear.in_features);
    add(linear.out_features);
  };
  const auto add_residual = [&](const ResidualHost& residual) noexcept {
    add_conv(residual.conv1, true);
    add_norm(residual.norm1);
    add_linear(residual.film);
    add_conv(residual.conv2, true);
    add_norm(residual.norm2);
    add_conv(residual.residual, !residual.identity_residual);
  };
  add_linear(spec.timestep_in);
  add_linear(spec.timestep_out);
  for (std::size_t stage = 0; stage < spec.stages; ++stage) {
    add_residual(spec.down[stage].residuals[0]);
    add_residual(spec.down[stage].residuals[1]);
    add_conv(spec.down[stage].downsample, !spec.down[stage].identity_downsample);
  }
  add_residual(spec.middle[0]);
  add_residual(spec.middle[1]);
  for (std::size_t stage = 0; stage + 1 < spec.stages; ++stage) {
    add_residual(spec.up[stage].residuals[0]);
    add_residual(spec.up[stage].residuals[1]);
    add_conv(spec.up[stage].upsample, true);
  }
  add_conv(spec.final_conv, true);
  add_norm(spec.final_norm);
  add_conv(spec.output_conv, true);
  add(spec.action_dim);
  add(spec.action_dim);
  return n;
}

bool CudaDiffusionResident::load(const LoadSpec& spec) noexcept
{
  release();
  if (!cuda_ops::device_available() || spec.stages < 2 || spec.stages > kMaxStages ||
      spec.action_dim == 0 || spec.horizon == 0 || spec.condition_dim == 0 || spec.groups == 0 ||
      spec.timestep_dim == 0 || spec.workspace_floats == 0 || spec.action_min == nullptr ||
      spec.action_max == nullptr)
    return false;
  action_dim_ = spec.action_dim;
  horizon_ = spec.horizon;
  condition_dim_ = spec.condition_dim;
  stages_ = spec.stages;
  groups_ = spec.groups;
  timestep_dim_ = spec.timestep_dim;
  values_ = spec.horizon * spec.action_dim;
  dims_ = spec.dims;

  const std::size_t packed = weight_floats(spec);
  weights_ = static_cast<float*>(cuda_ops::device_alloc(packed * sizeof(float)));
  Pack pack{weights_, packed, 0};
  bool ok = weights_ != nullptr && upload_linear(pack, spec.timestep_in, timestep_in_) &&
            upload_linear(pack, spec.timestep_out, timestep_out_);
  for (std::size_t stage = 0; ok && stage < spec.stages; ++stage) {
    ok = upload_residual(pack, spec.down[stage].residuals[0], down_[stage].residuals[0]) &&
         upload_residual(pack, spec.down[stage].residuals[1], down_[stage].residuals[1]);
    down_[stage].identity_downsample = spec.down[stage].identity_downsample;
    if (ok && !down_[stage].identity_downsample)
      ok = upload_conv(pack, spec.down[stage].downsample, down_[stage].downsample);
  }
  ok = ok && upload_residual(pack, spec.middle[0], middle_[0]) &&
       upload_residual(pack, spec.middle[1], middle_[1]);
  for (std::size_t stage = 0; ok && stage + 1 < spec.stages; ++stage)
    ok = upload_residual(pack, spec.up[stage].residuals[0], up_[stage].residuals[0]) &&
         upload_residual(pack, spec.up[stage].residuals[1], up_[stage].residuals[1]) &&
         upload_conv(pack, spec.up[stage].upsample, up_[stage].upsample);
  ok = ok && upload_conv(pack, spec.final_conv, final_conv_) &&
       upload_norm(pack, spec.final_norm, final_norm_) &&
       upload_conv(pack, spec.output_conv, output_conv_) &&
       pack_put(pack, spec.action_min, spec.action_dim, action_min_) &&
       pack_put(pack, spec.action_max, spec.action_dim, action_max_);

  std::size_t scratch_n = 0;
  const auto scratch_add = [&](std::size_t count) noexcept {
    scratch_n = align16(scratch_n);
    scratch_n += count;
  };
  scratch_add(spec.condition_dim);
  scratch_add(values_);
  scratch_add(values_);
  scratch_add(values_);
  scratch_add(spec.workspace_floats);
  scratch_ = static_cast<float*>(cuda_ops::device_alloc(scratch_n * sizeof(float)));
  if (!ok || scratch_ == nullptr) {
    release();
    return false;
  }
  std::size_t off = 0;
  const auto take = [&](std::size_t count) noexcept {
    off = align16(off);
    float* const ptr = scratch_ + off;
    off += count;
    return ptr;
  };
  condition_ = take(spec.condition_dim);
  x_ = take(values_);
  predicted_ = take(values_);
  gaussian_ = take(values_);
  bump_ = take(spec.workspace_floats);
  bump_cap_ = spec.workspace_floats;
  return true;
}

float* CudaDiffusionResident::bump_alloc(Bump& bump, std::size_t n) noexcept
{
  bump.used = align16(bump.used);
  if (n == 0 || bump.base == nullptr || bump.used + n > bump.cap)
    return nullptr;
  float* const ptr = bump.base + bump.used;
  bump.used += n;
  return ptr;
}

bool CudaDiffusionResident::residual_forward(float* input, std::size_t length,
                                             const Residual& weights, float* condition_mish,
                                             Bump& bump, float* output) noexcept
{
  const std::size_t in_channels = weights.conv1.in_channels;
  const std::size_t out_channels = weights.conv1.out_channels;
  const std::size_t mark = bump.used;
  float* const first = bump_alloc(bump, out_channels * length);
  float* const modulation = bump_alloc(bump, 2 * out_channels);
  if (first == nullptr || modulation == nullptr || input == nullptr || output == nullptr ||
      condition_mish == nullptr)
    return false;
  cuda_ops::conv1d_device(input, weights.conv1.weight, weights.conv1.bias, first, in_channels,
                          out_channels, length, length, weights.conv1.kernel, 1,
                          weights.conv1.kernel / 2);
  cuda_ops::matmul_f32_device(condition_mish, weights.film.weight, modulation, 1,
                              weights.film.in_features, weights.film.out_features);
  cuda_ops::add_bias_device(modulation, weights.film.bias, weights.film.out_features);
  cuda_ops::group_norm_mish_device(first, weights.norm1.weight, weights.norm1.bias, out_channels,
                                   length, groups_, kGroupNormEps, modulation);
  cuda_ops::conv1d_device(first, weights.conv2.weight, weights.conv2.bias, output, out_channels,
                          out_channels, length, length, weights.conv2.kernel, 1,
                          weights.conv2.kernel / 2);
  const float* residual = input;
  if (!weights.identity_residual) {
    cuda_ops::conv1d_device(input, weights.residual.weight, weights.residual.bias, first,
                            in_channels, out_channels, length, length, 1, 1, 0);
    residual = first;
  }
  cuda_ops::group_norm_mish_device(output, weights.norm2.weight, weights.norm2.bias, out_channels,
                                   length, groups_, kGroupNormEps, nullptr, residual);
  bump.used = mark;
  return true;
}

bool CudaDiffusionResident::denoise_from_x(float timestep) noexcept
{
  Bump bump{bump_, bump_cap_, 0};
  const std::size_t timestep_hidden = timestep_dim_ * 4;
  const std::size_t combined_features = timestep_dim_ + condition_dim_;
  float* const raw_time = bump_alloc(bump, timestep_dim_);
  float* const hidden_time = bump_alloc(bump, timestep_hidden);
  float* const condition_mish = bump_alloc(bump, combined_features);
  float* x = bump_alloc(bump, values_);
  if (raw_time == nullptr || hidden_time == nullptr || condition_mish == nullptr || x == nullptr)
    return false;
  cuda_ops::diffusion_timestep_device(timestep, raw_time, timestep_dim_);
  cuda_ops::matmul_f32_device(raw_time, timestep_in_.weight, hidden_time, 1, timestep_dim_,
                              timestep_hidden);
  cuda_ops::add_bias_device(hidden_time, timestep_in_.bias, timestep_hidden);
  cuda_ops::mish_device(hidden_time, timestep_hidden);
  cuda_ops::matmul_f32_device(hidden_time, timestep_out_.weight, raw_time, 1, timestep_hidden,
                              timestep_dim_);
  cuda_ops::add_bias_device(raw_time, timestep_out_.bias, timestep_dim_);
  if (!cuda_ops::device_copy_d2d(condition_mish, raw_time, timestep_dim_ * sizeof(float)) ||
      !cuda_ops::device_copy_d2d(condition_mish + timestep_dim_, condition_,
                                 condition_dim_ * sizeof(float)))
    return false;
  cuda_ops::mish_device(condition_mish, combined_features);
  cuda_ops::layout_horizon_to_channel_device(x_, x, horizon_, action_dim_);

  std::array<float*, kMaxStages> skips{};
  std::array<std::size_t, kMaxStages> skip_lengths{};
  std::size_t length = horizon_;
  std::size_t channels = action_dim_;
  for (std::size_t stage = 0; stage < stages_; ++stage) {
    for (std::size_t block = 0; block < 2; ++block) {
      const std::size_t out_channels = dims_[stage];
      float* const output = bump_alloc(bump, out_channels * length);
      if (output == nullptr ||
          !residual_forward(x, length, down_[stage].residuals[block], condition_mish, bump, output))
        return false;
      x = output;
      channels = out_channels;
    }
    skips[stage] = x;
    skip_lengths[stage] = length;
    if (!down_[stage].identity_downsample) {
      const std::size_t output_length = length / 2;
      float* const output = bump_alloc(bump, channels * output_length);
      if (output == nullptr)
        return false;
      const Conv& weights = down_[stage].downsample;
      cuda_ops::conv1d_device(x, weights.weight, weights.bias, output, channels, channels, length,
                              output_length, weights.kernel, 2, 1);
      x = output;
      length = output_length;
    }
  }

  for (const Residual& weights : middle_) {
    float* const output = bump_alloc(bump, channels * length);
    if (output == nullptr || !residual_forward(x, length, weights, condition_mish, bump, output))
      return false;
    x = output;
  }

  for (std::size_t stage = 0; stage + 1 < stages_; ++stage) {
    const std::size_t skip_index = stages_ - 1 - stage;
    float* const skip = skips[skip_index];
    const std::size_t x_n = channels * length;
    if (skip_lengths[skip_index] != length || skip == nullptr)
      return false;
    float* const concatenated = bump_alloc(bump, x_n + x_n);
    if (concatenated == nullptr ||
        !cuda_ops::device_copy_d2d(concatenated, x, x_n * sizeof(float)) ||
        !cuda_ops::device_copy_d2d(concatenated + x_n, skip, x_n * sizeof(float)))
      return false;
    x = concatenated;
    channels *= 2;
    for (std::size_t block = 0; block < 2; ++block) {
      const std::size_t out_channels = dims_[stages_ - 2 - stage];
      float* const output = bump_alloc(bump, out_channels * length);
      if (output == nullptr ||
          !residual_forward(x, length, up_[stage].residuals[block], condition_mish, bump, output))
        return false;
      x = output;
      channels = out_channels;
    }
    const std::size_t output_length = length * 2;
    float* const output = bump_alloc(bump, channels * output_length);
    if (output == nullptr)
      return false;
    const Conv& weights = up_[stage].upsample;
    cuda_ops::conv_transpose1d_device(x, weights.weight, weights.bias, output, channels, channels,
                                      length, output_length, weights.kernel, 2, 1, weights.k_major);
    x = output;
    length = output_length;
  }

  if (channels != dims_[0] || length != horizon_)
    return false;
  float* const final = bump_alloc(bump, channels * length);
  float* const result = bump_alloc(bump, values_);
  if (final == nullptr || result == nullptr)
    return false;
  cuda_ops::conv1d_device(x, final_conv_.weight, final_conv_.bias, final, channels, channels,
                          length, length, final_conv_.kernel, 1, final_conv_.kernel / 2);
  cuda_ops::group_norm_mish_device(final, final_norm_.weight, final_norm_.bias, channels, length,
                                   kFinalNormGroups, kGroupNormEps);
  cuda_ops::conv1d_device(final, output_conv_.weight, output_conv_.bias, result, channels,
                          action_dim_, length, length, 1, 1, 0);
  cuda_ops::layout_channel_to_horizon_device(result, predicted_, horizon_, action_dim_);
  return true;
}

bool CudaDiffusionResident::denoise(std::span<const float> condition,
                                    std::span<const float> normalized_sample, float timestep,
                                    std::span<float> predicted_noise) noexcept
{
  if (!begin(condition, normalized_sample) || predicted_noise.size() < values_)
    return false;
  if (!denoise_from_x(timestep))
    return false;
  return cuda_ops::device_copy_d2h(predicted_noise.data(), predicted_, values_ * sizeof(float));
}

bool CudaDiffusionResident::begin(std::span<const float> condition,
                                  std::span<const float> normalized_sample) noexcept
{
  if (condition.size() != condition_dim_ || normalized_sample.size() != values_)
    return false;
  return cuda_ops::device_copy_h2d(condition_, condition.data(), condition_dim_ * sizeof(float)) &&
         cuda_ops::device_copy_h2d(x_, normalized_sample.data(), values_ * sizeof(float));
}

bool CudaDiffusionResident::denoise_current(float timestep) noexcept
{
  return denoise_from_x(timestep);
}

void CudaDiffusionResident::ddim_update(float sqrt_alpha_t, float sqrt_beta_t,
                                        float sqrt_alpha_prev, float sqrt_one_minus_prev, bool clip,
                                        float clip_range) noexcept
{
  cuda_ops::ddim_update_device(x_, predicted_, values_, sqrt_alpha_t, sqrt_beta_t, sqrt_alpha_prev,
                               sqrt_one_minus_prev, clip, clip_range);
}

bool CudaDiffusionResident::ddpm_update(float sqrt_alpha_t, float sqrt_beta_t,
                                        float original_coefficient, float sample_coefficient,
                                        float sqrt_variance, bool clip, float clip_range,
                                        const float* host_noise) noexcept
{
  if (host_noise != nullptr &&
      !cuda_ops::device_copy_h2d(gaussian_, host_noise, values_ * sizeof(float)))
    return false;
  cuda_ops::ddpm_update_device(x_, predicted_, host_noise != nullptr ? gaussian_ : nullptr, values_,
                               sqrt_alpha_t, sqrt_beta_t, original_coefficient, sample_coefficient,
                               sqrt_variance, clip, clip_range);
  return true;
}

bool CudaDiffusionResident::copy_x(std::span<float> out) noexcept
{
  if (out.size() < values_)
    return false;
  return cuda_ops::device_copy_d2h(out.data(), x_, values_ * sizeof(float));
}

} // namespace fe
