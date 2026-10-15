#pragma once

#include <cstddef>
#include <cstdint>

// C++17 device-op surface used by the C++23 kernels.h bridge. Host pointers.
// Buffers on device are grow-only; the first call of a given size may allocate.
namespace fe::cuda_ops {

[[nodiscard]] bool device_available() noexcept;

void conv1d_causal(const float* x, const float* weight, const float* bias, float* y,
                   std::size_t channels, std::size_t length, std::size_t kernel) noexcept;
void conv1d_step(const float* window, const float* weight, const float* bias, float* y,
                 std::size_t channels, std::size_t kernel) noexcept;
void rmsnorm(const float* in, const float* weight, float* out, std::size_t rows,
             std::size_t dim) noexcept;
void gate_silu(const float* a, const float* g, float* out, std::size_t n) noexcept;
void silu(float* x, std::size_t n) noexcept;
void mish(float* x, std::size_t n) noexcept;
void softplus(float* x, std::size_t n) noexcept;
void matmul_f32(const float* in, const float* w, float* out, std::size_t rows, std::size_t in_dim,
                std::size_t out_dim) noexcept;
void matmul_bf16(const float* in, const std::uint16_t* w, float* out, std::size_t rows,
                 std::size_t in_dim, std::size_t out_dim) noexcept;
void discretize_and_scan(const float* delta, const float* a_neg, const float* b, const float* u,
                         const float* c_proj, const float* d_skip, float* h, float* y,
                         std::size_t length, std::size_t d_inner, std::size_t d_state,
                         bool reset_state, std::size_t row_stride) noexcept;

[[nodiscard]] std::uint64_t device_malloc_count() noexcept;
void device_census_begin() noexcept;
void device_census_end() noexcept;
void device_census_print() noexcept;
[[nodiscard]] void* device_alloc(std::size_t bytes) noexcept;
void device_free(void* ptr) noexcept;
[[nodiscard]] bool device_copy_h2d(void* dst, const void* src, std::size_t bytes) noexcept;
[[nodiscard]] bool device_copy_d2h(void* dst, const void* src, std::size_t bytes) noexcept;

void silu_device(float* x, std::size_t n) noexcept;
void matmul_f32_device(const float* in, const float* w, float* out, std::size_t rows,
                       std::size_t in_dim, std::size_t out_dim) noexcept;
void linear_f32_device(const float* in, const float* weight, const float* bias, float* out,
                       std::size_t in_dim, std::size_t out_dim, bool mish) noexcept;
void add3_device(float* x, const float* y, const float* z, std::size_t n) noexcept;
void add_scaled_device(float* x, const float* dx, float scale, std::size_t n) noexcept;
void scaled_sum_device(const float* x, const float* dx, float scale, float* out,
                       std::size_t n) noexcept;
void add_heun_device(float* x, const float* k1, const float* k2, float dt, std::size_t n) noexcept;
void add_rk4_device(float* x, const float* k1, const float* k2, const float* k3, const float* k4,
                    float dt, std::size_t n) noexcept;
void time_embed_device(float t, const float* freqs, float* sinu, std::size_t half) noexcept;

void conv1d(const float* x, const float* weight, const float* bias, float* y,
            std::size_t in_channels, std::size_t out_channels, std::size_t input_length,
            std::size_t output_length, std::size_t kernel, std::size_t stride,
            std::size_t padding) noexcept;
void conv_transpose1d(const float* x, const float* weight, const float* bias, float* y,
                      std::size_t in_channels, std::size_t out_channels, std::size_t input_length,
                      std::size_t output_length, std::size_t kernel, std::size_t stride,
                      std::size_t padding, bool k_major_weights) noexcept;
void group_norm(float* x, const float* weight, const float* bias, std::size_t channels,
                std::size_t length, std::size_t groups, float epsilon) noexcept;
void film(float* x, const float* scale, const float* bias, std::size_t channels,
          std::size_t length) noexcept;
void diffusion_timestep_embedding(float timestep, float* out, std::size_t n) noexcept;

[[nodiscard]] bool device_copy_d2d(void* dst, const void* src, std::size_t bytes) noexcept;
[[nodiscard]] bool device_synchronize() noexcept;

void mish_device(float* x, std::size_t n) noexcept;
void add_bias_device(float* x, const float* bias, std::size_t n) noexcept;
void conv1d_device(const float* x, const float* weight, const float* bias, float* y,
                   std::size_t in_channels, std::size_t out_channels, std::size_t input_length,
                   std::size_t output_length, std::size_t kernel, std::size_t stride,
                   std::size_t padding) noexcept;
void conv1d_f16_k5_l4_device(const float* x, const void* weight, const float* bias, float* y,
                             std::size_t in_channels, std::size_t out_channels) noexcept;
void conv_transpose1d_device(const float* x, const float* weight, const float* bias, float* y,
                             std::size_t in_channels, std::size_t out_channels,
                             std::size_t input_length, std::size_t output_length,
                             std::size_t kernel, std::size_t stride, std::size_t padding,
                             bool k_major_weights) noexcept;
void group_norm_device(float* x, const float* weight, const float* bias, std::size_t channels,
                       std::size_t length, std::size_t groups, float epsilon) noexcept;
// GroupNorm -> Mish -> optional FiLM -> optional residual, in that order.
// modulation is [scale(C), bias(C)]; residual has C*L elements.
void group_norm_mish_device(float* x, const float* weight, const float* bias, std::size_t channels,
                            std::size_t length, std::size_t groups, float epsilon,
                            const float* modulation = nullptr,
                            const float* residual = nullptr) noexcept;
void film_device(float* x, const float* scale, const float* bias, std::size_t channels,
                 std::size_t length) noexcept;
void diffusion_timestep_device(float timestep, float* out, std::size_t n) noexcept;
void layout_horizon_to_channel_device(const float* in, float* out, std::size_t horizon,
                                      std::size_t action_dim) noexcept;
void layout_channel_to_horizon_device(const float* in, float* out, std::size_t horizon,
                                      std::size_t action_dim) noexcept;
void ddim_update_device(float* x, const float* eps, std::size_t n, float sqrt_alpha_t,
                        float sqrt_beta_t, float sqrt_alpha_prev, float sqrt_one_minus_prev,
                        bool clip, float clip_range) noexcept;
void ddpm_update_device(float* x, const float* eps, const float* noise, std::size_t n,
                        float sqrt_alpha_t, float sqrt_beta_t, float original_coefficient,
                        float sample_coefficient, float sqrt_variance, bool clip,
                        float clip_range) noexcept;
void diffusion_unnormalize_device(const float* normalized, const float* action_min,
                                  const float* action_max, float* action, std::size_t action_start,
                                  std::size_t action_steps, std::size_t action_dim) noexcept;

void rmsnorm_device(const float* in, const float* weight, float* out, std::size_t rows,
                    std::size_t dim) noexcept;
void conv1d_causal_device(const float* x, const float* weight, const float* bias, float* y,
                          std::size_t channels, std::size_t length, std::size_t kernel) noexcept;
void conv1d_step_device(const float* window, const float* weight, const float* bias, float* y,
                        std::size_t channels, std::size_t kernel) noexcept;
void softplus_device(float* x, std::size_t n) noexcept;
void gate_silu_device(const float* a, const float* g, float* out, std::size_t n) noexcept;
void add_inplace_device(float* x, const float* y, std::size_t n) noexcept;
void add_bias_rows_device(float* x, const float* bias, std::size_t rows, std::size_t dim) noexcept;
void split_xz_device(const float* xz, float* x_cm, float* z, std::size_t length,
                     std::size_t d_inner) noexcept;
void channel_to_seq_device(const float* x_cm, float* x_sm, std::size_t length,
                           std::size_t d_inner) noexcept;
void gather_prefix_device(const float* rows, float* out, std::size_t length, std::size_t row_stride,
                          std::size_t width) noexcept;
void conv_shift_push_device(float* window, const float* xz, float* z, std::size_t channels,
                            std::size_t kernel) noexcept;
void discretize_and_scan_device(const float* delta, const float* a_neg, const float* b,
                                const float* u, const float* c_proj, const float* d_skip, float* h,
                                float* y, std::size_t length, std::size_t d_inner,
                                std::size_t d_state, bool reset_state,
                                std::size_t row_stride) noexcept;

} // namespace fe::cuda_ops
