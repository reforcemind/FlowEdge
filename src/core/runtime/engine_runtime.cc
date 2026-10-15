#include "runtime/engine_runtime.h"

#include "protocol/contracts.h"
#include "protocol/snapshot.h"
#ifdef FLOWEDGE_CUDA
#include "models/mamba/mamba_cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <thread>
#include <utility>

namespace fe {
namespace {

constexpr std::size_t k_max_decode_seq = 512uz; // scratch is sized for prefills up to this length

} // namespace

std::size_t EngineRuntime::required_slab_bytes(const ModelWeights& weights) noexcept
{
  const std::span<const TensorView> tensors = weights.tensors();
  const auto shape = [tensors](std::string_view name) noexcept {
    const TensorView* const tensor = find_tensor(tensors, name);
    return tensor != nullptr ? tensor->shape : std::array<std::size_t, 4>{};
  };
  const auto emb = shape("backbone.embeddings.weight");
  const auto a_log = shape("backbone.layers.0.mixer.A_log");
  const auto conv = shape("backbone.layers.0.mixer.conv1d.weight");
  const auto flow_in = shape("flow.in_proj.weight");
  const auto time_proj = shape("flow.time_proj.weight");
  const std::size_t d_model = emb[1];
  const std::size_t d_inner = a_log[0];
  const std::size_t d_state = a_log[1];
  const std::size_t d_conv = conv[2];
  const std::size_t flow_hidden = flow_in[0];
  const std::size_t action_dim = flow_in[1];
  const std::size_t flow_time_dim = time_proj[1];
  const bool has_backbone = d_model != 0uz && d_inner != 0uz && d_state != 0uz && d_conv != 0uz;
  const TensorView* const transformer_config = find_tensor(tensors, "transformer.config");
  const bool has_transformer_config =
      transformer_config != nullptr && transformer_config->is_f32() &&
      transformer_config->ndim == 1uz && transformer_config->shape[0] == 6uz;
  const float* const transformer_values =
      has_transformer_config ? transformer_config->as_f32() : nullptr;
  const auto dimension = [](float value, std::size_t maximum) noexcept {
    if (!std::isfinite(value) || value < 1.0F || value > static_cast<float>(maximum) ||
        std::trunc(value) != value)
      return 0uz;
    return static_cast<std::size_t>(value);
  };
  const std::size_t transformer_model =
      has_transformer_config ? dimension(transformer_values[1], 16'384uz) : 0uz;
  const std::size_t transformer_layers =
      has_transformer_config ? dimension(transformer_values[2], 48uz) : 0uz;
  const std::size_t transformer_prefix =
      has_transformer_config ? dimension(transformer_values[4], 4'096uz) : 0uz;
  const std::size_t transformer_mlp =
      has_transformer_config ? dimension(transformer_values[5], 65'536uz) : 0uz;
  const bool has_transformer = transformer_model != 0uz && transformer_layers != 0uz &&
                               transformer_prefix != 0uz && transformer_mlp != 0uz;
  const bool has_flow = flow_hidden != 0uz && action_dim != 0uz && flow_time_dim != 0uz;
  const std::size_t diffusion_workspace = DiffusionHead::required_workspace_floats(tensors);
  const std::size_t diffusion_persistent = DiffusionHead::required_persistent_floats(tensors);
  const bool has_diffusion = diffusion_workspace != 0uz && diffusion_persistent != 0uz;
  const auto smol_shape = [tensors](std::string_view name) noexcept {
    const TensorView* const tensor = find_tensor(tensors, name);
    return tensor != nullptr ? tensor->shape : std::array<std::size_t, 4>{};
  };
  const auto smol_action_in = smol_shape("model.action_in_proj.weight");
  const auto smol_state = smol_shape("model.state_proj.weight");
  const std::size_t smol_expert_width = smol_action_in[0];
  const std::size_t smol_action_dim = smol_action_in[1];
  const std::size_t smol_vlm_width = smol_state[0];
  const bool has_smolvla = smol_expert_width != 0uz && smol_expert_width <= 4096uz &&
                           smol_action_dim != 0uz && smol_action_dim <= 256uz &&
                           smol_vlm_width != 0uz && smol_vlm_width <= 4096uz;
  if (!has_backbone && !has_transformer && !has_flow && !has_diffusion && !has_smolvla)
    return 0uz;

  const std::size_t per_token =
      has_backbone ? (3uz * d_state * d_inner) + (16uz * d_inner) + (8uz * d_model) : 0uz;
  const std::size_t persistent_state =
      has_backbone ? 64uz * d_inner * (d_conv + (2uz * d_state)) * sizeof(float) : 0uz;
  const std::size_t transformer_persistent =
      has_transformer
          ? 2uz * transformer_layers * transformer_prefix * transformer_model * sizeof(float)
          : 0uz;
  const std::size_t transformer_scratch =
      has_transformer
          ? ((9uz * transformer_model) + transformer_mlp + transformer_prefix) * sizeof(float)
          : 0uz;
  const std::size_t flow_floats =
      has_flow ? (3uz * flow_hidden) + (6uz * action_dim) + ((3uz * flow_time_dim) / 2uz) : 0uz;
  const auto x_proj = shape("backbone.layers.0.mixer.x_proj.weight");
  const std::size_t mamba_x_out = x_proj[0];
#ifdef FLOWEDGE_CUDA
  const std::size_t flow_cond_dim = shape("flow.cond_proj.weight")[1];
  const std::size_t flow_upload =
      has_flow ? (flow_hidden * action_dim) + (flow_hidden * flow_time_dim) +
                     (flow_hidden * flow_cond_dim) + (action_dim * flow_hidden) +
                     (FlowHead::kMaxMlp * flow_hidden * flow_hidden) + (flow_time_dim / 2uz) + 64uz
               : 0uz;
  const std::size_t mamba_dt_rank =
      mamba_x_out > (2uz * d_state) ? mamba_x_out - (2uz * d_state) : 0uz;
  const std::size_t mamba_upload =
      has_backbone ? (sizeof(CudaMambaResident) / sizeof(float)) + 64uz +
                         (64uz * ((2uz * d_inner * d_model) + (mamba_x_out * d_inner) +
                                  (d_inner * mamba_dt_rank) + (d_model * d_inner)))
                   : 0uz;
#else
  const std::size_t flow_upload = 0uz;
  const std::size_t mamba_upload = 0uz;
#endif
  const std::size_t runtime =
      (kThreadRingSlots * sizeof(Task)) + (kThreadRingSlots * sizeof(std::size_t)) +
      (kMaxPoolThreads * sizeof(std::jthread)) + sizeof(ThreadPool) +
      ((flow_floats + flow_upload + mamba_upload + diffusion_workspace + diffusion_persistent) *
       sizeof(float)) +
      persistent_state + transformer_persistent + transformer_scratch +
      // The captured-VLM SmolVLA path needs the action-expert projections,
      // attention buffers, and two SwiGLU intermediates concurrently. Keep a
      // conservative fixed upper bound for its documented 512-token inputs.
      (has_smolvla ? (16uz * 512uz * smol_expert_width) : 0uz) * sizeof(float) + 4096uz;
  return (k_max_decode_seq * per_token * sizeof(float)) + runtime;
}

unsigned EngineRuntime::recommended_thread_count() noexcept
{
  const unsigned hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads < 2u)
    return 0u;
  // Short-sequence policy inference is normally memory-bandwidth-bound. Four
  // workers saturate typical desktop/edge memory without the WSL/SMT cliff
  // observed when every logical CPU is used.
  return std::min(std::max(hardware_threads / 2u, 2u), 4u);
}

EngineRuntime::EngineRuntime(std::shared_ptr<const ModelWeights> weights, std::size_t slab_bytes,
                             unsigned worker_threads, const char*& error)
    : EngineRuntime{std::move(weights), slab_bytes, worker_threads, DiffusionPrecision::ExactF32,
                    error}
{
}

EngineRuntime::EngineRuntime(std::shared_ptr<const ModelWeights> weights, std::size_t slab_bytes,
                             unsigned worker_threads, DiffusionPrecision diffusion_precision,
                             const char*& error)
    : weights_{std::move(weights)}, slab_{slab_bytes},
      arena_{std::span<std::byte>{slab_.data(), slab_.size()}},
      model_{weights_ ? weights_->tensors() : std::span<const TensorView>{}, arena_},
      smolvla_{weights_ ? weights_->tensors() : std::span<const TensorView>{}, arena_},
      transformer_{weights_ ? weights_->tensors() : std::span<const TensorView>{}, arena_},
      flow_{weights_ ? weights_->tensors() : std::span<const TensorView>{}, arena_},
      diffusion_{weights_ ? weights_->tensors() : std::span<const TensorView>{}, arena_,
                 diffusion_precision}
{
  if (!weights_) {
    error = "Immutable model weights are unavailable";
    return;
  }
  const std::span<const TensorView> tensors = weights_->tensors();
  identity_.digest = fingerprint_tensors(tensors);
  identity_.precision = checkpoint_precision(tensors);
  identity_.architecture = diffusion_.valid()                      ? FE_ARCH_DIFFUSION_HEAD
                           : transformer_.valid() && flow_.valid() ? FE_ARCH_TRANSFORMER_FLOW
                           : transformer_.valid()                  ? FE_ARCH_TRANSFORMER
                           : model_.valid() && flow_.valid()       ? FE_ARCH_MAMBA_FLOW
                           : model_.valid()                        ? FE_ARCH_MAMBA
                           : flow_.valid()                         ? FE_ARCH_FLOW_HEAD
                           : smolvla_.valid()                      ? FE_ARCH_SMOLVLA_ACTION_EXPERT
                                                                   : FE_ARCH_UNKNOWN;
  if (model_.valid()) {
    const MambaConfig& config = model_.config();
    identity_.d_model = config.d_model;
    identity_.n_layers = config.n_layers;
    identity_.d_inner = config.d_inner;
    identity_.d_state = config.d_state;
    identity_.d_conv = config.d_conv;
  }
  if (transformer_.valid()) {
    const TransformerConfig& config = transformer_.config();
    identity_.d_model = config.d_model;
    identity_.n_layers = config.n_layers;
    identity_.d_inner = config.mlp_dim;
  }
  if (flow_.valid()) {
    identity_.action_dim = flow_.config().action_dim;
    identity_.condition_dim = flow_.config().cond_dim;
  }
  if (diffusion_.valid()) {
    identity_.action_dim = diffusion_.config().action_dim;
    identity_.condition_dim = diffusion_.config().condition_dim;
  }
  if (smolvla_.valid()) {
    const SmolVLAActionExpertConfig& config = smolvla_.config();
    identity_.d_model = config.vlm_width;
    identity_.n_layers = config.expert_layers;
    identity_.d_inner = config.expert_width;
    identity_.action_dim = config.max_action_dim;
  }

  if (worker_threads > 0u) {
    auto* ring = arena_.alloc_array<Task, kSimdAlign>(kThreadRingSlots);
    auto* sequence = arena_.alloc_array<std::size_t, kSimdAlign>(kThreadRingSlots);
    auto* worker_mem = static_cast<std::jthread*>(
        arena_.alloc<alignof(std::jthread)>(sizeof(std::jthread) * worker_threads));
    auto* pool_mem = arena_.alloc<alignof(ThreadPool)>(sizeof(ThreadPool));
    if (ring != nullptr && sequence != nullptr && worker_mem != nullptr && pool_mem != nullptr) {
      workers_ = {worker_mem, worker_threads};
      for (unsigned i{0u}; i < worker_threads; ++i)
        std::construct_at(&workers_[i]);
      pool_ = std::construct_at(static_cast<ThreadPool*>(pool_mem),
                                std::span<Task>{ring, kThreadRingSlots},
                                std::span<std::size_t>{sequence, kThreadRingSlots}, workers_,
                                worker_threads);
    }
    if (pool_ == nullptr) {
      error = "Runtime slab cannot hold the requested worker pool";
      return;
    }
  }

  model_.set_pool(pool_);
  smolvla_.set_pool(pool_);
  transformer_.set_pool(pool_);
  flow_.set_pool(pool_);
  diffusion_.set_pool(pool_);
  if (model_.valid())
    if (auto* const state = arena_.alloc_array<float, kSimdAlign>(model_.state_size())) {
      decode_state_ = {state, model_.state_size()};
      std::ranges::fill(decode_state_, 0.0F);
    }
  if (flow_.valid())
    if (auto* const workspace =
            arena_.alloc_array<float, kSimdAlign>(flow_.sampler_workspace_size()))
      flow_workspace_ = {workspace, flow_.sampler_workspace_size()};
  if (diffusion_.valid())
    if (auto* const workspace =
            arena_.alloc_array<float, kSimdAlign>(diffusion_.sampler_workspace_size()))
      diffusion_workspace_ = {workspace, diffusion_.sampler_workspace_size()};
  if (model_.valid() && decode_state_.empty()) {
    error = "Runtime slab cannot hold the streaming model state";
    return;
  }
  if (flow_.valid() && flow_workspace_.empty()) {
    error = "Runtime slab cannot hold the flow sampler workspace";
    return;
  }
  if (diffusion_.valid() && diffusion_workspace_.empty()) {
    error = "Runtime slab cannot hold the diffusion sampler workspace";
    return;
  }
  ready_ = model_.valid() || smolvla_.valid() || transformer_.valid() || flow_.valid() ||
           diffusion_.valid();
}

EngineRuntime::~EngineRuntime()
{
  if (pool_ != nullptr)
    pool_->~ThreadPool();
  for (std::jthread& worker : workers_)
    std::destroy_at(&worker);
}

bool EngineRuntime::has_compatible_flow_head() const noexcept
{
  return !flow_.valid() || !has_backbone() || flow_.config().cond_dim == d_model();
}

std::size_t EngineRuntime::d_model() const noexcept
{
  return transformer_.valid() ? transformer_.config().d_model
         : model_.valid()     ? model_.config().d_model
         : smolvla_.valid()   ? smolvla_.config().vlm_width
                              : 0uz;
}

std::size_t EngineRuntime::n_layers() const noexcept
{
  return transformer_.valid() ? transformer_.config().n_layers
         : model_.valid()     ? model_.config().n_layers
         : smolvla_.valid()   ? smolvla_.config().expert_layers
                              : 0uz;
}

unsigned EngineRuntime::thread_count() const noexcept
{
  return pool_ != nullptr ? pool_->nthreads() : 0u;
}

std::size_t EngineRuntime::decode_state_bytes() const noexcept
{
  return decode_state_.empty() ? 0uz : decode_snapshot_bytes(decode_state_.size_bytes());
}

std::uint32_t EngineRuntime::flow_action_status() const noexcept
{
  if (flow_state_.cancelled)
    return FE_ACTION_CANCELLED;
  if (!flow_state_.active)
    return FE_ACTION_FAILED;
  return flow_state_.remaining() == 0uz ? FE_ACTION_COMPLETE : FE_ACTION_RUNNING;
}

std::uint64_t EngineRuntime::flow_remaining_nfe() const noexcept
{
  const std::uint64_t evaluations = flow_state_.method == FlowHead::kRK4    ? 4u
                                    : flow_state_.method == FlowHead::kHeun ? 2u
                                                                            : 1u;
  const std::uint64_t remaining = flow_state_.remaining();
  return remaining > (std::numeric_limits<std::uint64_t>::max() / evaluations)
             ? std::numeric_limits<std::uint64_t>::max()
             : remaining * evaluations;
}

std::size_t EngineRuntime::action_dim() const noexcept
{
  return diffusion_.valid() ? diffusion_.config().action_dim
         : flow_.valid()    ? flow_.config().action_dim
         : smolvla_.valid() ? smolvla_.config().max_action_dim
                            : 0uz;
}

std::size_t EngineRuntime::action_horizon() const noexcept
{
  return diffusion_.valid() ? diffusion_.config().horizon
         : flow_.valid()    ? 1uz
         : smolvla_.valid() ? 50uz
                            : 0uz;
}

std::size_t EngineRuntime::condition_dim() const noexcept
{
  return diffusion_.valid() ? diffusion_.config().condition_dim
         : flow_.valid()    ? flow_.config().cond_dim
                            : 0uz;
}

int EngineRuntime::run(const std::int32_t* tokens, std::size_t seq_len, float* out,
                       const char*& error) noexcept
{
  std::byte* const mark = arena_.mark();
  const int rc = run_backbone(tokens, seq_len, out, error);
  arena_.reset_to(mark);
  return rc;
}

int EngineRuntime::step(std::int32_t token, float* out, const char*& error) noexcept
{
  if (decode_state_.empty() && !transformer_.valid()) [[unlikely]] {
    error = "Model has no streaming state initialized";
    return 2;
  }
  if (transformer_.valid()) {
    if (!transformer_.decode_token(token, {out, transformer_.config().d_model})) {
      error = "Token ID is out of range or Transformer KV cache is full";
      return 3;
    }
    return 0;
  }
  const MambaConfig& c = model_.config();
  if (std::cmp_less(token, 0) || std::cmp_greater_equal(token, c.vocab)) {
    error = "Token ID out of vocabulary range";
    return 3;
  }

  std::byte* const mark = arena_.mark();
  auto* const x = arena_.alloc_array<float, kSimdAlign>(c.d_model);
  if (x == nullptr) {
    error = "Arena exhausted during streaming step";
    return 2;
  }
  const auto token_index = static_cast<std::size_t>(token);
  std::copy_n(model_.embedding() + (token_index * c.d_model), c.d_model, x);
  model_.decode({x, c.d_model}, decode_state_, {out, c.d_model});
  arena_.reset_to(mark);
  return 0;
}

void EngineRuntime::reset() noexcept
{
  std::ranges::fill(decode_state_, 0.0F);
  if (transformer_.valid())
    transformer_.reset();
}

int EngineRuntime::sample(const std::int32_t* tokens, std::size_t seq_len, const float* noise,
                          std::size_t steps, FlowHead::Method method, float* action,
                          const char*& error) noexcept
{
  if (!flow_.valid()) {
    error = "Model has no flow head to sample from";
    return 4;
  }
  if (!has_backbone()) {
    error = "Model has no backbone for token-conditioned sampling";
    return 4;
  }
  if (!has_compatible_flow_head()) {
    error = "Flow conditioning dimension does not match the backbone";
    return 4;
  }
  const std::size_t prefix_limit =
      transformer_.valid() ? transformer_.config().max_sequence : k_max_decode_seq;
  if (seq_len == 0uz || seq_len > prefix_limit) {
    error = "Token sequence exceeds the configured prefill limit";
    return 1;
  }

  std::byte* const mark = arena_.mark();
  const std::size_t width = d_model();
  auto* const hidden = arena_.alloc_array<float, kSimdAlign>(seq_len * width);
  if (hidden == nullptr) {
    error = "Arena exhausted allocating hidden states";
    return 2;
  }
  const int rc = run_backbone(tokens, seq_len, hidden, error);
  if (rc != 0) {
    arena_.reset_to(mark);
    return rc;
  }

  const std::span<const float> cond{hidden + ((seq_len - 1uz) * width), width};
  int sample_rc = flow_begin(cond.data(), noise, steps, method, error);
  if (sample_rc == 0) {
    std::size_t remaining{steps};
    sample_rc = flow_advance(steps, action, remaining, error);
  }
  arena_.reset_to(mark);
  return sample_rc;
}

int EngineRuntime::sample_condition(const float* condition, const float* noise, std::size_t steps,
                                    FlowHead::Method method, float* action,
                                    const char*& error) noexcept
{
  if (!flow_.valid()) {
    error = "Model has no flow head to sample from";
    return 4;
  }
  const int rc = flow_begin(condition, noise, steps, method, error);
  if (rc != 0)
    return rc;
  std::size_t remaining{steps};
  return flow_advance(steps, action, remaining, error);
}

int EngineRuntime::sample_diffusion(const float* condition, const float* noise, std::size_t steps,
                                    DiffusionHead::Scheduler scheduler, std::uint64_t seed,
                                    float* action, const char*& error) noexcept
{
  if (!diffusion_.valid()) {
    error = "Model has no Diffusion Policy head to sample from";
    return 4;
  }
  if (diffusion_workspace_.empty()) [[unlikely]] {
    error = "Diffusion sampler workspace is unavailable";
    return 2;
  }
  const DiffusionConfig& config = diffusion_.config();
  if (steps == 0uz || steps > config.train_timesteps) {
    error = "Diffusion inference steps must be between 1 and the training timestep count";
    return 1;
  }
  if (!diffusion_.sample({condition, config.condition_dim},
                         {noise, config.horizon * config.action_dim}, steps, scheduler, seed,
                         diffusion_workspace_, {action, config.horizon * config.action_dim})) {
    error = "Diffusion sampling failed";
    return 2;
  }
  return 0;
}

int EngineRuntime::sample_diffusion_device(const float* condition, const float* noise,
                                           std::size_t steps, DiffusionHead::Scheduler scheduler,
                                           std::uint64_t seed, std::size_t action_start,
                                           std::size_t action_steps, float* action,
                                           const char*& error) noexcept
{
  if (!diffusion_.valid()) {
    error = "Model has no Diffusion Policy head to sample from";
    return 4;
  }
#ifndef FLOWEDGE_CUDA
  static_cast<void>(condition);
  static_cast<void>(noise);
  static_cast<void>(steps);
  static_cast<void>(scheduler);
  static_cast<void>(seed);
  static_cast<void>(action_start);
  static_cast<void>(action_steps);
  static_cast<void>(action);
  error = "CUDA tensor sampling requires a CUDA Core build";
  return 7;
#else
  if (!diffusion_.cuda_resident()) {
    error = "CUDA tensor sampling requires a device-resident Diffusion Policy head";
    return 7;
  }
  if (diffusion_workspace_.empty()) [[unlikely]] {
    error = "Diffusion sampler workspace is unavailable";
    return 2;
  }
  const DiffusionConfig& config = diffusion_.config();
  if (steps == 0uz || steps > config.train_timesteps || action_steps == 0uz ||
      action_start > config.horizon || action_steps > config.horizon - action_start) {
    error = "Invalid diffusion steps or action slice";
    return 1;
  }
  if (!diffusion_.sample_device(condition, noise, steps, scheduler, seed, diffusion_workspace_,
                                action, action_start, action_steps)) {
    error = "CUDA tensor diffusion sampling failed";
    return 2;
  }
  return 0;
#endif
}

int EngineRuntime::denoise_diffusion(const float* condition, const float* normalized_sample,
                                     float timestep, float* predicted_noise,
                                     const char*& error) noexcept
{
  if (!diffusion_.valid()) {
    error = "Model has no Diffusion Policy head";
    return 4;
  }
  if (diffusion_workspace_.empty()) [[unlikely]] {
    error = "Diffusion workspace is unavailable";
    return 2;
  }
  const DiffusionConfig& config = diffusion_.config();
  const std::size_t values = config.horizon * config.action_dim;
  if (!diffusion_.denoise({condition, config.condition_dim}, {normalized_sample, values}, timestep,
                          diffusion_workspace_, {predicted_noise, values})) {
    error = "Diffusion denoiser execution failed";
    return 2;
  }
  return 0;
}

int EngineRuntime::flow_begin(const float* condition, const float* noise, std::size_t steps,
                              FlowHead::Method method, const char*& error) noexcept
{
  return flow_begin_request(condition, noise, FlowRequestMetadata{.steps = steps, .method = method},
                            error);
}

int EngineRuntime::flow_begin_request(const float* condition, const float* noise,
                                      const FlowRequestMetadata& metadata,
                                      const char*& error) noexcept
{
  if (!flow_.valid()) {
    error = "Model has no flow head to sample from";
    return 4;
  }
  if (flow_workspace_.empty()) [[unlikely]] {
    error = "Flow sampler workspace is unavailable";
    return 2;
  }
  flow_request_ = metadata;
  if (metadata.generation_tracking) {
    std::uint64_t observed = latest_generation_.load(std::memory_order_relaxed);
    while (observed < metadata.generation &&
           !latest_generation_.compare_exchange_weak(observed, metadata.generation,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed)) {
    }
  }
  const auto* cancellation = metadata.generation_tracking ? &latest_generation_ : nullptr;
  if (!flow_.sampler_begin({condition, flow_.config().cond_dim}, {noise, flow_.config().action_dim},
                           metadata.steps, metadata.method, flow_workspace_, flow_state_,
                           cancellation, metadata.generation)) [[unlikely]] {
    error = "Failed to initialize resumable flow sampler";
    return 2;
  }
  return 0;
}

int EngineRuntime::flow_advance(std::size_t step_budget, float* action,
                                std::size_t& steps_remaining, const char*& error) noexcept
{
  if (flow_state_.cancelled) {
    error = "Flow solve was cancelled by a newer generation";
    steps_remaining = flow_state_.remaining();
    return 8;
  }
  if (!flow_state_.active) {
    error = "No resumable flow sample is active";
    return 6;
  }
  static_cast<void>(
      flow_.sampler_advance(flow_state_, step_budget, {action, flow_.config().action_dim}));
  steps_remaining = flow_state_.remaining();
  if (flow_state_.cancelled) {
    error = "Flow solve was cancelled by a newer generation";
    return 8;
  }
  return 0;
}

void EngineRuntime::cancel_before(std::uint64_t generation) noexcept
{
  std::uint64_t observed = latest_generation_.load(std::memory_order_relaxed);
  while (observed < generation &&
         !latest_generation_.compare_exchange_weak(observed, generation, std::memory_order_release,
                                                   std::memory_order_relaxed)) {
  }
}

int EngineRuntime::export_decode_state(std::span<std::byte> destination,
                                       const char*& error) const noexcept
{
  const std::span<const std::byte> state = std::as_bytes(decode_state_);
  if (state.empty()) {
    error = "Model has no streaming decode state";
    return 7;
  }
  const auto result = export_decode_snapshot(identity_, state, destination);
  if (!result) {
    error = result.error().message;
    return result.error().code;
  }
  return 0;
}

int EngineRuntime::import_decode_state(std::span<const std::byte> source,
                                       const char*& error) noexcept
{
  const std::span<std::byte> state = std::as_writable_bytes(decode_state_);
  if (state.empty()) {
    error = "Model has no streaming decode state";
    return 7;
  }
  const auto result = import_decode_snapshot(identity_, source, state);
  if (!result) {
    error = result.error().message;
    return result.error().code;
  }
  return 0;
}

int EngineRuntime::run_backbone(const std::int32_t* tokens, std::size_t seq_len, float* hidden,
                                const char*& error) noexcept
{
  if (!has_backbone()) {
    error = "Model has no backbone";
    return 4;
  }
  const std::size_t width = d_model();
  const std::size_t prefix_limit =
      transformer_.valid() ? transformer_.config().max_sequence : k_max_decode_seq;
  if (seq_len == 0uz || seq_len > prefix_limit ||
      seq_len > (std::numeric_limits<std::size_t>::max() / width)) {
    error = "Token sequence length is outside the configured prefill limit";
    return 1;
  }
  const std::size_t hidden_size = seq_len * width;
  if (transformer_.valid()) {
    if (!transformer_.forward_tokens({tokens, seq_len}, {hidden, hidden_size})) {
      error = "Transformer token range or fixed prefix limit was violated";
      return 3;
    }
    return 0;
  }
  const MambaConfig& c = model_.config();
  auto* const input = arena_.alloc_array<float, kSimdAlign>(hidden_size);
  if (input == nullptr) {
    error = "Arena exhausted during backbone forward pass";
    return 2;
  }
  for (std::size_t t{0uz}; t < seq_len; ++t) {
    const std::int32_t token = tokens[t];
    if (std::cmp_less(token, 0) || std::cmp_greater_equal(token, c.vocab)) {
      error = "Token ID out of vocabulary range";
      return 3;
    }
    const auto token_index = static_cast<std::size_t>(token);
    std::copy_n(model_.embedding() + (token_index * c.d_model), c.d_model, input + (t * c.d_model));
  }
  model_.forward({input, hidden_size}, {hidden, hidden_size}, seq_len);
  return 0;
}

int EngineRuntime::run_embeddings(const float* embeddings, std::size_t seq_len, float* out,
                                  const char*& error) noexcept
{
  return run_embeddings_masked(embeddings, nullptr, seq_len, out, error);
}

int EngineRuntime::run_embeddings_masked(const float* embeddings,
                                         const std::uint8_t* attention_mask, std::size_t seq_len,
                                         float* out, const char*& error) noexcept
{
  if (!transformer_.valid()) {
    error = "Model has no Transformer backbone for external embeddings";
    return 4;
  }
  const std::size_t width = transformer_.config().d_model;
  const std::size_t prefix_limit = transformer_.config().max_sequence;
  if (seq_len == 0uz || seq_len > prefix_limit ||
      seq_len > (std::numeric_limits<std::size_t>::max() / width)) {
    error = "Embedding sequence length is outside the configured prefill limit";
    return 1;
  }
  const std::span<const std::uint8_t> mask =
      attention_mask == nullptr ? std::span<const std::uint8_t>{}
                                : std::span<const std::uint8_t>{attention_mask, seq_len};
  if (!transformer_.forward_embeddings({embeddings, seq_len * width}, mask,
                                       {out, seq_len * width})) {
    error = "Transformer embedding sequence or attention mask could not be executed";
    return 3;
  }
  return 0;
}

int EngineRuntime::smolvla_embed_suffix(const float* noisy_actions, std::size_t chunk_size,
                                        float timestep, float* out, const char*& error) noexcept
{
  if (!smolvla_.valid()) {
    error = "Model has no SmolVLA action-expert projection boundary";
    return 4;
  }
  const SmolVLAActionExpertConfig& config = smolvla_.config();
  if (chunk_size == 0uz || chunk_size > 512uz ||
      chunk_size > (std::numeric_limits<std::size_t>::max() / config.max_action_dim) ||
      chunk_size > (std::numeric_limits<std::size_t>::max() / config.expert_width)) {
    error = "SmolVLA chunk size is outside the fixed projection limit";
    return 1;
  }
  if (!smolvla_.embed_suffix({noisy_actions, chunk_size * config.max_action_dim}, timestep,
                             {out, chunk_size * config.expert_width})) {
    error = "SmolVLA action suffix embedding could not be executed";
    return 3;
  }
  return 0;
}

int EngineRuntime::smolvla_run_expert(const float* suffix, std::size_t chunk_size,
                                      const float* prefix_keys, const float* prefix_values,
                                      const std::uint8_t* prefix_mask, std::size_t prefix_length,
                                      float* out, const char*& error) noexcept
{
  if (!smolvla_.valid()) {
    error = "Model has no SmolVLA action expert";
    return 4;
  }
  const SmolVLAActionExpertConfig& config = smolvla_.config();
  if (chunk_size == 0uz || chunk_size > 512uz || prefix_length == 0uz || prefix_length > 512uz ||
      chunk_size > (std::numeric_limits<std::size_t>::max() / config.expert_width) ||
      prefix_length > (std::numeric_limits<std::size_t>::max() / config.expert_layers) ||
      prefix_length * config.expert_layers >
          (std::numeric_limits<std::size_t>::max() / config.key_value_width)) {
    error = "SmolVLA expert chunk or prefix length is outside the fixed execution limit";
    return 1;
  }
  const std::size_t suffix_values = chunk_size * config.expert_width;
  const std::size_t cache_values = prefix_length * config.expert_layers * config.key_value_width;
  if (!smolvla_.run_with_prefix_kv({suffix, suffix_values}, {prefix_keys, cache_values},
                                   {prefix_values, cache_values}, {prefix_mask, prefix_length},
                                   {out, suffix_values})) {
    error = "SmolVLA cached-VLM action expert could not be executed";
    return 3;
  }
  return 0;
}

int EngineRuntime::smolvla_project_actions(const float* hidden, std::size_t chunk_size, float* out,
                                           const char*& error) noexcept
{
  if (!smolvla_.valid()) {
    error = "Model has no SmolVLA action expert";
    return 4;
  }
  const SmolVLAActionExpertConfig& config = smolvla_.config();
  if (chunk_size == 0uz || chunk_size > 512uz ||
      chunk_size > (std::numeric_limits<std::size_t>::max() / config.expert_width) ||
      chunk_size > (std::numeric_limits<std::size_t>::max() / config.max_action_dim)) {
    error = "SmolVLA action chunk size is outside the fixed projection limit";
    return 1;
  }
  if (!smolvla_.project_actions({hidden, chunk_size * config.expert_width},
                                {out, chunk_size * config.max_action_dim})) {
    error = "SmolVLA action projection could not be executed";
    return 3;
  }
  return 0;
}

int EngineRuntime::smolvla_denoise(const float* noisy_actions, std::size_t chunk_size,
                                   float timestep, const float* prefix_keys,
                                   const float* prefix_values, const std::uint8_t* prefix_mask,
                                   std::size_t prefix_length, float* out,
                                   const char*& error) noexcept
{
  if (!smolvla_.valid()) {
    error = "Model has no SmolVLA action expert";
    return 4;
  }
  const SmolVLAActionExpertConfig& config = smolvla_.config();
  if (!std::isfinite(timestep) || chunk_size == 0uz || chunk_size > 512uz || prefix_length == 0uz ||
      prefix_length > 512uz ||
      chunk_size > (std::numeric_limits<std::size_t>::max() / config.max_action_dim) ||
      prefix_length > (std::numeric_limits<std::size_t>::max() / config.expert_layers) ||
      prefix_length * config.expert_layers >
          (std::numeric_limits<std::size_t>::max() / config.key_value_width)) {
    error = "SmolVLA denoise chunk or prefix length is outside the fixed execution limit";
    return 1;
  }
  const std::size_t action_values = chunk_size * config.max_action_dim;
  const std::size_t cache_values = prefix_length * config.expert_layers * config.key_value_width;
  if (!smolvla_.denoise_with_prefix_kv({noisy_actions, action_values}, timestep,
                                       {prefix_keys, cache_values}, {prefix_values, cache_values},
                                       {prefix_mask, prefix_length}, {out, action_values})) {
    error = "SmolVLA cached-VLM denoise step could not be executed";
    return 3;
  }
  return 0;
}

int EngineRuntime::smolvla_sample(const float* initial_noise, std::size_t chunk_size,
                                  std::size_t steps, const float* prefix_keys,
                                  const float* prefix_values, const std::uint8_t* prefix_mask,
                                  std::size_t prefix_length, float* out,
                                  const char*& error) noexcept
{
  if (!smolvla_.valid()) {
    error = "Model has no SmolVLA action expert";
    return 4;
  }
  const SmolVLAActionExpertConfig& config = smolvla_.config();
  if (steps == 0uz || steps > 100uz || chunk_size == 0uz || chunk_size > 512uz ||
      prefix_length == 0uz || prefix_length > 512uz ||
      chunk_size > (std::numeric_limits<std::size_t>::max() / config.max_action_dim) ||
      prefix_length > (std::numeric_limits<std::size_t>::max() / config.expert_layers) ||
      prefix_length * config.expert_layers >
          (std::numeric_limits<std::size_t>::max() / config.key_value_width)) {
    error = "SmolVLA Euler sample dimensions or step count are outside the fixed execution limit";
    return 1;
  }
  const std::size_t action_values = chunk_size * config.max_action_dim;
  const std::size_t cache_values = prefix_length * config.expert_layers * config.key_value_width;
  if (!smolvla_.sample_euler_with_prefix_kv({initial_noise, action_values}, steps,
                                            {prefix_keys, cache_values},
                                            {prefix_values, cache_values},
                                            {prefix_mask, prefix_length}, {out, action_values})) {
    error = "SmolVLA cached-VLM Euler sample could not be executed";
    return 3;
  }
  return 0;
}

} // namespace fe
