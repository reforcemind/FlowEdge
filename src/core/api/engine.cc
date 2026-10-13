#include "engine.h"

#include "runtime/engine_runtime.h"
#include "runtime/environment.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <string_view>

namespace {

const char*& last_error()
{
  thread_local const char* error = "";
  return error;
}

[[nodiscard]] bool solver_method(int method, fe::FlowHead::Method& solver) noexcept
{
  if (method < FE_SOLVER_EULER || method > FE_SOLVER_RK4)
    return false;
  solver = (method == FE_SOLVER_RK4)    ? fe::FlowHead::kRK4
           : (method == FE_SOLVER_HEUN) ? fe::FlowHead::kHeun
                                        : fe::FlowHead::kEuler;
  return true;
}

[[nodiscard]] bool diffusion_scheduler(int value, fe::DiffusionHead::Scheduler& scheduler) noexcept
{
  if (value == FE_DIFFUSION_DDIM) {
    scheduler = fe::DiffusionHead::kDDIM;
    return true;
  }
  if (value == FE_DIFFUSION_DDPM) {
    scheduler = fe::DiffusionHead::kDDPM;
    return true;
  }
  return false;
}

[[nodiscard]] std::uint64_t nfe_for(std::size_t steps, fe::FlowHead::Method method) noexcept
{
  const std::uint64_t per_step = method == fe::FlowHead::kRK4    ? 4u
                                 : method == fe::FlowHead::kHeun ? 2u
                                                                 : 1u;
  return steps > (std::numeric_limits<std::uint64_t>::max() / per_step)
             ? std::numeric_limits<std::uint64_t>::max()
             : static_cast<std::uint64_t>(steps) * per_step;
}

void copy_digest(fe_model_digest& destination, const fe::ModelDigest& source) noexcept
{
  std::ranges::copy(source, destination.bytes);
}

[[nodiscard]] bool same_digest(const fe_model_digest& supplied,
                               const fe::ModelDigest& expected) noexcept
{
  return std::ranges::equal(supplied.bytes, expected);
}

[[nodiscard]] std::optional<unsigned> environment_thread_count(const char*& error) noexcept
{
  // Environment mutation is a process-startup concern; engine loading only reads it.
  const char* const value = std::getenv("FLOWEDGE_THREADS"); // NOLINT(concurrency-mt-unsafe)
  if (value == nullptr || value[0] == '\0')
    return fe::EngineRuntime::recommended_thread_count();
  const std::string_view text{value};
  unsigned threads{0u};
  const auto [end, code] = std::from_chars(text.data(), text.data() + text.size(), threads);
  if (code != std::errc{} || end != text.data() + text.size() ||
      threads > fe::EngineRuntime::kMaxPoolThreads) {
    error = "FLOWEDGE_THREADS must be an integer from 0 through 8";
    return std::nullopt;
  }
  return threads;
}

} // namespace

struct FeEngine
{
  FeEngine(std::shared_ptr<const fe::ModelWeights> weights, std::size_t slab_bytes,
           unsigned worker_threads, const char*& error)
      : runtime{std::move(weights), slab_bytes, worker_threads, error}
  {
  }

  fe::EngineRuntime runtime;
};

struct FeWeights
{
  std::shared_ptr<const fe::ModelWeights> value{};
};

namespace {

fe_engine* create_engine(std::shared_ptr<const fe::ModelWeights> weights, unsigned worker_threads)
{
  if (worker_threads > fe::EngineRuntime::kMaxPoolThreads) {
    last_error() = "Worker thread count must be from 0 through 8";
    return nullptr;
  }
  if (!weights) {
    last_error() = "Invalid null model weights";
    return nullptr;
  }
  try {
    const std::size_t slab_bytes = fe::EngineRuntime::required_slab_bytes(*weights);
    if (slab_bytes == 0uz) {
      last_error() = "Failed to size model runtime state";
      return nullptr;
    }
    auto engine =
        std::make_unique<FeEngine>(std::move(weights), slab_bytes, worker_threads, last_error());
    if (!engine->runtime.valid()) {
      last_error() = "Model architecture initialization failed";
      return nullptr;
    }
    if (fe::environment_flag("FLOWEDGE_CUDA_REQUIRED")) {
#ifndef FLOWEDGE_CUDA
      last_error() = "FLOWEDGE_CUDA_REQUIRED=1 but this binary has no CUDA kernels "
                     "(CPU wheel or CPU source build)";
      return nullptr;
#else
      if (engine->runtime.cuda_capable() && !engine->runtime.cuda_resident()) {
        last_error() = "FLOWEDGE_CUDA_REQUIRED=1 but device-resident CUDA did not attach";
        return nullptr;
      }
#endif
    }
#ifdef FLOWEDGE_CUDA
    else if (engine->runtime.cuda_capable() && !engine->runtime.cuda_resident()) {
      std::fprintf(stderr, "FlowEdge: CUDA resident load did not attach; CPU kernels are active\n");
    }
#endif
    return engine.release();
  } catch (const std::exception&) {
    last_error() = "Exception during engine load (likely OOM)";
    return nullptr;
  }
}

fe_engine* load_engine(const char* path, unsigned worker_threads)
{
  if (path == nullptr) {
    last_error() = "Invalid null model path";
    return nullptr;
  }
  const auto weights = fe::ModelWeights::open(path);
  if (!weights) {
    last_error() = weights.error();
    return nullptr;
  }
  return create_engine(*weights, worker_threads);
}

} // namespace

const char* fe_engine_last_error(void)
{
  return last_error();
}

fe_engine* fe_engine_load(const char* path)
{
  last_error() = "";
  const std::optional<unsigned> worker_threads = environment_thread_count(last_error());
  return worker_threads ? load_engine(path, *worker_threads) : nullptr;
}

fe_engine* fe_engine_load_with_threads(const char* path, unsigned worker_threads)
{
  last_error() = "";
  return load_engine(path, worker_threads);
}

fe_weights* fe_weights_load(const char* path)
{
  last_error() = "";
  if (path == nullptr) {
    last_error() = "Invalid null model path";
    return nullptr;
  }
  const auto weights = fe::ModelWeights::open(path);
  if (!weights) {
    last_error() = weights.error();
    return nullptr;
  }
  try {
    return new FeWeights{*weights}; // NOLINT(cppcoreguidelines-owning-memory)
  } catch (const std::exception&) {
    last_error() = "Exception creating immutable weight handle (likely OOM)";
    return nullptr;
  }
}

std::size_t fe_weights_size_bytes(const fe_weights* weights)
{
  return weights != nullptr && weights->value ? weights->value->weight_bytes() : 0uz;
}

void fe_weights_free(fe_weights* weights)
{
  const std::unique_ptr<FeWeights> owner{weights};
}

fe_engine* fe_engine_create_from_weights(const fe_weights* weights, unsigned worker_threads)
{
  last_error() = "";
  if (weights == nullptr) {
    last_error() = "Invalid null model weights";
    return nullptr;
  }
  return create_engine(weights->value, worker_threads);
}

fe_engine* fe_engine_create_from_weights_auto(const fe_weights* weights)
{
  last_error() = "";
  if (weights == nullptr) {
    last_error() = "Invalid null model weights";
    return nullptr;
  }
  const std::optional<unsigned> worker_threads = environment_thread_count(last_error());
  return worker_threads ? create_engine(weights->value, *worker_threads) : nullptr;
}

void fe_engine_free(fe_engine* engine)
{
  const std::unique_ptr<FeEngine> owner{engine}; // adopt + delete
}

void fe_engine_dims(const fe_engine* engine, std::size_t* d_model, std::size_t* n_layers)
{
  if (engine == nullptr)
    return;
  if (d_model != nullptr)
    *d_model = engine->runtime.d_model();
  if (n_layers != nullptr)
    *n_layers = engine->runtime.n_layers();
}

unsigned fe_engine_thread_count(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.thread_count() : 0u;
}

int fe_engine_cuda_resident(const fe_engine* engine)
{
  return engine != nullptr && engine->runtime.cuda_resident() ? 1 : 0;
}

int fe_engine_diffusion_cuda_device(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.diffusion_cuda_device() : -1;
}

int fe_engine_model_metadata(const fe_engine* engine, fe_model_metadata* metadata)
{
  last_error() = "";
  if (engine == nullptr || metadata == nullptr) {
    last_error() = "Invalid arguments to fe_engine_model_metadata";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  const fe::ModelIdentity& identity = engine->runtime.identity();
  fe_model_metadata result{};
  result.struct_size = sizeof(result);
  result.protocol_version = FE_PROTOCOL_VERSION;
  result.architecture = identity.architecture;
  result.precision = identity.precision;
  copy_digest(result.model_digest, identity.digest);
  result.d_model = identity.d_model;
  result.n_layers = identity.n_layers;
  result.d_inner = identity.d_inner;
  result.d_state = identity.d_state;
  result.d_conv = identity.d_conv;
  result.action_dim = identity.action_dim;
  result.condition_dim = identity.condition_dim;
  result.decode_snapshot_bytes = engine->runtime.decode_state_bytes();
  *metadata = result;
  return 0;
}

int fe_engine_deployment_profile(const fe_engine* engine, fe_deployment_profile* profile)
{
  last_error() = "";
  if (engine == nullptr || profile == nullptr) {
    last_error() = "Invalid arguments to fe_engine_deployment_profile";
    return 1;
  }
  const fe::DeploymentProfile* const source = engine->runtime.deployment_profile();
  if (source == nullptr) {
    last_error() = "Checkpoint has no deployment profile";
    return 2;
  }
  fe_deployment_profile result{};
  result.struct_size = sizeof(result);
  result.profile_version = source->profile_version;
  result.model_compatibility_version = source->model_compatibility_version;
  result.action_units = source->action_units == fe::DeploymentActionUnits::kNormalized
                            ? FE_ACTION_UNITS_NORMALIZED
                            : FE_ACTION_UNITS_PHYSICAL;
  result.normalization_type = source->normalization_type == fe::DeploymentNormalization::kNone
                                  ? FE_NORMALIZATION_NONE
                                  : FE_NORMALIZATION_MINMAX;
  result.solver_default =
      source->solver_default == fe::DeploymentSolver::kEuler  ? FE_PROFILE_SOLVER_EULER
      : source->solver_default == fe::DeploymentSolver::kHeun ? FE_PROFILE_SOLVER_HEUN
                                                              : FE_PROFILE_SOLVER_RK4;
  result.observation_schema_hash = source->observation_schema_hash.data();
  result.action_dim = source->action_dim;
  result.action_horizon = source->action_horizon;
  result.normalization_count = source->normalization_min.size();
  result.normalization_min =
      source->normalization_min.empty() ? nullptr : source->normalization_min.data();
  result.normalization_max =
      source->normalization_max.empty() ? nullptr : source->normalization_max.data();
  result.solver_min_steps = source->solver_min_steps;
  result.solver_max_steps = source->solver_max_steps;
  *profile = result;
  return 0;
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_run(fe_engine* engine, const std::int32_t* tokens, std::size_t seq_len, float* out)
{
  last_error() = "";
  if (engine == nullptr || tokens == nullptr || out == nullptr || seq_len == 0uz) {
    last_error() = "Invalid arguments to fe_engine_run";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.run(tokens, seq_len, out, last_error());
}

int fe_engine_run_embeddings(fe_engine* engine, const float* embeddings, std::size_t seq_len,
                             float* out)
{
  last_error() = "";
  if (engine == nullptr || embeddings == nullptr || out == nullptr || seq_len == 0uz) {
    last_error() = "Invalid arguments to fe_engine_run_embeddings";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.run_embeddings(embeddings, seq_len, out, last_error());
}

int fe_engine_run_embeddings_masked(fe_engine* engine, const float* embeddings,
                                    const std::uint8_t* attention_mask, std::size_t seq_len,
                                    float* out)
{
  last_error() = "";
  if (engine == nullptr || embeddings == nullptr || attention_mask == nullptr || out == nullptr ||
      seq_len == 0uz) {
    last_error() = "Invalid arguments to fe_engine_run_embeddings_masked";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.run_embeddings_masked(embeddings, attention_mask, seq_len, out,
                                               last_error());
}

int fe_engine_smolvla_embed_suffix(fe_engine* engine, const float* noisy_actions,
                                   std::size_t chunk_size, float timestep, float* out)
{
  last_error() = "";
  if (engine == nullptr || noisy_actions == nullptr || out == nullptr || chunk_size == 0uz ||
      !std::isfinite(timestep)) {
    last_error() = "Invalid arguments to fe_engine_smolvla_embed_suffix";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.smolvla_embed_suffix(noisy_actions, chunk_size, timestep, out,
                                              last_error());
}

int fe_engine_smolvla_run_expert(fe_engine* engine, const float* suffix, std::size_t chunk_size,
                                 const float* prefix_keys, const float* prefix_values,
                                 const std::uint8_t* prefix_mask, std::size_t prefix_length,
                                 float* out)
{
  last_error() = "";
  if (engine == nullptr || suffix == nullptr || chunk_size == 0uz || prefix_keys == nullptr ||
      prefix_values == nullptr || prefix_mask == nullptr || prefix_length == 0uz ||
      out == nullptr) {
    last_error() = "Invalid arguments to fe_engine_smolvla_run_expert";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.smolvla_run_expert(suffix, chunk_size, prefix_keys, prefix_values,
                                            prefix_mask, prefix_length, out, last_error());
}

int fe_engine_smolvla_project_actions(fe_engine* engine, const float* hidden,
                                      std::size_t chunk_size, float* out)
{
  last_error() = "";
  if (engine == nullptr || hidden == nullptr || chunk_size == 0uz || out == nullptr) {
    last_error() = "Invalid arguments to fe_engine_smolvla_project_actions";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.smolvla_project_actions(hidden, chunk_size, out, last_error());
}

int fe_engine_smolvla_denoise(fe_engine* engine, const float* noisy_actions, std::size_t chunk_size,
                              float timestep, const float* prefix_keys, const float* prefix_values,
                              const std::uint8_t* prefix_mask, std::size_t prefix_length,
                              float* out)
{
  last_error() = "";
  if (engine == nullptr || noisy_actions == nullptr || chunk_size == 0uz ||
      !std::isfinite(timestep) || prefix_keys == nullptr || prefix_values == nullptr ||
      prefix_mask == nullptr || prefix_length == 0uz || out == nullptr) {
    last_error() = "Invalid arguments to fe_engine_smolvla_denoise";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.smolvla_denoise(noisy_actions, chunk_size, timestep, prefix_keys,
                                         prefix_values, prefix_mask, prefix_length, out,
                                         last_error());
}

int fe_engine_smolvla_sample(fe_engine* engine, const float* initial_noise, std::size_t chunk_size,
                             std::size_t steps, const float* prefix_keys,
                             const float* prefix_values, const std::uint8_t* prefix_mask,
                             std::size_t prefix_length, float* out)
{
  last_error() = "";
  if (engine == nullptr || initial_noise == nullptr || chunk_size == 0uz || steps == 0uz ||
      prefix_keys == nullptr || prefix_values == nullptr || prefix_mask == nullptr ||
      prefix_length == 0uz || out == nullptr) {
    last_error() = "Invalid arguments to fe_engine_smolvla_sample";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.smolvla_sample(initial_noise, chunk_size, steps, prefix_keys,
                                        prefix_values, prefix_mask, prefix_length, out,
                                        last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_step(fe_engine* engine, std::int32_t token, float* out)
{
  last_error() = "";
  if (engine == nullptr || out == nullptr) {
    last_error() = "Invalid null arguments to fe_engine_step";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.step(token, out, last_error());
}

void fe_engine_reset(fe_engine* engine)
{
  if (engine != nullptr)
    engine->runtime.reset();
}

std::size_t fe_engine_action_dim(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.action_dim() : 0uz;
}

std::size_t fe_engine_action_horizon(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.action_horizon() : 0uz;
}

int fe_engine_diffusion_metadata(const fe_engine* engine, fe_diffusion_metadata* metadata)
{
  last_error() = "";
  if (engine == nullptr || metadata == nullptr) {
    last_error() = "Invalid arguments to fe_engine_diffusion_metadata";
    return 1;
  }
  const fe::DiffusionConfig* const config = engine->runtime.diffusion_config();
  if (config == nullptr) {
    last_error() = "Model has no Diffusion Policy head";
    return 4;
  }
  fe_diffusion_metadata result{};
  result.struct_size = sizeof(result);
  result.protocol_version = FE_PROTOCOL_VERSION;
  result.clip_sample = config->clip_sample ? 1u : 0u;
  result.action_dim = config->action_dim;
  result.condition_dim = config->condition_dim;
  result.horizon = config->horizon;
  result.action_steps = config->action_steps;
  result.observation_steps = config->observation_steps;
  result.train_timesteps = config->train_timesteps;
  result.clip_sample_range = config->clip_sample_range;
  *metadata = result;
  return 0;
}

std::size_t fe_engine_condition_dim(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.condition_dim() : 0uz;
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_sample_diffusion(fe_engine* engine, const float* condition, const float* noise,
                               std::size_t steps, int scheduler, std::uint64_t seed, float* action)
{
  last_error() = "";
  if (engine == nullptr || condition == nullptr || noise == nullptr || action == nullptr ||
      steps == 0uz) {
    last_error() = "Invalid arguments to fe_engine_sample_diffusion";
    return 1;
  }
  fe::DiffusionHead::Scheduler selected{};
  if (!diffusion_scheduler(scheduler, selected)) {
    last_error() = "Diffusion scheduler must be FE_DIFFUSION_DDIM or FE_DIFFUSION_DDPM";
    return 5;
  }
  return engine->runtime.sample_diffusion(condition, noise, steps, selected, seed, action,
                                          last_error());
}

int fe_engine_sample_diffusion_device(fe_engine* engine, const float* condition_device,
                                      const float* noise_device, std::size_t steps, int scheduler,
                                      std::uint64_t seed, std::size_t action_start,
                                      std::size_t action_steps, float* action_device)
{
  last_error() = "";
  if (engine == nullptr || condition_device == nullptr || noise_device == nullptr ||
      action_device == nullptr || steps == 0uz || action_steps == 0uz) {
    last_error() = "Invalid arguments to fe_engine_sample_diffusion_device";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  fe::DiffusionHead::Scheduler selected{};
  if (!diffusion_scheduler(scheduler, selected)) {
    last_error() = "Diffusion scheduler must be FE_DIFFUSION_DDIM or FE_DIFFUSION_DDPM";
    return FE_STATUS_INVALID_SOLVER;
  }
  return engine->runtime.sample_diffusion_device(condition_device, noise_device, steps, selected,
                                                 seed, action_start, action_steps, action_device,
                                                 last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_diffusion_denoise(fe_engine* engine, const float* condition,
                                const float* normalized_sample, float timestep,
                                float* predicted_noise)
{
  last_error() = "";
  if (engine == nullptr || condition == nullptr || normalized_sample == nullptr ||
      predicted_noise == nullptr || !std::isfinite(timestep)) {
    last_error() = "Invalid arguments to fe_engine_diffusion_denoise";
    return 1;
  }
  return engine->runtime.denoise_diffusion(condition, normalized_sample, timestep, predicted_noise,
                                           last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_sample(fe_engine* engine, const std::int32_t* tokens, std::size_t seq_len,
                     const float* noise, std::size_t steps, int method, float* action)
{
  last_error() = "";
  if (engine == nullptr || tokens == nullptr || noise == nullptr || action == nullptr ||
      seq_len == 0uz || steps == 0uz) {
    last_error() = "Invalid arguments to fe_engine_sample";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  fe::FlowHead::Method solver{};
  if (!solver_method(method, solver)) {
    last_error() = "Invalid solver method";
    return FE_STATUS_INVALID_SOLVER;
  }
  return engine->runtime.sample(tokens, seq_len, noise, steps, solver, action, last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_sample_condition(fe_engine* engine, const float* condition, const float* noise,
                               std::size_t steps, int method, float* action)
{
  last_error() = "";
  if (engine == nullptr || condition == nullptr || noise == nullptr || action == nullptr ||
      steps == 0uz) {
    last_error() = "Invalid arguments to fe_engine_sample_condition";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  fe::FlowHead::Method solver{};
  if (!solver_method(method, solver)) {
    last_error() = "Invalid solver method";
    return FE_STATUS_INVALID_SOLVER;
  }
  return engine->runtime.sample_condition(condition, noise, steps, solver, action, last_error());
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_flow_begin(fe_engine* engine, const float* condition, const float* noise,
                         std::size_t steps, int method)
{
  last_error() = "";
  if (engine == nullptr || condition == nullptr || noise == nullptr || steps == 0uz) {
    last_error() = "Invalid arguments to fe_engine_flow_begin";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  fe::FlowHead::Method solver{};
  if (!solver_method(method, solver)) {
    last_error() = "Invalid solver method";
    return FE_STATUS_INVALID_SOLVER;
  }
  return engine->runtime.flow_begin(condition, noise, steps, solver, last_error());
}

int fe_engine_make_condition_metadata(const fe_engine* engine, std::uint64_t timestamp_ns,
                                      std::uint64_t deadline_ns, std::uint64_t generation,
                                      std::size_t steps, int method,
                                      fe_condition_metadata* metadata)
{
  last_error() = "";
  if (engine == nullptr || metadata == nullptr || steps == 0uz ||
      (deadline_ns != 0u && deadline_ns < timestamp_ns)) {
    last_error() = "Invalid arguments to fe_engine_make_condition_metadata";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  fe::FlowHead::Method solver{};
  if (!solver_method(method, solver)) {
    last_error() = "Invalid solver method";
    return FE_STATUS_INVALID_SOLVER;
  }
  const fe::ModelIdentity& identity = engine->runtime.identity();
  fe_condition_metadata result{};
  result.struct_size = sizeof(result);
  result.protocol_version = FE_PROTOCOL_VERSION;
  result.solver = static_cast<std::uint32_t>(method);
  copy_digest(result.model_digest, identity.digest);
  result.timestamp_ns = timestamp_ns;
  result.deadline_ns = deadline_ns;
  result.generation = generation;
  result.condition_dim = identity.condition_dim;
  result.action_dim = identity.action_dim;
  result.solver_steps = steps;
  result.remaining_nfe = nfe_for(steps, solver);
  *metadata = result;
  return 0;
}

int fe_engine_flow_begin_request(fe_engine* engine, const float* condition, const float* noise,
                                 const fe_condition_metadata* metadata)
{
  last_error() = "";
  if (engine == nullptr || condition == nullptr || noise == nullptr || metadata == nullptr ||
      metadata->struct_size < sizeof(*metadata) ||
      metadata->protocol_version != FE_PROTOCOL_VERSION || metadata->reserved != 0u ||
      metadata->solver_steps == 0u ||
      metadata->solver_steps > std::numeric_limits<std::size_t>::max()) {
    last_error() = "Invalid versioned flow request metadata";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  fe::FlowHead::Method solver{};
  if (!solver_method(static_cast<int>(metadata->solver), solver)) {
    last_error() = "Invalid solver method in flow request metadata";
    return FE_STATUS_INVALID_SOLVER;
  }
  const fe::ModelIdentity& identity = engine->runtime.identity();
  if (!same_digest(metadata->model_digest, identity.digest) ||
      metadata->condition_dim != identity.condition_dim ||
      metadata->action_dim != identity.action_dim ||
      metadata->remaining_nfe != nfe_for(metadata->solver_steps, solver) ||
      (metadata->deadline_ns != 0u && metadata->deadline_ns < metadata->timestamp_ns)) {
    last_error() = "Flow request metadata is incompatible with this model";
    return FE_STATUS_INCOMPATIBLE_STATE;
  }
  return engine->runtime.flow_begin_request(
      condition, noise,
      fe::FlowRequestMetadata{.timestamp_ns = metadata->timestamp_ns,
                              .deadline_ns = metadata->deadline_ns,
                              .generation = metadata->generation,
                              .steps = static_cast<std::size_t>(metadata->solver_steps),
                              .method = solver,
                              .generation_tracking = true},
      last_error());
}

void fe_engine_cancel_before(fe_engine* engine, std::uint64_t generation)
{
  if (engine != nullptr)
    engine->runtime.cancel_before(generation);
}

#if defined(__MINGW32__) && defined(__AVX2__)
__attribute__((force_align_arg_pointer))
#endif
int fe_engine_flow_advance(fe_engine* engine, std::size_t step_budget, float* action,
                           std::size_t* steps_remaining)
{
  last_error() = "";
  if (engine == nullptr || action == nullptr || step_budget == 0uz) {
    last_error() = "Invalid arguments to fe_engine_flow_advance";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  std::size_t remaining{0uz};
  const int rc = engine->runtime.flow_advance(step_budget, action, remaining, last_error());
  if ((rc == 0 || rc == 8) && steps_remaining != nullptr)
    *steps_remaining = remaining;
  return rc;
}

int fe_engine_flow_action_metadata(const fe_engine* engine, fe_action_metadata* metadata)
{
  last_error() = "";
  if (engine == nullptr || metadata == nullptr) {
    last_error() = "Invalid arguments to fe_engine_flow_action_metadata";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  const fe::ModelIdentity& identity = engine->runtime.identity();
  const fe::FlowRequestMetadata& request = engine->runtime.flow_request_metadata();
  fe_action_metadata result{};
  result.struct_size = sizeof(result);
  result.protocol_version = FE_PROTOCOL_VERSION;
  result.solver = static_cast<std::uint32_t>(request.method);
  result.status = engine->runtime.flow_action_status();
  copy_digest(result.model_digest, identity.digest);
  result.timestamp_ns = request.timestamp_ns;
  result.deadline_ns = request.deadline_ns;
  result.generation = request.generation;
  result.condition_dim = identity.condition_dim;
  result.action_dim = identity.action_dim;
  result.remaining_nfe = engine->runtime.flow_remaining_nfe();
  *metadata = result;
  return 0;
}

std::size_t fe_engine_decode_state_bytes(const fe_engine* engine)
{
  return engine != nullptr ? engine->runtime.decode_state_bytes() : 0uz;
}

int fe_engine_export_decode_state(const fe_engine* engine, void* destination, std::size_t bytes)
{
  last_error() = "";
  if (engine == nullptr || destination == nullptr) {
    last_error() = "Invalid arguments to fe_engine_export_decode_state";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t required = engine->runtime.decode_state_bytes();
  if (required != 0uz && bytes < required) {
    last_error() = "Decode snapshot destination is too small";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.export_decode_state({static_cast<std::byte*>(destination), bytes},
                                             last_error());
}

int fe_engine_import_decode_state(fe_engine* engine, const void* source, std::size_t bytes)
{
  last_error() = "";
  if (engine == nullptr || source == nullptr) {
    last_error() = "Invalid arguments to fe_engine_import_decode_state";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t required = engine->runtime.decode_state_bytes();
  if (required != 0uz && bytes != required) {
    last_error() = bytes < required ? "Decode snapshot source is truncated"
                                    : "Decode snapshot source size is incompatible";
    return FE_STATUS_INVALID_ARGUMENT;
  }
  return engine->runtime.import_decode_state({static_cast<const std::byte*>(source), bytes},
                                             last_error());
}
