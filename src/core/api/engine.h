#ifndef FE_ENGINE_H
#define FE_ENGINE_H

#include "protocol/contracts.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FeEngine fe_engine;
typedef struct FeWeights fe_weights;

enum
{
  FE_SOLVER_EULER = 0,
  FE_SOLVER_HEUN = 1,
  FE_SOLVER_RK4 = 2,
};

enum
{
  FE_DIFFUSION_DDIM = 0,
  FE_DIFFUSION_DDPM = 1,
};

typedef struct fe_diffusion_metadata
{
  uint32_t struct_size;
  uint32_t protocol_version;
  uint32_t clip_sample;
  uint32_t reserved;
  uint64_t action_dim;
  uint64_t condition_dim;
  uint64_t horizon;
  uint64_t action_steps;
  uint64_t observation_steps;
  uint64_t train_timesteps;
  float clip_sample_range;
  float reserved_float;
} fe_diffusion_metadata;

/** Stable return values shared by all C ABI operations that return int. */
enum
{
  FE_STATUS_OK = 0,
  FE_STATUS_INVALID_ARGUMENT = 1,
  FE_STATUS_RESOURCE_EXHAUSTED = 2,
  FE_STATUS_TOKEN_OUT_OF_RANGE = 3,
  FE_STATUS_UNSUPPORTED_MODEL = 4,
  FE_STATUS_INVALID_SOLVER = 5,
  FE_STATUS_NO_ACTIVE_OPERATION = 6,
  FE_STATUS_STATE_UNAVAILABLE = 7,
  FE_STATUS_CANCELLED = 8,
  FE_STATUS_INCOMPATIBLE_STATE = 9,
  FE_STATUS_CORRUPT_STATE = 10,
};
/**
 * Last C ABI error on this thread. Valid until the next ABI call; never NULL.
 */
const char* fe_engine_last_error(void);

/**
 * @brief Load a FlowEdge model from a .safetensors file.
 * @param path Path to the .safetensors file.
 * @return Pointer to the initialized engine, or NULL on failure. Check
 * fe_engine_last_error() on failure.
 */
fe_engine* fe_engine_load(const char* path);

/**
 * @brief Load a model with an exact number of background worker threads.
 *
 * Passing 0 selects caller-thread-only execution. Values above 8 are rejected.
 * fe_engine_load() instead uses FLOWEDGE_THREADS when set, otherwise a
 * bandwidth-aware automatic default. FLOWEDGE_CUDA_REQUIRED=1 fails load
 * when device-resident CUDA did not attach. FLOWEDGE_CUDA_FORCE_HOST=1 skips
 * the CUDA upload and keeps CPU kernels (tests / 4 GB cards).
 */
fe_engine* fe_engine_load_with_threads(const char* path, unsigned worker_threads);

/** Load one immutable checkpoint store that can be shared by multiple engines. */
fe_weights* fe_weights_load(const char* path);

/** Return the checkpoint tensor bytes retained by an immutable weight store. */
size_t fe_weights_size_bytes(const fe_weights* weights);

/** Release a weight handle. Engines created from it retain their own shared reference. */
void fe_weights_free(fe_weights* weights);

/** Create an engine with private state and a shared reference to immutable weights. */
fe_engine* fe_engine_create_from_weights(const fe_weights* weights, unsigned worker_threads);

/** Create a shared-weight engine using FLOWEDGE_THREADS or the automatic default. */
fe_engine* fe_engine_create_from_weights_auto(const fe_weights* weights);

/**
 * @brief Get the underlying backbone model dimensions.
 * @param engine The engine instance.
 * @param d_model Output pointer for the model dimension (can be NULL).
 * @param n_layers Output pointer for the number of layers (can be NULL).
 */
void fe_engine_dims(const fe_engine* engine, size_t* d_model, size_t* n_layers);

/**
 * @brief Process a sequence of tokens in batch mode (prefill).
 * @param engine The engine instance.
 * @param tokens Array of input token IDs.
 * @param seq_len Length of the token array.
 * @param out Output buffer of size [seq_len * d_model] for the hidden states.
 * @return FE_STATUS_OK on success, otherwise one of the FE_STATUS_* values.
 */
int fe_engine_run(fe_engine* engine, const int32_t* tokens, size_t seq_len, float* out);

/**
 * @brief Process a sequence of caller-supplied Transformer embeddings.
 *
 * The input and output buffers have size [seq_len * d_model]. This boundary
 * is supported only by the generic Transformer backbone; Mamba and action-only
 * checkpoints return FE_STATUS_UNSUPPORTED_MODEL.
 */
int fe_engine_run_embeddings(fe_engine* engine, const float* embeddings, size_t seq_len,
                             float* out);

/**
 * @brief Process caller-supplied embeddings with a batch-one prefix attention mask.
 *
 * `attention_mask` contains one byte per row: leading 1 values are attended and
 * trailing 0 values are padding. The mask is the condition-sequence boundary
 * for VLA adapters; it does not implement a SmolVLA action expert.
 */
int fe_engine_run_embeddings_masked(fe_engine* engine, const float* embeddings,
                                    const uint8_t* attention_mask, size_t seq_len, float* out);

/**
 * @brief Build the trained SmolVLA action-expert suffix embedding.
 *
 * This is the exact action/time projection boundary for a padded action
 * chunk. Call `fe_engine_smolvla_run_expert` with a VLM K/V cache to execute
 * the trained action expert.
 * `noisy_actions` has [chunk_size * max_action_dim] values and `out` has
 * [chunk_size * expert_width] values; dimensions are read from the checkpoint.
 */
int fe_engine_smolvla_embed_suffix(fe_engine* engine, const float* noisy_actions, size_t chunk_size,
                                   float timestep, float* out);

/**
 * @brief Execute the SmolVLA action expert from an external VLM K/V cache.
 *
 * `suffix` and `out` have [chunk_size * expert_width] values. `prefix_keys`
 * and `prefix_values` are layer-major F32 VLM K/V caches with
 * [expert_layers * prefix_length * 320] values. The cache keys must already
 * include the VLM RoPE transform. `prefix_mask` has one binary validity byte
 * per physical prefix slot; valid positions may be sparse because source
 * SmolVLA pads language tokens before appending state. This does not run
 * LeRobot preprocessing or the vision/language VLM encoder.
 */
int fe_engine_smolvla_run_expert(fe_engine* engine, const float* suffix, size_t chunk_size,
                                 const float* prefix_keys, const float* prefix_values,
                                 const uint8_t* prefix_mask, size_t prefix_length, float* out);

/**
 * @brief Project SmolVLA expert hidden rows into padded action coordinates.
 *
 * `hidden` has [chunk_size * expert_width] values; `out` has
 * [chunk_size * max_action_dim] values. The caller removes padded action
 * coordinates according to its checkpoint configuration.
 */
int fe_engine_smolvla_project_actions(fe_engine* engine, const float* hidden, size_t chunk_size,
                                      float* out);

/**
 * @brief Run one complete SmolVLA action-expert denoising step.
 *
 * This composes suffix embedding, cached-VLM expert execution, and action
 * projection without runtime allocation. `noisy_actions` and `out` have
 * [chunk_size * max_action_dim] values. The external VLM K/V cache follows
 * the contract of `fe_engine_smolvla_run_expert`.
 */
int fe_engine_smolvla_denoise(fe_engine* engine, const float* noisy_actions, size_t chunk_size,
                              float timestep, const float* prefix_keys, const float* prefix_values,
                              const uint8_t* prefix_mask, size_t prefix_length, float* out);

/**
 * @brief Sample a SmolVLA action chunk with the source deterministic Euler schedule.
 *
 * `initial_noise` and `out` have [chunk_size * max_action_dim] values. `steps`
 * is bounded to [1, 100]. `out` may exactly alias `initial_noise`; other
 * overlap is unsupported. The VLM cache contract is the same as
 * `fe_engine_smolvla_run_expert`.
 */
int fe_engine_smolvla_sample(fe_engine* engine, const float* initial_noise, size_t chunk_size,
                             size_t steps, const float* prefix_keys, const float* prefix_values,
                             const uint8_t* prefix_mask, size_t prefix_length, float* out);

/**
 * @brief Advance the streaming decode state by a single token.
 * @param engine The engine instance.
 * @param token The input token ID.
 * @param out Output buffer of size [d_model] for the current hidden state.
 * @return FE_STATUS_OK on success, otherwise one of the FE_STATUS_* values.
 */
int fe_engine_step(fe_engine* engine, int32_t token, float* out);

/**
 * @brief Reset the internal streaming state for a new sequence.
 * @param engine The engine instance.
 */
void fe_engine_reset(fe_engine* engine);

/**
 * @brief Get the number of background worker threads in the engine.
 * @param engine The engine instance.
 * @return The number of worker threads (0 if single-threaded).
 */
unsigned fe_engine_thread_count(const fe_engine* engine);

/**
 * @brief 1 if Mamba, flow, or Diffusion Policy weights are device-resident.
 *
 * A CUDA Core binary still returns 0 when attach failed (OOM, no GPU) or
 * FLOWEDGE_CUDA_FORCE_HOST=1. CPU wheels always return 0. Load itself succeeds
 * on CPU kernels unless FLOWEDGE_CUDA_REQUIRED=1.
 */
int fe_engine_cuda_resident(const fe_engine* engine);

/** CUDA device owning the resident Diffusion Policy head, or -1 when unavailable. */
int fe_engine_diffusion_cuda_device(const fe_engine* engine);

/** Fill versioned architecture, precision, dimension, and model-digest metadata. */
int fe_engine_model_metadata(const fe_engine* engine, fe_model_metadata* metadata);

/**
 * @brief Borrow the validated deployment profile retained by the engine.
 *
 * The returned pointers remain valid until fe_engine_free(). Return value 2 means that the
 * checkpoint uses the legacy path and has no deployment profile.
 */
int fe_engine_deployment_profile(const fe_engine* engine, fe_deployment_profile* profile);

/**
 * @brief Get the action dimension of the flow-matching head.
 * @param engine The engine instance.
 * @return The action dimension, or 0 if the checkpoint lacks a flow head.
 */
size_t fe_engine_action_dim(const fe_engine* engine);

/**
 * @brief Get the fixed number of action timesteps returned by the action head.
 * @return Diffusion Policy horizon, 1 for a flow head, or 0 without an action head.
 */
size_t fe_engine_action_horizon(const fe_engine* engine);

/** Fill the Diffusion Policy shape, action-slice, and scheduler metadata. */
int fe_engine_diffusion_metadata(const fe_engine* engine, fe_diffusion_metadata* metadata);

/**
 * @brief Get the condition-vector dimension accepted by the flow head.
 * @param engine The engine instance.
 * @return The condition dimension, or 0 if the checkpoint lacks a flow head.
 */
size_t fe_engine_condition_dim(const fe_engine* engine);

/**
 * @brief Sample a fixed Diffusion Policy action horizon from an external condition.
 *
 * The checkpoint's MIN_MAX action normalization is inverted before returning.
 * DDIM is deterministic from condition and initial noise. DDPM additionally
 * uses `seed` for deterministic per-step Gaussian noise.
 *
 * @param condition Input array of size [condition_dim].
 * @param noise Initial normalized Gaussian noise [action_horizon * action_dim].
 * @param steps Scheduler inference steps, from 1 through the checkpoint's train timesteps.
 * @param scheduler FE_DIFFUSION_DDIM or FE_DIFFUSION_DDPM.
 * @param seed DDPM random seed; ignored by deterministic DDIM.
 * @param action Output [action_horizon * action_dim], in the original action units.
 */
int fe_engine_sample_diffusion(fe_engine* engine, const float* condition, const float* noise,
                               size_t steps, int scheduler, uint64_t seed, float* action);

/**
 * @brief Sample directly between caller-owned CUDA tensors.
 *
 * All pointers must address contiguous float32 storage on
 * fe_engine_diffusion_cuda_device(engine). The output is compact and contains
 * action_steps rows beginning at action_start. This call synchronizes the CUDA
 * device before returning; it does not allocate device memory.
 */
int fe_engine_sample_diffusion_device(fe_engine* engine, const float* condition_device,
                                      const float* noise_device, size_t steps, int scheduler,
                                      uint64_t seed, size_t action_start, size_t action_steps,
                                      float* action_device);

/** Run one ConditionalUnet1D epsilon-prediction pass in normalized action space. */
int fe_engine_diffusion_denoise(fe_engine* engine, const float* condition,
                                const float* normalized_sample, float timestep,
                                float* predicted_noise);

/**
 * @brief Sample an action trajectory from the flow-matching head.
 *
 * This function processes the input tokens to generate a conditioning context,
 * then solves the ODE from the provided noise to the final action.
 *
 * @param engine The engine instance.
 * @param tokens Array of conditioning token IDs (prefix).
 * @param seq_len Length of the token array.
 * @param noise Initial Gaussian noise array of size [action_dim].
 * @param steps Number of ODE solver steps.
 * @param method One of FE_SOLVER_EULER, FE_SOLVER_HEUN, or FE_SOLVER_RK4.
 * @param action Output buffer of size [action_dim] for the final action.
 * @return FE_STATUS_OK on success, otherwise one of the FE_STATUS_* values.
 */
int fe_engine_sample(fe_engine* engine, const int32_t* tokens, size_t seq_len, const float* noise,
                     size_t steps, int method, float* action);

/**
 * @brief Sample directly from an externally produced condition vector.
 *
 * This bypasses the built-in backbone and lets a vision encoder, VLA, or other
 * runtime reuse FlowEdge's deterministic action head without copying through a
 * token representation.
 *
 * @param engine The engine instance.
 * @param condition Input array of size [condition_dim].
 * @param noise Initial Gaussian noise array of size [action_dim].
 * @param steps Number of ODE solver steps.
 * @param method One of FE_SOLVER_EULER, FE_SOLVER_HEUN, or FE_SOLVER_RK4.
 * @param action Output buffer of size [action_dim].
 * @return FE_STATUS_OK on success, otherwise one of the FE_STATUS_* values.
 */
int fe_engine_sample_condition(fe_engine* engine, const float* condition, const float* noise,
                               size_t steps, int method, float* action);

/**
 * @brief Begin an allocation-free, resumable flow solve from an external condition.
 *
 * Only one resumable solve may be active per engine. Starting another replaces
 * the previous solve. Use fe_engine_flow_advance() to run bounded work units.
 *
 * @return FE_STATUS_OK on success, otherwise one of the FE_STATUS_* values.
 */
int fe_engine_flow_begin(fe_engine* engine, const float* condition, const float* noise,
                         size_t steps, int method);

/**
 * Fill condition metadata for a generation-tracked request.
 * `remaining_nfe` is derived from solver steps and method.
 */
int fe_engine_make_condition_metadata(const fe_engine* engine, uint64_t timestamp_ns,
                                      uint64_t deadline_ns, uint64_t generation, size_t steps,
                                      int method, fe_condition_metadata* metadata);

/** Begin a solve after validating the complete versioned request metadata. */
int fe_engine_flow_begin_request(fe_engine* engine, const float* condition, const float* noise,
                                 const fe_condition_metadata* metadata);

/**
 * Atomically cancel active work whose generation is lower than `generation`.
 * This is the only engine operation safe to call concurrently with flow advance.
 */
void fe_engine_cancel_before(fe_engine* engine, uint64_t generation);

/**
 * @brief Advance a resumable solve by at most `step_budget` complete ODE steps.
 *
 * The current solver state is copied to `action`; publish it as a final action
 * when `steps_remaining` reaches zero.
 *
 * @param engine The engine instance.
 * @param step_budget Maximum number of ODE steps to execute in this call.
 * @param action Output buffer of size [action_dim].
 * @param steps_remaining Optional output for the number of unfinished steps.
 * @return FE_STATUS_OK on success, otherwise one of the FE_STATUS_* values.
 */
int fe_engine_flow_advance(fe_engine* engine, size_t step_budget, float* action,
                           size_t* steps_remaining);

/** Return metadata for the current action candidate or completed action. */
int fe_engine_flow_action_metadata(const fe_engine* engine, fe_action_metadata* metadata);

/**
 * @brief Return the byte size of the model's streaming decode state.
 *
 * The returned size includes a versioned, checksummed compatibility header.
 */
size_t fe_engine_decode_state_bytes(const fe_engine* engine);

/**
 * @brief Copy the streaming decode state into caller-owned storage.
 * @param destination Non-null caller-owned buffer with at least
 * fe_engine_decode_state_bytes() bytes. The buffer is unchanged on failure.
 * @return FE_STATUS_OK on success, otherwise one of the FE_STATUS_* values.
 */
int fe_engine_export_decode_state(const fe_engine* engine, void* destination, size_t bytes);

/**
 * @brief Restore a validated streaming state from the same model.
 * @param source Non-null caller-owned snapshot buffer of exactly
 * fe_engine_decode_state_bytes() bytes. The engine state is unchanged on failure.
 * @return FE_STATUS_OK on success, otherwise one of the FE_STATUS_* values.
 */
int fe_engine_import_decode_state(fe_engine* engine, const void* source, size_t bytes);

/**
 * @brief Free the engine resources.
 * @param engine The engine instance to destroy.
 */
void fe_engine_free(fe_engine* engine);

#ifdef __cplusplus
}
#endif

#endif // FE_ENGINE_H
