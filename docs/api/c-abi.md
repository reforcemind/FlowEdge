# C-ABI

The only public surface. C linkage, opaque handle.

```c
fe_engine* fe_engine_load(const char* path);
fe_engine* fe_engine_load_with_threads(const char* path, unsigned worker_threads);
fe_weights* fe_weights_load(const char* path);
size_t      fe_weights_size_bytes(const fe_weights*);
void        fe_weights_free(fe_weights*);
fe_engine*  fe_engine_create_from_weights(const fe_weights*, unsigned worker_threads);
fe_engine*  fe_engine_create_from_weights_auto(const fe_weights*);
void       fe_engine_free(fe_engine* e);

void   fe_engine_dims(const fe_engine*, size_t* d_model, size_t* n_layers);
size_t fe_engine_action_dim(const fe_engine*);
size_t fe_engine_action_horizon(const fe_engine*);
size_t fe_engine_condition_dim(const fe_engine*);
unsigned fe_engine_thread_count(const fe_engine*);
int      fe_engine_cuda_resident(const fe_engine*);
int      fe_engine_diffusion_cuda_device(const fe_engine*);
int fe_engine_model_metadata(const fe_engine*, fe_model_metadata*);
int fe_engine_deployment_profile(const fe_engine*, fe_deployment_profile*);

int fe_engine_run(fe_engine*, const int32_t* tokens, size_t n, float* out);
int fe_engine_run_embeddings(fe_engine*, const float* embeddings, size_t seq_len, float* out);
int fe_engine_run_embeddings_masked(fe_engine*, const float* embeddings,
                                    const uint8_t* attention_mask, size_t seq_len, float* out);
int fe_engine_sample(fe_engine*, const int32_t* tokens, size_t n,
                     const float* noise, size_t steps, int method, float* action);
int fe_engine_sample_condition(fe_engine*, const float* condition,
                               const float* noise, size_t steps,
                               int method, float* action);
int fe_engine_sample_diffusion(fe_engine*, const float* condition,
                               const float* noise, size_t steps,
                               int scheduler, uint64_t seed, float* action);
int fe_engine_sample_diffusion_device(fe_engine*, const float* condition_device,
                                      const float* noise_device, size_t steps,
                                      int scheduler, uint64_t seed,
                                      size_t action_start, size_t action_steps,
                                      float* action_device);
int fe_engine_diffusion_denoise(fe_engine*, const float* condition,
                                const float* normalized_sample, float timestep,
                                float* predicted_noise);

int fe_engine_flow_begin(fe_engine*, const float* condition,
                         const float* noise, size_t steps, int method);
int fe_engine_flow_advance(fe_engine*, size_t step_budget, float* action,
                           size_t* steps_remaining);
int fe_engine_make_condition_metadata(const fe_engine*, uint64_t timestamp_ns,
                                      uint64_t deadline_ns, uint64_t generation,
                                      size_t steps, int method,
                                      fe_condition_metadata*);
int fe_engine_flow_begin_request(fe_engine*, const float* condition,
                                 const float* noise,
                                 const fe_condition_metadata*);
void fe_engine_cancel_before(fe_engine*, uint64_t generation);
int fe_engine_flow_action_metadata(const fe_engine*, fe_action_metadata*);

int  fe_engine_step(fe_engine*, int32_t token, float* out);
void fe_engine_reset(fe_engine*);

size_t fe_engine_decode_state_bytes(const fe_engine*);
int fe_engine_export_decode_state(const fe_engine*, void* destination, size_t bytes);
int fe_engine_import_decode_state(fe_engine*, const void* source, size_t bytes);

const char* fe_engine_last_error(void);
```

Rules:

- Everything runtime-relevant is a parameter. Nothing is baked in.
- The handle owns the whole runtime. Free it with `fe_engine_free`.
- `fe_weights` owns immutable checkpoint tensors. Engines created from it retain a shared reference,
  so the weight handle may be released immediately after construction. Mutable engine state is never
  shared.
- `fe_engine_load` reads `FLOWEDGE_THREADS=0..8` when present and otherwise uses a
  bandwidth-aware automatic default. `fe_engine_load_with_threads` bypasses the environment;
  zero selects caller-thread-only execution.
- `fe_engine_cuda_resident` is 1 only when Mamba, flow, or Diffusion Policy weights
  are device-resident. A CUDA binary still returns 0 after OOM, no GPU, or
  `FLOWEDGE_CUDA_FORCE_HOST=1`; load keeps CPU kernels unless `FLOWEDGE_CUDA_REQUIRED=1`.
- `fe_engine_sample_diffusion_device` accepts contiguous float32 CUDA pointers on
  `fe_engine_diffusion_cuda_device`. It keeps condition, initial noise, and the requested compact
  action slice on device and synchronizes before return. It performs no device allocation.
- No exception crosses the boundary. Errors return `nullptr` or a non-zero code.
- `method` is 0 for Euler, 1 for Heun, 2 for RK4.
- Prefer the named constants `FE_SOLVER_EULER`, `FE_SOLVER_HEUN`, and `FE_SOLVER_RK4` from
  `engine.h` instead of literal method values. Unknown values fail with a non-zero return code.
- `fe_engine_sample_condition` accepts the output of an encoder owned by another runtime. A
  checkpoint may therefore contain only `flow.*` tensors and no built-in backbone.
- `fe_engine_deployment_profile` returns the validated checkpoint profile through borrowed pointers.
  Those pointers remain valid until `fe_engine_free`; return code `2` explicitly identifies a
  legacy checkpoint without a profile. The profile is descriptive metadata: FlowEdge does not
  execute observation preprocessing from it.
- `fe_engine_flow_begin` projects the condition once. Each `fe_engine_flow_advance` executes at
  most `step_budget` complete solver steps and reports how many remain. Starting a new solve
  replaces the previous one.
- `fe_engine_make_condition_metadata` fills protocol version, model digest, dimensions, solver,
  generation, timestamps, and initial NFE. `fe_engine_flow_begin_request` validates every field.
  `fe_engine_cancel_before` atomically makes older generations stale; an advance notices this
  between complete solver steps and returns code 8.
- `fe_engine_flow_action_metadata` preserves the source timestamp/deadline/generation and reports
  running, complete, cancelled, or failed status plus remaining NFE.
- Decode snapshots remain caller-owned and allocation-free, but are no longer raw floats. The
  fixed little-endian envelope contains a version, architecture, precision, dimensions, model
  digest, payload size, and checksum. Imports reject incompatible, truncated, and corrupt data
  before modifying engine state.
- One handle has mutable scratch and sampler state and is not safe for concurrent calls. Use one
  engine per concurrently executing session. `fe_engine_cancel_before` is the sole operation
  designed for a concurrent scheduler thread.
- Prefill and token-conditioned sampling accept 1 to 512 tokens per call; streaming `step` has no
  growing sequence buffer.
- `fe_engine_smolvla_embed_suffix` accepts one padded action chunk and returns the real
  SmolVLA action/time suffix embedding. `fe_engine_smolvla_run_expert` executes the trained
  expert from a caller-supplied, RoPE-applied VLM K/V cache, and
  `fe_engine_smolvla_project_actions` produces padded action coordinates. These calls do not run
  image/language preprocessing or the VLM encoder, so a complete policy still needs an external
  cache producer and real-capture parity evidence.
- `fe_engine_smolvla_denoise` composes those three operations in the native runtime for one flow
  step and writes caller-owned padded action velocity storage without allocating in the hot path.
- `fe_engine_smolvla_sample` applies the source deterministic Euler schedule from caller-supplied
  noise and the same captured VLM cache. It supports 1--100 steps and permits exact in-place
  noise/output storage. It still does not construct that cache or execute the VLM.

## Link with CMake

```cmake
find_package(FlowEdge REQUIRED)
target_link_libraries(my_node PRIVATE FlowEdge::Core)
```

Source: `src/core/api/engine.h` and `src/core/protocol/contracts.h`. See
[ADR 0003](../decisions/0003-api-boundary) and
[ADR 0011](../decisions/0011-versioned-state-contracts).
