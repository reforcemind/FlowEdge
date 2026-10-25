# Python

Built with pybind11. Install with pip.

```bash
python -m pip install flowedge-0.1.2-*.whl   # CPU wheel from the v0.1.2 GitHub Release
# or, from a clone with a C++23 toolchain:
python -m pip install .
```

Wheels are CPU. CUDA is a source build. PyPI is not published. [Python quickstart](../python-quickstart).

```python
import numpy as np, flowedge

e = flowedge.Engine("models/mamba_flow.safetensors")
print(e.action_dim, e.condition_dim, e.d_model, e.thread_count, e.cuda_resident)
print(e.model_metadata)  # architecture, precision, dimensions, digest, snapshot size

tokens = np.array([1, 2, 3, 4], dtype=np.int32)
hidden = e.run(tokens)
noise = np.zeros(e.action_dim, dtype=np.float32)
action = e.sample(prefix=tokens,
                  noise=noise,
                  steps=10, method="euler")
```

The convenience calls accept array-like inputs, normalize their dtype/layout when necessary, and
allocate returned NumPy arrays. A real-time loop can instead own every output and use the zero-copy
forms; those inputs and outputs must expose C-contiguous `int32` or `float32` buffers:

```python
hidden = np.empty((tokens.size, e.d_model), dtype=np.float32)
e.run_into(tokens, hidden)

current = np.empty(e.d_model, dtype=np.float32)
e.step_into(7, current)

action = np.empty(e.action_dim, dtype=np.float32)
e.sample_into(tokens, noise, action, steps=10, method="euler")
```

All inference calls release the GIL while C++ runs.

Pass `threads=0..8` to override the automatic worker pool for a specific engine. If omitted,
`FLOWEDGE_THREADS` is honored and then the automatic default is used.
`e.cuda_resident` is runtime residency, not `flowedge.cuda` (compile-time).
CPU wheels always report false. A CUDA source build reports false when attach
failed and the engine is on CPU kernels. `FLOWEDGE_CUDA_REQUIRED=1` fails load
in that case.

```python
single_threaded = flowedge.Engine("models/mamba_flow.safetensors", threads=0)
```

## External encoders and cooperative solving

A head-only checkpoint accepts a condition vector produced by PyTorch, TensorRT, ONNX Runtime, a
shared-memory camera process, or another model server. Output is caller-owned:

```python
condition = encoder(observation).astype(np.float32, copy=False)
noise = np.zeros(e.action_dim, dtype=np.float32)
action = np.empty(e.action_dim, dtype=np.float32)

e.sample_condition(condition, noise, action, 8, "heun")
```

For a generic Transformer checkpoint, the experimental boundary accepts
caller-owned embedding rows. A batch-one prefix mask may be supplied for padded
VLA sequences; it must contain
leading `1` values followed by `0` padding:

```python
transformer = flowedge.Engine("models/tiny-gpt2.flowedge.safetensors")
embeddings = external_vlm(observation).astype(np.float32, copy=False)
mask = np.array([1, 1, 1, 0], dtype=np.uint8)
hidden = transformer.run_embeddings(embeddings, attention_mask=mask)
```

This is an embedding boundary, not a native SmolVLA encoder.

The pinned SmolVLA checkpoint exposes the native action expert when the caller
supplies a captured VLM K/V cache. Its three allocation-free primitives are
suffix embedding, expert execution, and action projection:

```python
smolvla = flowedge.Engine("models/smolvla_base/model.safetensors")
noisy_actions = np.zeros((50, 32), dtype=np.float32)
suffix = smolvla.smolvla_embed_suffix(noisy_actions, timestep=1.0)
assert suffix.shape == (50, 720)

# Cache layout: [16 expert layers, prefix length, 320 VLM K/V values].
# Keys have already received the VLM RoPE transform. The uint8 mask has one
# validity byte per prefix slot; source SmolVLA may have zero-padded language
# tokens before a valid state token.
hidden = smolvla.smolvla_run_expert(suffix, prefix_keys, prefix_values, prefix_mask)
velocity_padded = smolvla.smolvla_project_actions(hidden)

# Prefer this equivalent single call in the control loop.
velocity_padded = smolvla.smolvla_denoise(
    noisy_actions, 1.0, prefix_keys, prefix_values, prefix_mask
)

# Source SmolVLA's deterministic 10-step Euler loop, seeded by the caller.
actions_padded = smolvla.smolvla_sample(
    initial_noise, prefix_keys, prefix_values, prefix_mask, steps=10
)

# Reuse a caller-owned output buffer in a control loop to avoid allocating a
# new result array for each action chunk.
action_buffer = np.empty_like(initial_noise)
smolvla.smolvla_sample_into(
    initial_noise, prefix_keys, prefix_values, prefix_mask, action_buffer, steps=10
)
```

This does not run LeRobot observation preprocessing, image/language encoding,
or build the VLM cache. A real captured-cache *trajectory* replay artifact is
still required before making a native SmolVLA inference or control-quality claim.

The same solve can be split across scheduler quanta without changing its result:

```python
e.flow_begin(condition, noise, 8, "heun", generation=42,
             timestamp_ns=observation_time, deadline_ns=control_deadline)
remaining = e.flow_advance(action, 2)
while remaining:
    remaining = e.flow_advance(action, 1)
```

Only publish `action` as final when `remaining == 0`. One engine owns one resumable solve.
`e.flow_metadata` reports the generation, original timestamp/deadline, status, model digest, and
remaining NFE. A scheduler can call `e.cancel_before(43)` concurrently; the older solve stops at
the next complete solver-step boundary and raises `RuntimeError` from `flow_advance`.

## Streaming state

`step(token)` advances the Mamba recurrence, `reset()` starts a fresh stream, and
`decode_state()` / `restore_decode_state(bytes)` create deterministic branches. Snapshots are
versioned and checksummed; restore rejects a different model digest, architecture, precision,
dimensions, truncated payload, or corruption before changing state.

Source: `python/flowedge_ext.cc`.

## Diffusion Policy

```python
e = flowedge.Engine("models/diffusion_pusht.flowedge.safetensors")
condition = observation_encoder(history).astype(np.float32, copy=False)
noise = np.random.default_rng(7).standard_normal(
    (e.action_horizon, e.action_dim), dtype=np.float32)
actions = e.sample_diffusion(condition, noise, steps=10, scheduler="ddim")
```

`actions` is `[action_horizon, action_dim]` and is already in dataset action
units. `sample_diffusion_into` writes into a caller-owned C-contiguous float32
buffer. Seeded `scheduler="ddpm"` is deterministic; the DDIM result depends only
on the condition and supplied initial noise. `diffusion_denoise` runs one
normalized U-Net pass for verification or profiling.

`diffusion_precision="exact-f32"` is the default. CUDA builds may opt into the calibrated
approximate profile explicitly:

```python
e = flowedge.Engine(
    "models/diffusion_pusht.flowedge.safetensors",
    diffusion_precision="approx-fp16-deep",
)
assert e.cuda_resident and e.diffusion_precision == "approx-fp16-deep"
```

Approximate mode fails closed if the diffusion head is not CUDA-resident. It keeps activations,
accumulation, and output in FP32; only six calibrated deep-convolution weight tensors use FP16.
The exact path and its parity contract are unchanged.

A CUDA source build also accepts any contiguous CUDA float32 tensors implementing DLPack:

```python
import torch

start = e.diffusion_metadata["observation_steps"] - 1
count = e.diffusion_metadata["action_steps"]
condition = encoder(history).flatten().contiguous()  # CUDA tensor
noise = torch.randn(e.action_horizon, e.action_dim, device=condition.device)
actions = torch.empty(count, e.action_dim, device=condition.device)
e.sample_diffusion_dlpack_into(
    condition, noise, actions, steps=10, scheduler="ddim", action_start=start
)
```

`e.diffusion_cuda_device` identifies the required CUDA device. The call synchronizes before
returning and writes only the requested action slice. No NumPy staging or device allocation occurs
inside the call. DDPM still creates its seeded per-step scheduler noise on the host; the default
DDIM path has no host tensor transfer.

The LeRobot companion adapter exposes the same allocation-conscious pattern through
`predict_action_chunk_into` and `select_action_into`; reuse those buffers in a control loop.

`e.diffusion_metadata` exposes `horizon`, `action_steps`, `observation_steps`,
`train_timesteps`, and the clipping policy. LeRobot normally executes
`actions[observation_steps - 1: observation_steps - 1 + action_steps]` from the
returned horizon; the controller owns that slicing decision.
