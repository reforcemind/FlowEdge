# Python Quickstart

FlowEdge provides Python bindings built with pybind11, giving direct access to C++ inference with zero dynamic allocations on hot paths.

## Installation

### Prebuilt Wheels (CPU)

Prebuilt CPU wheels for CPython 3.10–3.12 are available on the [FlowEdge GitHub Releases](https://github.com/reforcemind/FlowEdge/releases).

```bash
# Download the wheel matching your platform, then install:
python -m pip install flowedge-0.1.2-*.whl
```

| Platform | Wheel Tag |
|---|---|
| Linux x86_64 | `manylinux_2_28_x86_64` |
| Linux aarch64 | `manylinux_2_28_aarch64` |
| Windows | `win_amd64` |
| macOS ARM | `macosx_arm64` |

### Building from Source

To build from source with your local C++23 toolchain:

```bash
git clone https://github.com/reforcemind/FlowEdge.git
cd FlowEdge
python -m pip install .
```

For CUDA acceleration, see the [CUDA Architecture Guide](architecture/cuda).

---

## Basic Inference

Download a reference Mamba flow matching checkpoint and sample actions:

```bash
mkdir -p models
curl -L "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors" \
  -o models/mamba_flow.safetensors
```

```python
import numpy as np
import flowedge

# Load checkpoint into fixed arena
engine = flowedge.Engine("models/mamba_flow.safetensors")

print(f"Action dim: {engine.action_dim}")
print(f"Condition dim: {engine.condition_dim}")
print(f"Model dim: {engine.d_model}")

# Generate action chunk via flow matching
prefix = np.array([1, 2, 3, 4], dtype=np.int32)
noise = np.random.randn(engine.action_dim).astype(np.float32)

action = engine.sample(prefix=prefix, noise=noise, steps=10, method="euler")
print("Action chunk:", action)
```

Supported integration solvers for flow matching: `"euler"`, `"heun"`, and `"rk4"`.

---

## Zero-Allocation Hot Paths

Standard calls like `engine.sample()` allocate a new NumPy array on each invocation. In high-frequency control loops, use the zero-copy buffer methods (`run_into`, `step_into`, `sample_into`) to write directly into pre-allocated memory:

```python
import numpy as np
import flowedge

engine = flowedge.Engine("models/mamba_flow.safetensors")

tokens = np.array([1, 2, 3, 4], dtype=np.int32)
noise = np.random.randn(engine.action_dim).astype(np.float32)

# Pre-allocate output buffer once during setup
action_buf = np.empty(engine.action_dim, dtype=np.float32)

# Control tick: runs completely allocation-free and releases the Python GIL
engine.sample_into(tokens, noise, action_buf, steps=10, method="euler")
```

Buffers must be C-contiguous and have matching shapes and data types (`int32` or `float32`).

---

## CUDA Support

CUDA acceleration requires building from source with `nvcc`:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BACKEND=cuda -DFLOWEDGE_PYTHON=ON
cmake --build build-cuda -j
```

Verify CUDA availability and device residency:

```python
import flowedge

print("CUDA compiled in:", flowedge.cuda)

engine = flowedge.Engine("models/diffusion_pusht.flowedge.safetensors")
print("Weights device-resident:", engine.cuda_resident)
```

- `flowedge.cuda`: Compile-time flag indicating whether the binary contains CUDA kernels.
- `engine.cuda_resident`: Runtime flag indicating whether weights were successfully uploaded to GPU memory. If GPU memory allocation fails, FlowEdge gracefully retains CPU execution unless `FLOWEDGE_CUDA_REQUIRED=1` is set to fail closed.

For detailed architecture and configuration options, see the [CUDA Architecture Guide](architecture/cuda).

---

## Next Steps

- [Python API Reference](api/python) — Complete class and method definitions.
- [LeRobot Guide](guides/lerobot) — End-to-end deployment with LeRobot policies.
- [Performance Benchmarks](performance) — Latency comparisons and hardware results.
