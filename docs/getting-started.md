# Getting Started

FlowEdge is an inference runtime for robotics policies on CPU and CUDA. It executes the latency-sensitive policy head with fixed memory and predictable timing, while upstream frameworks (such as LeRobot) handle training, cameras, and hardware control.

```{image} _static/figures/purpose.svg
:alt: Hardware runs the loop, FlowEdge runs the policy, LeRobot trains and drives the robot
:class: fe-fig
```

## Quick Navigation

Choose the entry point that matches your setup:

| Objective | Recommended Guide |
|---|---|
| **Python Inference** | [Python Quickstart](python-quickstart) — Install wheels and run inference from Python. |
| **LeRobot Integration** | [LeRobot Guide](guides/lerobot) — Package and deploy Diffusion Policy or SmolVLA models. |
| **Checkpoint Conversion** | [Converter Guide](guides/converter) — Convert PyTorch weights into FlowEdge `.safetensors`. |
| **Custom Encoders** | [Python API](api/python) & [C-ABI](api/c-abi) — Feed external condition vectors directly into FlowEdge heads. |
| **Inter-Process Control** | [Relay Quickstart](guides/relay-quickstart) — Split vision and controller across processes via lock-free shared memory. |
| **Performance Benchmarks** | [Policy Evaluation](guides/policy-evaluation) & [Edge Benchmarks](guides/edge-benchmarks) — Measure latency against PyTorch baselines. |

---

## Prerequisites

- **Python**: 3.10–3.12 (for prebuilt CPU wheels)
- **C++ Toolchain** (when building from source): CMake 3.21+ and a C++23 compiler (Clang 16+, GCC 13+, or MSVC 19.38+)
- **CUDA** (optional): CUDA Toolkit 12.0+ with `nvcc` for GPU-resident execution. See the [CUDA Architecture Guide](architecture/cuda).

---

## 3-Minute Quickstart

1. **Install FlowEdge**:
   ```bash
   # Download the wheel for your platform from GitHub Releases, then:
   python -m pip install flowedge-0.1.2-*.whl
   ```

2. **Download Model Checkpoint**:
   ```bash
   mkdir -p models
   curl -L "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors" \
     -o models/mamba_flow.safetensors
   ```

3. **Run Inference**:
   ```python
   import numpy as np
   import flowedge

   engine = flowedge.Engine("models/mamba_flow.safetensors")
   prefix = np.array([1, 2, 3, 4], dtype=np.int32)
   noise = np.random.randn(engine.action_dim).astype(np.float32)

   action = engine.sample(prefix=prefix, noise=noise, steps=10, method="euler")
   print("Sampled action shape:", action.shape)
   ```
