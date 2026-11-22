# System Capabilities

FlowEdge is structured around two deployment modes: **Core** (in-process engine for direct, low-overhead inference) and **Relay** (optional lock-free shared-memory sidecar for multi-process robotics pipelines).

```{image} _static/figures/stack.svg
:alt: Layers from plugin down to the arena
:class: fe-fig
```

## Workflows and Entry Points

| Objective | Entry Point / Example |
|---|---|
| **Flow Matching Inference** | `./build/flow_sample` / `flowedge.Engine.sample()` |
| **Diffusion Policy Rollout** | `python -m flowedge_dev pipeline rollout` |
| **External Vision / Condition Encoder** | `engine.sample_condition()` / `external_flow_sample` |
| **Streaming State Space Model (Mamba)** | `engine.step()` / `./build/mamba_forward` |
| **State Snapshot & Exact Restore** | `engine.export_decode_state()` / `streaming_snapshot` |
| **Multi-Process Shared Memory IPC** | `./tools/scripts/relay_demo.sh` |
| **Action Chunk Delivery & Clamping** | `fe::relay::ActionDeliveryGate` / `action_delivery_sample` |
| **Benchmarking vs PyTorch** | `python -m flowedge_dev bench policy` |

---

## Feature Matrix

| Domain | Supported Capabilities |
|---|---|
| **Hardware Backends** | CPU (AVX2, NEON, and portable scalar). CUDA (`FLOWEDGE_BACKEND=cuda` via `nvcc`) with device-resident Mamba and diffusion heads. Tenstorrent (TT-Metal) is an active research track. |
| **Backbones** | Mamba (selective state-space model with O(1) state). Transformer decoder fixture for testing. |
| **Policy Heads** | Flow matching (Euler, Heun, RK4). Diffusion Policy (DDIM deterministic, seeded DDPM). SmolVLA action expert adapter. |
| **Model Checkpoints** | `mamba_flow.safetensors`, `diffusion_pusht.flowedge.safetensors`, custom converted LeRobot checkpoints. |
| **Precision** | FP32 and BF16 linear weights. Calibrated approximate FP16 deep convolution layers on CUDA. |
| **Weight Storage** | Zero-copy `.safetensors` loader via memory mapping. Shared immutable weights across parallel worker engines. |
| **State Management** | Versioned, checksummed state snapshots with bit-identical reconstruction. |
| **Relay IPC** | Single-producer single-consumer (SPSC) shared-memory rings. Earliest Deadline First (EDF) scheduler. 1–8 worker threads. |
| **API Bindings** | Pure C ABI, modern C++23 API, and Python bindings (pybind11). |
| **LeRobot Integration** | First-class policy adapter (`flowedge`, `flowedge_smolvla`), configurable deadline handling (`hold`, `drop`, `raise`). |

---

## Runtime Guarantees & Contracts

| Contract | Implementation |
|---|---|
| **Zero Hot-Path Allocations** | The memory arena is sized and pre-faulted at load time. No heap allocations occur on supported inference paths. |
| **Deterministic State** | Each engine instance owns its mutable state. Concurrent inference lanes use dedicated engine instances over shared immutable weights. |
| **Explicit Deadline Handling** | Configurable action policies determine behavior when control loop periods are overrun. |
| **Safe Action Delivery** | Action delivery gates enforce timestamp freshness, rate-of-change limits, and workspace bounds before publishing to controllers. |

---

## Scope & Non-Goals

FlowEdge focuses exclusively on deterministic, low-latency execution of trained robot policy action heads. Upstream robotics stacks retain ownership of:
- Perception pipelines, camera drivers, and raw vision encoders.
- Model training, dataset generation, and loss optimization.
- Hardware joint limits, motor controllers, and physical emergency stops.
