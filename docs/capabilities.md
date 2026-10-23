# Capabilities

**Core** is the in-process engine: convert a policy, pin RSS, sample an action. **Relay** is an optional sidecar for deadline-aware IPC. Neither replaces LeRobot training, cameras, or e-stop.

```{image} _static/figures/purpose.svg
:alt: Hardware runs the loop, FlowEdge runs the policy, LeRobot trains and drives the robot
:class: fe-fig
```

Use Core when the policy and the controller share a process. Use Relay when a sensor or VLA process must hand a condition across a ring and still meet a deadline. The default Diffusion Policy replay loads `flowedge.Engine` in-process and does not start Relay.

```{image} _static/figures/stack.svg
:alt: Layers from plugin down to the arena
:class: fe-fig
```

## Choose a path

| Goal | Start here |
|---|---|
| Flow matching | `flow_sample` |
| LeRobot Diffusion Policy, period, `--on-miss` | `python -m flowedge_dev pipeline rollout` |
| Keep an existing encoder | `external_flow_sample` |
| Streaming Mamba | `mamba_forward` |
| Snapshot / restore | `streaming_snapshot` |
| Cross-process actions | `tools/scripts/relay_demo.sh` |
| Faster control loop | `action_delivery_sample` |
| vs PyTorch | `python -m flowedge_dev bench policy` |

## Available now

| Area | Capabilities |
|---|---|
| Backends | CPU scalar / AVX2 / NEON. `FLOWEDGE_BACKEND=cuda` (nvcc): Mamba, flow, and DP heads device-resident; CUDA OOM at load keeps CPU kernels. Tenstorrent (TT-Metal) is a research track, not a download |
| Backbones | Mamba. Transformer CPU decoder fixture (not a policy) |
| Heads | flow matching (Euler / Heun / RK4); Diffusion Policy (DDIM / DDPM); experimental SmolVLA action expert |
| Models | `mamba_flow`; `diffusion_pusht`. π0 / DiT / native VLM no |
| Precisions | FP32 / BF16; experimental approximate FP16 for selected CUDA Diffusion Policy layers. INT8 no |
| Weights | `.safetensors`; shared immutable worker weights |
| State | Versioned snapshots; exact restore |
| Relay | Shared memory; EDF; 1–8 workers; action delivery |
| Jobs | Iterative, streaming, speculative; drain / quarantine |
| APIs | C, C++, Python |
| LeRobot | `--on-miss hold\|drop\|raise`; `--device cuda` Core rollout; `--policy.type=flowedge` / `flowedge_smolvla` |
| Evaluation | Matched CPU and CUDA replay; period misses; PushT runner |

## Contracts

| Contract | Expectation |
|---|---|
| Allocation | Nothing after setup on supported hot paths |
| CUDA attach | Device-resident upload is best-effort; CPU kernels remain unless `FLOWEDGE_CUDA_REQUIRED=1` |
| Mutable state | One engine per concurrent lane |
| Action delivery | Binding, overlap, bounds, per-step delta |
| Deadlines | Predictive admission, not an OS hard guarantee |
| Transport | Shared-memory rings are SPSC |

## Not yet

π0 / DiT · native VLM · ROS 2. The Transformer decoder fixture is not a robotics policy. FlowEdge is not a graph runtime, trainer, or safety controller.
