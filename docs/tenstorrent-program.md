# Tenstorrent Research Track

This document outlines the exploration track for deploying robotics policy heads on Tenstorrent hardware using TT-Metal and TTNN.

```{note}
Tenstorrent acceleration is currently an active research track. Runtime execution on TT-Metal is experimental and does not imply vendor endorsement.
```

## Objectives & Roadmap

1. **Diffusion Policy & SmolVLA Reference Parity**:
   Validate reference trajectories and numerical parity using captured encoder embeddings and action expert targets.
2. **Fixed-Shape Device Graph**:
   Target a fixed-shape action-head path on Tenstorrent hardware with persistent device-resident weights and ODE / denoising state buffers.
3. **Execution Profiling**:
   Measure initialization, host-to-device transfers, kernel execution latency, and synchronization overhead against CPU and CUDA baselines.
4. **TTNN Optimization**:
   Evaluate buffer reuse, native TTNN tensor layouts, and trace capture for repetitive denoising iterations.
5. **Deadline Scheduling**:
   Integrate deadline-aware scheduling with bounded action delivery and underrun prevention.

---

## Architecture & Boundary

- **Core Runtime**: FlowEdge Core provides the native execution interface, memory management, and deterministic solver steps.
- **Hardware Integration**: Accelerator backends implement persistent-device memory buffers and avoid host round-trips during iterative denoising.
- **Ecosystem Adapters**: Upstream vision encoders and multi-modal models run on host or companion accelerators, passing fixed condition tensors to the Tenstorrent device.

Collaborations and discussions follow Tenstorrent's public developer and academic initiatives. For the general project roadmap, see [Roadmap](roadmap).
