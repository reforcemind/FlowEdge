# Architecture Overview

FlowEdge is designed around worst-case latency and deterministic execution for robotics control loops, prioritizing bounded memory and zero runtime allocation over generic graph flexibility.

```{image} ../_static/figures/big-flow.svg
:alt: End-to-end FlowEdge architecture from checkpoint to motor controller
:class: fe-fig
```

---

## Design Philosophy

General graph compilers often introduce dynamic memory management, kernel dispatch overhead, and unpredictable working sets. For physical robots operating on strict control deadlines (e.g. 10–100 Hz), jitter in inference time directly translates to control instability.

FlowEdge achieves predictable, low-latency execution through a hand-assembled execution pipeline:
- **Fixed Memory Arena**: All weight tensors and intermediate execution scratch are pre-sized and allocated once at initialization.
- **Dedicated Policy Heads**: Optimized kernel paths for flow matching and Diffusion Policy U-Nets.
- **Decoupled Observation Pipeline**: Vision backbones and image encoders remain in high-level frameworks (such as PyTorch or LeRobot), passing compact condition vectors into FlowEdge.

```{image} ../_static/figures/stack.svg
:alt: FlowEdge layer architecture from application down to memory arena
:class: fe-fig
```

---

## Architectural Layers

1. **Memory Arena (`src/core/arena/`)**:
   A single contiguous byte slab managed with an aligned bump allocator. Memory is pre-faulted at load time to avoid page faults during control ticks. Layers reuse scratch memory via save-and-restore marks (`mark()` / `reset_to()`).

2. **Zero-Copy Loader (`src/core/loader/`)**:
   Reads `.safetensors` files directly via memory mapping (`mmap`). Validates tensor shapes and types without external library dependencies, aligning weights to 64-byte boundaries.

3. **Kernel Layer (`src/core/kernels/`)**:
   A single unified interface (`kernels.h`) backed by architecture-specific implementations:
   - **CPU**: SIMD vectorized kernels with AVX2 and ARM NEON intrinsics, alongside portable scalar implementations.
   - **CUDA**: Device-resident kernels with warp-level split-K reductions and fused operator launches.

4. **Representation Backbones (`src/core/models/`)**:
   Computes conditioning vectors from inputs. FlowEdge natively supports Mamba (selective state-space model with $O(1)$ memory footprint), with constant-time streaming step decoding.

5. **Policy Heads (`src/core/heads/`)**:
   Takes condition vectors and noise to generate action trajectories:
   - **Flow Matching**: Solves straight-path ODEs via Euler, Heun, or Runge-Kutta 4.
   - **Diffusion Policy**: Runs reverse denoising schedules via DDIM or seeded DDPM over a 1D convolutional U-Net.

6. **Relay IPC (Optional, `src/relay/`)**:
   Lock-free shared-memory rings supporting inter-process communication between vision pipelines and motor control loops with Earliest Deadline First (EDF) request scheduling.

---

## Cooperative Execution

In robotics, an inference task must be able to yield to higher-priority control deadlines or safety events.

FlowEdge provides cooperative scheduling primitives (`flow_begin` and `flow_advance`) that allow solvers to be broken down across multiple integration steps. Solver stages run in bounded increments without altering numerical results, maintaining bit-identical output with monolithic solves.

For more details, see [Cooperative Execution](cooperative-execution).
