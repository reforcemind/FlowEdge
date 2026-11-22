# CUDA Backend Architecture

Configuring FlowEdge with `-DFLOWEDGE_BACKEND=cuda` compiles the native CUDA kernel implementations in `src/core/kernels/cuda/` using `nvcc`.

---

## Build Requirements

- **Toolchain**: CMake 3.21+ and CUDA Toolkit 12.0+ with `nvcc`.
- **Host Compiler**: An MSVC-compatible compiler is required on Windows; the MinGW/Clang toolchain used for CPU builds cannot host `nvcc`. Use MSVC or WSL2.
- **Python Bindings**: To compile Python bindings with CUDA support, pass `-DFLOWEDGE_PYTHON=ON`.

```bash
# Configure and build CUDA backend
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BACKEND=cuda -DFLOWEDGE_PYTHON=ON
cmake --build build-cuda -j
```

---

## Device Residency & Fallback Semantics

FlowEdge maintains a device-resident memory model to eliminate host-device transfer overhead during iterative solving:

1. **One-Time Upload at Load**: Weight tensors and intermediate scratch spaces for Mamba, flow-matching, and Diffusion Policy heads are allocated and uploaded to GPU VRAM once during engine initialization.
2. **Zero Dynamic Allocation**: Sampling steps perform zero calls to `cudaMalloc` or `cudaFree`.
3. **Residency Verification**:
   - `fe_engine_cuda_resident(e)` / `Engine.cuda_resident`: Returns `1` if all weights and scratch buffers are confirmed device-resident on GPU memory.
   - If GPU VRAM is exhausted during initialization, FlowEdge defaults to CPU kernels rather than crashing.
   - Set environment variable `FLOWEDGE_CUDA_REQUIRED=1` to enforce strict failure if device residency cannot be established.
   - Set `FLOWEDGE_CUDA_FORCE_HOST=1` to intentionally bypass GPU memory upload.

---

## Device-Resident Execution Paths

### 1. Diffusion Policy Head
- **Weight Residency**: Convolutional filters, normalization parameters, and FiLM linear weights live in device memory.
- **Iterative Denoising**: All DDIM (or seeded DDPM) steps execute entirely within GPU scratch buffers.
- **DLPack Tensor Interface**: `sample_diffusion_dlpack_into` accepts framework-owned CUDA tensors via DLPack. Observation embeddings and initial noise stay on device, eliminating host staging. Only the final predicted action chunk needed by the hardware driver is transferred back to host memory.

### 2. Flow Matching Head
- Projected condition vectors and ODE solver state (Euler, Heun, RK4) remain on the GPU across solver steps.

### 3. Mamba Backbone
- State-space parameters and layer scratch are uploaded at initialization; step decode advances internal state directly in device memory.

---

## Kernel Specializations & Optimizations

- **Short-Horizon Coalescing**: Launch grids match trajectory horizon length ($L$), ensuring coalesced writes across time steps.
- **Closed-Form Transpose**: Inverse mapping for `conv_transpose1d` avoids iterative input scans.
- **Warp Split-K Reductions**: For narrow horizons ($L=4$ and $L=8$), the input channel reduction (`IC`) is distributed across warp threads, saturating GPU compute resources.
- **Block-per-Group GroupNorm**: Thread blocks map directly to channel groups, minimizing barrier synchronizations.
- **Launch Fusion**: Timestep sinusoidal projection and Mish activations are fused into single kernel launches, eliminating launch latency.
- **Calibrated Approximate FP16 (`approx-fp16-deep`)**: An optional mode stores deep convolution weights in FP16 while preserving FP32 accumulation and output, reducing memory bandwidth pressure.

---

## Jetson Orin Target Configuration

FlowEdge includes a dedicated CMake preset for NVIDIA Jetson Orin (`sm_87`):

```bash
# Build native preset on Jetson Orin
cmake --preset jetson-orin
cmake --build --preset jetson-orin
```

To install directly into a Python environment on Orin:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install . \
  -Ccmake.define.FLOWEDGE_BACKEND=cuda \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=87
```

Ensure the environment includes a PyTorch build compatible with JetPack.

For detailed performance measurements on CUDA, see [Performance Documentation](../performance).
