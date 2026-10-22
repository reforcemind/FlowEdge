# CUDA backend

`FLOWEDGE_BACKEND=cuda` links the ISA kernel surface (`matmul`, activations
including `mish`, causal `conv1d`, `rmsnorm`, device SSM scan) through
`src/core/kernels/cuda/`. CMake fails closed without `nvcc`. On Windows, `nvcc`
needs an MSVC-compatible host compiler; the MinGW/Clang gnu-target tree used
for CPU Release builds cannot host `nvcc`. WSL or MSVC is the CUDA build.
CPU wheels (v0.1.2, not PyPI) have no CUDA kernels.

Device-resident attach is best-effort. If `cudaMalloc` / H2D fails (typical on a
4 GB GTX 1650 under WDDM/WSL for the ~959 MiB PushT U-Net), the engine keeps the
already-loaded **CPU kernels**. `fe_engine_cuda_resident` / `Engine.cuda_resident`
is then 0. That is not `fe_engine_load` failure. `FLOWEDGE_CUDA_REQUIRED=1` fails
load when attach did not happen. `FLOWEDGE_CUDA_FORCE_HOST=1` skips the upload.
`--device cuda` still fails closed unless residency is true.

Published CUDA matched replay on GTX 1650 stays **131 vs 345 ms** for the exact path. An explicit
`approx-fp16-deep` mode is separately gated: six deep convolution weight tensors use FP16 with
FP32 accumulation/output. Five alternating native 10-step trials measured 104.216 ms exact versus
82.620 ms approximate (20.7% lower); matched PushT seeds 7–16 remained 8/10 successful in both
modes, with mean maximum reward 0.998138 exact versus 0.998099 approximate. This is simulation and
GTX 1650 evidence, not SO-101 or Jetson evidence. See the
{download}`raw artifact <../../tools/benchmark/artifacts/policy/cuda-fp16-deep-2026-10-15/README.md>`.

The **Mamba backbone**, **flow head**, and **Diffusion Policy head** are
device-resident. `Mamba` uploads `backbone.*` weights and layer scratch once at
load. `forward` / `decode` copy the token activations (and host decode state)
in, run the mixer on device, and copy the hidden state out. `FlowHead`
uploads `flow.*` weights and ODE scratch once at load. `sample` /
`sampler_advance` copy condition and noise in, run Euler/Heun/RK4 on device, and
copy the action out. `DiffusionHead` uploads `dp.*` weights and U-Net/DDIM
scratch once. `denoise` / `sample` copy condition and the horizon in, run the
Conv1D U-Net and DDIM (seeded DDPM) on device, and copy the action or epsilon
out. `sample_diffusion_dlpack_into` instead accepts framework-owned CUDA tensors,
copies condition/noise device-to-device into persistent scratch, and unnormalizes only the
requested action slice into caller-owned CUDA storage. The LeRobot CUDA plugin uses this path, so
the RGB/state condition, initial noise, and queued action chunk do not stage through NumPy. Its
postprocessor still transfers the small selected action to CPU for the robot driver. Seeded DDPM
retains a host-generated per-step noise copy; DDIM has no host tensor transfer. No `cudaMalloc`
after load. `kernels.h` stays host `std::span` for the
generic op surface. Dense DP ops also remain available as host-span launches
from the same CUDA TU. Short PushT horizons launch with `block.x` matching `L`
so time-axis stores stay coalesced; `conv_transpose1d` recovers `it` from `ot`,
`k`, and stride instead of scanning `input_length`. L=4 and L=8 dense conv
split `IC` across the block so the inner reduction is not serial; L=16 stays
2D. GroupNorm launches one block per group. FiLM GEMM `rows=1` uses a 1D launch and fuses its
bias; timestep input also fuses Mish. This removes 150 launches from a 10-step sample (1,000 to
850). Five alternating exact-FP32 native trials were 105.815 ms before versus 104.257 ms after
(1.47% lower), with bitwise-identical actions. K-major L=8 upsample uses the same split-K pattern.
`flowedge_cuda_dp_mix`
isolates those shapes and, with a checkpoint, prints a launch census for one
native 10-step DDIM.
The {download}`launch-fusion artifact
<../../tools/benchmark/artifacts/policy/cuda-launch-fusion-2026-10-15/README.md>` contains the end-to-end
confirmation; the previously tested full CUDA Graph doubled latency and remains rejected in issue
#181.
See [#174](https://github.com/reforcemind/FlowEdge/issues/174),
[#176](https://github.com/reforcemind/FlowEdge/issues/176),
[#179](https://github.com/reforcemind/FlowEdge/issues/179), and
[#182](https://github.com/reforcemind/FlowEdge/issues/182).

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BACKEND=cuda
cmake --build build-cuda -j
./build-cuda/flow_sample models/mamba_flow.safetensors euler 10
./build-cuda/mamba_forward models/mamba_flow.safetensors 1 2 3 4
./build-cuda/diffusion_sample models/diffusion_pusht.flowedge.safetensors 10
./build-cuda/flowedge_cuda_dp_mix models/diffusion_pusht.flowedge.safetensors
PYTHONPATH=build-cuda python tools/verification/benchmark_cuda_tensor_boundary.py \
  models/diffusion_pusht.flowedge.safetensors
```

Same checkpoint, same noise: compare those binaries to CPU `flow_sample` /
`diffusion_sample`. Matched full-policy CUDA vs PyTorch CUDA on GTX 1650 is
{download}`diffusion-pusht-cuda-replay.md <../../tools/benchmark/artifacts/policy/diffusion-pusht-cuda-replay.md>`:
policy p50 131 ms vs 345 ms, max abs 7.63e-5, same CPU observation file.
The LeRobot CUDA period log is
{download}`diffusion-pusht-cuda-period10-hold.md <../../tools/benchmark/artifacts/policy/diffusion-pusht-cuda-period10-hold.md>`:
10 ms hold missed 20 / 20 (step p50 589 ms). Not TensorRT/ONNX. Not Jetson/ARM.

WSL2, nvcc 12.0, GTX 1650 (`sm_75`): `FlowHead.*`, `DiffusionHead.*`, and
`Mamba.*` including `CudaResidentDoesNotMallocAfterLoad` pass. Same
`mamba_flow` checkpoint, `mamba_forward` tokens `1 2 3 4` prints `out[0]=-1.10703`
on CUDA and CPU; `flow_sample` Euler/Heun/RK4 `action[0..2]` matches a CPU
binary at printed precision. Same `diffusion_pusht` checkpoint and noise,
10-step DDIM `diffusion_sample` matches the CPU horizon at printed precision
(iostream rounding on two components: `226.249` vs `226.25`).
`flowedge_cuda_dp_mix` native 10-step DDIM p50 on this card is 136 ms after
block-per-group GroupNorm (152 ms after L=4 split-K, 321 ms after #174, 448 ms
before). That is not a policy p50.

See [#160](https://github.com/reforcemind/FlowEdge/issues/160). Device-resident
flow is [#161](https://github.com/reforcemind/FlowEdge/issues/161); device-resident
DDIM is [#162](https://github.com/reforcemind/FlowEdge/issues/162); matched GPU
replay [#163](https://github.com/reforcemind/FlowEdge/issues/163); LeRobot CUDA
rollout is `--device cuda` on `flowedge-lerobot-rollout` with the same
`--on-miss` contract ([#164](https://github.com/reforcemind/FlowEdge/issues/164));
device-resident Mamba is [#172](https://github.com/reforcemind/FlowEdge/issues/172);
CUDA DP conv occupancy is [#174](https://github.com/reforcemind/FlowEdge/issues/174);
L=4 split-K conv is [#176](https://github.com/reforcemind/FlowEdge/issues/176);
the native DDIM launch census is [#179](https://github.com/reforcemind/FlowEdge/issues/179);
L=8 split-K and L=4 occupancy is [#182](https://github.com/reforcemind/FlowEdge/issues/182).
The RGB encoder stays in LeRobot. Not Jetson/ARM.
