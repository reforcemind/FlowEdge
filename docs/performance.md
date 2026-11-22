# Performance

FlowEdge benchmarks measure end-to-end policy replay against PyTorch references under identical conditions: the same observation sequences, preprocessing, initial noise, and 10-step DDIM sampling schedules.

```{image} _static/figures/compare.svg
:alt: Policy p50 comparison: 851 ms vs 1409 ms on CPU (threads=1)
:class: fe-fig
```

```{image} _static/matched_replay.gif
:alt: Matched replay benchmark: FlowEdge CUDA 131 ms vs PyTorch 345 ms on GTX 1650; FlowEdge CPU 851 ms vs PyTorch 1409 ms (threads=1)
:class: fe-fig
```

---

## Diffusion Policy Replay

### CPU Benchmarks

Evaluated on the matched PushT Diffusion Policy checkpoint. All runs use identical observation files and 10 DDIM denoising steps.

| Metric | FlowEdge | LeRobot / PyTorch | Comparison |
|---|---:|---:|---:|
| **Policy Latency (p50)** | **851 ms** | **1409 ms** | **1.66× speedup** (0.60× runtime) |
| Preprocess-to-chunk (p50) | 889 ms | 1448 ms | 1.63× speedup |
| Max absolute action error | 3.05e-5 | — | Bit-accurate parity |
| Hot-path memory allocations | 0 | Dynamic | Zero allocations |

#### Thread Scaling

| Worker Threads | FlowEdge Policy p50 | LeRobot Policy p50 | Relative Ratio | Native DDIM p50 |
|---:|---:|---:|---:|---:|
| 1 | 851 ms | 1409 ms | 0.60× | 810 ms |
| 2 | 800 ms | 1031 ms | 0.78× | 714 ms |
| 4 | 633 ms | 819 ms | 0.77× | 567 ms |
| 6 | — | — | — | 565 ms |

#### CPU Kernel Optimizations

Specialized AVX2/NEON convolution kernels provide substantial speedups over naive loops:

| Kernel Operation | Baseline p50 | Optimized p50 | Speedup | 4 Workers |
|---|---:|---:|---:|---:|
| Conv1D 512×512, K=5, L=16 | 29.3 ms | 9.21 ms | 3.18× | 3.30 ms |
| Conv1D 1024×1024, K=5, L=8 | 57.9 ms | 31.6 ms | 1.83× | 11.0 ms |
| Conv1D 2048×2048, K=5, L=4 | 118 ms | 108 ms | 1.09× | 41.3 ms |
| Strided Conv1D 512×512, K=3, L=16→8 | 9.61 ms | 5.78 ms | 1.66× | — |
| ConvTranspose1D 512×512, K=4, L=8→16 | 10.5 ms | 6.00 ms | 1.75× | — |

---

### CUDA Benchmarks

Evaluated on an NVIDIA GeForce GTX 1650 (`sm_75`) running Ubuntu/WSL2 with CUDA 12:

| Metric | FlowEdge CUDA | PyTorch CUDA | Comparison |
|---|---:|---:|---:|
| **Policy Latency (p50)** | **131 ms** | **345 ms** | **2.63× speedup** (0.38× runtime) |
| Preprocess-to-chunk (p50) | 136 ms | 350 ms | 2.57× speedup |
| Max absolute action error | 7.63e-5 | — | High numerical parity |
| GPU memory allocation on hot path | 0 | Dynamic | Device-resident scratch |

#### CUDA Kernel Optimizations

Step-by-step kernel progression on native 10-step DDIM:

| Optimization Step | Native DDIM Latency (p50) | Key Kernel Changes |
|---|---:|---|
| Initial CUDA baseline | 448 ms | Baseline device port |
| Short-horizon & closed-form transpose | 321 ms | Coalesced time-axis launch; closed-form index arithmetic |
| Warp split-K (L=4, L=8) | 152 ms | Split `IC` reduction across warp lanes for small horizons |
| Block-per-group GroupNorm | 136 ms | One thread block per channel group, minimizing syncthreads |
| Launch fusion & tuned reductions | 104 ms | Fused FiLM GEMM bias, timestep Mish fusion, 1D row launches |

---

## Flow Matching

Flow matching generates actions by integrating a learned velocity field from noise:

| Metric | Measured Value | Requirement / Gate |
|---|---|---|
| Reference Checkpoint | `models/mamba_flow.safetensors` | Verified `.safetensors` format |
| Max Relative Error vs PyTorch | **~1e-6** | ≤ 2e-3 |
| Max ULP difference | **Within tolerance** | ≤ 4096 ULP |
| Solver options | Euler, Heun, RK4 | Bit-identical across split solves |

Parity is verified using `python -m flowedge_dev verify ulp models/mamba_flow.safetensors`.

---

## Relay (Inter-Process Communication)

Relay provides lock-free, zero-copy shared memory IPC for multi-process robotics architectures:

### IPC Latency and Throughput

| Benchmark | Windows | Linux |
|---|---:|---:|
| Synchronous roundtrip (p99) | 47.60 µs | 40.10 µs |
| Worker pool throughput (1 worker) | 37,240 req/s | 55,719 req/s |
| Worker pool throughput (2 workers) | 73,187 req/s | 109,507 req/s |

### Cooperative Job Routing

| Operation | Windows | Linux |
|---|---:|---:|
| Job migration & handoff | 478.56 ns | 292.08 ns |
| Direct route, execute, and return | 274.80 ns | 88.94 ns |
| EDF queue dispatch + metrics | 3.49 µs | 1.19 µs |
| Shared-memory service + pool + metrics | 4.60 µs | 1.78 µs |
| Action delivery gate (accept + publish) | 49.13 ns | 60.75 ns |

---

## Reproducing the Benchmarks

### CPU Replay Benchmark

```bash
# Run matched Diffusion Policy benchmark
python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --threads 1

# Run kernel microbenchmarks
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_BENCH=ON
cmake --build build-bench --target flowedge_kernels_bench -j
./build-bench/flowedge_kernels_bench --benchmark_filter='BM_diffusion_'
```

### CUDA Replay Benchmark

```bash
# Requires FLOWEDGE_BACKEND=cuda
python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --device cuda \
  --observations tools/benchmark/artifacts/policy/diffusion-pusht-cpu-replay.observations.npz \
  --output bench-cuda.json
```

### Relay Benchmarks

```bash
cmake -S . -B build-relay -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_RELAY=ON -DFLOWEDGE_BENCH=ON
cmake --build build-relay -j

# Run synchronous IPC benchmark
./build-relay/flowedge_relay_bench models/mamba_flow.safetensors 5000 0

# Run multi-worker pool benchmark
./build-relay/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 2 0
```
