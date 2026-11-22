# Verification Contract

FlowEdge maintains an extensive verification suite to validate numerical parity, memory bounds, and latency across backends.

Because full checkpoint parity and end-to-end policy replay require model weights and platform-specific dependencies, the comprehensive gate runs locally on target hardware, while GitHub Actions validates compilation, formatting, and unit tests (`ctest`).

```{image} ../_static/figures/verify.svg
:alt: Numerical parity workflow: checkpoint verification through FlowEdge and PyTorch
:class: fe-fig
```

---

## Release Verification Gates

| Gate | Verification Target |
|---|---|
| **Kernel Unit Tests** | Validates that C++ kernels match independent mathematical references. |
| **PyTorch Parity** | Verifies that inference outputs match PyTorch within strict ULP and relative error thresholds. |
| **Relay IPC Lifecycle** | Exercises daemon startup, shared-memory queues, client requests, deadline enforcement, and clean shutdown. |
| **Generic Job Service** | Validates worker routing, typed result delivery, backpressure handling, and graceful draining. |
| **Worker Pool Operations** | Validates multi-lane execution, QoS slot reservations, rolling drain, and quarantine recovery. |
| **Action Delivery Gate** | Verifies action chunk replacement, timestamp freshness, clamping, and per-step rate-of-change bounds. |
| **Mamba Adapter Tests** | Verifies state snapshot migration and streaming token decode parity. |
| **Job Observability** | Checks bounded circular event buffers, metric kind counters, and trace format serializations. |
| **Trace Portability** | Verifies canonical little-endian byte layouts across platforms for action and job records. |
| **Consumer Integration** | Validates that installed CMake targets (`FlowEdge::Core`, `FlowEdge::Relay`) configure and link cleanly. |

---

## Running the Verification Suite

### Complete Test Gate

```bash
# Linux / macOS / Git Bash
./tools/scripts/verify_all.sh models/mamba_flow.safetensors
```

```powershell
# Windows PowerShell
powershell -File tools/scripts/verify_local.ps1 -BuildDir build-win-clang
```

The verification script builds Core, Relay, unit tests, microbenchmarks, and examples, and executes the complete parity suite when model files are present.

### Focused Verification Commands

| Scope | Command |
|---|---|
| **C++ Unit Tests** | `ctest --test-dir build --output-on-failure` |
| **PyTorch ULP Parity** | `python -m flowedge_dev verify ulp models/mamba_flow.safetensors` |
| **Diffusion Policy Parity** | `python -m flowedge_dev verify diffusion build` |
| **Transformer Decoder Fixture** | `python -m flowedge_dev verify transformer models/tiny-gpt2 models/tiny-gpt2.flowedge.safetensors` |
| **Full Policy Replay Benchmark** | `python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors --source models/diffusion_pusht --threads 1` |
| **Simulated Period Rollout** | `python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors --steps 20 --period-ms 10 --on-miss hold` |
| **Relay Lifecycle Demo** | `FLOWEDGE_BUILD_DIR=build ./tools/scripts/relay_demo.sh models/mamba_flow.safetensors` |
| **Code Formatting & Linting** | `FLOWEDGE_BUILD_DIR=build ./tools/scripts/lint.sh` |

---

## Architectural Invariants

Every release must uphold the following core invariants:

| Invariant | Verification Mechanism |
|---|---|
| **Zero Dynamic Allocations** | Verified via heap profiling on hot paths (`sample`, `run_into`, `sample_into`). |
| **Stale Result Elimination** | Results from older generations or cancelled requests cannot publish as current actions. |
| **Backpressure Safety** | SPSC rings preserve typed error codes on ring exhaustion without losing in-flight state. |
| **State Snapshot Integrity** | Checksummed state capsules restore bit-identical execution without prefix recomputation. |
| **Non-Stranding Worker Drain** | Worker shutdown drains queued jobs before termination without dropping accepted requests. |
| **QoS Priority Preservation** | Lower-priority traffic cannot consume pre-allocated high-priority deadline slots. |
| **Deterministic Error Codes** | Failed lanes enter quarantine and require explicit recovery before rejoining the pool. |

---

## Downstream Consumer Verification

To ensure installed libraries work cleanly in external projects:

```bash
# Install to temporary prefix
cmake --install build --prefix build/install-check

# Configure and build test consumer
cmake -S test/install_consumer -B build/install-consumer \
  -DCMAKE_PREFIX_PATH="$PWD/build/install-check"
cmake --build build/install-consumer -j
./build/install-consumer/flowedge_install_consumer
```

---

## CUDA Kernel Optimization Workflow

When developing or optimizing CUDA kernels:

1. **Profile operator and end-to-end latency**: Run `flowedge_cuda_dp_mix` to measure isolated kernel execution and native 10-step DDIM latency.
2. **Validate numerical parity**: Verify that maximum absolute error remains within the parity gate ($\le 1\times 10^{-3}$) against PyTorch CUDA references:
   ```bash
   python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors \
     --source models/diffusion_pusht --device cuda --build-dir build-cuda \
     --observations tools/benchmark/artifacts/policy/diffusion-pusht-cpu-replay.observations.npz
   ```
3. **Verify memory residency**: Ensure weights and scratch buffers are confirmed device-resident (`cuda_resident == 1`). If memory is insufficient on the device, ensure the engine fails closed when `FLOWEDGE_CUDA_REQUIRED=1` is specified.
