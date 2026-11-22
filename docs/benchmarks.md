# Benchmark Suite

FlowEdge organizes its verification and performance benchmarks into distinct tiers, from isolated kernel microbenchmarks to full-policy rollouts.

```{image} _static/figures/benches.svg
:alt: Benchmark suite overview: kernels, runtime tail, policy vs PyTorch, period misses
:class: fe-fig
```

## Benchmark Directory

| Suite | Entry Point / Location | Primary Metric |
|---|---|---|
| **Kernel Microbenchmarks** | `tools/benchmark/kernels/` | Operator latency (Conv1D, GEMM, scan, GroupNorm) |
| **Relay IPC Runtime** | `tools/benchmark/runtime/` | IPC roundtrip latency, throughput, EDF scheduling |
| **Flow Matching Parity** | `python -m flowedge_dev verify ulp` | Numerical error & ULP difference vs PyTorch reference |
| **Diffusion Policy Replay** | `python -m flowedge_dev bench policy` | End-to-end policy latency (p50/p95/p99) vs LeRobot |
| **Period Rollout Simulator** | `python -m flowedge_dev pipeline rollout` | Deadline miss counts and timing under target period |
| **SmolVLA Expert** | `python -m flowedge_dev verify smolvla` | Action expert numerical parity with cached VLM prefix |
| **Transformer Decoder** | `python -m flowedge_dev verify transformer` | GPT-2 attention and decoder block verification |
| **Closed-loop Simulation** | `flowedge_lerobot.evaluate` | Task completion rate and cumulative reward (PushT) |

---

## Benchmark Guidelines & Standards

To ensure benchmarks remain reproducible and meaningful, claims in FlowEdge follow strict standards:

| Benchmark Claim | Verification Standard |
|---|---|
| **Framework Speedup** | Must compare against identical checkpoints, preprocessing, batch size, thread counts, and scheduler steps on the same machine. |
| **Real-time Deadline Feasibility** | Must evaluate the full control interval, report missed deadline counts, and specify the fallback strategy (`hold`, `drop`, or `raise`). |
| **Zero Dynamic Allocations** | Verified via heap profiling or memory arena tracking on warm inference paths. |
| **Hardware-Specific Results** | Must explicitly document CPU architecture, GPU model, driver versions, and power/thermal constraints. |

For detailed benchmark figures and hardware configurations, see [Performance](performance). To generate standardized benchmark artifacts, see the [Edge Benchmarks Guide](guides/edge-benchmarks).
