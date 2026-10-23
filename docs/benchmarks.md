# Benchmark map

```{image} _static/figures/benches.svg
:alt: kernels, Relay tail, policy vs PyTorch, period misses
:class: fe-fig
```

| Suite | Location | Measures |
|---|---|---|
| Kernels | `tools/benchmark/kernels/` | Dense ops, including DP Conv1D shapes |
| Runtime | `tools/benchmark/runtime/` | Relay admission, queues, delivery |
| Flow matching vs PyTorch | `python -m flowedge_dev verify ulp` | ULP / rel-error on `mamba_flow` — not a p50 |
| Diffusion Policy vs PyTorch | `python -m flowedge_dev bench policy` | Same observations, processor, noise, DDIM vs LeRobot |
| Period loop | `python -m flowedge_dev pipeline rollout` | `--period-ms`, `--on-miss`, RSS; `--device cuda` on a CUDA Core build |
| Closed loop | `flowedge_lerobot.evaluate` | PushT outcome — not a timing proof |
| SmolVLA expert | `python -m flowedge_dev verify smolvla` | Cached-VLM expert vs source; VLM stays in PyTorch |
| Transformer fixture | `python -m flowedge_dev verify transformer`, `transformer_latency` | GPT-2 smoke vs Hugging Face; host decoder p50 is not a policy result |
| ONNX companion | `integrations/onnx` | Fixed-shape ORT adapter; measure separately |

The two product heads are flow matching and Diffusion Policy. Without FlowEdge
you keep them in PyTorch/LeRobot or a graph compiler. With FlowEdge you convert
the head, pin RSS, and compare against that same PyTorch reference. Mix ULP
with p50 and the claim is wrong. [Performance](performance).

For the exact report command, artifact contents, and reproducibility contract,
use the [policy benchmark guide](guides/edge-benchmarks). For measured results
and their hardware conditions, see [Performance](performance).

## Interpretation

| Claim | Allowed when |
|---|---|
| PyTorch speedup | Same checkpoint, processor, host, threads, schedule |
| Real-time | Full period, delivery path, miss counts |
| Zero hot allocations | Native path instrumentation |
| Hardware result | Artifact names the board |

**CPU available · CUDA matched PushT replay on GTX 1650 · Tenstorrent planned**. CUDA policy p50 is not the README CPU headline. Not TensorRT/ONNX. Not Jetson/ARM ([#70](https://github.com/reforcemind/FlowEdge/issues/70)).
