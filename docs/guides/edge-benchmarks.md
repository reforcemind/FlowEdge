# Reproducible policy reports

Matched **Diffusion Policy** p50 vs LeRobot/PyTorch is this recipe. Flow matching
ULP and graph compilers live on [Performance](../performance.md) — do not mix
those rows.

```{image} ../_static/figures/benches.svg
:alt: kernels, Relay tail, policy vs PyTorch, period misses
:class: fe-fig
```

## Run one report

```bash
python -m flowedge_dev bench policy models/policy.safetensors \
  --source models/diffusion_pusht --revision <revision> \
  --steps 10 --iterations 100 --warmup 5 --threads 1 \
  --output tools/benchmark/artifacts/policy/diffusion-pusht-report.json
```

The published CPU replay used `--threads 1`. CUDA matched replay uses
`--device cuda` and records the CUDA device name; `threads=1` is not the GPU
headline. Do not mix thread counts or devices across backends.

| Output | Includes |
|---|---|
| JSON | model/processor hashes, contract, host/build, raw samples, parity, limitations |
| Markdown | p50/p95/p99 table, FlowEdge/LeRobot ratios, exact commands |
| `.observations.npz` | fixed observation history used by the matched replay |

The reference is not a toy implementation: it is LeRobot running through
PyTorch. End-to-end includes preprocessing, observation history, encoder, and
iterative sampling; it excludes camera capture, IPC, and robot delivery.

## Contract

| Field | Rule |
|---|---|
| Quantiles | `p50 <= p95 <= p99` |
| Backend status | `measured`, `not_measured`, `unavailable`, or `blocked` |
| Missing values | Keep them null and explain the reason; never write zero |
| Reproducibility | Pin revision, hashes, host, compiler, threads, steps, warmup, and iterations |

Validate an existing artifact with:

```bash
python -m flowedge_dev bench artifact \
  tools/benchmark/artifacts/policy/diffusion-pusht-report.json \
  --report tools/benchmark/artifacts/policy/diffusion-pusht-report.md
```

Small runs are smoke checks, not stable p99 evidence. Repeat on the target
machine and publish the JSON, report, and raw samples together.
