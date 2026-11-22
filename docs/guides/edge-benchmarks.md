# Reproducible Policy Benchmarks

This guide describes how to run and generate standardized, reproducible benchmark reports comparing FlowEdge against upstream PyTorch and LeRobot baselines.

```{image} ../_static/figures/benches.svg
:alt: Benchmark structure: kernels, runtime tail, policy vs PyTorch, period misses
:class: fe-fig
```

---

## Generating a Benchmark Report

To run a matched Diffusion Policy comparison:

```bash
# CPU comparison (single-threaded baseline)
python -m flowedge_dev bench policy models/policy.safetensors \
  --source models/diffusion_pusht --revision <revision> \
  --steps 10 --iterations 100 --warmup 5 --threads 1 \
  --output tools/benchmark/artifacts/policy/diffusion-pusht-report.json
```

For CUDA evaluation:

```bash
# CUDA comparison (device-resident GPU execution)
python -m flowedge_dev bench policy models/policy.safetensors \
  --source models/diffusion_pusht --revision <revision> \
  --steps 10 --iterations 100 --warmup 5 --device cuda \
  --output tools/benchmark/artifacts/policy/diffusion-pusht-cuda-report.json
```

### Report Contents

The benchmark runner generates matched JSON and Markdown reports containing:
- **Latencies**: Full action-chunk generation times (`p50`, `p95`, `p99`).
- **Parity Verification**: Maximum absolute error and relative error against PyTorch outputs.
- **Hardware & Environment**: CPU model, GPU device, driver versions, compiler, and thread count.
- **Provenance**: SHA256 hashes of input models, configs, processors, and observation vectors.

---

## Reproducibility Standards

To ensure comparability across runs and systems:
- **Identical Inputs**: All runs must use the same observation sequences (`.observations.npz`), initial noise tensors, and scheduler step counts.
- **Warmup Iterations**: Always include sufficient warmup iterations (e.g. 5) to allow CPU caches or GPU contexts to reach steady state.
- **Quantile Validity**: Reports enforce strict monotonicity: `p50 <= p95 <= p99`.
- **Accurate Reporting**: Missing or uninstrumented metrics must be explicitly noted as `null` with a stated reason rather than reported as zero.

To validate an existing JSON artifact against the schema:

```bash
python -m flowedge_dev bench artifact \
  tools/benchmark/artifacts/policy/diffusion-pusht-report.json \
  --report tools/benchmark/artifacts/policy/diffusion-pusht-report.md
```

For published benchmark figures and comparisons, see [Performance Documentation](../performance).
