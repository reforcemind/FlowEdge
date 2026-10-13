# Maintainer tools

![do not mix the evidence](../docs/_static/figures/benches.svg)

Use one module from the checkout root:

```bash
python -m flowedge_dev --help
```

| Need | Command |
|---|---|
| Matched DP vs PyTorch | `python -m flowedge_dev bench policy` |
| Kernel mix → `data/` | `python -m flowedge_dev bench mix` |
| Size/setup budgets | `python -m flowedge_dev bench budgets` |
| ULP / diffusion / external head | `python -m flowedge_dev verify ulp\|diffusion\|head` |
| Convert / inspect / period rollout | `python -m flowedge_dev pipeline convert\|inspect\|rollout` |
| Cached SmolVLA hybrid | `python -m flowedge_dev verify smolvla` |
| GPT-2 incubator | `python -m flowedge_dev verify transformer` |
| Checkpoint JSON | `flowedge-inspect` (`tools/checkpoint_inspect.cc`) |

Scripts under `tools/` remain the implementations. The local gate is `scripts/verify_all.sh` / `scripts/verify_local.ps1`.

For a compiled PyTorch U-Net reference, add `--reference-compile default` to
`bench policy` (at least one warmup is required). The encoder and scheduler stay
eager; the report records the mode. Compilation errors are not silently retried
with eager execution.

For alternating-order native A/B measurements, use
`python tools/benchmark/compare_native_diffusion.py CHECKPOINT --baseline BASE_BUILD --candidate NEW_BUILD --steps 1 2 4 10 --output bench-results/native-ab.json`.
This records raw timings and checks action parity in separate processes. It
measures synthetic-condition sampling, not visual policy or closed-loop quality.

The A/B report measures relative performance; it does not impose a universal
latency target. Deployment budgets belong to the declared checkpoint, host,
control loop and workload. See [product direction](../docs/product-direction.md).
