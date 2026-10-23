# CUDA FP16-deep policy mode

Decision: **ship as an explicit approximate mode**. The exact FP32 path remains the default.

## Gate

GTX 1650 4 GB, WSL2, Release CUDA build, pinned `diffusion_pusht` checkpoint, 10-step DDIM,
1 warm process per trial, 5 alternating trials, 5 warmups and 30 measured iterations:

| Mode | Median native p50 | Result |
|---|---:|---:|
| `exact-f32` | 104.216 ms | baseline |
| `approx-fp16-deep` | 82.620 ms | 20.72% lower, 1.261x |

All five paired trials favored approximate mode. Per-trial maximum action error was
0.00116–0.00220. Tensor closeness is recorded but was not used as the quality gate.

Matched closed-loop `gym_pusht/PushT-v0`, seeds 7–16, 300-step limit:

| Mode | Success | Mean max reward |
|---|---:|---:|
| `exact-f32` | 8/10 | 0.998138 |
| `approx-fp16-deep` | 8/10 | 0.998099 |

The seven-layer calibration candidate was rejected because success fell to 7/10. The shipped
six-layer profile keeps the final deepest residual output in FP32. This is PushT simulation
evidence only; SO-101 task success remains unmeasured.

## Files

- `native-six-layer.json`: complete alternating native timing samples, actions, and parity.
- `pusht.json`: exact baseline plus the rejected seven-layer calibration run.
- `pusht-six-layer-approx.json`: accepted six-layer approximate PushT run.
- `native-final.json`: exact versus approximate timing after the separately gated linear fusion.

Reproduce native timing with `tools/benchmark/compare_native_diffusion.py`. Reproduce the
cross-environment task run with `tools/benchmark/evaluate_wsl_cuda_pusht.py`; its JSON IPC is not
part of latency measurement.
