# LeRobot / FlowEdge edge benchmark

> These are measurements of one checkpoint on one declared host, not universal deployment guarantees.

## Run identity

| Field | Value |
|---|---|
| Captured | `2026-09-16T16:43:09.625545+00:00` |
| Model | `lerobot/diffusion_pusht` @ `84a7c23178445c6bbf7e1a884ff497017910f653` |
| Model SHA-256 | `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8` |
| Processor | `lerobot/diffusion_pusht` @ `84a7c23178445c6bbf7e1a884ff497017910f653` |
| Processor stats | `{"observation.image":{"mean":[[[0.48500001430511475]],[[0.4560000002384186]],[[0.4059999883174896]]],"std":[[[0.2290000021457672]],[[0.2240000069141388]],[[0.22499999403953552]]]},"observation.state":{"max":[496.14617919921875,510.9578857421875],"min":[13.45642375946045,32.93829345703125]}}` |
| Observation contract | `2dde368a238be30a27b8a7c243b724c51d7c744946f28e439c2afc625e2904fe`; 2 steps |
| Action contract | 2 dims x 8 steps; dataset |
| Run shape | batch 1; 10 inference steps |
| Host | `Jankowski`; Windows-10-10.0.26200-SP0; Intel64 Family 6 Model 158 Stepping 13, GenuineIntel; 1 threads |
| Build | `Clang 23.1.0 (https://github.com/llvm/llvm-project ea7d852a70e8bdfaf601d6626a760f9771b2c4b4)`; Release |
| Comparison | FlowEdge native runtime vs LeRobot policy executed with PyTorch |
| Reference framework | `PyTorch` |

## Measurements

Encoder and policy are reported separately. End-to-end includes both and is the number to use for control-loop budgeting. The reference row is the same LeRobot policy executed through PyTorch.

| Backend | Startup | Encoder p50 / p95 / p99 | Policy p50 / p95 / p99 | E2E p50 / p95 / p99 | Throughput | RSS | Setup / hot allocations |
|---|---:|---:|---:|---:|---:|---:|---:|
| flowedge | 962.549 ms | 38.596 ms / 40.185 ms / 40.997 ms | 851.228 ms / 865.971 ms / 867.270 ms | 889.243 ms / 906.596 ms / 906.988 ms | 1.122 Hz | 3357.395 MiB | not measured / not measured |
| lerobot | 1011.828 ms | 38.537 ms / 39.630 ms / 40.013 ms | 1409.156 ms / 1429.392 ms / 1446.317 ms | 1447.994 ms / 1469.475 ms / 1486.018 ms | 0.689 Hz | 3357.395 MiB | not measured / not measured |

## Relative comparison

| Metric | FlowEdge / LeRobot |
|---|---:|
| Policy p50 | 0.604x |
| End-to-end p50 | 0.614x |
| Throughput | 1.629x |

## Commands

```text
FlowEdge: C:\Users\igorj\Desktop\FlowEdge\venv\Scripts\python.exe -m flowedge_lerobot.benchmark models\diffusion_pusht.flowedge.safetensors --source models\diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 --model-id lerobot/diffusion_pusht --steps 10 --iterations 20 --warmup 2 --threads 1 --seed 7 --output bench\artifacts\policy\diffusion-pusht-cpu-replay.json
LeRobot:  C:\Users\igorj\Desktop\FlowEdge\venv\Scripts\python.exe -m flowedge_lerobot.benchmark models\diffusion_pusht.flowedge.safetensors --source models\diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 --model-id lerobot/diffusion_pusht --steps 10 --iterations 20 --warmup 2 --threads 1 --seed 7 --output bench\artifacts\policy\diffusion-pusht-cpu-replay.json
```

## Limits

- Repeat on the target CPU, OS, compiler, thread count, power mode, and checkpoint before making a deployment decision.
- RSS includes process/runtime state and is not a per-inference allocation measurement.
- The zero-allocation contract applies to supported native paths; unknown full-policy counts remain unmeasured.
- CPU full-policy replay of one fixed PushT observation history; not closed-loop task success.
- End-to-end includes preprocessing, encoder, history and sampling; excludes camera capture, IPC and robot delivery.
- RSS is shared process high-water memory with both implementations resident, not per-backend memory.
- Allocations are unmeasured; no zero-allocation claim applies to this Python visual pipeline.
- Startup excludes interpreter/import time and uses the existing filesystem cache.
- Small iteration counts do not establish stable p99 estimates.
