# Fake-robot period loop

> One host, one converted `diffusion_pusht` checkpoint, simulator seam only. Not Jetson, not SO-100, not a PyTorch compare.

| Field | Value |
|---|---|
| Captured | 2026-09-17, same Windows/Clang host as the packed visual replay |
| Command | `python tools/verification/run_edge_dp_rollout.py models/diffusion_pusht.flowedge.safetensors --steps 20 --threads 4 --period-ms 10 --on-miss hold` |
| Period | 10 ms |
| On miss | `hold` (repeat last sent action; zeros before the first on-time send) |
| Steps | 20 |
| Missed deadlines | 20 / 20 |
| Step p50 / p99 / max | 655 ms / 812 ms / 834 ms |
| Startup | 2292 ms |
| Peak RSS | 1945 MiB |
| Threads | 4 |

Every 10 ms period missed. Joint-limit clamps were not applied; the seam does not import a robot driver. Use this command on Jetson or SO-100 and attach JSON to issue #70.
