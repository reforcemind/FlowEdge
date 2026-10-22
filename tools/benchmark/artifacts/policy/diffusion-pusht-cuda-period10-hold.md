# Fake-robot CUDA period loop

> One host, one converted `diffusion_pusht` checkpoint, CUDA Core, simulator seam only. Not Jetson, not SO-100, not a PyTorch compare, not a success-rate claim.

| Field | Value |
|---|---|
| Captured | 2026-09-18, WSL2, NVIDIA GeForce GTX 1650 |
| Command | `python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors --steps 20 --threads 1 --period-ms 10 --on-miss hold --device cuda --host-facts --output tools/benchmark/artifacts/policy/diffusion-pusht-cuda-period10-hold.json` |
| Period | 10 ms |
| On miss | `hold` (repeat last sent action; zeros before the first on-time send) |
| Steps | 20 |
| Missed deadlines | 20 / 20 |
| Step p50 / p99 / max | 589 ms / 1571 ms / 1795 ms |
| Startup | 30.2 s |
| Peak RSS | 1948 MiB |
| Threads | 1 |
| Device | NVIDIA GeForce GTX 1650 |

Every 10 ms period missed. Synthetic encoded zeros; the RGB encoder stays in LeRobot. No warmup; the first step is the cold tail. Joint-limit clamps were not applied. This is not the matched CUDA replay p50 (420 ms vs PyTorch CUDA 346 ms) and not Jetson/ARM.
