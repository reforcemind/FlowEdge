# CUDA diffusion linear fusion

Decision: **ship**. The change is exact and wins end-to-end, though the gain is small.

The diffusion U-Net previously launched a separate bias kernel after every 1xK FiLM and timestep
linear, plus a separate Mish after the first timestep linear. Fusing those operations reduces the
native 10-step launch census from 1,000 to 850.

GTX 1650 4 GB, WSL2, Release CUDA build, exact FP32, pinned `diffusion_pusht` checkpoint,
5 alternating trials, 5 warmups and 30 measured iterations:

| Build | Median native p50 | Result |
|---|---:|---:|
| before fusion | 105.815 ms | baseline |
| after fusion | 104.257 ms | 1.47% lower, 1.015x |

The candidate won all five paired trials. Actions were bitwise identical in every trial. A prior
whole-denoiser CUDA Graph experiment doubled native latency and remains rejected; this artifact
supports only the narrower linear/bias/activation fusion.

`native.json` contains the complete timing samples and actions.
