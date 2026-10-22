# Memory-mapped checkpoint load

This capture compares `fe1039a` with the memory-mapped `ModelWeights`
candidate on the same Windows host and the real converted PushT Diffusion
Policy checkpoint.

The process is `flowedge-inspect`, so each sample includes safetensors
validation, checkpoint fingerprinting, and runtime construction. It is a model
load measurement, not policy inference latency. One warmup precedes five
alternating trials. The executable hashes and raw samples are retained in
`windows-gtx1650.json`.

| Metric (median) | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Wall time | 877.834 ms | 613.298 ms | -30.1% |
| Peak private bytes | 1,048,662,016 | 42,536,960 | -1,006,125,056 bytes |
| Peak working set | 2,005,897,216 | 1,050,349,568 | -955,547,648 bytes |

The removed private allocation is approximately one checkpoint payload
(`1,006,047,844` supported tensor bytes). The mapped pages remain file-backed,
so the candidate retains CPU fallback without a second anonymous weight copy.

Command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools/benchmark/measure_checkpoint_load.ps1 `
  -Baseline ..\FlowEdge-mmap-baseline\build\flowedge-inspect.exe `
  -Candidate build-win-clang\flowedge-inspect.exe `
  -Checkpoint models\diffusion_pusht.flowedge.safetensors `
  -Output bench\artifacts\policy\checkpoint-mmap-2026-10-14\windows-gtx1650.json `
  -Trials 5 -Warmup 1
```
