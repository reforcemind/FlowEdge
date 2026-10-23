# Resident CUDA GEMV follow-up

Keep the FP32 warp-per-output GEMV for resident single-row matrices with at
least 128 inputs and outputs. It needs no repacking, extra workspace, or hot
device allocation. The original checkpoint, normalization, precision, horizon,
and ten-step DDIM schedule are unchanged. **10 ms was not reached.**

Baseline is the existing fused GroupNorm/Mish/FiLM/residual implementation,
saved before this experiment in `/home/igor/fe-baseline-memory-20261011`.
This is an incremental comparison, not a comparison against the unfused branch
base. Candidate source: `src/core/kernels/cuda/kernels_cuda.cu`, SHA-256
`3c9c0863988b3168ea24f9b648f6a910044b76af228cf459f2bd7e46ba319cce`.
Both builds use GCC 13.3.0 Release, CUDA 12.0, GTX 1650 4 GB, Windows/WSL Ubuntu.
Native module SHA-256 identities:

- Baseline: `399f59c0a1149adaac9ffef7b210f3f63d5556947d6102fc137cfe084fafbfa1`.
- Candidate: `3e9be4d20dcc14232035e2a3c6203d0fd41bfd2f1daf5f9301ea438e8cffcfc8`.

The converted checkpoint SHA-256 is
`3367c7236d7a0d6da74e28cde1514ecc7b31470ec028e244cf28c93f93e0b5d6`.

## Measurements and decision

The first mix screen reduced the 1x260x4096 GEMV p50 from 76.928 to
43.008 us and native ten-step sampling from 107.892 to 106.630 ms.
The first three alternating native trials had a median gain of only 0.256%
and one paired win. That was insufficient evidence of a stable improvement.

Before confirmation, `confirmation-gate.md` declared five alternating trials,
ten warmups, 100 measured samples, new seeds 107-111, at least four paired wins,
and more than 1% median latency reduction. Confirmation passed all five pairs:

| Seed | Baseline p50 ms | Candidate p50 ms |
|---|---:|---:|
| 107 | 105.348 | 103.777 |
| 108 | 105.028 | 103.805 |
| 109 | 104.902 | 103.604 |
| 110 | 104.751 | 103.587 |
| 111 | 104.867 | 103.622 |
| Median of trial medians | 104.902 | 103.622 |

This is a **1.22% latency reduction**, not an order-of-magnitude improvement.
Every native comparison passed `atol=1e-3, rtol=1e-4`. Raw timings and action
outputs are in `confirmation.json`; `native-ab.json` preserves the initial
mixed result. CUDA residency and Release builds were checked in every worker.

Matched visual replay against eager LeRobot/PyTorch also passed: maximum
absolute action error **3.05176e-5**, below the existing 1e-3 gate.

| Matched replay | FlowEdge p50 ms | PyTorch eager p50 ms |
|---|---:|---:|
| Policy | 103.880 | 346.709 |
| Encoder through action chunk | 109.873 | 352.477 |

This replay uses one fixed real PushT observation history, five warmups and
30 samples. It does not establish task success or stable tail latency. Camera
capture, IPC, and physical robot delivery are excluded. It is not a new
compiled-PyTorch comparison. Both U-Nets are loaded sequentially on the 4 GB
card; PyTorch emitted allocation warnings during setup but completed every
iteration and parity check. See `cuda-policy.json` and its generated report
for hashes, versions, raw timings, and exact scope. Neither the published
README headline nor robot/Jetson claims are changed.

## Changes and checks

- The CUDA multiply shares an output reduction across a warp; the existing
  row-major weights are read by adjacent lanes. Small and multi-row calls
  keep their previous paths. Floating-point reduction order changes.
- `CudaMatmul.ResidentRowsMatchDoubleReferenceWithoutAllocation` covers
  widths 37/127/128/129/260/512, outputs 17/127/128/129/4096, one/two rows,
  repeated calls, independent double-precision results and device malloc counts.
- The CUDA mix now includes all five projection shapes from the native PushT
  census, rather than only the largest FiLM matrix. The extended short screens
  are diagnostic and noisy, not a replacement for native paired trials.
- CUDA CTest: 89/89 passed.
- CUDA conversion/denoiser/DDIM/DDPM verification passed; maximum absolute
  errors were 1.49e-8 / 4.77e-7 / 2.86e-6 for the three numerical checks.
- CUDA Mamba/flow ULP passed: maximum 2 ULP, relative error 1.74e-7.
  The first attempt failed at import because the native module loaded the
  system CUDA 12.0 runtime before PyTorch. Selecting PyTorch's newer compatible
  `libcudart` with `LD_LIBRARY_PATH` resolved that local environment mismatch;
  no inference code or tolerance was changed.
- Windows local verification passed: 82/82 CTest checks, 108 integration tests
  with one optional-dependency skip, conversion/diffusion, external-head,
  delivery, Mamba/flow ULP (maximum 3 ULP), and tiny-GPT-2 reference checks.
  `git diff --check` also passed.

### Selected PR gate (2026-10-12)

The student-policy tools and their six tests were removed when the universal
10 ms objective was retired. After that cleanup, the Windows full local gate
passed again: 82 CTest checks and 102 integration tests with the same optional
skip, plus delivery/head/diffusion/ULP/Transformer verification. CUDA CTest
passed 89/89 again. The eight benchmark-artifact tests, changed Python Ruff
checks and formatting checks, and `git diff --check` passed. No kernel behavior
changed after the measured confirmation and matched replay above. Verbose local
verification logs are intentionally not stored as benchmark artifacts.

The new A/B runner has no hard-coded latency target; archived JSON retains its
original target fields. The product goal is now
[faster, memory-efficient edge inference](../../../../docs/product-direction.md),
not a universal 10 ms sampler. Output parity is an acceptance guardrail for
performance work, not the project's focus. Full-file clang-format checking of
`kernels_cuda.cu` still reports
pre-existing formatting debt (also present at the branch base); it is not a
claimed passing check. Other changed C++ files passed the focused format check.

The access-pattern hypothesis follows NVIDIA's
[coalesced-memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory);
the keep decision is based on the measurements above, not that guidance alone.

## Reproduce

Commands run from the repository inside WSL. Baseline and candidate must be
separate native build directories; builds and GPU runs were serialized.

```sh
cmake --build /home/igor/fe-cuda --target flowedge_cuda_dp_mix flowedge flowedge_tests -j2
ctest --test-dir /home/igor/fe-cuda --output-on-failure
FLOWEDGE_CUDA_REQUIRED=1 /home/igor/fe-cuda/flowedge_cuda_dp_mix models/diffusion_pusht.flowedge.safetensors bench-results/runtime-gemv-20261011/candidate.json
/home/igor/fe-cuda-venv/bin/python tools/benchmark/compare_native_diffusion.py models/diffusion_pusht.flowedge.safetensors --baseline /home/igor/fe-baseline-memory-20261011 --candidate /home/igor/fe-cuda --steps 10 --iterations 100 --warmup 10 --trials 5 --seed 107 --output bench-results/runtime-gemv-20261011/confirmation.json
FLOWEDGE_CUDA_REQUIRED=1 /home/igor/fe-cuda-venv/bin/python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 --steps 10 --iterations 30 --warmup 5 --threads 1 --device cuda --observations tools/benchmark/artifacts/policy/diffusion-pusht-cpu-replay.observations.npz --build-dir /home/igor/fe-cuda --output bench-results/runtime-gemv-20261011/cuda-policy.json
PYTHONPATH=/home/igor/fe-cuda FLOWEDGE_CUDA_REQUIRED=1 /home/igor/fe-cuda-venv/bin/python -m flowedge_dev verify diffusion /home/igor/fe-cuda
LD_LIBRARY_PATH=/home/igor/fe-cuda-venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib PYTHONPATH=/home/igor/fe-cuda FLOWEDGE_CUDA_REQUIRED=1 /home/igor/fe-cuda-venv/bin/python -m flowedge_dev verify ulp models/mamba_flow.safetensors
```

Windows gate, after rebuilding the Release test target with two build jobs:

```powershell
cmake --build build-win-clang --target flowedge_tests -j2
powershell -NoProfile -File tools/scripts/verify_local.ps1 -BuildDir build-win-clang
```
