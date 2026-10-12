# Fixed-shape Diffusion Policy optimization

10 ms was **not reached**. Keep the small CUDA epilogue fusion; do not claim an
order-of-magnitude runtime improvement or replace the published README numbers.

## Contract and machine

Batch 1, FP32, the existing PushT checkpoint, deterministic DDIM, 10 steps.
The target is a newly generated action chunk, not delivery of a buffered action.
Visual end-to-end latency additionally includes the existing observation encoder.
No scheduler, checkpoint, precision, action horizon, or normalization change was
used for the retained optimization. `papers/` and the root README were untouched.

Baseline source: `69233f6`. Candidate: this branch's CUDA GroupNorm/Mish/FiLM/
residual fusion. GTX 1650 4 GB, Windows/WSL Ubuntu, CUDA toolkit 12.0, GCC 13.3
Release; PyTorch 2.14.0+cu126 and LeRobot 0.4.4. CPU screens used Windows Clang
Release on the same host. The model revision and hashes are in the JSON files.

## Native A/B result

Three trials alternate baseline/candidate process order. Each trial uses a new
synthetic condition/noise seed, five warmups and 30 timed samples per step count.
Numbers below are medians of the three per-trial medians, not pooled quantiles.
`native-ab.json` contains every timing and action output. All paired actions pass
`atol=1e-3, rtol=1e-4`. Reduced-step rows are timing probes, not policy-quality evidence.

| DDIM steps | Baseline ms | Fused ms | Speedup |
|---|---:|---:|---:|
| 1 | 11.532 | 11.239 | 1.026x |
| 2 | 22.445 | 21.870 | 1.026x |
| 4 | 43.553 | 43.142 | 1.010x |
| 10 | 108.882 | 107.737 | 1.011x |

The 10-step candidate is faster in all three trials, but the approximately 1%
gain is small and host-dependent. No confidence-interval or stable-tail claim is
made from these short runs.

## Experiments

The matched visual replay (`cuda-policy-candidate.json`) passes with maximum
action error **3.05176e-5**. Policy p50 is **105.885 ms FlowEdge vs 332.048 ms
PyTorch eager** (3.14x); visual end-to-end p50 is **111.766 vs 338.057 ms**.
The earlier baseline replay measured 108.529 vs 349.134 ms. Both backends became
faster between those runs, so that cross-run difference is not the fusion's
causal speedup: use the alternating native A/B above. CUDA replay times one
backend after the other to avoid holding both networks on the GPU at once.
These results do not establish closed-loop task success.

The additional `torch.compile(mode="default", fullgraph=True)` U-Net run also
passes parity at **3.05176e-5**: **103.845 ms FlowEdge vs 328.838 ms compiled
PyTorch** policy p50 (3.17x), and **109.161 vs 334.433 ms** visual end-to-end.
It uses three warmups and 20 measured samples. This is one compile mode, not an
exhaustive PyTorch tuning study. The encoder and DDIM scheduler remain eager.
See `cuda-policy-compiled.json` for the explicit scope and raw samples.

| Change | Evidence | Decision |
|---|---|---|
| Repack CUDA convolution weights from OC/IC/K to OC/K/IC | Hottest isolate 731.3 -> 748.7 us; native 108.059 -> 108.845 ms | Rejected and removed |
| Fuse GroupNorm, Mish, optional FiLM and residual addition | Native A/B above; launch census 1490 -> 1000 per chunk | Retained |
| CPU K-blocked row-4 GEMM | Hottest single-thread GEMM 4297.9 -> 4875.9 us; four-worker native 582.026 -> 607.406 ms | Rejected and removed |

The fusion keeps the existing double-precision GroupNorm reduction and FP32
epilogues. It allocates no new hot-path storage. The isolated combined epilogue
also improved, but WDDM launch gaps make its speedup highly variable; use the
native A/B measurement for the actual benefit. CPU single-worker full sampling
improved in the screen while its hottest isolate regressed: that is not a reason
to keep an inconsistent candidate. Raw screen results are included beside this file.

The final rebuilt mix (`cuda-final.json`) confirms the epilogue isolate improves
from 16.5 to 13.8 us and the census remains 1000 launches. This smaller isolate
gain is a useful reminder not to extrapolate the unusually large first-screen
isolate difference to the entire policy.

Nsight Compute hardware counters were unavailable (`ERR_NVGPUCTRPERM`). We did
not alter privileged profiling settings. The bandwidth diagnosis below is an
estimate, not a measured DRAM-counter result.

## Why fusion cannot deliver 10 ms

The converted checkpoint has 1,006,047,844 tensor bytes. Under an approximate
stream-each-weight-once-per-step model, ten steps require about 10 GB of reads.
The device reports a 128-bit bus and 4,001,000 kHz memory clock, giving a
theoretical 128.032 GB/s (`cuda-final.json`). Even ideal utilization would take
about 79 ms for those reads alone.
This is a model of this implementation, not a universal algorithmic lower bound:
cache reuse, compression, precision changes, and a different network change it.
The hot 2048-channel convolution already accounts for about 51 ms of the baseline
mix estimate. Eliminating small pointwise launches cannot remove that cost.

FP16 weight storage would at best halve this particular traffic term, before
conversion and computation costs; it is not a plausible standalone 10 ms fix.
Simply setting DDIM to one step is also insufficient here (11.239 ms native) and
does not preserve the 10-step policy's task quality.

## Roadmap decision (2026-10-12)

Retain the measured runtime improvement and stop treating 10 ms as a universal
product goal. The compute/traffic estimates above explain the limit of this
experiment; they do not establish that every model or accelerator has that limit.
The follow-up training investigation is not part of the runtime PR. FlowEdge's
next milestone is qualified deployment of an existing trained policy: matched
outputs, explicit resource budgets and traceable action delivery on a real task.
See [product direction](../../../../docs/product-direction.md).

Historical JSON keeps the original 10 ms target and failed gate for auditability.
New native A/B reports measure relative performance without imposing that target.

## Reproduction

Preserve separate Release builds of baseline and candidate; do not benchmark
while builds, tests, or other inference jobs run. On this memory-constrained host
use two build jobs and run heavyweight checks sequentially.
An initial attempt failed CUDA residency, and running replay alongside the
Windows gate exhausted host commit/pagefile capacity. Sequential retries passed;
neither failed attempt is included in the timing results. The archived replay
JSONs reference per-run copies of the observation fixture; use the existing
`diffusion-pusht-cpu-replay.observations.npz` one directory up (SHA-256
`0e2616b6312a6f0682a7307682e4701011499a473f80ee734f570d438cccd60f`).

```bash
python tools/benchmark/compare_native_diffusion.py \
  models/diffusion_pusht.flowedge.safetensors \
  --baseline /home/igor/fe-perf-baseline --candidate /home/igor/fe-cuda \
  --steps 1 2 4 10 --iterations 30 --warmup 5 --trials 3 \
  --output bench-results/perf-10ms/native-ab.json

FLOWEDGE_CUDA_REQUIRED=1 /home/igor/fe-cuda/flowedge_cuda_dp_mix \
  models/diffusion_pusht.flowedge.safetensors bench-results/perf-10ms/cuda-final.json

python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --device cuda --build-dir /home/igor/fe-cuda \
  --observations bench/artifacts/policy/diffusion-pusht-cpu-replay.observations.npz \
  --steps 10 --iterations 30 --warmup 5 --threads 1 \
  --output bench-results/perf-10ms/cuda-policy-candidate.json
```

`bench policy --reference-compile default` also supports a compiled U-Net
reference. The encoder and scheduler remain eager. Warmup excludes the first
compilation; failures are not replaced by eager results. See
[PyTorch's compile contract](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

## Verification

`powershell -File scripts/verify_local.ps1 -BuildDir build-win-clang` passed:
82 CTest entries, 102 integration tests (one existing optional action-delivery
artifact test skipped), delivery smoke, head/streaming smoke, CPU diffusion
parity, ULP (maximum 3 ULP), and tiny-GPT2 parity. The eight benchmark-artifact
tests also passed. CUDA focused fusion/reference and resident-allocation tests
passed. Final `ctest --test-dir /home/igor/fe-cuda --output-on-failure` passed
all 88 entries after building the two missing help-test executables.
`FLOWEDGE_CUDA_REQUIRED=1 python -m flowedge_dev verify diffusion /home/igor/fe-cuda`
passed converter/normalization checks and denoiser/DDIM/DDPM parity (maximum
absolute errors 1.49e-8 / 4.77e-7 / 2.86e-6). Both real-checkpoint policy replays
passed. Ruff and `git diff --check` passed. The A/B runner's wrong-build import
guard was checked to fail without writing an artifact.
