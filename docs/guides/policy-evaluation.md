# Policy evaluation

```{image} ../_static/figures/verify.svg
:alt: Same checkpoint, FlowEdge vs PyTorch
:class: fe-fig
```

Smoke tests, matched full-policy replay, and closed-loop PushT. Replay uses the original RGB encoder, stats, history, and trained DP weights. Fixture: `lerobot/diffusion_pusht` revision `84a7c23178445c6bbf7e1a884ff497017910f653`.

## Setup

```bash
python -m pip install .
python -m pip install -e 'integrations/lerobot[evaluation]'
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --include config.json model.safetensors --local-dir models/diffusion_pusht
python -m flowedge_dev pipeline convert models/diffusion_pusht models/diffusion_pusht.flowedge.safetensors \
  --arch diffusion --dtype f32
```

The pinned gym-pusht release needs Pymunk 6; its upstream dependency also permits
incompatible Pymunk 7. The evaluation extra constrains that dependency.

## Plugin semantics

`FlowEdgeConfig.input_mode="encoded"` accepts one already-normalized, complete
condition vector in `observation.state`. It rejects raw dataset statistics and images.
`input_mode="visual"` also requires `source_checkpoint_path` pointing at the original
model directory. Its processor loads source statistics or validates explicit
`dataset_stats`; its encoder maintains the source observation history. Use source
feature definitions in the plugin config. Policy methods accept processor outputs.
Converted checkpoints bind the source model and config hashes. Reconvert older artifacts
before visual deployment; a mismatched encoder/checkpoint pair is rejected at setup.

`select_action` updates observations every tick, consumes the current chunk, then
resamples on depletion. `action_steps=None` adopts the converted executable horizon;
a shorter execution horizon is allowed. `reset` clears history/actions and restarts
the configured seed. `predict_action_chunk` predicts a fresh chunk and invalidates
queued actions. Optional full-horizon `noise` overrides sampling at the next refill.

Native output is already in dataset action units; the output processor must not
unnormalize it again. Encoding uses PyTorch on CPU, outside Core's allocation contract.
Unsupported source layouts fail during setup; this is not a universal LeRobot adapter.

## Matched replay

```bash
python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --steps 10 --iterations 100 --warmup 5 --threads 1 \
  --output bench/artifacts/policy/diffusion-pusht-report.json
```

CUDA matched replay uses the same observations, processor, noise, and DDIM
schedule, with FlowEdge CUDA Core against PyTorch CUDA. `threads=1` is not the
GPU headline; the artifact records the CUDA device name. Reuse the CPU fixture
when you want the same observation file. On a 4GB card the runner frees the
FlowEdge engine before loading the PyTorch U-Net, so both U-Nets are not
device-resident together:

```bash
python -m flowedge_dev bench policy models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --steps 10 --iterations 10 --warmup 2 --threads 1 --device cuda \
  --observations bench/artifacts/policy/diffusion-pusht-cpu-replay.observations.npz \
  --build-dir /path/to/cuda-build \
  --output bench/artifacts/policy/diffusion-pusht-cuda-replay.json
```

Fail closed if `torch.cuda` is missing or FlowEdge was not built with
`FLOWEDGE_BACKEND=cuda`. Do not invent a GPU p50, do not compare TensorRT/ONNX
unless those runs are also matched, and do not treat a desktop GPU JSON as
Jetson/ARM evidence.

The runner alternates execution order and checks actions against the upstream LeRobot
PyTorch U-Net and diffusers DDIM scheduler. It saves raw timings, source/converted/config hashes,
processor statistics, versions, and replay observations. End-to-end means preprocessing
through executable action chunk; capture, IPC, and robot delivery are excluded. RSS is
shared process high-water memory, not per-backend usage. Uninstrumented allocations are
`null` with a reason, never an invented zero. One fixed history is a replay workload,
not task quality. Small sample counts do not establish stable tail-latency estimates.

## Closed-loop tasks

```bash
python -m flowedge_lerobot.evaluate models/diffusion_pusht.flowedge.safetensors \
  --source models/diffusion_pusht --backend flowedge --episodes 5 --max-steps 300 \
  --steps 10 --threads 1 --device cpu --precision exact-f32 \
  --output bench-results/pusht-flowedge.json
```

Repeat with `--backend lerobot` and identical seeds/settings. Reports include seeds,
reward, success, steps, and hashes. This mode advances simulated time after an action
becomes available; wall-clock latency does not penalize its score.

For a CUDA source build, use `--device cuda`. `--precision approx-fp16-deep` is an explicit,
fail-closed approximate policy mode; compare it against `exact-f32` with identical seeds. Do not
promote a latency result unless task success is unchanged on the declared evaluation set.

Add `--period-ms 100` for periodic asynchronous delivery. Startup bootstraps the first
chunk before timing; later chunks discard actions whose observation-relative ticks have
elapsed. Reports include underruns, delivery deadline misses, absolute inter-delivery
jitter, and observation age. PushT's explicit fallback holds its current agent position.
Real robots must supply appropriate fallback and stop behavior. The Python runner can
allocate and is subject to OS and GIL scheduling.

`flowedge-lerobot-rollout` remains a synthetic integration smoke test. Its
`missed_deadlines` counts per-call overruns, not periodic deadline misses.

CUDA Core rollout uses the same `--on-miss hold|drop|raise` contract. Fail closed
if FlowEdge was not built with `FLOWEDGE_BACKEND=cuda`. The RGB encoder stays in
LeRobot (`input_mode=visual`, `device=cuda`); Core still sees a host condition.
Do not treat a desktop GPU JSON as Jetson/ARM evidence. On GTX 1650 a 10 ms
period missed 20 / 20 steps (step p50 589 ms); that is a miss log, not a
success-rate claim.

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 1 --period-ms 10 --on-miss hold --device cuda --host-facts \
  --output bench/artifacts/policy/diffusion-pusht-cuda-period10-hold.json
```
