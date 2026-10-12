# FlowEdge + LeRobot

Out-of-tree `lerobot_policy_flowedge` plugin for LeRobot 0.6. Training, cameras,
limits, and e-stop stay in LeRobot. FlowEdge runs the converted action path.

## Scope

- Compatible LeRobot `DiffusionConditionalUnet1d` checkpoints (`--policy.type=flowedge`)
- Visual observations via the source RGB encoder, or a flattened condition
- `--on-miss hold|drop|raise` on a control period
- Cached SmolVLA expert (`--policy.type=flowedge_smolvla`): LeRobot VLM cache, native Euler expert — not a native VLM

Action slice: `actions[observation_steps - 1 : observation_steps - 1 + action_steps]`.

## Install

```bash
python -m pip install -e "integrations/lerobot[dp]"
python -m unittest discover -s integrations/lerobot/tests
```

The plugin requires Python 3.12+ and `lerobot>=0.6.1,<0.7`. An explicit CUDA
request never falls back to CPU: both the PyTorch encoder and native FlowEdge
engine must be CUDA-capable.

## Hardware and sim

Fake robot (CI and CPU):

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 4 --period-ms 10 --on-miss hold
```

CUDA Core, same `--on-miss` contract (`FLOWEDGE_BACKEND=cuda`):

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 1 --period-ms 10 --on-miss hold --device cuda
```

The RGB encoder stays in LeRobot when `input_mode=visual`. GTX 1650 10 ms hold
missed 20 / 20. Not Jetson/ARM.

Physical Jetson CUDA smoke:

```bash
FLOWEDGE_EDGE_REPORT=jetson.json \
  scripts/edge_dp_rollout.sh <converted.safetensors>
```

The helper requires Linux aarch64, `/etc/nv_tegra_release`, Python 3.12+, a
CUDA build of FlowEdge, and device-resident weights. Its JSON identifies the
Jetson model and L4T release. It is a synthetic inference result, not evidence
of robot task success.

SO-100/SO-101: wrap `reset` / `observe` / `send_action` / `stop` and pass `encode_condition` into `run_rollout`. The plugin does not import a robot driver.

```python
from flowedge_lerobot import FlowEdgeDiffusionPolicy, run_rollout

policy = FlowEdgeDiffusionPolicy.from_checkpoint("models/diffusion_pusht.flowedge.safetensors")
result = run_rollout(policy, robot, encode_condition=processor, steps=100, seed=7)
```

## SmolVLA expert

VLM stays in LeRobot. Hybrid parity / timing:

```bash
python -m flowedge_dev verify smolvla \
  models/smolvla_base/model.safetensors models/smolvla_base \
  bench/artifacts/smolvla/eslab-frame-000000.capture.npz \
  --module-path build-research
python -m flowedge_dev bench hybrid ...   # stage p50/p95/p99
```

Not native SmolVLA. Not a control-loop claim. Details: [policy evaluation](../../docs/guides/policy-evaluation.md).
