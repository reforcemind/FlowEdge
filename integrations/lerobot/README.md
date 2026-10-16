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
python -m pip install -e "integrations/lerobot[so101]"
python -m unittest discover -s integrations/lerobot/tests
```

The plugin requires Python 3.12+ and `lerobot>=0.6.1,<0.7`. An explicit CUDA
request never falls back to CPU: both the PyTorch encoder and native FlowEdge
engine must be CUDA-capable. Use the smaller `rollout` extra for a non-Feetech
robot, or `dp` if you only need policy conversion and tests.

On CUDA, the plugin passes the encoded condition, initial noise, and executable action chunk
through DLPack. These tensors stay on device until LeRobot's postprocessor moves the selected
action to CPU for the robot driver. CPU mode keeps the NumPy path.

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

## Package and run on a robot

Create a relocatable LeRobot policy directory from the matching converted and
source Diffusion Policy checkpoints:

```bash
flowedge-lerobot-package \
  models/my_so101.flowedge.safetensors \
  outputs/train/diffusion_so101/checkpoints/last/pretrained_model \
  deploy/my_so101_flowedge
```

The package contains the native U-Net, the source RGB encoder and normalization
state, and a LeRobot `config.json`. It does not duplicate the source PyTorch
U-Net. The source identity is checked before packaging and whenever the package
loads. Upload the directory to a Hugging Face model repository to use its repo
ID as `--policy.path` on another machine.

Run it through LeRobot's standard hardware-independent deployment command:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=deploy/my_so101_flowedge \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="put the red block in the bowl" \
  --duration=30
```

Use `--policy.device=cuda` only with a CUDA FlowEdge build. Add
`--policy.diffusion_precision=approx-fp16-deep` only after task calibration.
LeRobot owns connection, cameras, action mapping, joint limits, and emergency
stop. Commission with an unloaded arm and a short duration.

The lower-level `run_rollout` protocol remains available for non-LeRobot drivers:

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
