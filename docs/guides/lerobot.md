# LeRobot deployment adapter

LeRobot trains the policy, runs cameras, and talks to the robot. FlowEdge is the native action head behind that plugin: a compatible Diffusion Policy U-Net or cached SmolVLA expert. It provides fixed-memory execution, an explicit period, and a miss policy without putting e-stop or joint limits inside Core.

## Boundary

| LeRobot | FlowEdge |
|---|---|
| Cameras, RGB/state encoder, normalization, history | Fixed-memory U-Net / expert |
| Robot driver, joint limits, e-stop | `--on-miss hold\|drop\|raise` |
| Training | Converted `.safetensors` |

Action slice: `actions[observation_steps - 1 : observation_steps - 1 + action_steps]`.
`input_mode="visual"` uses the source encoder. A flattened condition remains valid. [Policy evaluation](policy-evaluation).

## Install

```bash
python -m pip install -e "integrations/lerobot[so101]"
python -m unittest discover -s integrations/lerobot/tests
```

The adapter supports Python 3.12+, LeRobot 0.6, and only `cpu` or `cuda`.
Unavailable CUDA is an error; it is never silently replaced with CPU.
The `so101` extra includes LeRobot's rollout, Diffusion Policy, and Feetech
dependencies. Use the `rollout` extra for another robot family.
The CUDA plugin uses DLPack end to end for the encoded condition, diffusion noise, and queued
action chunk. Only the final action handed to the robot driver crosses back to host memory.

```python
from flowedge_lerobot import FlowEdgeDiffusionPolicy
policy = FlowEdgeDiffusionPolicy.from_checkpoint(
    "models/diffusion_pusht.flowedge.safetensors", threads=4
)
```

## Real robot package

Package the converted U-Net with only the matching source RGB encoder and
normalization state. This avoids shipping a second PyTorch copy of the U-Net and
produces a standard LeRobot policy directory:

```bash
flowedge-lerobot-package \
  models/my_so101.flowedge.safetensors \
  outputs/train/diffusion_so101/checkpoints/last/pretrained_model \
  deploy/my_so101_flowedge

lerobot-rollout \
  --strategy.type=base \
  --policy.path=deploy/my_so101_flowedge \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="put the red block in the bowl" \
  --duration=30
```

`lerobot-rollout` owns the driver lifecycle and maps the policy action to the
robot. FlowEdge's plugin loads its native checkpoint without a dummy PyTorch
`model.safetensors`. Relative package paths also work after moving the directory
or downloading it from the Hub; a Hugging Face model repo ID can be passed
directly as `--policy.path`. Commission with an unloaded arm and a short duration;
limits and emergency stop remain LeRobot responsibilities.

Run a no-robot preflight after copying the package to the deployment machine:

```bash
python -m flowedge_dev pipeline preflight deploy/my_so101_flowedge \
  --iterations 5 --output flowedge-preflight.json
```

The installed-wheel equivalent is `flowedge-lerobot-preflight`. It verifies
package hashes, loads the registered plugin, and runs synthetic RGB/state input
through preprocessing, the encoder, and a fresh native action chunk. Its JSON
records the feature contract, backend, CUDA residency, startup/RSS, and latency.
This is a deployment diagnostic, not camera, safety, or task-success evidence.

## Period loop (sim and Jetson smoke)

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 4 --period-ms 10 --on-miss hold
```

CUDA Core (`FLOWEDGE_BACKEND=cuda`), same miss contract:

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 1 --period-ms 10 --on-miss hold --device cuda
```

`--device cuda` requires a CUDA Core binary **and** `Engine.cuda_resident`. A 4 GB
card that cannot upload the U-Net stays on CPU kernels; that is a load success,
not a CUDA replay. `FLOWEDGE_CUDA_REQUIRED=1` fails closed. GTX 1650, 10 ms hold:
20 / 20 misses. Not a Jetson/ARM claim.

| `--on-miss` | On overrun |
|---|---|
| `hold` | Repeat last sent action; zeros before the first on-time send |
| `drop` | Skip `send_action` |
| `raise` | `DeadlineMissed` after `stop` |

On a physical Jetson, run:

```bash
FLOWEDGE_EDGE_REPORT=jetson.json \
  scripts/edge_dp_rollout.sh models/diffusion.flowedge.safetensors
```

The helper requires Linux aarch64, the L4T release file, Python 3.12+, and
device-resident native CUDA weights. The resulting JSON is a synthetic
inference benchmark. SO-101 task success is measured through the packaged
policy and `lerobot-rollout`; CI uses a fake robot, so ARM64 CI alone is not a
board claim.

`--policy.type=flowedge` is DP. `--policy.type=flowedge_smolvla` is the native expert plus a LeRobot VLM cache — not a native VLM. Tenstorrent is a separate backend, not this plugin.
