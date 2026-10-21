# LeRobot deployment adapter

LeRobot trains the policy, runs cameras, and talks to the robot. FlowEdge is the native action head behind that plugin: a compatible Diffusion Policy U-Net or cached SmolVLA expert. The Python policy API only produces actions; it does not schedule a robot period. `flowedge-lerobot-rollout` is a simulator-only smoke test; its `hold`, `drop`, or `raise` behavior does not control a physical robot. Use LeRobot's `lerobot-rollout` for hardware. Neither path replaces joint limits or an emergency stop.

## Boundary

| LeRobot | FlowEdge |
|---|---|
| Cameras, RGB/state encoder, normalization, history | Fixed-memory U-Net / expert |
| Robot driver, joint limits, e-stop | Simulator smoke timing and miss-policy measurements |
| Training | Converted `.safetensors` |

Action slice: `actions[observation_steps - 1 : observation_steps - 1 + action_steps]`.
`input_mode="visual"` uses the source encoder. A flattened condition remains valid. [Policy evaluation](policy-evaluation).

## Install

```bash
python -m pip install .
python -m pip install -e "integrations/lerobot[so101]"
python -m unittest discover -s integrations/lerobot/tests
```

The adapter supports Python 3.12+, LeRobot 0.6, and only `cpu` or `cuda`.
Unavailable CUDA is an error; it is never silently replaced with CPU.
The command above builds the CPU runtime from the checkout. For CUDA, build
FlowEdge with CUDA and install that runtime plus CUDA-enabled PyTorch; see the
[CUDA build guide](../architecture/cuda). Published Python wheels are CPU-only.
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

Use the same Python entry point to package either supported policy type.
Diffusion Policy needs its converted FlowEdge checkpoint; SmolVLA is read
directly from its trained LeRobot directory:

Convert the Diffusion Policy checkpoint once:

```bash
python -m flowedge_dev pipeline convert \
  outputs/train/diffusion_so101/checkpoints/last/pretrained_model \
  models/so101.flowedge.safetensors --arch diffusion
```

```python
from flowedge_lerobot import load_policy, package_policy

package_policy(
    "outputs/train/diffusion_so101/checkpoints/last/pretrained_model",
    "deploy/so101",
    converted_checkpoint="models/so101.flowedge.safetensors",
)
policy = load_policy("deploy/so101", device="cpu")
policy.reset()
# Application sketch: pass the latest observation mapping from your camera/state
# reader. Its keys and shapes must match policy.input_features.
action = policy.select_action(observation)
```

The result is a postprocessed LeRobot tensor shaped `(1, action_dim)`, not the
named joint mapping accepted by a robot driver. LeRobot's standard rollout
performs that mapping.

For SmolVLA, call `package_policy("trained_smolvla", "deploy/so100")` without
`converted_checkpoint`. Its native expert is currently CPU-only; this does not
accelerate the VLM or establish a Jetson speedup. `policy.input_features`
shows expected observation keys. Pass the task with
`policy.select_action(observation, task="pick up the cube")` for SmolVLA.

For physical operation, LeRobot remains the robot interface. Package the
converted U-Net with only the matching source RGB encoder and normalization
state, then run:

```bash
flowedge-lerobot-package \
  outputs/train/diffusion_so101/checkpoints/last/pretrained_model \
  deploy/so101 \
  --converted-checkpoint models/my_so101.flowedge.safetensors

lerobot-rollout \
  --strategy.type=base \
  --policy.path=deploy/so101 \
  --device=cpu \
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

The example explicitly uses CPU. For a CUDA-enabled Diffusion Policy build,
use `--device=cuda`; this requires both the CUDA FlowEdge runtime and
CUDA-enabled PyTorch. The packaged config otherwise defaults to CPU. SmolVLA
currently supports CPU only and rejects a CUDA request.

Run the installed no-robot preflight after copying the package to the deployment
machine:

```bash
flowedge-lerobot-preflight deploy/so101 \
  --iterations 5 --output flowedge-preflight.json
```

It verifies package hashes, loads the registered plugin, and runs synthetic RGB/state input
through preprocessing, the encoder, and a fresh native action chunk. Its JSON
records the feature contract, backend, CUDA residency, startup/RSS, and latency.
This is a deployment diagnostic, not camera, safety, or task-success evidence.
From a source checkout, the maintainer equivalent is
`python -m flowedge_dev pipeline preflight ...`.

## Simulator timing smoke test

This built-in simulator does not connect to a robot, and its missed-period
counts are not physical control-loop evidence. For robot operation, use the
`lerobot-rollout` command above.

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
inference benchmark, not a robot rollout. Evaluate physical task success with
LeRobot's `lerobot-rollout`; CI uses a fake robot, so ARM64 CI alone is not a
board claim.

`--policy.type=flowedge` is DP. `--policy.type=flowedge_smolvla` is the native expert plus a LeRobot VLM cache — not a native VLM. Tenstorrent is a separate backend, not this plugin.

## Compact SmolVLA package

Install the plugin with both robot and VLA dependencies in Python 3.12+:

```bash
python -m pip install .
python -m pip install -e 'integrations/lerobot[so100,smolvla]'
```

The source must be a trained LeRobot SmolVLA directory containing
`model.safetensors`, `config.json`, `policy_preprocessor.json`, and
`policy_postprocessor.json`. These files carry the trained camera, state,
language, and action contracts. Package and exercise one synthetic action
chunk before connecting a robot:

```bash
flowedge-lerobot-package /path/to/trained_smolvla deploy/smolvla
flowedge-lerobot-preflight deploy/smolvla \
  --task='pick up the cube' --iterations 1 --output smolvla-preflight.json
```

If the training base revision is known, add
`--vlm-revision REVISION` to pin its tokenizer/config assets.

The package separates source VLM and native expert weights and bundles the VLM
config/tokenizer assets; packaging may download those assets once. Pin the
revision used by training when it is known. The manifest records the resolved
revision and hashes of the bundled assets. The runtime drops
the redundant source PyTorch expert after loading. This path currently runs
the native expert on CPU and copies the VLM prefix cache to CPU. Do not use
`--device cuda` with this SmolVLA package. The no-robot preflight may take
minutes on a CPU and does not establish Jetson latency or robot task success.

For a calibrated SO-100, LeRobot's rollout command uses the same package:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=deploy/smolvla \
  --device=cpu \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task='pick up the cube' \
  --duration=10
```

Replace the port, camera names, resolution, and task with the trained policy's
contract. An SO-101 uses `so101_follower` instead. For Python applications,
`load_policy('deploy/smolvla')` returns an object with `reset()`,
`input_features`, and `select_action(observation, task=...)`. The physical
rollout remains subject to the observed preflight latency and the robot's
calibration and stop controls. A fast local Jetson SmolVLA remains unfinished;
the action expert needs a CUDA implementation and a measured whole-policy run.
