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
python -m pip install -e "integrations/lerobot[dp]"
python -m unittest discover -s integrations/lerobot/tests
```

The adapter supports Python 3.12+, LeRobot 0.6, and only `cpu` or `cuda`.
Unavailable CUDA is an error; it is never silently replaced with CPU.
The CUDA plugin uses DLPack end to end for the encoded condition, diffusion noise, and queued
action chunk. Only the final action handed to the robot driver crosses back to host memory.

```python
from flowedge_lerobot import FlowEdgeDiffusionPolicy
policy = FlowEdgeDiffusionPolicy.from_checkpoint(
    "models/diffusion_pusht.flowedge.safetensors", threads=4
)
```

## Period loop (sim, Jetson, SO-100)

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
inference benchmark. SO-101 task success still requires a robot adapter with
`reset` / `observe` / `send_action` / `stop` and an `encode_condition`
callback. CI uses a fake robot; ARM64 CI alone is not a board claim.

`--policy.type=flowedge` is DP. `--policy.type=flowedge_smolvla` is the native expert plus a LeRobot VLM cache — not a native VLM. Tenstorrent is a separate backend, not this plugin.
