# LeRobot Deployment Adapter

FlowEdge integrates with [LeRobot](https://github.com/huggingface/lerobot) as an optimized native action head. LeRobot manages training, cameras, observation preprocessing, and hardware drivers; FlowEdge executes the latency-sensitive policy head (Diffusion Policy Conv1D U-Net or SmolVLA action expert) within a pre-allocated C++ memory arena.

---

## Architecture & Boundary

| LeRobot | FlowEdge |
|---|---|
| Cameras, video streams, and image preprocessing | Native fixed-memory U-Net / action expert |
| Observation normalization and history tracking | Deterministic ODE / DDIM denoising |
| Robot hardware drivers, joint limits, and emergency stops | Action chunk prediction and missed-deadline handling |
| Model training and dataset management | Converted `.safetensors` weight loading |

The executable action slice is defined as:
```text
actions[observation_steps - 1 : observation_steps - 1 + action_steps]
```

---

## Installation

```bash
# Install FlowEdge and the LeRobot integration with SO-101 support
python -m pip install .
python -m pip install -e "integrations/lerobot[so101]"

# Run adapter test suite
python -m unittest discover -s integrations/lerobot/tests
```

- **Requirements**: Python 3.10+ (Python 3.12+ recommended) and LeRobot `>=0.6.1,<0.7`.
- **Backend**: Defaults to CPU. For CUDA acceleration, build FlowEdge with `-DFLOWEDGE_BACKEND=cuda` and run within a CUDA-enabled PyTorch environment.
- **Zero-Copy CUDA Transfer**: The CUDA plugin uses DLPack end-to-end for condition features, diffusion noise, and queued action chunks. Only the final action returned to the robot driver transfers back to host memory.

---

## Packaging a Trained Policy

### 1. Diffusion Policy

Convert the trained LeRobot checkpoint once:

```bash
python -m flowedge_dev pipeline convert \
  outputs/train/diffusion_so101/checkpoints/last/pretrained_model \
  models/so101.flowedge.safetensors --arch diffusion
```

Package the model with its normalization parameters:

```python
from flowedge_lerobot import load_policy, package_policy

package_policy(
    "outputs/train/diffusion_so101/checkpoints/last/pretrained_model",
    "deploy/so101",
    converted_checkpoint="models/so101.flowedge.safetensors",
)

# Load packaged policy
policy = load_policy("deploy/so101", device="cpu")
policy.reset()

# Query action from current observation mapping
action = policy.select_action(observation)
```

To predict an entire unconsumed action chunk directly, call:
```python
chunk = policy.predict_action_chunk(observation)  # Shape: (1, action_steps, action_dim)
```

### 2. SmolVLA

For SmolVLA models, package directly from the trained LeRobot directory (no separate conversion needed):

```python
package_policy("trained_smolvla", "deploy/smolvla")
policy = load_policy("deploy/smolvla", device="cpu")
policy.reset()

action = policy.select_action(observation, task="pick up the cube")
```

---

## Deployment on Physical Robots

For physical robot control, deploy using LeRobot's standard `lerobot-rollout`:

```bash
# Package the policy
flowedge-lerobot-package \
  outputs/train/diffusion_so101/checkpoints/last/pretrained_model \
  deploy/so101 \
  --converted-checkpoint models/my_so101.flowedge.safetensors

# Execute rollout on hardware
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

### No-Robot Preflight Diagnostic

Before connecting hardware, verify package integrity, startup latency, and memory footprint on the deployment machine:

```bash
flowedge-lerobot-preflight deploy/so101 \
  --iterations 5 --output flowedge-preflight.json
```

This diagnostic runs synthetic observations through preprocessing, encoder, and native action-head inference, logging latency, memory high-water mark, and CUDA residency.

---

## Control Loop Period & Miss Policies

When running rollouts under explicit control periods, specify `--on-miss` to define behavior if inference exceeds the allocated period:

| `--on-miss` Policy | Behavior on Period Overrun |
|---|---|
| `hold` | Repeats the previous valid action (or zeros before the first completed inference). |
| `drop` | Skips publishing an action for the overrun cycle. |
| `raise` | Halts execution immediately and raises `DeadlineMissed`. |

Example rollout test:

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 4 --period-ms 10 --on-miss hold
```

For CUDA execution:

```bash
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 20 --threads 1 --period-ms 10 --on-miss hold --device cuda
```

If GPU memory allocation fails, FlowEdge retains CPU execution unless `FLOWEDGE_CUDA_REQUIRED=1` is set to fail closed.

For full benchmark figures and hardware comparisons, see [Performance Documentation](../performance).
