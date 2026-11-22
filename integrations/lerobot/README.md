# FlowEdge + LeRobot Integration

An optimized deployment adapter for [LeRobot](https://github.com/huggingface/lerobot) policies. LeRobot manages model training, dataset handling, vision pipelines, and robot hardware drivers; FlowEdge executes the compute-intensive action-generation head natively.

---

## Use Cases

- **LeRobot Diffusion Policy**: Accelerate the 1D convolutional U-Net action head using FlowEdge's native C++ runtime with zero hot-path allocations.
- **SmolVLA (Hybrid Execution)**: Run the Vision-Language Model in PyTorch while executing the action expert natively in FlowEdge.
- **Hardware Deployment**: Package trained checkpoints for real-robot deployment via standard `lerobot-rollout`.

---

## Installation

Requires Python 3.10+ (Python 3.12+ recommended) and LeRobot `>=0.6.1,<0.7`:

```bash
# Install FlowEdge and the LeRobot integration with robot extras
python -m pip install flowedge-0.1.2-*.whl
python -m pip install "integrations/lerobot[so101,smolvla]"
```

- For Diffusion Policy only, `[so100]` or `[so101]` is sufficient. Add `[smolvla]` for SmolVLA models.
- The prebuilt wheels provide CPU execution. For GPU acceleration, build FlowEdge with `-DFLOWEDGE_BACKEND=cuda` in an environment with CUDA-enabled PyTorch.

---

## Python API

### 1. Diffusion Policy

Convert the trained LeRobot checkpoint once:

```bash
python -m flowedge_dev pipeline convert \
  outputs/train/my_policy/checkpoints/last/pretrained_model \
  models/my_policy.flowedge.safetensors --arch diffusion
```

Package and execute the policy:

```python
from flowedge_lerobot import load_policy, package_policy

# Package model with normalization parameters
package = package_policy(
    "outputs/train/my_policy/checkpoints/last/pretrained_model",
    "deploy/my_policy",
    converted_checkpoint="models/my_policy.flowedge.safetensors",
)

policy = load_policy(package, device="cpu")
policy.reset()

# Sample next action from current observation
action = policy.select_action(observation)
```

To predict an entire unconsumed action chunk directly:
```python
chunk = policy.predict_action_chunk(observation)  # Shape: (1, action_steps, action_dim)
```

### 2. SmolVLA

Package directly from the trained LeRobot checkpoint directory:

```python
package = package_policy("outputs/train/smolvla/pretrained_model", "deploy/smolvla")
policy = load_policy(package, device="cpu")
policy.reset()

action = policy.select_action(observation, task="pick up the cube")
```

---

## Deployment CLI (`flowedge-edge`)

The `flowedge-edge` command groups common deployment utilities:

```bash
# View available deployment commands
flowedge-edge --help

# Run diagnostic preflight without connecting a robot
flowedge-edge preflight deploy/my_policy --device cpu --output preflight.json

# Run end-to-end benchmark comparison against PyTorch
flowedge-edge bench models/my_policy.flowedge.safetensors \
  --source outputs/train/my_policy/checkpoints/last/pretrained_model \
  --revision <source-revision> --observations observations.npz \
  --device cpu --output comparison.json
```

---

## Physical Robot Rollout

Deploy on hardware using LeRobot's standard rollout runner:

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=deploy/my_policy \
  --device=cpu \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="place the object in the bin" \
  --duration=30
```

For full rollout configurations and deadline policies, see the [LeRobot Deployment Guide](../../docs/guides/lerobot.md).
