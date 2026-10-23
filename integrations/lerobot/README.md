# FlowEdge + LeRobot

An optional deployment plugin for supported LeRobot policies. LeRobot keeps
training, camera processing, robot drivers, joint limits, and emergency stops;
FlowEdge replaces the supported action-generation step.

## Who should use this

- You have a compatible LeRobot Diffusion Policy and want its action head in
  FlowEdge's native runtime.
- You have a trained SmolVLA and want to try the experimental hybrid path
  (LeRobot VLM plus FlowEdge action expert).
- You want a measured local action-inference path, not a general VLA host.

FlowEdge's current published comparisons are PushT only: 851 vs 1409 ms on
CPU (`threads=1`), and 131 vs 345 ms on a desktop GTX 1650. They are not
Jetson or SO-100 performance results. The SmolVLA expert currently runs on CPU;
the VLM is not native.

## Install

Use Python 3.12+. Install the matching CPU FlowEdge wheel and this plugin's
robot extra in the same environment:

```bash
python -m pip install ./path/to/downloaded-flowedge-wheel.whl
python -m pip install "integrations/lerobot[so100,smolvla]"
```

Replace the example path with the CPython/platform wheel downloaded from the
[v0.1.2 release](https://github.com/reforcemind/FlowEdge/releases/tag/v0.1.2).
For Diffusion Policy only, `[so100]` is enough; add `smolvla` for SmolVLA. Use
`[so101]` instead of `so100` for an SO-101 follower. The LeRobot plugin supports LeRobot
`>=0.6.1,<0.7`. CUDA requires a separately built FlowEdge CUDA runtime; it is
not included in the published wheel.

## Python API

Package and load a supported policy. A Diffusion Policy requires its matching
converted checkpoint; SmolVLA is detected from its trained policy directory.

Create the Diffusion Policy checkpoint once from its trained LeRobot directory:

```bash
python -m flowedge_dev pipeline convert \
  outputs/train/my_policy/checkpoints/last/pretrained_model \
  models/my_policy.flowedge.safetensors --arch diffusion
```

```python
from flowedge_lerobot import load_policy, package_policy

package = package_policy(
    "outputs/train/my_policy/checkpoints/last/pretrained_model",
    "deploy/my_policy",
    converted_checkpoint="models/my_policy.flowedge.safetensors",
)
policy = load_policy(package, device="cpu")
policy.reset()
# Application sketch: provide the latest observation mapping from your
# camera/state reader, matching policy.input_features.
action = policy.select_action(observation)
```

For SmolVLA, omit `converted_checkpoint` and pass the language instruction:

```python
package = package_policy("outputs/train/smolvla/pretrained_model", "deploy/smolvla")
policy = load_policy(package, device="cpu")
action = policy.select_action(observation, task="pick up the cube")
```

`select_action` accepts an unbatched LeRobot observation and returns a
postprocessed tensor shaped `(1, action_dim)`. It is not the named joint
mapping accepted by a robot driver; LeRobot's rollout path performs that
mapping. `policy.input_features` lists the expected observation keys and
shapes. SmolVLA CUDA execution is not supported yet. For a remote SmolVLA base,
pass `vlm_revision="<commit>"` to pin its tokenizer/config assets; the package
manifest records the resolved revision and asset hashes.

## Robot rollout

Use LeRobot's standard `lerobot-rollout` with the packaged policy. The Python
action API does not schedule a control period. `flowedge-lerobot-rollout` is a
simulator-only smoke test; its period and miss-policy results do not apply to
physical hardware. Commission physical motion with an unloaded arm and a short
duration; see the [deployment guide](../../docs/guides/lerobot.md).

The low-level `flowedge.Engine`, native sampling calls, benchmarks, and
verification tools remain available to advanced users and maintainers.
