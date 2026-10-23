# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">FlowEdge is a C++23 inference runtime for robotics policies with fixed memory, deterministic execution, and explicit deadline handling. It runs two heads: <strong>flow matching</strong> (a short ODE from noise to action) and <strong>Diffusion Policy</strong> (DDIM on a Conv1D U-Net). LeRobot still owns training, cameras, and the robot driver.</p>

```{raw} html
<ul class="fe-chips">
  <li>C++23 Core</li>
  <li>CPU · CUDA</li>
  <li>Tenstorrent preview</li>
  <li>LeRobot adapter</li>
</ul>

<div class="fe-metrics">
  <div class="fe-metric">
    <span class="fe-metric-value">851 vs 1409</span>
    <span class="fe-metric-label">CPU DDIM p50 · ms · threads=1</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">131 vs 345</span>
    <span class="fe-metric-label">CUDA DDIM p50 · ms · GTX 1650</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">malloc = 0</span>
    <span class="fe-metric-label">Supported hot paths after load</span>
  </div>
</div>

<nav class="fe-jump" aria-label="Start here">
  <a class="fe-jump-card" href="getting-started.html">
    <span class="fe-jump-kicker">01</span>
    <strong>Get started</strong>
    <span>Build Core, convert a mix, run the first sample.</span>
  </a>
  <a class="fe-jump-card" href="python-quickstart.html">
    <span class="fe-jump-kicker">02</span>
    <strong>Python</strong>
    <span>Wheel, Engine, predict, and DDIM from CPython.</span>
  </a>
  <a class="fe-jump-card" href="guides/lerobot.html">
    <span class="fe-jump-kicker">03</span>
    <strong>LeRobot</strong>
    <span>Plugin, period, and <code>--on-miss</code> around the native head.</span>
  </a>
  <a class="fe-jump-card" href="performance.html">
    <span class="fe-jump-kicker">04</span>
    <strong>Performance</strong>
    <span>Fair compare vs PyTorch. Not a 10 ms loop.</span>
  </a>
  <a class="fe-jump-card" href="architecture/overview.html">
    <span class="fe-jump-kicker">05</span>
    <strong>Architecture</strong>
    <span>Arena, kernels, CUDA residency, two product heads.</span>
  </a>
  <a class="fe-jump-card" href="api/c-abi.html">
    <span class="fe-jump-kicker">06</span>
    <strong>C API</strong>
    <span><code>fe_engine_*</code> load, sample, and miss policy.</span>
  </a>
</nav>
```

## News

**September 2026**

```{raw} html
<ul class="fe-news">
  <li class="fe-news-item">
    <span class="fe-news-date">19</span>
    <p>Repo lives at <a href="https://github.com/reforcemind/FlowEdge">reforcemind/FlowEdge</a>. <a href="https://github.com/reforcemind/FlowEdge/releases/tag/v0.1.2">v0.1.2</a> CPU wheels are <code>flowedge-0.1.2-*</code>. Not PyPI. CUDA stays a source backend.</p>
  </li>
  <li class="fe-news-item">
    <span class="fe-news-date">19</span>
    <p>Split-K CUDA Conv1D on horizon 8 (<a href="https://github.com/reforcemind/FlowEdge/pull/183">#183</a>). GTX 1650 matched DDIM replay stays <strong>131 vs 345 ms</strong> vs PyTorch CUDA. Not a 10 ms loop.</p>
  </li>
  <li class="fe-news-item">
    <span class="fe-news-date">18</span>
    <p>Device-resident CUDA after load: flow (<a href="https://github.com/reforcemind/FlowEdge/pull/167">#167</a>), Diffusion Policy (<a href="https://github.com/reforcemind/FlowEdge/pull/168">#168</a>, <a href="https://github.com/reforcemind/FlowEdge/pull/169">#169</a>), Mamba (<a href="https://github.com/reforcemind/FlowEdge/pull/173">#173</a>). Same-process CUDA vs PyTorch CUDA (<a href="https://github.com/reforcemind/FlowEdge/pull/170">#170</a>). <code>--device cuda</code> period rollouts (<a href="https://github.com/reforcemind/FlowEdge/pull/171">#171</a>).</p>
  </li>
  <li class="fe-news-item">
    <span class="fe-news-date">17</span>
    <p><a href="https://github.com/reforcemind/FlowEdge/releases/tag/v0.1.1">v0.1.1</a> CPU wheels (manylinux, Windows, macOS ARM). Not on PyPI. <code>--on-miss hold|drop|raise</code> (<a href="https://github.com/reforcemind/FlowEdge/pull/152">#152</a>). CPU fair-compare <strong>851 vs 1409 ms</strong>, <code>threads=1</code>.</p>
  </li>
</ul>
```

<p class="fe-inline-links"><a href="https://github.com/reforcemind/FlowEdge/releases">GitHub Releases</a> · <a href="performance.html">Performance</a> · <a href="https://github.com/reforcemind/FlowEdge">Source</a></p>

## Who should use FlowEdge

A robot stack already has a trainer, perception pipeline, and hardware driver.
Use FlowEdge when you want to keep those pieces and run a supported action head
in a small native runtime with explicit memory and deadline behavior.

| Your situation | What FlowEdge does |
|---|---|
| A trained LeRobot Diffusion Policy | Packages its matching native action head while retaining the source RGB encoder and processors. |
| A compatible SmolVLA checkpoint | Experimental hybrid: LeRobot runs the VLM; FlowEdge runs the CPU action expert. |
| A custom robot or simulator | Exposes a Python policy API or lower-level C++/C interfaces; the caller keeps control of observations and actions. |

The practical advantage is a specialized action-inference path: fixed-shape
native kernels and a sized runtime arena. The optional period runner is a
simulator-only smoke test with explicit `hold`, `drop`, or `raise` behavior;
those deadline results do not apply to physical hardware. `load_policy()` alone
does not schedule a robot loop or enforce a deadline. FlowEdge does not replace
model training, general-purpose graph compilers, robot drivers, joint limits,
or an emergency stop.

## Start from Python

The optional LeRobot integration packages a trained policy and exposes a
single action API. Diffusion Policy also needs its matching converted
FlowEdge checkpoint; the [converter guide](guides/converter) shows that step.

```python
from flowedge_lerobot import load_policy, package_policy

package = package_policy(
    "outputs/train/my_policy/checkpoints/last/pretrained_model",
    "deploy/my_policy",
    converted_checkpoint="models/my_policy.flowedge.safetensors",
)
policy = load_policy(package, device="cpu")
policy.reset()
# Build this mapping from your camera/state reader; keys and shapes must match
# policy.input_features.
action = policy.select_action(observation)
```

For SmolVLA, omit `converted_checkpoint` and pass the task instruction to
`select_action`. The result is a postprocessed tensor shaped
`(1, action_dim)`, not a named joint mapping for a robot driver. Its VLM is not
native and the expert currently runs on CPU. See the
[LeRobot guide](guides/lerobot) for installation and rollout details.

## What the results mean

On one matched PushT Diffusion Policy checkpoint, FlowEdge measured 851 ms
versus PyTorch's 1409 ms on CPU (`threads=1`), and 131 ms versus 345 ms on a
desktop GTX 1650. These are policy replay medians, not Jetson/SO-100 results
or a 10 ms control-loop claim. Check the [benchmark conditions](performance)
before comparing results.

## Explore

[Getting started](getting-started) · [Python API](python-quickstart) ·
[LeRobot deployment](guides/lerobot) · [Performance](performance) ·
[Architecture](architecture/overview) · [Capabilities](capabilities) ·
[Relay](guides/relay-quickstart)

```{toctree}
:hidden:
:maxdepth: 1
:caption: Start
Overview <self>
getting-started
python-quickstart
guides/lerobot
performance
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Guides
guides/converter
guides/diffusion-policy
guides/transformer-backbone
guides/model-import
guides/checkpoint-preflight
guides/verification
guides/policy-evaluation
guides/edge-benchmarks
guides/relay-quickstart
guides/action-delivery
guides/deadline-flow
guides/deadline-profile
guides/observability
guides/cooperative-jobs
guides/generic-job-daemon
guides/sanitizers
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Architecture
architecture/overview
architecture/memory
architecture/loader
architecture/kernels
architecture/cuda
architecture/backbones
architecture/heads
architecture/deployment-profile
architecture/cooperative-execution
architecture/model-porting
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: API
api/c-abi
api/python
capabilities
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Contribute
contributing
guides/add-a-head
guides/add-a-backbone
benchmarks
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project
roadmap
product-direction
tenstorrent-program
decisions/index
FlowEdge Relay <ecosystem/relay-proposal>
LeRobot edge-inference RFC draft <ecosystem/lerobot-rfc>
```
