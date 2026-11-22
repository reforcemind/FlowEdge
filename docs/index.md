# FlowEdge

<img class="fe-hero-img" src="_static/hero.png" alt="FlowEdge" />

<p class="fe-lede">FlowEdge is a C++23 inference runtime for robotics policies engineered for fixed memory, deterministic execution, and explicit deadline handling. It natively executes two primary heads: <strong>flow matching</strong> (a short ODE from noise to action) and <strong>Diffusion Policy</strong> (DDIM on a Conv1D U-Net). LeRobot continues to own training, observation pipelines, and robot hardware drivers.</p>

```{raw} html
<ul class="fe-chips">
  <li>C++23 Core</li>
  <li>CPU · CUDA</li>
  <li>Zero hot allocations</li>
  <li>LeRobot adapter</li>
</ul>

<div class="fe-metrics">
  <div class="fe-metric">
    <span class="fe-metric-value">851 vs 1409</span>
    <span class="fe-metric-label">CPU DDIM p50 · ms (threads=1)</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">131 vs 345</span>
    <span class="fe-metric-label">CUDA DDIM p50 · ms (GTX 1650)</span>
  </div>
  <div class="fe-metric">
    <span class="fe-metric-value">malloc = 0</span>
    <span class="fe-metric-label">Zero hot-path allocations</span>
  </div>
</div>

<nav class="fe-jump" aria-label="Start here">
  <a class="fe-jump-card" href="getting-started.html">
    <span class="fe-jump-kicker">01</span>
    <strong>Get started</strong>
    <span>Build Core, convert checkpoints, run first sample.</span>
  </a>
  <a class="fe-jump-card" href="python-quickstart.html">
    <span class="fe-jump-kicker">02</span>
    <strong>Python</strong>
    <span>Install wheels, load Engine, sample actions.</span>
  </a>
  <a class="fe-jump-card" href="guides/lerobot.html">
    <span class="fe-jump-kicker">03</span>
    <strong>LeRobot</strong>
    <span>Adapter, rollout, and deadline miss policies.</span>
  </a>
  <a class="fe-jump-card" href="performance.html">
    <span class="fe-jump-kicker">04</span>
    <strong>Performance</strong>
    <span>Matched end-to-end benchmarks vs PyTorch.</span>
  </a>
  <a class="fe-jump-card" href="architecture/overview.html">
    <span class="fe-jump-kicker">05</span>
    <strong>Architecture</strong>
    <span>Memory arena, kernels, CUDA residency, design.</span>
  </a>
  <a class="fe-jump-card" href="api/c-abi.html">
    <span class="fe-jump-kicker">06</span>
    <strong>C API</strong>
    <span>ABI functions, lifecycle, and zero-overhead calls.</span>
  </a>
</nav>
```

## Key Highlights

- **Deterministic Memory**: All weight and scratch memory is allocated and pre-faulted once at engine initialization. Hot-path execution performs zero dynamic allocations.
- **Explicit Deadline Handling**: Control loops can select explicit behaviors (`hold`, `drop`, or `raise`) when policy inference exceeds the allocated control period.
- **Matched Parity**: Native kernels maintain bit-level and ULP parity with upstream PyTorch and LeRobot references.
- **Modular Ecosystem Fit**: FlowEdge operates as a dedicated policy accelerator, leaving perception, data collection, joint safety limits, and hardware control to frameworks like LeRobot.

## Who should use FlowEdge

| Use Case | How FlowEdge helps |
|---|---|
| **Trained LeRobot Diffusion Policy** | Executes the compute-intensive action head with low latency and fixed memory while keeping source RGB encoders and processors. |
| **SmolVLA Checkpoints** | Hybrid path: LeRobot runs the VLM; FlowEdge executes the action expert. |
| **Custom Robots & Simulators** | Provides low-level C and C++ ABIs alongside high-level Python bindings, with complete caller control over observation buffers and action chunks. |

## Quick Python Example

The LeRobot adapter packages a trained checkpoint and provides an inference API:

```python
from flowedge_lerobot import load_policy, package_policy

# Package converted checkpoint with normalization statistics
package = package_policy(
    "outputs/train/my_policy/checkpoints/last/pretrained_model",
    "deploy/my_policy",
    converted_checkpoint="models/my_policy.flowedge.safetensors",
)

policy = load_policy(package, device="cpu")
policy.reset()

# Query action from current sensor observation
action = policy.select_action(observation)
```

## Benchmark Summary

Matched PushT Diffusion Policy replay (median latency, lower is better):

| Backend | Configuration | FlowEdge | PyTorch / LeRobot | Speedup |
|---|---|---:|---:|---:|
| CPU | 1 thread | **851 ms** | 1409 ms | 1.66× |
| CPU | 4 threads | **633 ms** | 819 ms | 1.29× |
| CUDA | GeForce GTX 1650 | **131 ms** | 345 ms | 2.63× |

All comparisons use identical observations, normalization statistics, initial noise, and 10-step DDIM schedules. See [Performance](performance) for detailed breakdowns, hardware specs, and reproduction steps.

## Documentation Index

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
