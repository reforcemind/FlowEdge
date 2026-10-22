# Getting started

FlowEdge runs supported policy heads on CPU or CUDA. It does not train policies
or replace the robot's controller and safety system. Pick the path that matches
what you already have:

```{image} _static/figures/purpose.svg
:alt: Hardware runs the loop, FlowEdge runs the policy, LeRobot trains and drives the robot
:class: fe-fig
```

| What you have / want | Start here |
|---|---|
| Run a model from Python | [Python quickstart](python-quickstart) |
| Use a LeRobot Diffusion Policy or SmolVLA package | [LeRobot guide](guides/lerobot) |
| Convert a supported checkpoint | [Converter guide](guides/converter) |
| Keep your own encoder and run a FlowEdge head | [Python API](api/python) |
| Split perception and control across local processes | [Relay quickstart](guides/relay-quickstart) |
| Compare full policy latency with PyTorch | [Policy benchmark guide](guides/edge-benchmarks) |

For CUDA build requirements and residency behavior, see the [CUDA guide](architecture/cuda).
