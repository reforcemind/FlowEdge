# Getting Started

FlowEdge is the deploy runtime, not the trainer. Convert a checkpoint, load it
once, sample **flow matching** or **Diffusion Policy**. Optional Relay sits
between processes when perception and control are split.

```{image} _static/figures/purpose.svg
:alt: Hardware runs the loop, FlowEdge runs the policy, LeRobot trains and drives the robot
:class: fe-fig
```

| Goal | Continue at |
|---|---|
| Python wheel (CPU) | [Python](python-quickstart) |
| Included flow-matching policy | [Sample an action](sample-an-action) |
| Diffusion Policy on Jetson / SO-100 / sim | [LeRobot adapter](guides/lerobot) |
| Your encoder, our flow or DP head | [External encoder](use-an-external-encoder) |
| Local IPC | [Relay](relay-end-to-end-demo) |
| Cached SmolVLA expert (not a product head) | [Transformer / SmolVLA](guides/transformer-backbone) |
| Flow matching vs Diffusion Policy vs PyTorch | [Performance](performance) |

GPT-2 conversion is a kernel incubator, not a product path.

## Install (CPU wheel)

Download the wheel for your platform from the [v0.1.2](https://github.com/reforcemind/FlowEdge/releases/tag/v0.1.2) Release assets. Filenames are `flowedge-0.1.2-*`. PyPI is not published.
Those wheels are CPU; CUDA needs `nvcc` ([CUDA](architecture/cuda)).

```bash
python -m pip install flowedge-0.1.2-*.whl
mkdir -p models
curl -L "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors" \
  -o models/mamba_flow.safetensors
python examples/core/flow_sample.py models/mamba_flow.safetensors euler 10
```

Full Python API: [Python](python-quickstart).

## From source

CMake 3.21+ and C++23. CI uses LLVM/Clang 23; GCC 13+ is supported.

```bash
git clone https://github.com/reforcemind/FlowEdge.git
cd FlowEdge
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

`FLOWEDGE_BACKEND=cuda` needs `nvcc` (WSL or MSVC on Windows) and tries to keep
Mamba, flow, and Diffusion Policy heads on device. A 4 GB card that cannot
allocate the U-Net falls back to CPU kernels instead of failing load. Check
`Engine.cuda_resident`. [CUDA](architecture/cuda).

## Get a model

```bash
mkdir -p models
wget -qO models/mamba_flow.safetensors \
  "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
```

LeRobot: `python -m flowedge_dev pipeline convert` ([converter](guides/converter)).

(sample-an-action)=
## Sample an action

```bash
./build/flow_sample models/mamba_flow.safetensors euler 10
```

Solver is `euler`, `heun`, or `rk4`. Last argument is NFE (how many velocity-net evaluations). That is the cost of flow matching: a short ODE, not a long denoiser. The published gate is ULP vs PyTorch (~1e-6 rel), not a policy p50.

Converted Diffusion Policy — plugin owns the LeRobot processor; Core owns the U-Net:

```bash
python -m pip install -e integrations/lerobot
python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors \
  --steps 10 --period-ms 10 --on-miss hold
```

`--on-miss`: `hold` last sent action, `drop` the send, or `raise`. Limits and e-stop stay in the robot adapter. On a physical Jetson, `scripts/edge_dp_rollout.sh` fails closed unless native CUDA weights are resident and writes host evidence to JSON. A 10 ms period on current PushT DDIM is a miss log, not a hit-rate claim ([performance](performance)).

(use-an-external-encoder)=
## Use an external encoder

Keep ResNet / VLM / your own encoder where it already runs. Pass the condition vector in; FlowEdge only integrates the head.

```bash
./build/external_flow_sample models/mamba_flow.safetensors
./build/streaming_snapshot models/mamba_flow.safetensors
```

Caller owns condition, noise, and output. Engine owns solver scratch.

(relay-end-to-end-demo)=
## Relay

Optional local IPC when perception and control are separate processes. Matched DP replay does not start Relay.

```bash
cmake -S . -B build-relay -DCMAKE_BUILD_TYPE=Release -DFLOWEDGE_RELAY=ON
cmake --build build-relay -j
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./scripts/relay_demo.sh models/mamba_flow.safetensors
```

[Relay quickstart](guides/relay-quickstart).
