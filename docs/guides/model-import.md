# Model import paths

```{image} ../_static/figures/convert.svg
:alt: convert once
:class: fe-fig
```

| Artifact | Path | Runtime | Allocation contract | Status |
|---|---|---|---|---|
| Mamba / supported flow checkpoint | `tools/conversion/convert.py` → `.safetensors` | FlowEdge Core | Zero after setup | Supported |
| LeRobot `diffusion_pusht` directory | `policy_inspect.py` → converter → `flowedge-lerobot` | Core + companion | Zero in native inference after setup | Supported |
| Fixed-shape one-I/O ONNX policy | `flowedge-onnx` | Optional ONNX Runtime | External runtime; measure separately | Supported adapter |
| ExecuTorch `.pte` | Future companion adapter | ExecuTorch | External runtime; measure separately | Planned |

Use the smallest path that preserves your policy. Native conversion is the only path covered by
FlowEdge's zero-hot-path-allocation contract. The ONNX adapter validates one fixed-shape float32
input/output pair at setup, has no dependency from Core or Relay, and reports operator/provider
failures instead of silently falling back.

```bash
python tools/conversion/policy_inspect.py policy_dir --json
python -m pip install -e 'integrations/onnx[runtime]'
```

The CI fixture exports a deterministic PyTorch policy and checks its ONNX Runtime output through
`OnnxAdapter.run_into`. Repeat that parity check for your model before a control loop. Dynamic shapes,
multiple I/O, and unsupported operators need a model-specific adapter or native converter rather than a
hidden graph runtime.
