# Checkpoint Converter

FlowEdge loads models in fixed, standardized tensor layouts: `backbone.*` for representation backbones, `flow.*` for flow-matching heads, and `dp.*` for Diffusion Policy heads. The converter maps upstream PyTorch and Hugging Face weights into these layouts and serializes them as `.safetensors`.

```{image} ../_static/figures/convert.svg
:alt: Workflow: convert PyTorch/HuggingFace checkpoints to FlowEdge safetensors
:class: fe-fig
```

---

## Supported Architectures

### 1. Mamba Flow Matching

Converts a Mamba state-space backbone with a flow-matching action head:

```bash
python -m flowedge_dev pipeline convert <source_path> models/mamba_flow.safetensors --arch mamba
```

To extract only the action head (for use with an external vision or state encoder):

```bash
python -m flowedge_dev pipeline convert models/full.safetensors models/head.safetensors \
  --arch mamba --component head --dtype bf16
```

### 2. LeRobot Diffusion Policy

Converts a trained LeRobot Diffusion Policy checkpoint (`model.safetensors` and `config.json`):

```bash
# Download reference checkpoint
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir models/diffusion_pusht

# Convert checkpoint
python -m flowedge_dev pipeline convert models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion
```

The converter maps `ConditionalUnet1D`, standardizes tensor naming, records horizon and timestep metadata, and embeds dataset action un-normalization statistics (`action.min` and `action.max`).

### 3. Transformer Decoder (Test Fixture)

Converts a GPT-2 style decoder block for testing:

```bash
hf download sshleifer/tiny-gpt2 --local-dir models/tiny-gpt2
python -m flowedge_dev pipeline convert models/tiny-gpt2 \
  models/tiny-gpt2.flowedge.safetensors --arch transformer
```

---

## Conversion Options

| Flag | Default | Description |
|---|---|---|
| `--arch` | Required | Target architecture: `mamba`, `diffusion`, or `transformer`. |
| `--component` | `all` | Elements to preserve: `all`, `backbone`, or `head`. |
| `--dtype` | `f32` | Weight precision: `f32` or `bf16`. In `bf16` mode, large 2D weight matrices are narrowed while normalization and biases remain FP32. |
| `--processor` | None | Explicit path to `policy_postprocessor.json` or `dataset_stats.json` for action un-normalization parameters. |

---

## Validation Contract

During conversion, FlowEdge enforces structural validation:
- All required tensor weights for the requested architecture must be present and match expected dimensions.
- Tensor dimensions are verified against checkpoint metadata to prevent shape mismatches at runtime.
- Normalization statistics are checked for finiteness and monotonicity.
