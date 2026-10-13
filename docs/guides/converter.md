# Converter

FlowEdge loads fixed tensor layouts: `backbone.*` for the backbone, `flow.*` for
the flow head, and `dp.*` for the Diffusion Policy head. The converter maps a
PyTorch or Hugging Face checkpoint into those layouts.

```{image} ../_static/figures/convert.svg
:alt: HF/torch to map names to validate to Engine
:class: fe-fig
```

```bash
python -m flowedge_dev pipeline convert <source> models/out.safetensors --arch mamba
```

Convert a GPT-2-style decoder (`model_type: gpt2`) into the Transformer
fixture. `sshleifer/tiny-gpt2` is the CI fixture; `openai-community/gpt2` is optional:

```bash
hf download sshleifer/tiny-gpt2 --local-dir models/tiny-gpt2
python -m flowedge_dev pipeline convert models/tiny-gpt2 \
  models/tiny-gpt2.flowedge.safetensors --arch transformer
```

This maps learned positions, affine LayerNorm, GELU MLPs, and GPT-2 fused QKV
attention. It does not convert the tokenizer, LM head, visual encoder, or the
SmolVLA action expert.

Extract only the action head for an external encoder:

```bash
python -m flowedge_dev pipeline convert models/full.safetensors models/head.safetensors \
  --arch mamba --component head --dtype bf16
```

`--component backbone` drops the head; `all` preserves both. Head-only output validates the four
required flow projections and is directly loadable through `fe_engine_sample_condition`.

## A mapping

An architecture is one function in the `ARCH` registry. It takes the source state dict and returns a dict keyed by FlowEdge names.

```python
def mamba(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace("backbone.embedding.weight", "backbone.embeddings.weight")
        if k.startswith("backbone.") or k.startswith("flow."):
            out[k] = v
    return out

ARCH = {"mamba": mamba}
```

FlowEdge uses PyTorch `[out, in]` linear layout, so most weights need no transpose. Rename keys, drop what the engine does not use, and transpose only where the source layout differs.

## Validation and normalization

`REQUIRED_BACKBONE` and `REQUIRED_FLOW` list the tensors the model constructors look for. The
converter checks the selected component and stops if it is incomplete. Keep these lists in sync
with the constructors.

With `--dtype bf16`, only two-dimensional matmul weights are narrowed. Embeddings, normalization,
biases, convolution weights, `A_log`, and `D` remain F32 because the current kernels and model
constructors require them in that format.

If the source normalizes actions with dataset stats, carry those stats through and un-normalize the sampled action. Otherwise the output stays in normalized space.

## LeRobot Diffusion Policy

Pass a LeRobot 0.6 Diffusion Policy directory so the converter can read
`model.safetensors`, `config.json`, and modern processor sidecars. The public
PushT checkpoint is the pinned parity fixture, not a hard-coded shape:

```bash
hf download lerobot/diffusion_pusht --revision 84a7c23178445c6bbf7e1a884ff497017910f653 \
  --local-dir models/diffusion_pusht
python -m flowedge_dev pipeline convert models/diffusion_pusht \
  models/diffusion_pusht.flowedge.safetensors --arch diffusion
```

This maps `ConditionalUnet1D`, shortens tensor names for the fixed loader,
stores horizon/dimension/scheduler metadata, and preserves action min/max. It
drops the RGB encoder by design. Action dimension, horizon, observation steps,
action steps, and U-Net widths come from the checkpoint. Unsupported schedules,
prediction modes, normalization, geometry, or U-Net variants fail with a
specific conversion error.

Source: `convert/convert.py`. See [ADR 0006](../decisions/0006-obs-encoder-out-of-scope).
