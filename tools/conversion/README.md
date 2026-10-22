# Checkpoint conversion

Convert supported model weights once, before loading them in FlowEdge:

```bash
python -m flowedge_dev pipeline convert <source> models/policy.safetensors \
  --arch mamba|diffusion|transformer
```

Inspect a LeRobot policy directory with `python -m flowedge_dev pipeline
inspect <policy-directory> --json`. Supported architectures and conversion
limits are documented in [Model import](../../docs/guides/model-import.md).
