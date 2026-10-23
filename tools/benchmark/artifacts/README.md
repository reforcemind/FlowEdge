# Benchmark artifacts

- `policy/` — matched replay, thread sweep, period-loop miss counts
- `smolvla/` — cached-expert / hybrid captures; not native VLM
- `transformer/` — GPT-2 incubator parity

```bash
python -m flowedge_dev bench policy ...
python -m flowedge_dev pipeline rollout ...
```
