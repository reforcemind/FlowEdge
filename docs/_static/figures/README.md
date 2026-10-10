# Figures

Excalidraw sources (`.excalidraw`) plus GitHub/Sphinx SVGs. GitHub does not render `.excalidraw`.

```bash
python docs/_static/figures/draw.py
```

Also writes `docs/_static/flowedge.gif` (flow-matching ODE) and
`docs/_static/matched_replay.gif` (one loop: CUDA 131 vs 345 ms stacked over
CPU 851 vs 1409 ms).
`python docs/_static/figures/draw.py perf` regenerates the replay GIF.

`purpose.svg` is Hardware / FlowEdge / LeRobot. `big-flow.svg` is convert → arena → flow or DP → period → motors (no backends; those live on `status.svg`). `relay.svg` is the optional IPC path under that flow. `policies.svg` is the two heads. `why-this.svg` is the runtime comparison table. Tenstorrent and Metal are one backend (TT-Metal).
