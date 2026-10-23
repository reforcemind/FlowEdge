# SmolVLA recorded-cache replay

| File group | Evidence |
| --- | --- |
| `eslab-frame-000000.capture.*` | One recorded public LeRobot v3 frame, task tokens, and seeded flow noise. |
| `*.source-action.*` | Upstream `SmolVLAPolicy` 50 by 6 action chunk from that capture. |
| `*.expert-reference.*`, `*.trajectory-reference.*` | Upstream VLM K/V cache plus one expert velocity and ten-step Euler target. |
| `*.cached-expert.json`, `*.cached-trajectory.json` | FlowEdge replay reports for the cached expert boundary. |
| `*.hybrid-action.json` | Source LeRobot preprocessing/VLM-prefix provider plus native FlowEdge action-expert parity for the complete 50 by 6 action chunk. |
| `*.hybrid-report.json` | Matched prepared-capture PyTorch and hybrid stage p50/p95/p99 measurements. |

The capture manifest pins the source dataset revision, license, parquet/video
digests, task, parquet and video frame indices, and noise seed. It maps the recorded top and wrist
cameras to two configured source keys without synthesizing, resizing, or
duplicating pixels.

The hybrid report establishes action parity across the source-owned
preprocessing/VLM-prefix boundary and the native FlowEdge action expert. It
does not establish a native FlowEdge VLM or preprocessing implementation,
end-to-end control-loop latency, control quality, or task success. The timing
report measures prepared-capture CPU stages only and excludes camera capture,
transport, controller delivery, and native VLM implementation.
