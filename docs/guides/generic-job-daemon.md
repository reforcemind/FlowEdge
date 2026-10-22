# Generic job daemon

`flowedge-jobd` runs bounded cooperative jobs as a managed local service. The first built-in route is
Mamba streaming; custom iterative, streaming, and speculative backends use the same `JobService`
contract in an embedding process.

```{image} ../_static/figures/jobs.svg
:alt: JobClient, shared rings, EDF pool, Mamba lane
:class: fe-fig
```

## Run the complete demo

```bash
cmake -S . -B build-relay -DCMAKE_BUILD_TYPE=Release \
  -DFLOWEDGE_RELAY=ON
cmake --build build-relay --config Release -j
FLOWEDGE_BUILD_DIR="$PWD/build-relay" \
  ./tools/scripts/job_demo.sh models/mamba_flow.safetensors
```

The script submits interactive and critical streams, drains and resumes one lane, validates traces
and metrics, then shuts down through the control plane.

## Run it manually

Terminal 1:

```bash
./build-relay/src/relay/flowedge-jobd \
  --model models/mamba_flow.safetensors --create \
  --workers 2 --threads 0 --max-tokens 512 \
  --trace jobs.trace --metrics-prometheus jobs.prom
```

Terminal 2:

```bash
./build-relay/src/relay/flowedge-jobctl status
./build-relay/src/relay/flowedge-jobctl mamba \
  --model models/mamba_flow.safetensors --tokens 1,2,3,4 \
  --session 42 --generation 1 --class interactive
./build-relay/src/relay/flowedge-jobctl drain --worker 0
./build-relay/src/relay/flowedge-jobctl resume --worker 0
./build-relay/src/relay/flowedge-jobctl shutdown
```

`flowedge-jobctl mamba` loads model identity for diagnostics. Long-lived producers should retain
their route metadata and use `JobClient` directly.

## Operations

| Command | Effect |
|---|---|
| `status` | Queue, lane, QoS, failure, quarantine, and transport state |
| `drain --worker N` | Stop new dispatch to a lane; migrate compatible active state when possible |
| `resume --worker N` | Return a fully drained lane to service |
| `recover --worker N` | Clear a quarantined, drained lane and return it to service |
| `shutdown` | Publish an acknowledgement, then stop |

Data and administration use different SPSC ring pairs. A blocked data consumer therefore cannot
prevent status or lane administration. Each pair still requires one producer and one consumer.

## Production settings

| Option | Contract |
|---|---|
| `--workers 1..8` | One mutable adapter instance and outer thread per lane |
| `--threads 0..8` | Core background threads inside each lane |
| `--work-quantum N` | Cancellation and migration boundary |
| `--streaming-unit-ns N` | Calibrated deadline cost per token; `0` disables predictive rejection |
| `--admission-reserve-ns N` | Fixed safety margin added to predicted work |
| `--interactive-reserve N` | Slots reserved from best-effort traffic; default `2` |
| `--critical-reserve N` | Slots reserved from lower-class traffic; default `1` |
| `--failure-threshold N` | Consecutive failures before quarantine; default `3`, `0` disables |
| `--placement compact\|spread` | NUMA-aware outer-worker placement |

Deadlines remain primary; service class breaks equal-deadline ties. Reserved slots prevent lower
classes from consuming the whole queue. A quarantined lane must drain before explicit recovery.
Lifecycle traces contain metadata, not model payloads. See [Observability](observability).
