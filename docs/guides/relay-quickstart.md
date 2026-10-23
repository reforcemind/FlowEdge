# Relay quickstart

```{image} ../_static/figures/relay.svg
:alt: Optional Relay flow from sensor or VLA through shared-memory rings to Core
:class: fe-fig
```

Local Relay lifecycle: start the daemon, submit a head request from another process, get a typed result, stop.

## Before you start

You need a C++23 compiler, CMake 3.21+, and a compatible flow-head checkpoint. Relay supports native
Windows and Linux. Shared memory is local to one host.

Build Core, Relay, tools, examples, and benchmarks:

```bash
cmake -S . -B build-relay -DCMAKE_BUILD_TYPE=Release \
  -DFLOWEDGE_RELAY=ON -DFLOWEDGE_BENCH=ON
cmake --build build-relay --config Release -j
```

On Windows PowerShell, executables end in `.exe`. Depending on the generator, Relay tools may be in
`build-relay/src/relay/`.

## Migrate a live Mamba stream

```bash
./build-relay/mamba_relay_stream models/mamba_flow.safetensors
```

The example runs two tokens, exports a checksummed state capsule, restores it into an engine sharing
the same immutable weights, and finishes without replaying the prefix.

## Easiest complete demo

```bash
FLOWEDGE_BUILD_DIR="$PWD/build-relay" \
  ./tools/scripts/relay_demo.sh models/mamba_flow.safetensors
```

The script performs all of these operations:

1. Starts two preallocated workers with shared immutable weights.
2. Sends one valid request through `RelayClient` and receives an action.
3. Sends one deliberately impossible deadline and receives `rejected_deadline`.
4. Requests graceful shutdown.
5. Inspects and replays the portable trace.
6. Validates Prometheus, compact JSON, and OTLP JSON metrics files.

A successful run includes output similar to:

```text
sequence=1 generation=1 status=1 outcome=inference action0=...
sequence=2 generation=2 status=3 outcome=rejected_deadline ...
replay compared=1 mismatched=0 ...
```

`rejected_deadline` is a normal typed action result, not a broken connection.

## Run the processes manually

Terminal 1 owns the shared-memory rings:

```bash
./build-relay/src/relay/flowedge-relayd \
  --model models/mamba_flow.safetensors \
  --create --workers 2 --threads 0 --placement compact
```

Terminal 2 submits a diagnostic request, then stops the daemon:

```bash
./build-relay/src/relay/flowedge-relayctl request \
  --model models/mamba_flow.safetensors
./build-relay/src/relay/flowedge-relayctl shutdown
```

For a long-lived application, use `RelayClient` as shown in `examples/relay/relay_client_sample.cc` rather
than loading model metadata for every diagnostic request.

## Choose production settings

| Setting | Start with | Meaning |
|---|---:|---|
| `--workers` | `2` | Independent mutable model sessions/outer threads |
| `--threads` | `0` | Core background threads per outer worker; `0` keeps each engine caller-only |
| `--placement` | `compact` | Favor local shared-weight reads; compare with `spread` on multi-node hosts |
| `--nfe-ns` | `0` | Deadline admission disabled until calibrated on the deployment host |
| `--admission-reserve-ns` | deployment-specific | Extra transport/controller safety margin |
| `--metrics-interval-ms` | `0` | Export only at shutdown; periodic file I/O is opt-in |

Measure before enabling deadline admission:

```bash
./build-relay/flowedge_relay_bench models/mamba_flow.safetensors 5000 0
./build-relay/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 2 0
```

Use the reported `p99_ns_per_nfe` as a starting observation, add a safety reserve, then repeat under
sustained load, fixed affinity, and the real power/thermal policy. It is not a portable constant.

## Common outcomes

| Outcome | What it means | Typical response |
|---|---|---|
| `inference` + complete | Valid current action | Publish after the application safety gate |
| `rejected_stale` | A newer generation already exists | Drop it; do not retry the old observation |
| `rejected_deadline` | Calibrated work cannot finish in time | Reduce work, add capacity, or use a later deadline |
| `rejected_capacity` | Bounded queue retained earlier-deadline work | Backpressure or retry only if still fresh |
| `expired` | Deadline passed before dispatch | Drop it and inspect load/clock behavior |
| cancelled | A fresher generation stopped active work | Normal freshness behavior |
| failed | Model or worker execution failed | Inspect daemon error, trace, and metrics |

## Choose the service

| Service | Use it for |
|---|---|
| `flowedge-relayd` | Condition vectors to flow-matching action chunks |
| `flowedge-jobd` | Managed cooperative jobs; currently built-in Mamba streaming |

See [Generic job daemon](generic-job-daemon) or embed the contracts in [Cooperative jobs](cooperative-jobs).
