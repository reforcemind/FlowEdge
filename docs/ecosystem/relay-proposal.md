# FlowEdge Relay

FlowEdge Core remains the small, allocation-free execution library. An optional component in this
repository, **FlowEdge Relay**, solves the system problem around it:
freshness-aware, deadline-aware inference between perception/VLA services and robot controllers.

The first local MVP is implemented behind `-DFLOWEDGE_RELAY=ON`. It provides versioned,
fixed-capacity condition/action messages with compact wire payloads, checksummed SPSC shared-memory
rings, bounded earliest-deadline-first
scheduling, generation cancellation, a head-only worker, and replayable binary traces. Core has no
dependency on Relay; Relay links only `FlowEdge::Core` and the platform shared-memory API.

The production-hardening track adds a move-only `RelayClient`, the `flowedge-relayctl` smoke/control
tool, and a cross-process test that launches the real daemon. The client owns all message construction,
model-identity validation, compact ring I/O, and control-message details while preserving an
allocation-free request/response hot path.

## Problem

Companies assembling embodied-AI systems repeatedly build the same fragile layer: shared-memory or
network transport, action-chunk queues, stale-request cancellation, model-session state, deadline
admission, and replay traces. Generic model servers optimize request throughput; robotics needs the
newest valid action before a physical deadline. LLM infrastructure has a related need for resumable
state, preemption, and fair scheduling of long-lived sessions.

## Shape

```{image} ../_static/figures/relay.svg
:alt: Optional Relay flow from sensor or VLA through shared-memory rings to Core
:class: fe-fig
```

The current MVP provides:

- a versioned condition/action protocol with timestamps, model digest, solver budget,
  and cancellation generation;
- local checksummed shared-memory rings with no ROS 2, gRPC, or cloud dependency;
- bounded earliest-deadline-first admission and expiration before dispatch;
- optional calibrated multi-worker EDF simulation that includes each active lane and queued NFE;
- stale-request pruning and cancellation between complete solver steps;
- typed stale, unreachable-deadline, capacity, and expiry outcomes returned on the action ring;
- a preallocated 1..8 worker pool with one independent Core engine and outer thread per slot;
- one immutable checkpoint store shared by every worker, with optional compact/spread NUMA-aware
  outer-thread placement;
- portable little-endian condition/action traces with checksum validation and v1 read compatibility;
- `flowedge-relayd`, which opens or creates rings and executes compatible head-only checkpoints;
- `RelayClient`, a typed SPSC producer/action-consumer endpoint for external C++ services;
- `flowedge-relayctl`, which submits a deterministic smoke request or asks a daemon to shut down;
- `flowedge-relay-trace`, which inspects traces as text/JSONL or replays actions against a model;
- a Windows/POSIX integration test that crosses a real process and shared-memory boundary;
- fixed-memory counters and latency histograms with Prometheus, JSON, and OTLP/HTTP JSON exporters.
- a non-owning C++23 cooperative-job contract with checked work budgets;
- canonical state capsules bound to job kind, model digest, schema, session, generation, and progress;
- concept-based iterative, streaming, and speculative adapters for external runtimes;
- a production Mamba streaming adapter with shared weights and exact cross-engine continuation;
- rolling worker drain with preallocated live state handoff to a compatible lane;
- controller-side action chunk validation, timed overlap, freshness, bounds, and rate limiting;
- validated variable-size generic job request/result messages;
- frozen, fixed-capacity adapter registration keyed by job kind, model digest, and state schema;
- `JobClient` and embeddable `JobService` endpoints over bounded shared-memory rings;
- `flowedge-jobd`, a managed Mamba streaming service over the generic job contract;
- a separate `JobControlClient` data path for status, drain, resume, recovery, and shutdown;
- best-effort, interactive, and critical queue reservations with deadline-first ordering; and
- repeated-failure lane quarantine with explicit, observable recovery.

Generic jobs cross a real process boundary, migrate during rolling worker drain, retain results under
backpressure, protect urgent capacity, isolate failing lanes, and emit bounded lifecycle records.
Action chunks cross an allocation-free final delivery gate. The next layer is release hardening and
optional ecosystem adapters over these stable local contracts.

## Repository boundary

Transport libraries, ROS distributions, authentication, telemetry, and daemon lifecycles have a
different dependency footprint from a static C++ inference engine. The repository therefore keeps
the boundary explicit instead of putting these concerns into the engine:

- `src/core/` builds `FlowEdge::Core`, owns inference state and kernels, and has no Relay dependency;
- `src/relay/` builds `FlowEdge::Relay` and the `flowedge-relayd` executable when Relay is enabled;
- transport and robot integrations remain optional Relay adapters;
- Relay may be packaged and versioned independently even while both components share one repository.

Keeping Core and Relay together lets state capsules, protocols, model metadata, and cooperative
execution evolve atomically. The one-way dependency preserves Core's embeddability. A repository
split remains possible later if the release cadence or contributor community genuinely diverges.
The first milestone stays narrow: one-machine shared memory, one producer, one head-only FlowEdge
engine, cancellation by generation number, and a replay log. Distributed scheduling should follow
only after real traces show that local scheduling is insufficient.

## Build and measure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFLOWEDGE_RELAY=ON -DFLOWEDGE_TESTS=ON -DFLOWEDGE_BENCH=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/flowedge_relay_bench models/mamba_flow.safetensors 500
```

The benchmark measures the complete allocation-free local path: compact producer serialization,
condition ring, EDF queue, cooperative head worker, action ring, and consumer validation.
`tools/scripts/relay_bench.sh` performs the
same build and run on Linux or Git Bash. The daemon uses `--create` when it owns both rings and
`--trace FILE` to record accepted inputs and published outputs.

For a manual two-process smoke test, keep the daemon in one terminal and use the control tool from a
second terminal:

```bash
./build/src/relay/flowedge-relayd --model models/mamba_flow.safetensors --create
./build/src/relay/flowedge-relayctl request --model models/mamba_flow.safetensors
./build/src/relay/flowedge-relayctl shutdown
```

The daemon owns ring creation in this example. `RelayClient::create` is available when the embedding
service should own their lifetime instead. A ring remains strictly SPSC: sharing one `RelayClient`
between concurrent producer threads is unsupported; use one endpoint or an application-side MPSC
gate per ring pair. Model loading in `flowedge-relayctl` is intentionally a diagnostic convenience.
Long-lived producers should retain model metadata from their own model/configuration layer and call
`RelayClient` directly.

Captured traces are durable cross-platform artifacts rather than dumps of C++ object memory:

```bash
./build/src/relay/flowedge-relay-trace inspect run.trace
./build/src/relay/flowedge-relay-trace inspect run.trace --jsonl
./build/src/relay/flowedge-relay-trace replay run.trace \
  --model models/mamba_flow.safetensors --tolerance 1e-5
```

Replay compares completed action vectors. Cancelled/failed records and accepted conditions without a
published action are counted separately because the trace does not record every cooperative solver
step or scheduler interleaving.

Deadline admission is opt-in because its estimate is deployment-specific. Start with the benchmark's
`p99_ns_per_nfe`, repeat under representative load and thermal conditions, then add a safety reserve:

```bash
./build/flowedge_relay_bench models/mamba_flow.safetensors 5000 0
./build/src/relay/flowedge-relayd --model models/mamba_flow.safetensors --create \
  --nfe-ns 3000 --admission-reserve-ns 20000
```

The scheduler assigns retained EDF work to the earliest available simulated worker lane and checks
every completion, not just the new request in isolation.
Passing `--nfe-ns 0` (the default) disables calibrated rejection while expiry-at-dispatch remains
active. A producer receives a normal, model-compatible action record with `status=failed` and an
outcome code such as `rejected_deadline`; `RelayClient::try_receive` therefore remains one typed path
for inference and scheduling results.

Parallel model workers are explicit:

```bash
./build/src/relay/flowedge-relayd --model models/mamba_flow.safetensors --create \
  --workers 2 --threads 0
./build/flowedge_relay_pool_bench models/mamba_flow.safetensors 5000 2 0
```

`--workers` creates independent mutable engines and dedicated outer threads; `--threads` configures the Core
background thread pool inside each engine. Start with caller-only engines when using multiple outer
workers, then measure alternatives. Each slot keeps a completed action until the ring consumer makes
space, while the other slots and the transport loop continue. Immutable checkpoint tensors are loaded
once and shared; scratch, decode/sampler state, and transformed constants remain private per worker.
Use `--placement compact` to favor local shared-weight reads or `--placement spread` to distribute
outer workers across available NUMA nodes.

For a runnable source-level `RelayClient` integration plus deadline and trace handling, use:

```bash
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./tools/scripts/relay_demo.sh models/mamba_flow.safetensors
```

The client implementation is also demonstrated directly in `examples/relay/relay_client_sample.cc`.

The managed generic-job path is runnable with one command:

```bash
FLOWEDGE_BUILD_DIR="$PWD/build-relay" ./tools/scripts/job_demo.sh models/mamba_flow.safetensors
```

`flowedge-jobd` keeps job request/result traffic separate from status, drain, resume, recovery, and shutdown
traffic. See [Generic job daemon](../guides/generic-job-daemon).

The runtime-neutral job layer has no model dependency beyond Relay. `JobClient` and `JobService`
carry typed records across process-compatible rings; `JobWorkerPool` routes them through bounded EDF
lanes with per-kind work costs and lifecycle events.
The examples cover
migration, streaming cancellation, speculative classification, and queued execution:

```bash
./build/cooperative_job_sample
./build/routed_job_sample
./build/mamba_relay_stream models/mamba_flow.safetensors
./build/action_delivery_sample
./build/flowedge_cooperative_job_bench 100000
./build/flowedge_mamba_stream_bench models/mamba_flow.safetensors 5000
./build/flowedge_worker_drain_bench 10000
./build/flowedge_job_qos_bench 1000000
```

See [Cooperative jobs and state migration](../guides/cooperative-jobs) for the adapter contract and
capsule invariants.
See [Action delivery](../guides/action-delivery) for the controller boundary.

## Reuse outside robotics

The protocol is deliberately condition-vector and state-capsule oriented rather than robot-message
specific. ML systems teams could use the same worker and scheduler for latent decoders, speculative
SSM/LLM branches, iterative generative heads, or any model stage that exposes bounded cooperative
steps. Robot adapters translate actions and safety semantics at the edge of the system.
