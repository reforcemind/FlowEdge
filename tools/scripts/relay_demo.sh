#!/usr/bin/env bash
# End-to-end Relay lifecycle: daemon, typed client, deadline rejection, trace, and replay.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${1:-$ROOT/models/mamba_flow.safetensors}"
BUILD_DIR="${FLOWEDGE_BUILD_DIR:-$ROOT/build-relay}"
TRACE="${FLOWEDGE_RELAY_DEMO_TRACE:-$BUILD_DIR/relay-demo.trace}"
PROMETHEUS_METRICS="$BUILD_DIR/relay-demo.prom"
JSON_METRICS="$BUILD_DIR/relay-demo-metrics.json"
OTLP_METRICS="$BUILD_DIR/relay-demo-otlp.json"
CONDITION_SHM="flowedge-relay-demo-conditions-$$"
ACTION_SHM="flowedge-relay-demo-actions-$$"

case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

resolve_executable() {
  local name=$1
  local candidate
  for candidate in "$BUILD_DIR/$name" "$BUILD_DIR/$name.exe" \
                   "$BUILD_DIR/src/relay/$name" "$BUILD_DIR/src/relay/$name.exe"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  echo "run tools/scripts/bench.sh once to download the sample checkpoint" >&2
  exit 1
fi

if ! DAEMON="$(resolve_executable flowedge-relayd)" ||
   ! CLIENT_SAMPLE="$(resolve_executable relay_client_sample)" ||
   ! CONTROL="$(resolve_executable flowedge-relayctl)" ||
   ! TRACE_TOOL="$(resolve_executable flowedge-relay-trace)"; then
  FLOWEDGE_BUILD_DIR="$BUILD_DIR" "$ROOT/tools/scripts/build.sh" Release -DFLOWEDGE_RELAY=ON
  DAEMON="$(resolve_executable flowedge-relayd)"
  CLIENT_SAMPLE="$(resolve_executable relay_client_sample)"
  CONTROL="$(resolve_executable flowedge-relayctl)"
  TRACE_TOOL="$(resolve_executable flowedge-relay-trace)"
fi
mkdir -p "$BUILD_DIR"
rm -f "$TRACE" "$PROMETHEUS_METRICS" "$JSON_METRICS" "$OTLP_METRICS"

"$DAEMON" --model "$MODEL" --create --condition-shm "$CONDITION_SHM" \
  --action-shm "$ACTION_SHM" --workers "${FLOWEDGE_RELAY_DEMO_WORKERS:-2}" --threads 0 \
  --placement "${FLOWEDGE_RELAY_DEMO_PLACEMENT:-spread}" \
  --nfe-ns "${FLOWEDGE_RELAY_DEMO_NFE_NS:-1000000}" --trace "$TRACE" \
  --metrics-prometheus "$PROMETHEUS_METRICS" --metrics-json "$JSON_METRICS" \
  --metrics-otlp-json "$OTLP_METRICS" \
  >"$BUILD_DIR/relay-demo-daemon.log" 2>&1 &
DAEMON_PID=$!

cleanup() {
  if kill -0 "$DAEMON_PID" >/dev/null 2>&1; then
    "$CONTROL" shutdown --condition-shm "$CONDITION_SHM" >/dev/null 2>&1 || true
    kill "$DAEMON_PID" >/dev/null 2>&1 || true
    wait "$DAEMON_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

connected=false
for _ in $(seq 1 200); do
  if "$CLIENT_SAMPLE" "$MODEL" "$CONDITION_SHM" "$ACTION_SHM" \
      >"$BUILD_DIR/relay-demo-client.log" 2>&1; then
    connected=true
    break
  fi
  sleep 0.01
done
if [[ "$connected" != true ]]; then
  cat "$BUILD_DIR/relay-demo-daemon.log" >&2
  cat "$BUILD_DIR/relay-demo-client.log" >&2
  exit 1
fi
cat "$BUILD_DIR/relay-demo-client.log"

# This request is deliberately impossible under the configured NFE calibration.
set +e
DEADLINE_OUTPUT="$("$CONTROL" request --model "$MODEL" --condition-shm "$CONDITION_SHM" \
  --action-shm "$ACTION_SHM" --sequence 2 --session 42 --generation 2 \
  --steps 64 --solver heun --deadline-ms 1 2>&1)"
DEADLINE_STATUS=$?
set -e
printf '%s\n' "$DEADLINE_OUTPUT"
if [[ "$DEADLINE_STATUS" -ne 1 || "$DEADLINE_OUTPUT" != *"outcome=rejected_deadline"* ]]; then
  echo "expected a typed rejected_deadline outcome" >&2
  exit 1
fi
"$CONTROL" shutdown --condition-shm "$CONDITION_SHM" --sequence 3 --session 42
wait "$DAEMON_PID"
trap - EXIT

cat "$BUILD_DIR/relay-demo-daemon.log"
"$TRACE_TOOL" inspect "$TRACE"
"$TRACE_TOOL" replay "$TRACE" --model "$MODEL" --tolerance 1e-5
grep -q '^flowedge_relay_inference_complete_total 1$' "$PROMETHEUS_METRICS"
grep -q '"inference_complete":1' "$JSON_METRICS"
grep -q '"resourceMetrics"' "$OTLP_METRICS"
echo "metrics_prometheus=$PROMETHEUS_METRICS metrics_json=$JSON_METRICS metrics_otlp=$OTLP_METRICS"
