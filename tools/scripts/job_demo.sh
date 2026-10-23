#!/usr/bin/env bash
# Standalone generic-job daemon: stream, inspect, drain, resume, metrics, and shutdown.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${1:-$ROOT/models/mamba_flow.safetensors}"
BUILD_DIR="${FLOWEDGE_BUILD_DIR:-$ROOT/build-relay}"
REQUEST_SHM="flowedge-job-demo-requests-$$"
RESULT_SHM="flowedge-job-demo-results-$$"
CONTROL_SHM="flowedge-job-demo-control-$$"
STATUS_SHM="flowedge-job-demo-status-$$"
TRACE="$BUILD_DIR/job-demo.trace"
PROMETHEUS_METRICS="$BUILD_DIR/job-demo.prom"
JSON_METRICS="$BUILD_DIR/job-demo-metrics.json"
OTLP_METRICS="$BUILD_DIR/job-demo-otlp.json"

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
  exit 1
fi
if ! DAEMON="$(resolve_executable flowedge-jobd)" ||
   ! CONTROL="$(resolve_executable flowedge-jobctl)" ||
   ! TRACE_TOOL="$(resolve_executable flowedge-relay-trace)"; then
  FLOWEDGE_BUILD_DIR="$BUILD_DIR" "$ROOT/tools/scripts/build.sh" Release -DFLOWEDGE_RELAY=ON
  DAEMON="$(resolve_executable flowedge-jobd)"
  CONTROL="$(resolve_executable flowedge-jobctl)"
  TRACE_TOOL="$(resolve_executable flowedge-relay-trace)"
fi
mkdir -p "$BUILD_DIR"
rm -f "$TRACE" "$PROMETHEUS_METRICS" "$JSON_METRICS" "$OTLP_METRICS"

COMMON_CONTROL=(--control-shm "$CONTROL_SHM" --status-shm "$STATUS_SHM")
COMMON_DATA=(--request-shm "$REQUEST_SHM" --result-shm "$RESULT_SHM")
"$DAEMON" --model "$MODEL" --create "${COMMON_DATA[@]}" "${COMMON_CONTROL[@]}" \
  --capacity 8 --workers 2 --threads 0 --max-tokens 8 --work-quantum 1 \
  --trace "$TRACE" --metrics-prometheus "$PROMETHEUS_METRICS" \
  --metrics-json "$JSON_METRICS" --metrics-otlp-json "$OTLP_METRICS" \
  >"$BUILD_DIR/job-demo-daemon.log" 2>&1 &
DAEMON_PID=$!

cleanup() {
  if kill -0 "$DAEMON_PID" >/dev/null 2>&1; then
    "$CONTROL" shutdown "${COMMON_CONTROL[@]}" >/dev/null 2>&1 || true
    kill "$DAEMON_PID" >/dev/null 2>&1 || true
    wait "$DAEMON_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

connected=false
for _ in $(seq 1 200); do
  if "$CONTROL" status "${COMMON_CONTROL[@]}" >"$BUILD_DIR/job-demo-status.log" 2>&1; then
    connected=true
    break
  fi
  sleep 0.01
done
if [[ "$connected" != true ]]; then
  cat "$BUILD_DIR/job-demo-daemon.log" >&2
  cat "$BUILD_DIR/job-demo-status.log" >&2
  exit 1
fi
cat "$BUILD_DIR/job-demo-status.log"

"$CONTROL" mamba --model "$MODEL" "${COMMON_DATA[@]}" --tokens 1,2,3,4 \
  --sequence 10 --session 42 --generation 1 --class interactive
"$CONTROL" drain --worker 0 "${COMMON_CONTROL[@]}" --sequence 11
"$CONTROL" mamba --model "$MODEL" "${COMMON_DATA[@]}" --tokens 4,3,2,1 \
  --sequence 12 --session 42 --generation 2 --class critical
"$CONTROL" resume --worker 0 "${COMMON_CONTROL[@]}" --sequence 13
"$CONTROL" shutdown "${COMMON_CONTROL[@]}" --sequence 14
wait "$DAEMON_PID"
trap - EXIT

cat "$BUILD_DIR/job-demo-daemon.log"
"$TRACE_TOOL" inspect "$TRACE"
grep -q '^flowedge_relay_job_completed_total 2$' "$PROMETHEUS_METRICS"
grep -q '"completed":2' "$JSON_METRICS"
grep -q '"resourceMetrics"' "$OTLP_METRICS"
echo "metrics_prometheus=$PROMETHEUS_METRICS metrics_json=$JSON_METRICS metrics_otlp=$OTLP_METRICS"
