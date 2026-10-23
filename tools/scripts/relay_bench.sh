#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${1:-$ROOT/models/mamba_flow.safetensors}"
BUILD_DIR="${FLOWEDGE_BUILD_DIR:-$ROOT/build-relay}"
ITERS="${FLOWEDGE_RELAY_BENCH_ITERS:-500}"
REPORT_DIR="${FLOWEDGE_RELAY_BENCH_REPORT_DIR:-}"

if [[ ! -f "$MODEL" ]]; then
    echo "Downloading mamba_flow.safetensors from Hugging Face..."
    mkdir -p "$ROOT/models"
    wget -qO "$MODEL" "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors"
fi

case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

FLOWEDGE_BUILD_DIR="$BUILD_DIR" "$ROOT/tools/scripts/build.sh" Release \
    -DFLOWEDGE_RELAY=ON -DFLOWEDGE_BENCH=ON -DFLOWEDGE_TESTS=OFF

executable() {
    local path="$BUILD_DIR/$1"
    [[ -f "$path.exe" ]] && path="$path.exe"
    printf '%s\n' "$path"
}

report=""
if [[ -n "$REPORT_DIR" ]]; then
    mkdir -p "$REPORT_DIR"
    report="$REPORT_DIR/relay-bench-$(date -u +%Y%m%dT%H%M%SZ).txt"
    { printf 'system='; uname -a; printf 'commit='; git -C "$ROOT" rev-parse --short HEAD; } > "$report"
fi

run() {
    if [[ -n "$report" ]]; then
        { printf 'command='; printf '%q ' "$@"; printf '\n'; "$@"; } | tee -a "$report"
    else
        "$@"
    fi
}

relay_args=("$(executable flowedge_relay_bench)" "$MODEL" "$ITERS")
[[ -n "${FLOWEDGE_THREADS:-}" ]] && relay_args+=("$FLOWEDGE_THREADS")
run "${relay_args[@]}"

POOL_WORKERS="${FLOWEDGE_RELAY_POOL_WORKERS:-2}"
POOL_THREADS="${FLOWEDGE_RELAY_POOL_THREADS:-0}"
run "$(executable flowedge_relay_pool_bench)" "$MODEL" "$ITERS" "$POOL_WORKERS" "$POOL_THREADS"
run "$(executable flowedge_cooperative_job_bench)" "${FLOWEDGE_RELAY_COOPERATIVE_BENCH_ITERS:-1000000}"
run "$(executable flowedge_job_queue_bench)" "${FLOWEDGE_RELAY_QUEUE_BENCH_ROUNDS:-3000}"
run "$(executable flowedge_job_qos_bench)" "${FLOWEDGE_RELAY_QOS_BENCH_ITERS:-1000000}"
run "$(executable flowedge_mamba_stream_bench)" "$MODEL" "$ITERS"
run "$(executable flowedge_worker_drain_bench)" "${FLOWEDGE_RELAY_DRAIN_BENCH_ITERS:-10000}"
run "$(executable flowedge_action_delivery_bench)" "${FLOWEDGE_RELAY_ACTION_BENCH_ITERS:-1000000}"

[[ -n "$report" ]] && echo "wrote $report"
