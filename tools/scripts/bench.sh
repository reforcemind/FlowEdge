#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="${FLOWEDGE_BUILD_DIR:-$ROOT/build}"


# MinGW runtime DLLs live in Strawberry
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) export PATH="/c/Strawberry/c/bin:$PATH" ;; esac

"$ROOT/tools/scripts/build.sh" Release -DFLOWEDGE_BENCH=ON -DFLOWEDGE_TESTS=OFF

exe_k="$BENCH/flowedge_kernels_bench"
exe_t="$BENCH/flowedge_threaded_matmul_bench"
if [[ -f "$BENCH/flowedge_kernels_bench.exe" ]]; then
    exe_k="$BENCH/flowedge_kernels_bench.exe"
    exe_t="$BENCH/flowedge_threaded_matmul_bench.exe"
fi

echo "== kernel microbenchmarks =="
"$exe_k" --benchmark_counters_tabular=true --benchmark_color=true --benchmark_out="$BENCH/kernels.json" --benchmark_out_format=json

echo ""
echo "== threaded matmul scaling =="
"$exe_t" --benchmark_counters_tabular=true --benchmark_color=true

echo ""
echo "Policy measurements require captured observations and an explicit replay command; see docs/performance.md."
