#!/usr/bin/env bash
# Usage: ./tools/scripts/lint.sh [--fix]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${FLOWEDGE_BUILD_DIR:-$ROOT/build}"
FIX="${1:-}"
FORMAT_FLAG=$([[ "$FIX" == "--fix" ]] && echo "-i" || echo "--dry-run")

reject_source_pattern() {
  local pattern="$1"
  local message="$2"
  shift 2
  if command -v rg >/dev/null; then
    if rg -n "$pattern" "$@"; then
      echo "error: $message"
      exit 1
    fi
  elif grep -RInE "$pattern" "$@"; then
    echo "error: $message"
    exit 1
  fi
}

find "$ROOT/src" "$ROOT/bench" "$ROOT/test" "$ROOT/examples" "$ROOT/convert" \
  -type f \( -name "*.h" -o -name "*.cc" -o -name "*.cpp" \) -print0 \
  | xargs -0 clang-format --style=file --Werror "$FORMAT_FLAG"

reject_source_pattern 'std::mutex|memory_order_seq_cst' \
  'blocking mutexes and sequentially consistent atomics are forbidden in Core and Relay' \
  "$ROOT/src/core" "$ROOT/src/relay"
reject_source_pattern '\.load\(\)' \
  'atomic loads require an explicit memory order' "$ROOT/src/core" "$ROOT/src/relay"
reject_source_pattern '\bnew\s|\bdelete\s|\bdelete\[|std::(vector|map|unordered_map|deque|list)<' \
  'Core compute paths must use caller-owned or arena-backed fixed storage' \
  "$ROOT/src/core/kernels" "$ROOT/src/core/heads" "$ROOT/src/core/models"

PY="$(command -v py || command -v python3 || command -v python)"
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
  if grep -qi microsoft /proc/version 2>/dev/null &&
     grep -q -- '--target=x86_64-w64-mingw32' "$BUILD_DIR/compile_commands.json"; then
    echo "note: skipping clang-tidy because WSL clang-tidy cannot consume the Windows/MinGW compile database"
    exit 0
  fi
  if [[ "$(uname -s)" == "Linux" ]]; then
    PROBE="$(mktemp --suffix=.cc)"
    trap 'rm -f "$PROBE"' EXIT
    printf '#include <expected>\nstd::expected<int, int> f();\n' > "$PROBE"
    if ! clang-tidy "$PROBE" -- -std=c++23 >/dev/null 2>&1; then
      echo "note: skipping clang-tidy because this Linux clang-tidy cannot parse C++23 std::expected"
      exit 0
    fi
  fi
  "$PY" -c "import json, sys
print('\0'.join(e['file'] for e in json.load(sys.stdin)
                if not any(x in e['file'].replace(chr(92), '/') for x in ['/bench/', '/test/', '/python/', '/_deps/'])), end='')" \
    < "$BUILD_DIR/compile_commands.json" \
    | xargs -0 -n 1 clang-tidy -p "$BUILD_DIR" ${FIX:+--fix}
else
  echo "note: skipping clang-tidy because $BUILD_DIR/compile_commands.json is missing"
fi
