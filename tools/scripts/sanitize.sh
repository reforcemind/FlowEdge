#!/usr/bin/env bash
# Bounded Linux ASan/UBSan verification for Core, protocol, and Relay tests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${FLOWEDGE_SANITIZER_BUILD_DIR:-$ROOT/build-sanitizers}"
BUILD_JOBS="${FLOWEDGE_SANITIZER_BUILD_JOBS:-2}"

[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "FLOWEDGE_SANITIZER_BUILD_JOBS must be a positive integer" >&2
  exit 2
}

command -v cmake >/dev/null || { echo "cmake not found" >&2; exit 1; }
command -v ctest >/dev/null || { echo "ctest not found" >&2; exit 1; }

mkdir -p "$BUILD_DIR/Testing/Temporary"

# ASan reserves a large virtual shadow range, so an address-space ulimit would
# make the sanitizer itself fail. Instead, bound peak RSS by compiling with two
# workers and running one test process at a time.
SANITIZER_FLAGS=(-fsanitize=address,undefined -fno-omit-frame-pointer)
cmake -S "$ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="${CC:-clang}" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DFLOWEDGE_RELAY=ON \
  -DFLOWEDGE_TESTS=ON \
  -DFLOWEDGE_BENCH=OFF \
  -DFLOWEDGE_PYTHON=OFF \
  -DCMAKE_C_FLAGS="${SANITIZER_FLAGS[*]}" \
  -DCMAKE_CXX_FLAGS="${SANITIZER_FLAGS[*]}" \
  -DCMAKE_EXE_LINKER_FLAGS="${SANITIZER_FLAGS[*]}" \
  -DCMAKE_SHARED_LINKER_FLAGS="${SANITIZER_FLAGS[*]}"

cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" 2>&1 \
  | tee "$BUILD_DIR/Testing/Temporary/sanitizer-build.log"

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:halt_on_error=1:abort_on_error=1:color=never}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1:color=never}"
ctest --test-dir "$BUILD_DIR" --output-on-failure --parallel 1 2>&1 \
  | tee "$BUILD_DIR/Testing/Temporary/sanitizer-tests.log"

echo "ASan/UBSan verification passed (build jobs=$BUILD_JOBS, test jobs=1)"
