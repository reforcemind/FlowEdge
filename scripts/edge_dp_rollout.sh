#!/usr/bin/env bash
# Fail-closed Jetson CUDA smoke benchmark. This does not drive a robot.
set -euo pipefail

checkpoint=${1:?usage: $0 <converted.safetensors> [flowedge-lerobot-rollout args]}
shift || true

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "error: this evidence helper must run on Linux aarch64" >&2
  exit 2
fi
if [[ ! -r /etc/nv_tegra_release ]]; then
  echo "error: /etc/nv_tegra_release is missing; no physical Jetson was detected" >&2
  exit 2
fi
if ! python -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
  echo "error: LeRobot 0.6 requires Python 3.12 or newer" >&2
  exit 2
fi

report=${FLOWEDGE_EDGE_REPORT:-flowedge-jetson-report.json}
export FLOWEDGE_CUDA_REQUIRED=1
flowedge-lerobot-rollout "$checkpoint" \
  --device cuda --steps 20 --threads 1 --period-ms 10 --on-miss hold \
  --host-facts --output "$report" "$@"
echo "wrote $report"
