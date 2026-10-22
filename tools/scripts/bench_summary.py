#!/usr/bin/env python3
"""Turn Google Benchmark JSON into a short Markdown job summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("label")
    args = parser.parse_args()
    if not args.json_path.is_file():
        print(f"### {args.label}\n\nMissing `{args.json_path}`.")
        return 1
    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    rows = payload.get("benchmarks") or []
    print(f"### {args.label}")
    print()
    print("| Benchmark | Time | Unit | Iterations |")
    print("|---|---:|---|---:|")
    for row in rows:
        name = str(row.get("name", ""))
        if name.endswith("_mean") or name.endswith("_median") or name.endswith("_stddev"):
            continue
        print(
            f"| `{name}` | {row.get('real_time', '')} | {row.get('time_unit', '')} | "
            f"{row.get('iterations', '')} |"
        )
    if not rows:
        print("| _(none)_ | | | |")
    host = (payload.get("context") or {}).get("host_name", "")
    mhz = (payload.get("context") or {}).get("mhz_per_cpu", "")
    if host or mhz:
        print()
        print(f"Host `{host}` · `{mhz}` MHz/CPU. Not a policy-latency result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
