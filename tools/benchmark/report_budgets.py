#!/usr/bin/env python3
"""Report checked-in binary-size and model-initialization budgets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_library(build_dir: Path, names: tuple[str, ...]) -> Path:
    candidates = [path for name in names for path in build_dir.rglob(name)]
    candidates = [path for path in candidates if "_deps" not in path.parts]
    if not candidates:
        raise FileNotFoundError(f"could not find {', '.join(names)} below {build_dir}")
    return max(candidates, key=lambda path: path.stat().st_size)


def initialization_allocations(executable: Path, model: Path, tokens: list[int]) -> int:
    completed = subprocess.run(
        [
            str(executable),
            str(model),
            *(str(token) for token in tokens),
            "--cycles",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
        hot_allocations = int(result["hot_path_allocations"])
        setup_allocations = int(result["setup_allocations"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError("model lifecycle check did not emit a valid result") from error
    if hot_allocations != 0:
        raise RuntimeError(f"model hot path allocated {hot_allocations} times")
    return setup_allocations


def render(measured: dict[str, int], budgets: dict[str, int]) -> tuple[str, bool]:
    rows = [
        "# FlowEdge size and initialization budget",
        "",
        "| Metric | Measured | Budget | Result |",
        "|---|---:|---:|---|",
    ]
    failed = False
    for name, label in (
        ("core_bytes", "Core library bytes"),
        ("relay_bytes", "Relay library bytes"),
        ("model_initialization_allocations", "Model initialization allocations"),
    ):
        actual = measured[name]
        budget = budgets[name]
        result = "ok" if actual <= budget else "OVER BUDGET"
        failed |= result != "ok"
        rows.append(f"| {label} | {actual} | {budget} | {result} |")
    rows.extend(
        [
            "",
            "The initialization allocation count covers engine construction and is checked across "
            "three load/run/free cycles; the lifecycle check also fails if the measured hot path allocates.",
        ]
    )
    return "\n".join(rows) + "\n", failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--token", type=int, action="append", required=True)
    parser.add_argument("--budget", type=Path, default=Path("tools/benchmark/config/budgets.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        budget_document = json.loads(args.budget.read_text(encoding="utf-8"))
        if budget_document.get("version") != 1:
            raise ValueError("unsupported budget version")
        max_library = budget_document["max_library_bytes"]
        budgets = {
            "core_bytes": int(max_library["core"]),
            "relay_bytes": int(max_library["relay"]),
            "model_initialization_allocations": int(
                budget_document["max_model_initialization_allocations"]
            ),
        }
        core = find_library(args.build_dir, ("libflowedge_engine.a", "flowedge_engine.lib"))
        relay = find_library(args.build_dir, ("libflowedge_relay.a", "flowedge_relay.lib"))
        lifecycle_check = next(
            (
                path
                for path in (
                    args.build_dir / "flowedge_model_lifecycle_check",
                    args.build_dir / "flowedge_model_lifecycle_check.exe",
                    args.build_dir / "Release" / "flowedge_model_lifecycle_check.exe",
                )
                if path.is_file()
            ),
            None,
        )
        if lifecycle_check is None:
            raise FileNotFoundError("flowedge_model_lifecycle_check was not found")
        measured = {
            "core_bytes": core.stat().st_size,
            "relay_bytes": relay.stat().st_size,
            "model_initialization_allocations": initialization_allocations(
                lifecycle_check, args.model, args.token
            ),
        }
        report, failed = render(measured, budgets)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"budget report failed: {error}", file=sys.stderr)
        return 2

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
