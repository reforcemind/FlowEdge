#!/usr/bin/env python3
"""Run and validate the canonical LeRobot/PyTorch vs FlowEdge report.

This is the single entry point used by the manual CI workflow and by hardware
operators.  It deliberately delegates measurement collection to the installed
LeRobot integration, then validates and renders the portable JSON artifact.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-id", default="lerobot/diffusion_pusht")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--reference-compile",
        default="none",
        choices=(
            "none",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        help="reuse a previously captured observations.npz instead of gym-pusht",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="optional native build directory to prepend to PYTHONPATH",
    )
    args = parser.parse_args(argv)

    environment = os.environ.copy()
    pythonpath = [
        str(Path(__file__).resolve().parents[2] / "integrations" / "lerobot" / "src")
    ]
    if args.build_dir:
        pythonpath.insert(0, str(args.build_dir.resolve()))
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)

    benchmark_args = [
        sys.executable,
        "-m",
        "flowedge_lerobot.benchmark",
        str(args.checkpoint),
        "--source",
        str(args.source),
        "--revision",
        args.revision,
        "--model-id",
        args.model_id,
        "--steps",
        str(args.steps),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--threads",
        str(args.threads),
        "--device",
        args.device,
        "--reference-compile",
        args.reference_compile,
        "--seed",
        str(args.seed),
        "--output",
        str(args.output),
    ]
    if args.observations is not None:
        benchmark_args.extend(["--observations", str(args.observations)])
    subprocess.run(benchmark_args, check=True, env=environment)

    report = args.output.with_suffix(".md")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("benchmark_artifact.py")),
            str(args.output),
            "--report",
            str(report),
        ],
        check=True,
        env=environment,
    )
    print(f"wrote {args.output}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
