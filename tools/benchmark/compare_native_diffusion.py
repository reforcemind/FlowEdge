#!/usr/bin/env python3
"""Compare two native builds in separate processes, preserving raw timing and parity.

This measures condition-to-action, not visual policy or robot latency. Each worker
loads one build and one engine. CUDA fallback is an error. Repetitions alternate
build order; timing samples use a fixed condition/noise within each trial.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, nargs="+", default=[10])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(args.iterations, args.trials, args.threads, *args.steps) <= 0
        or args.warmup < 0
    ):
        parser.error("counts must be positive; warmup must be nonnegative")
    if args.worker:
        import flowedge

        if (
            args.module_dir is None
            or Path(flowedge.__file__).resolve().parent != args.module_dir.resolve()
        ):
            raise RuntimeError("imported module does not come from the requested build")
        engine = flowedge.Engine(str(args.checkpoint), threads=args.threads)
        if bool(engine.cuda_resident) != (args.device == "cuda"):
            raise RuntimeError("engine residency does not match requested device")
        rng = np.random.default_rng(args.seed)
        condition = rng.standard_normal(engine.condition_dim, dtype=np.float32) * 0.1
        noise = rng.standard_normal(
            engine.action_horizon * engine.action_dim, dtype=np.float32
        )
        result = {
            "module": flowedge.__file__,
            "cuda_resident": engine.cuda_resident,
            "compiler": flowedge.compiler,
            "build_type": flowedge.build_type,
            "seed": args.seed,
            "rows": [],
        }
        if flowedge.build_type != "Release":
            raise RuntimeError("Release build required")
        for steps in args.steps:
            samples = []
            for iteration in range(args.warmup + args.iterations):
                start = perf_counter_ns()
                action = engine.sample_diffusion(condition, noise, steps, "ddim")
                elapsed = (perf_counter_ns() - start) / 1e6
                if iteration >= args.warmup:
                    samples.append(elapsed)
            result["rows"].append(
                {
                    "steps": steps,
                    "samples_ms": samples,
                    "p50_ms": float(np.median(samples)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "action": np.asarray(action).tolist(),
                }
            )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    if args.baseline is None or args.candidate is None:
        parser.error("--baseline and --candidate are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for trial in range(args.trials):
        pair = {}
        order = (
            ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
        )
        for label in order:
            output = args.output.with_name(f"{args.output.stem}-{trial}-{label}.json")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(getattr(args, label).resolve())
            if args.device == "cuda":
                environment["FLOWEDGE_CUDA_REQUIRED"] = "1"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.checkpoint.resolve()),
                "--worker",
                "--module-dir",
                str(getattr(args, label).resolve()),
                "--device",
                args.device,
                "--threads",
                str(args.threads),
                "--steps",
                *map(str, args.steps),
                "--iterations",
                str(args.iterations),
                "--warmup",
                str(args.warmup),
                "--seed",
                str(args.seed + trial),
                "--output",
                str(output.resolve()),
            ]
            subprocess.run(command, env=environment, check=True)
            pair[label] = json.loads(output.read_text())
            print(
                trial,
                label,
                [(r["steps"], r["p50_ms"]) for r in pair[label]["rows"]],
                flush=True,
            )
        for baseline, candidate in zip(
            pair["baseline"]["rows"], pair["candidate"]["rows"]
        ):
            if not np.allclose(
                baseline["action"], candidate["action"], atol=1e-3, rtol=1e-4
            ):
                raise RuntimeError("candidate action parity failed")
        results.append(pair)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    checkpoint_hash = digest.hexdigest()
    document = {
        "checkpoint_sha256": checkpoint_hash,
        "argv": sys.argv,
        "scope": "native synthetic-condition DDIM; not visual policy or task quality",
        "trials": results,
        "summary": [],
    }
    for index, steps in enumerate(args.steps):
        baseline = float(
            np.median([p["baseline"]["rows"][index]["p50_ms"] for p in results])
        )
        candidate = float(
            np.median([p["candidate"]["rows"][index]["p50_ms"] for p in results])
        )
        document["summary"].append(
            {
                "steps": steps,
                "baseline_p50_ms": baseline,
                "candidate_p50_ms": candidate,
                "speedup": baseline / candidate,
            }
        )
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
