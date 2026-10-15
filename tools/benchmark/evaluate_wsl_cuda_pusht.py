#!/usr/bin/env python3
"""Run matched PushT episodes against a WSL CUDA FlowEdge build.

The parent process keeps the Windows LeRobot/PushT stack. A persistent WSL
worker owns one CUDA-resident engine and exchanges condition/noise/action
arrays over newline-delimited JSON. IPC is outside the latency measurement;
this tool exists only to compare closed-loop task quality across precision
modes when the CUDA build and simulator live in different Python environments.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np


def worker(args: argparse.Namespace) -> int:
    import flowedge

    engine = flowedge.Engine(
        args.checkpoint,
        threads=args.threads,
        diffusion_precision=args.precision,
    )
    if not engine.cuda_resident:
        raise RuntimeError("WSL worker did not attach a CUDA-resident diffusion head")
    metadata = dict(engine.diffusion_metadata)
    metadata["precision"] = engine.diffusion_precision
    print(json.dumps(metadata), flush=True)
    action_start = int(metadata["observation_steps"]) - 1
    action_stop = action_start + int(metadata["action_steps"])
    for line in sys.stdin:
        request = json.loads(line)
        condition = np.asarray(request["condition"], dtype=np.float32)
        noise = np.asarray(request["noise"], dtype=np.float32)
        horizon = np.asarray(
            engine.sample_diffusion(
                condition,
                noise,
                int(request["steps"]),
                "ddim",
                int(request["seed"]),
            ),
            dtype=np.float32,
        ).reshape(int(metadata["horizon"]), int(metadata["action_dim"]))
        print(json.dumps(horizon[action_start:action_stop].tolist()), flush=True)
    return 0


def wsl_path(path: Path) -> str:
    result = subprocess.run(
        ["wsl.exe", "--exec", "wslpath", "-a", "-u", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class WslPolicy:
    def __init__(self, args: argparse.Namespace, precision: str):
        script = wsl_path(Path(__file__))
        checkpoint = wsl_path(Path(args.checkpoint))
        command = [
            "wsl.exe",
            "--exec",
            "env",
            f"PYTHONPATH={args.wsl_build}",
            "FLOWEDGE_CUDA_REQUIRED=1",
            args.wsl_python,
            script,
            checkpoint,
            "--worker",
            "--precision",
            precision,
            "--threads",
            str(args.threads),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        ready = self.process.stdout.readline()
        if not ready:
            error = (
                self.process.stderr.read() if self.process.stderr is not None else ""
            )
            raise RuntimeError(
                "WSL CUDA worker exited before publishing metadata: " + error.strip()
            )
        self.metadata = json.loads(ready)
        self.precision = precision

    def predict_action_chunk(
        self, condition: np.ndarray, noise: np.ndarray, *, steps: int, seed: int = 0
    ) -> np.ndarray:
        request = {
            "condition": np.asarray(condition, dtype=np.float32).reshape(-1).tolist(),
            "noise": np.asarray(noise, dtype=np.float32).tolist(),
            "steps": steps,
            "seed": seed,
        }
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("WSL CUDA worker exited during inference")
        return np.asarray(json.loads(response), dtype=np.float32)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=30)


def evaluate(args: argparse.Namespace, precision: str) -> dict:
    from flowedge_lerobot.evaluate import PushTRobot
    from flowedge_lerobot.observation import DiffusionObservationEncoder

    encoder = DiffusionObservationEncoder(args.source)
    policy = WslPolicy(args, precision)
    metadata = policy.metadata
    reports = []
    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            robot = PushTRobot(seed=episode_seed, max_steps=args.max_steps)
            encoder.reset()
            rng = np.random.default_rng(episode_seed)
            queue: deque[np.ndarray] = deque()
            started = perf_counter()
            try:
                robot.reset()
                while not robot.done:
                    encoder.observe(robot.observe())
                    if not queue:
                        noise = rng.standard_normal(
                            (int(metadata["horizon"]), int(metadata["action_dim"])),
                            dtype=np.float32,
                        )
                        queue.extend(
                            policy.predict_action_chunk(
                                encoder.condition(), noise, steps=args.steps
                            )
                        )
                    robot.send_action(queue.popleft())
            finally:
                robot.stop()
            report = {
                "seed": episode_seed,
                "success": robot.success,
                "steps": robot.steps,
                "max_reward": robot.max_reward,
                "elapsed_s": perf_counter() - started,
            }
            reports.append(report)
            print(precision, report, flush=True)
    finally:
        policy.close()
    return {
        "precision": precision,
        "episodes": reports,
        "success_rate": sum(row["success"] for row in reports) / len(reports),
        "mean_max_reward": float(np.mean([row["max_reward"] for row in reports])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--source")
    parser.add_argument(
        "--precisions", nargs="+", default=["exact-f32", "approx-fp16-deep"]
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wsl-build")
    parser.add_argument("--wsl-python")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--precision", default="exact-f32", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)
    if any(
        value is None
        for value in (args.source, args.output, args.wsl_build, args.wsl_python)
    ):
        parser.error("--source, --output, --wsl-build, and --wsl-python are required")
    if os.name != "nt":
        parser.error("the parent evaluator must run on Windows")
    if min(args.episodes, args.max_steps, args.steps, args.threads) <= 0:
        parser.error("counts must be positive")
    from flowedge_lerobot.observation import validate_source_pair

    validate_source_pair(args.checkpoint, args.source)
    document = {
        "schema_version": 1,
        "scope": "matched gym_pusht/PushT-v0 simulation; IPC excluded from latency",
        "seed": args.seed,
        "steps": args.steps,
        "max_steps": args.max_steps,
        "results": [evaluate(args, precision) for precision in args.precisions],
        "limitations": [
            "Simulation only; no SO-101 hardware validation.",
            "The image encoder runs on Windows CPU and the native denoiser in WSL CUDA.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document["results"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
