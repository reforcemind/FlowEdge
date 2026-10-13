"""Compare host-buffer and CUDA DLPack Diffusion Policy boundaries."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise SystemExit(
            "steps and iterations must be positive; warmup must be non-negative"
        )

    import flowedge
    import torch

    engine = flowedge.Engine(str(args.checkpoint), threads=0)
    if not engine.cuda_resident or engine.diffusion_cuda_device < 0:
        raise SystemExit(
            "checkpoint did not attach a device-resident CUDA diffusion head"
        )

    metadata = engine.diffusion_metadata
    start = int(metadata["observation_steps"]) - 1
    action_steps = int(metadata["action_steps"])
    stop = start + action_steps
    condition_host = np.linspace(-0.2, 0.2, engine.condition_dim, dtype=np.float32)
    noise_host = np.linspace(
        -1.0,
        1.0,
        engine.action_horizon * engine.action_dim,
        dtype=np.float32,
    ).reshape(engine.action_horizon, engine.action_dim)
    horizon_host = np.empty_like(noise_host)
    chunk_host = np.empty((action_steps, engine.action_dim), dtype=np.float32)
    device = torch.device("cuda", engine.diffusion_cuda_device)
    condition_device = torch.from_numpy(condition_host).to(device)
    noise_device = torch.from_numpy(noise_host).to(device)
    horizon_device = torch.empty_like(noise_device)
    chunk_device = torch.empty(
        (action_steps, engine.action_dim), dtype=torch.float32, device=device
    )

    def host_call() -> None:
        engine.sample_diffusion_into(
            condition_host, noise_host, horizon_host, args.steps, "ddim", 0
        )
        np.copyto(chunk_host, horizon_host[start:stop])

    def device_call() -> None:
        engine.sample_diffusion_dlpack_into(
            condition_device,
            noise_device,
            chunk_device,
            args.steps,
            "ddim",
            0,
            start,
        )

    def device_full_call() -> None:
        engine.sample_diffusion_dlpack_into(
            condition_device,
            noise_device,
            horizon_device,
            args.steps,
            "ddim",
            0,
            0,
        )

    for _ in range(args.warmup):
        host_call()
        device_call()

    host_ms: list[float] = []
    device_ms: list[float] = []

    def measure(function, samples: list[float]) -> None:
        begin = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - begin) / 1_000_000.0)

    for iteration in range(args.iterations):
        if iteration % 2 == 0:
            measure(host_call, host_ms)
            measure(device_call, device_ms)
        else:
            measure(device_call, device_ms)
            measure(host_call, host_ms)

    device_full_call()
    horizon_device_host = horizon_device.cpu().numpy()
    chunk_device_host = chunk_device.cpu().numpy()
    host_median = statistics.median(host_ms)
    device_median = statistics.median(device_ms)
    report = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "gpu": torch.cuda.get_device_name(device),
        "steps": args.steps,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "host_chunk_ms": host_ms,
        "dlpack_chunk_ms": device_ms,
        "host_chunk_median_ms": host_median,
        "dlpack_chunk_median_ms": device_median,
        "median_reduction_percent": (host_median - device_median) / host_median * 100.0,
        "full_horizon_max_abs": float(
            np.max(np.abs(horizon_host - horizon_device_host))
        ),
        "compact_chunk_max_abs": float(
            np.max(np.abs(horizon_host[start:stop] - chunk_device_host))
        ),
    }
    if report["full_horizon_max_abs"] > 1.0e-3:
        raise SystemExit("DLPack full-horizon parity exceeded 1e-3")
    if report["compact_chunk_max_abs"] > 1.0e-3:
        raise SystemExit("DLPack compact-chunk parity exceeded 1e-3")
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
