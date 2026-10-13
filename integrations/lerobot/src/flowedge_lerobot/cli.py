"""Simulator-first command line entry point for LeRobot rollout smoke tests."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .device import require_native_device
from .diffusion import FlowEdgeDiffusionPolicy
from .rollout import RolloutResult, run_rollout


def peak_rss_bytes() -> int | None:
    """Return process high-water RSS without making a runtime dependency mandatory."""

    if platform.system() == "Windows":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_faults", ctypes.c_ulong),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("quota_peak_paged", ctypes.c_size_t),
                ("quota_paged", ctypes.c_size_t),
                ("quota_peak_non_paged", ctypes.c_size_t),
                ("quota_non_paged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("peak_pagefile", ctypes.c_size_t),
            ]

        counters = Counters(ctypes.sizeof(Counters))
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
        if psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(-1), ctypes.byref(counters), ctypes.sizeof(counters)
        ):
            return counters.peak_working_set
        return None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss if platform.system() == "Darwin" else rss * 1024
    except (ImportError, OSError):
        return None


def _cpu_summary() -> str | None:
    if platform.system() == "Linux" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    try:
        return subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or None


class _SimulatedRobot:
    """Deterministic robot seam that keeps observation and action buffers owned."""

    def __init__(self, condition_dim: int) -> None:
        self._observation = np.zeros(condition_dim, dtype=np.float32)
        self.actions = 0
        self.stopped = False

    def reset(self) -> None:
        self.actions = 0
        self.stopped = False

    def observe(self) -> np.ndarray:
        return self._observation

    def send_action(self, action: np.ndarray) -> None:
        del action
        self.actions += 1

    def stop(self) -> None:
        self.stopped = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowedge-lerobot-rollout",
        description="Run a bounded simulator smoke test for a converted LeRobot policy.",
    )
    parser.add_argument(
        "checkpoint", help="FlowEdge-converted Diffusion Policy checkpoint"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="control-loop steps (default: 10)"
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=10, help="denoising steps (default: 10)"
    )
    parser.add_argument("--scheduler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="FlowEdge Core backend; cuda requires FLOWEDGE_BACKEND=cuda",
    )
    parser.add_argument(
        "--period-ms",
        type=float,
        default=None,
        help="deadline period for missed-step counts",
    )
    parser.add_argument(
        "--on-miss",
        choices=("hold", "drop", "raise"),
        default="hold",
        help="hold last action, drop the send, or raise when a period is missed",
    )
    parser.add_argument(
        "--host-facts",
        action="store_true",
        help="include checkpoint path, thread count, and CPU summary in JSON",
    )
    parser.add_argument("--output", type=Path, default=None, help="write JSON to this path")
    return parser


def run_simulator(
    policy: FlowEdgeDiffusionPolicy,
    *,
    steps: int,
    diffusion_steps: int,
    scheduler: str,
    seed: int,
    period_ms: float | None = None,
    on_miss: str = "hold",
    sync=None,
) -> RolloutResult:
    """Run the built-in simulator seam; hardware adapters stay outside this package."""

    robot = _SimulatedRobot(policy.metadata.condition_dim)
    return run_rollout(
        policy,
        robot,
        lambda observation: observation,
        steps=steps,
        seed=seed,
        diffusion_steps=diffusion_steps,
        scheduler=scheduler,
        period_ms=period_ms,
        on_miss=on_miss,
        sync=sync,
    )


def _cuda_sync():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_device_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    if not output.splitlines():
        return None
    return output.splitlines()[0].strip() or None


def _jetson_facts() -> dict[str, object]:
    """Return evidence that distinguishes a Jetson run from generic ARM64 CI."""

    release_path = Path("/etc/nv_tegra_release")
    model_path = Path("/proc/device-tree/model")
    detected = (
        platform.system() == "Linux"
        and platform.machine().lower() in {"aarch64", "arm64"}
        and release_path.exists()
    )
    facts: dict[str, object] = {"detected": detected}
    if release_path.exists():
        facts["tegra_release"] = release_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
    if model_path.exists():
        facts["model"] = model_path.read_text(
            encoding="utf-8", errors="replace"
        ).rstrip("\x00\n")
    return facts


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    flowedge_module = None
    if args.device == "cuda":
        try:
            import flowedge as flowedge_module

            require_native_device(args.device, flowedge_module)
        except ImportError:
            parser.error("CUDA requested but FlowEdge is not importable")
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))
    started = time.perf_counter()
    policy = FlowEdgeDiffusionPolicy.from_checkpoint(
        args.checkpoint, threads=args.threads, device=args.device
    )
    startup_ms = (time.perf_counter() - started) * 1_000
    if args.device == "cuda":
        try:
            require_native_device(args.device, flowedge_module, engine=policy)
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))
    result = run_simulator(
        policy,
        steps=args.steps,
        diffusion_steps=args.diffusion_steps,
        scheduler=args.scheduler,
        seed=args.seed,
        period_ms=args.period_ms,
        on_miss=args.on_miss,
        sync=_cuda_sync if args.device == "cuda" else None,
    )
    report = asdict(result)
    report.update(
        platform=platform.platform(),
        machine=platform.machine(),
        startup_ms=startup_ms,
        evaluation_kind="synthetic_integration_smoke",
        peak_rss_bytes=peak_rss_bytes(),
        device=args.device,
    )
    jetson = _jetson_facts() if args.host_facts else None
    if args.device == "cuda":
        cuda_device = _cuda_device_name()
        if cuda_device:
            report["cuda_device"] = cuda_device
        limitations = [
            "Synthetic encoded-condition smoke; RGB encoder stays in LeRobot for input_mode=visual.",
            "missed_deadlines counts per-call overruns against --period-ms, not closed-loop task success.",
            "Native CUDA Core only; not TensorRT/ONNX.",
        ]
        limitations.append(
            "Physical Jetson host detected; this is not a robot task-success result."
            if jetson and jetson["detected"]
            else "No physical Jetson host was detected."
        )
        report["limitations"] = limitations
    if args.host_facts:
        report.update(
            evaluation_kind="edge_dp_rollout",
            checkpoint=os.path.abspath(args.checkpoint),
            threads=args.threads,
            period_ms=args.period_ms,
            processor=_cpu_summary(),
            python=platform.python_version(),
            jetson=jetson,
        )
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
