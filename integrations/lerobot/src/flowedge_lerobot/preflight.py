"""No-robot validation for portable FlowEdge LeRobot policy packages."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from flowedge_lerobot.cli import (
    _cpu_summary,
    _cuda_device_name,
    _jetson_facts,
    peak_rss_bytes,
)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_facts() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_summary(),
        "python": platform.python_version(),
        "versions": {
            "flowedge": _version("flowedge"),
            "lerobot": _version("lerobot"),
            "lerobot_policy_flowedge": _version("lerobot_policy_flowedge"),
        },
        "jetson": _jetson_facts(),
    }


def _feature_contract(features) -> dict[str, dict[str, object]]:
    result = {}
    for name, feature in sorted(features.items()):
        feature_type = getattr(feature.type, "value", str(feature.type))
        result[name] = {"type": feature_type, "shape": list(feature.shape)}
    return result


def _synthetic_observation(config):
    import torch

    return {
        name: torch.zeros(tuple(feature.shape), dtype=torch.float32)
        for name, feature in config.input_features.items()
    }


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(samples: list[float]) -> dict[str, float]:
    return {
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
    }


def _synchronize(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def _exercise(policy, preprocessor, *, warmup: int, iterations: int) -> dict:
    import torch

    raw = _synthetic_observation(policy.config)
    preprocess_ms = []
    action_chunk_ms = []
    end_to_end_ms = []
    total_runs = warmup + iterations
    for index in range(total_runs):
        policy.reset()
        _synchronize(str(policy.config.device))
        started = time.perf_counter()
        batch = preprocessor(raw)
        _synchronize(str(policy.config.device))
        preprocessed = time.perf_counter()
        chunk = policy.predict_action_chunk(batch)
        _synchronize(str(policy.config.device))
        finished = time.perf_counter()
        expected = (
            1,
            int(policy.config.action_steps),
            int(policy.config.action_feature.shape[0]),
        )
        if tuple(chunk.shape) != expected or not bool(torch.isfinite(chunk).all()):
            raise RuntimeError(
                f"policy returned an invalid action chunk: {tuple(chunk.shape)}"
            )
        if index >= warmup:
            preprocess_ms.append((preprocessed - started) * 1_000)
            action_chunk_ms.append((finished - preprocessed) * 1_000)
            end_to_end_ms.append((finished - started) * 1_000)
    return {
        "warmup": warmup,
        "iterations": iterations,
        "preprocess": _latency_summary(preprocess_ms),
        "fresh_action_chunk": _latency_summary(action_chunk_ms),
        "end_to_end": _latency_summary(end_to_end_ms),
    }


def run_preflight(
    reference: str | Path,
    *,
    device: str = "cpu",
    precision: str = "exact-f32",
    warmup: int = 1,
    iterations: int = 3,
) -> dict:
    """Verify a package and run its complete synthetic visual action path."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or cuda")
    if precision not in {"exact-f32", "approx-fp16-deep"}:
        raise ValueError("unsupported FlowEdge diffusion precision")
    if precision == "approx-fp16-deep" and not device.startswith("cuda"):
        raise ValueError("approx-fp16-deep precision requires device=cuda")

    from flowedge_lerobot.deployment import validate_deployment_package

    preflight_started = time.perf_counter()
    integrity_started = time.perf_counter()
    package = validate_deployment_package(reference)
    integrity_ms = (time.perf_counter() - integrity_started) * 1_000

    # Importing the plugin registers both configuration types with LeRobot.
    import flowedge
    import lerobot_policy_flowedge  # noqa: F401
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot_policy_flowedge.processor_flowedge import (
        make_flowedge_pre_post_processors,
    )

    config = PreTrainedConfig.from_pretrained(package["root"])
    if config.type != "flowedge":
        raise ValueError("preflight currently supports packaged FlowEdge Diffusion Policy")
    config.device = device
    config.diffusion_precision = precision
    load_started = time.perf_counter()
    policy = get_policy_class(config.type).from_pretrained(
        package["root"], config=config
    )
    preprocessor, _ = make_flowedge_pre_post_processors(policy.config)
    load_ms = (time.perf_counter() - load_started) * 1_000
    startup_ms = (time.perf_counter() - preflight_started) * 1_000
    timings = _exercise(
        policy, preprocessor, warmup=warmup, iterations=iterations
    )
    engine = policy._policy
    cuda_requested = device.startswith("cuda")
    native_resident = bool(getattr(engine, "cuda_resident", False))
    if cuda_requested and not native_resident:
        raise RuntimeError("CUDA was requested but native policy weights are not resident")

    return {
        "schema_version": 1,
        "status": "pass",
        "scope": "synthetic_visual_policy_preflight_no_robot",
        "package": {
            "reference": Path(reference).name if Path(reference).exists() else str(reference),
            "manifest": package["manifest"],
            "integrity_ms": integrity_ms,
        },
        "contract": {
            "inputs": _feature_contract(policy.config.input_features),
            "outputs": _feature_contract(policy.config.output_features),
            "observation_steps": int(policy.config.n_obs_steps),
            "action_steps": int(policy.config.action_steps),
        },
        "execution": {
            "requested_device": device,
            "resolved_device": str(policy.config.device),
            "requested_precision": precision,
            "resolved_precision": str(getattr(engine, "precision", precision)),
            "flowedge_cuda_build": bool(getattr(flowedge, "cuda", False)),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "native_cuda_resident": native_resident,
            "cuda_device": _cuda_device_name() if cuda_requested else None,
            "startup_ms": startup_ms,
            "load_ms": load_ms,
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "timings": timings,
        "runtime": _runtime_facts(),
        "limitations": [
            "Synthetic zero observations; this does not test a camera or robot driver.",
            "No claim about task success, joint limits, or emergency-stop behavior.",
            "Latency percentiles are diagnostic when the iteration count is small.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowedge-lerobot-preflight",
        description=(
            "Validate a packaged policy and exercise visual-to-action inference "
            "without a robot."
        ),
    )
    parser.add_argument("policy", help="local package directory or Hugging Face repo ID")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--precision",
        choices=("exact-f32", "approx-fp16-deep"),
        default="exact-f32",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser


def _write_report(report: dict, output: Path | None) -> None:
    encoded = json.dumps(report, sort_keys=True)
    print(encoded)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_preflight(
            args.policy,
            device=args.device,
            precision=args.precision,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "error",
            "scope": "synthetic_visual_policy_preflight_no_robot",
            "error": {"type": type(error).__name__, "message": str(error)},
            "runtime": _runtime_facts(),
        }
        _write_report(report, args.output)
        return 1
    _write_report(report, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
