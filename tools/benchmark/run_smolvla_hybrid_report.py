#!/usr/bin/env python3
"""Benchmark matched SmolVLA source and hybrid action generation on one capture.

The input is a real capture produced by the upstream LeRobot pipeline. The
source measurement runs the installed PyTorch ``SmolVLAPolicy`` from prepared
observation tensors. The hybrid measurement times source VLM-prefix/cache
production followed by the native FlowEdge action expert. Camera capture,
dataset decoding, network transport, and robot delivery are outside this
benchmark. Pass ``--device cuda`` to run the source VLM on PyTorch CUDA; the
FlowEdge expert still runs on CPU, and CUDA work is synchronized for timing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np


def quantiles(samples: list[float]) -> dict[str, float]:
    return dict(
        zip(("p50", "p95", "p99"), map(float, np.percentile(samples, (50, 95, 99))))
    )


def synchronize(device, torch) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="FlowEdge SmolVLA action-expert safetensors"
    )
    parser.add_argument("model", type=Path, help="local lerobot/smolvla_base directory")
    parser.add_argument(
        "capture", type=Path, help="recorded source-pipeline .npz capture"
    )
    parser.add_argument("--module-path", type=Path, required=True)
    parser.add_argument(
        "--plugin-path",
        type=Path,
        default=Path("integrations/lerobot/src"),
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device for the source PyTorch SmolVLA VLM; FlowEdge expert remains on CPU",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.iterations, args.threads) <= 0 or args.warmup < 0:
        parser.error(
            "iterations and threads must be positive; warmup cannot be negative"
        )

    verification_path = Path(__file__).resolve().parents[1] / "verification"
    sys.path.insert(0, str(verification_path))
    sys.path.insert(0, str(args.module_path))
    sys.path.insert(0, str(args.plugin_path))
    try:
        import flowedge
        import lerobot
        import torch
        import transformers
        from flowedge_lerobot import (
            FlowEdgeSmolVLACachedExpert,
            LeRobotSmolVLACacheProvider,
        )
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from verify_smolvla_hybrid import (
            array_digest,
            clone_batch,
            digest,
            present_images,
            required,
        )

        if (
            not args.checkpoint.is_file()
            or not args.model.is_dir()
            or not args.capture.is_file()
        ):
            raise ValueError("checkpoint, model directory, and capture must exist")
        torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is unavailable")
        policy = SmolVLAPolicy.from_pretrained(args.model).eval()
        policy.config.device = args.device
        policy.to(device)
        if policy.config.adapt_to_pi_aloha:
            raise ValueError(
                "benchmark does not support SmolVLA Aloha action postprocessing"
            )

        with np.load(args.capture, allow_pickle=False) as capture:
            batch_size = 1
            state_dim = policy.config.robot_state_feature.shape[0]
            action_dim = policy.config.action_feature.shape[0]
            state = required(
                capture,
                "observation.state",
                np.dtype("float32"),
                (batch_size, state_dim),
            )
            tokens = required(
                capture,
                "observation.language.tokens",
                np.dtype("int64"),
                (batch_size, policy.config.tokenizer_max_length),
            )
            language_mask = required(
                capture,
                "observation.language.attention_mask",
                np.dtype("bool"),
                (batch_size, policy.config.tokenizer_max_length),
            )
            noise = required(
                capture,
                "noise",
                np.dtype("float32"),
                (batch_size, policy.config.chunk_size, policy.config.max_action_dim),
            )
            images = present_images(capture, policy.config.image_features, batch_size)

        if not np.isfinite(state).all() or not np.isfinite(noise).all():
            raise ValueError("capture state and noise must be finite")
        batch = {
            "observation.state": torch.from_numpy(state).to(device),
            "observation.language.tokens": torch.from_numpy(tokens).to(device),
            "observation.language.attention_mask": torch.from_numpy(language_mask).to(
                device
            ),
        }
        batch.update(
            {key: torch.from_numpy(value).to(device) for key, value in images.items()}
        )
        noise_tensor = torch.from_numpy(noise).to(device)
        expected_shape = (policy.config.n_action_steps, action_dim)

        source_samples: list[float] = []
        source_actions: list[np.ndarray] = []
        prefix_samples: list[float] = []
        expert_samples: list[float] = []
        hybrid_samples: list[float] = []
        max_error = 0.0
        final_cache = None
        total_runs = args.warmup + args.iterations

        def source_action() -> np.ndarray:
            with torch.no_grad():
                result = policy.predict_action_chunk(
                    clone_batch(batch, torch), noise=noise_tensor
                )
            result = result.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
            if result.shape != expected_shape or not np.isfinite(result).all():
                raise RuntimeError("source policy emitted an invalid action chunk")
            return result

        for iteration in range(total_runs):
            policy.reset()
            synchronize(device, torch)
            source_start = perf_counter_ns()
            source = source_action()
            synchronize(device, torch)
            source_elapsed = (perf_counter_ns() - source_start) / 1e6
            source_actions.append(source)
            if iteration >= args.warmup:
                source_samples.append(source_elapsed)

        # The deployed hybrid retains only the source VLM. Release the PyTorch
        # action expert before timing it so its CUDA weights do not crowd the
        # actual CUDA-VLM/CPU-expert path.
        LeRobotSmolVLACacheProvider.release_source_expert(policy)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            synchronize(device, torch)
        native = flowedge.Engine(str(args.checkpoint), threads=args.threads)
        provider = LeRobotSmolVLACacheProvider(policy)
        hybrid = FlowEdgeSmolVLACachedExpert(
            native,
            action_dim=action_dim,
            action_steps=policy.config.n_action_steps,
        )

        def hybrid_action() -> tuple[np.ndarray, float, float, float, object]:
            policy.reset()
            synchronize(device, torch)
            started = perf_counter_ns()
            cache = provider(clone_batch(batch, torch))
            synchronize(device, torch)
            prefix_done = perf_counter_ns()
            result = hybrid.predict_action_chunk(
                cache,
                noise=np.ascontiguousarray(noise[0]),
                steps=policy.config.num_steps,
            )
            synchronize(device, torch)
            finished = perf_counter_ns()
            if result.shape != expected_shape or not np.isfinite(result).all():
                raise RuntimeError("hybrid path emitted an invalid action chunk")
            return (
                result,
                (prefix_done - started) / 1e6,
                (finished - prefix_done) / 1e6,
                (finished - started) / 1e6,
                cache,
            )

        for iteration in range(total_runs):
            result, prefix_elapsed, expert_elapsed, total_elapsed, final_cache = (
                hybrid_action()
            )
            error = np.abs(result - source_actions[iteration])
            current_max = float(error.max())
            max_error = max(max_error, current_max)
            if current_max > 3e-2:
                raise RuntimeError(
                    f"hybrid action parity exceeded tolerance: actions={current_max}"
                )
            if iteration >= args.warmup:
                prefix_samples.append(prefix_elapsed)
                expert_samples.append(expert_elapsed)
                hybrid_samples.append(total_elapsed)

    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"SmolVLA hybrid benchmark failed: {error}", file=sys.stderr)
        return 2

    report = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "smolvla_hybrid",
        "checkpoint": args.checkpoint.name,
        "capture": args.capture.name,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "sample_count": len(source_samples),
        "measurements_ms": {
            "pytorch_prepared_capture_to_action": quantiles(source_samples),
            "source_vlm_prefix_and_cache": quantiles(prefix_samples),
            "flowedge_cached_action_expert": quantiles(expert_samples),
            "hybrid_source_prefix_to_action": quantiles(hybrid_samples),
        },
        "raw_samples_ms": {
            "pytorch_prepared_capture_to_action": source_samples,
            "source_vlm_prefix_and_cache": prefix_samples,
            "flowedge_cached_action_expert": expert_samples,
            "hybrid_source_prefix_to_action": hybrid_samples,
        },
        "parity": {
            "action_shape": list(expected_shape),
            "max_absolute_error": max_error,
            "tolerance": 3e-2,
        },
        "provenance": {
            "capture_sha256": digest(args.capture),
            "noise_sha256": array_digest(noise),
            "cache_prefix_length": int(final_cache.mask.size),
            "cache_keys_sha256": array_digest(final_cache.keys),
            "cache_values_sha256": array_digest(final_cache.values),
        },
        "hardware": {
            "host": platform.node(),
            "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "threads": args.threads,
            "native_threads": args.threads,
            "source_vlm_device": str(device),
            "flowedge_expert_device": "cpu",
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "compiler": flowedge.compiler,
            "build_type": flowedge.build_type,
        },
        "versions": {
            "torch": torch.__version__,
            "lerobot": lerobot.__version__,
            "transformers": transformers.__version__,
            "numpy": importlib.metadata.version("numpy"),
        },
        "status": "passed",
        "limitations": [
            "Quantiles are not stable estimates when iterations is small; use a larger matched run for p99 claims.",
            "Prepared capture starts after camera capture, dataset decoding, tokenizer, and public processor I/O.",
            "The hybrid path keeps the VLM and preprocessing in LeRobot/PyTorch; it does not measure a native VLM implementation.",
            "The source PyTorch and hybrid paths run in separate passes; source expert weights are released before hybrid timing.",
            "No controller queue, deadline, underrun, task-success, or control-quality measurement is included.",
        ],
        "scope": (
            "prepared real LeRobot capture through source PyTorch or source VLM-prefix plus native "
            "FlowEdge action expert; excludes camera capture, dataset decode, transport, controller "
            "delivery, native VLM implementation, and task-quality evaluation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
