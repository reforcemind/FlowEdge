"""Capture a matched full-policy replay benchmark on real PushT observations.

Both paths use the source encoder, processor, observation history, initial noise,
and complete DDIM schedule. `--device cuda` runs FlowEdge CUDA Core against
PyTorch CUDA. Closed-loop success is measured by evaluate.py.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(samples):
    return dict(
        zip(("p50", "p95", "p99"), map(float, np.percentile(samples, (50, 95, 99))))
    )


def require_replay_device(device, flowedge_module):
    from .device import require_native_device

    require_native_device(device, flowedge_module)
    if device == "cpu":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda is unavailable")


def load_observation_frames(path):
    data = np.load(path)
    names = list(data.files)
    if not names:
        raise ValueError(f"empty observation fixture: {path}")
    steps = int(data[names[0]].shape[0])
    if any(int(data[name].shape[0]) != steps for name in names):
        raise ValueError(f"observation fixture length mismatch: {path}")
    return [
        {name: torch.from_numpy(np.array(data[name][index])) for name in names}
        for index in range(steps)
    ]


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def flush_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def time_policy_loop(
    name, policy, encoder, frames, noise, steps, device, warmup, iterations, samples
):
    chunks = []
    total = warmup + iterations
    use_cuda_tensors = device == "cuda" and bool(
        getattr(policy, "supports_cuda_tensors", False)
    )
    device_noise = None
    device_output = None
    if use_cuda_tensors:
        device_noise = torch.as_tensor(
            noise, dtype=torch.float32, device=device
        ).contiguous()
        device_output = torch.empty(
            (policy.action_steps, policy.action_dim),
            dtype=torch.float32,
            device=device,
        )
    for iteration in range(total):
        begin = perf_counter_ns()
        encoder.reset()
        for frame in frames:
            encoder.observe(frame)
        condition = encoder.condition()
        synchronize(device)
        encoded = perf_counter_ns()
        if use_cuda_tensors:
            policy.predict_action_chunk_dlpack_into(
                condition.contiguous(), device_noise, device_output, steps=steps
            )
            chunk = device_output
        else:
            chunk = policy.predict_action_chunk(condition, noise, steps=steps)
        synchronize(device)
        end = perf_counter_ns()
        if torch.is_tensor(chunk):
            chunk = chunk.detach().cpu().numpy()
        chunks.append(np.array(chunk, copy=True))
        if iteration >= warmup:
            samples[name]["encoder_ms"].append((encoded - begin) / 1e6)
            samples[name]["policy_ms"].append((end - encoded) / 1e6)
            samples[name]["end_to_end_ms"].append((end - begin) / 1e6)
        print(f"{name} replay {iteration + 1}/{total}", file=sys.stderr)
    return chunks


def max_abs_error(left, right):
    return float(np.max(np.abs(left - right)))


def _has_distribution(name):
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
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
        help="compile the reference U-Net; warmup excludes compilation from measured samples",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        help="reuse a previously captured observations.npz instead of gym-pusht",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.steps, args.iterations, args.threads) <= 0 or args.warmup < 0:
        parser.error("invalid step, iteration, thread, or warmup count")
    if args.reference_compile != "none" and args.warmup == 0:
        parser.error("compiled reference requires at least one warmup iteration")
    torch.set_num_threads(args.threads)
    import flowedge
    from .cli import peak_rss_bytes
    from .diffusion import FlowEdgeDiffusionPolicy
    from .observation import (
        DiffusionObservationEncoder,
        source_config,
        source_stats,
        validate_source_pair,
    )
    from .reference import TorchDiffusionReference

    try:
        require_replay_device(args.device, flowedge)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    validate_source_pair(args.checkpoint, args.source)
    startup = perf_counter_ns()
    encoder = DiffusionObservationEncoder(args.source, device=args.device)
    encoder_startup = (perf_counter_ns() - startup) / 1e6
    startup = perf_counter_ns()
    native = FlowEdgeDiffusionPolicy.from_checkpoint(
        args.checkpoint, threads=args.threads, device=args.device
    )
    native_startup = (perf_counter_ns() - startup) / 1e6
    config = source_config(args.source, device=args.device)
    if (
        config.horizon != native.metadata.horizon
        or config.n_obs_steps != native.metadata.observation_steps
        or config.n_action_steps != native.action_steps
    ):
        raise ValueError("reference and native checkpoint contracts disagree")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = args.output.with_suffix(".observations.npz")
    if args.observations is not None:
        shutil.copyfile(args.observations, fixture)
        frames = load_observation_frames(fixture)
    else:
        from .evaluate import PushTRobot

        robot = PushTRobot(
            seed=args.seed, max_steps=max(2, native.metadata.observation_steps)
        )
        frames = []
        try:
            robot.reset()
            for _ in range(native.metadata.observation_steps):
                frames.append(robot.observe())
                robot.send_action(np.array([256, 256], dtype=np.float32))
        finally:
            robot.stop()
        np.savez_compressed(
            fixture,
            **{k: np.stack([frame[k].numpy() for frame in frames]) for k in frames[0]},
        )
    noise = np.random.default_rng(args.seed).standard_normal(
        (native.metadata.horizon, native.action_dim), dtype=np.float32
    )
    contract_steps = {
        "observation_steps": native.metadata.observation_steps,
        "action_steps": native.action_steps,
        "action_dim": native.action_dim,
        "condition_dim": native.metadata.condition_dim,
    }
    samples = {
        name: {"encoder_ms": [], "policy_ms": [], "end_to_end_ms": []}
        for name in ("flowedge", "lerobot")
    }
    loop_args = dict(
        encoder=encoder,
        frames=frames,
        noise=noise,
        steps=args.steps,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        samples=samples,
    )
    if args.device == "cuda":
        flowedge_chunks = time_policy_loop("flowedge", native, **loop_args)
        del native
        flush_cuda()
        startup = perf_counter_ns()
        reference = TorchDiffusionReference(
            args.source,
            contract_steps["condition_dim"],
            device=args.device,
            compile_mode=args.reference_compile,
        )
        reference_startup = (perf_counter_ns() - startup) / 1e6
        lerobot_chunks = time_policy_loop("lerobot", reference, **loop_args)
    else:
        startup = perf_counter_ns()
        reference = TorchDiffusionReference(
            args.source,
            contract_steps["condition_dim"],
            device=args.device,
            compile_mode=args.reference_compile,
        )
        reference_startup = (perf_counter_ns() - startup) / 1e6
        flowedge_chunks = []
        lerobot_chunks = []
        for iteration in range(args.warmup + args.iterations):
            outputs = {}
            order = (("flowedge", native), ("lerobot", reference))
            if iteration % 2:
                order = order[::-1]
            for name, policy in order:
                begin = perf_counter_ns()
                encoder.reset()
                for frame in frames:
                    encoder.observe(frame)
                condition = encoder.condition()
                encoded = perf_counter_ns()
                outputs[name] = policy.predict_action_chunk(
                    condition, noise, steps=args.steps
                )
                end = perf_counter_ns()
                if iteration >= args.warmup:
                    samples[name]["encoder_ms"].append((encoded - begin) / 1e6)
                    samples[name]["policy_ms"].append((end - encoded) / 1e6)
                    samples[name]["end_to_end_ms"].append((end - begin) / 1e6)
            flowedge_chunks.append(outputs["flowedge"])
            lerobot_chunks.append(outputs["lerobot"])
            error = max_abs_error(outputs["flowedge"], outputs["lerobot"])
            if not np.allclose(
                outputs["flowedge"], outputs["lerobot"], atol=1e-3, rtol=1e-4
            ):
                raise ValueError(f"full-policy parity failed: max_abs_error={error}")
            print(
                f"replay {iteration + 1}/{args.warmup + args.iterations}: max_abs_error={error:.6g}",
                file=sys.stderr,
            )
    max_error = 0.0
    for left, right in zip(flowedge_chunks, lerobot_chunks, strict=True):
        error = max_abs_error(left, right)
        max_error = max(error, max_error)
        if not np.allclose(left, right, atol=1e-3, rtol=1e-4):
            raise ValueError(f"full-policy parity failed: max_abs_error={error}")
        print(f"parity max_abs_error={error:.6g}", file=sys.stderr)
    measurements = {}
    rss = peak_rss_bytes()
    for name, startup_ms in (
        ("flowedge", native_startup),
        ("lerobot", reference_startup),
    ):
        measurements[name] = {
            "status": "measured",
            "startup_ms": startup_ms + encoder_startup,
            **{k: quantiles(v) for k, v in samples[name].items()},
            "throughput_hz": 1000 / float(np.mean(samples[name]["end_to_end_ms"])),
            "rss_mb": None if rss is None else rss / 2**20,
            "rss_reason": "process high-water RSS is unavailable on this host"
            if rss is None
            else "",
            "allocations": {
                "setup": None,
                "hot_path": None,
                "reason": "Python/PyTorch/native process allocations are not instrumented by this runner",
            },
        }
    stats = {
        k: {n: torch.as_tensor(v).tolist() for n, v in values.items()}
        for k, values in source_stats(args.source, encoder.config).items()
    }
    schema = {k: list(v.shape) for k, v in frames[0].items()}
    command = subprocess.list2cmdline(
        [sys.executable, "-m", "flowedge_lerobot.benchmark", *(argv or sys.argv[1:])]
    )
    document = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "sha256": sha256(args.source / "model.safetensors"),
            "converted_sha256": sha256(args.checkpoint),
        },
        "processor": {
            "id": args.model_id,
            "revision": args.revision,
            "stats": stats,
            "config_sha256": sha256(args.source / "config.json"),
        },
        "contract": {
            "observation_schema_hash": hashlib.sha256(
                json.dumps(schema, sort_keys=True).encode()
            ).hexdigest(),
            "observation_steps": contract_steps["observation_steps"],
            "action_steps": contract_steps["action_steps"],
            "action_dim": contract_steps["action_dim"],
            "batch_size": 1,
            "inference_steps": args.steps,
            "action_units": "dataset",
            "normalization": "minmax",
            "solver": "ddim",
        },
        "hardware": {
            "host": platform.node(),
            "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "threads": args.threads,
            "accelerator": args.device,
            "compiler": flowedge.compiler,
            "build_type": flowedge.build_type,
        },
        "commands": {"flowedge": command, "lerobot": command},
        "comparison": {
            "candidate": "FlowEdge native runtime",
            "reference": "LeRobot policy executed with PyTorch",
            "reference_framework": "PyTorch",
            "reference_compile": args.reference_compile,
            "compiled_scope": (
                "none; eager reference"
                if args.reference_compile == "none"
                else "U-Net only; encoder and DDIM scheduler remain eager"
            ),
            "scope": (
                "matched observation encoder, history, noise, and DDIM schedule on CUDA; "
                "FlowEdge engine is freed before the PyTorch U-Net is loaded; "
                "not TensorRT/ONNX; not Jetson/ARM"
                if args.device == "cuda"
                else "matched observation encoder, history, noise, and DDIM schedule"
            ),
        },
        "measurements": measurements,
        "raw_samples": samples,
        "parity": {"max_abs_error": max_error, "atol": 1e-3, "rtol": 1e-4},
        "fixture": {"file": fixture.name, "sha256": sha256(fixture), "seed": args.seed},
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("torch", "lerobot", "gym-pusht", "numpy")
            if _has_distribution(k)
        },
        "iterations": args.iterations,
        "warmup": args.warmup,
        "limitations": [
            (
                "CUDA full-policy replay of one fixed PushT observation history; not closed-loop task success."
                if args.device == "cuda"
                else "CPU full-policy replay of one fixed PushT observation history; not closed-loop task success."
            ),
            "End-to-end includes preprocessing, encoder, history and sampling; excludes camera capture, IPC and robot delivery.",
            (
                "RSS is peak process high-water memory; CUDA frees the FlowEdge engine before loading the PyTorch U-Net, so both U-Nets are not device-resident together."
                if args.device == "cuda"
                else "RSS is shared process high-water memory with both implementations resident, not per-backend memory."
            ),
            "Allocations are unmeasured; no zero-allocation claim applies to this Python visual pipeline.",
            "Startup excludes interpreter/import time and uses the existing filesystem cache.",
            "Small iteration counts do not establish stable p99 estimates.",
        ],
    }
    if args.device == "cuda":
        document["hardware"]["cuda_device"] = torch.cuda.get_device_name(0)
        document["limitations"].extend(
            [
                "CUDA device name is the headline; host threads record leftover CPU work, not a GPU fair-compare.",
                "Not Jetson/ARM evidence. Not TensorRT/ONNX.",
                "A 4GB GPU cannot hold the FlowEdge resident DDIM and the PyTorch U-Net at once.",
            ]
        )
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
