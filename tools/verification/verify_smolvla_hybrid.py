#!/usr/bin/env python3
"""Verify one full hybrid SmolVLA action chunk from a recorded source capture.

LeRobot remains responsible for the public observation pipeline and VLM prefix:
image preparation, state preparation, language tokens, prefix masking, and VLM
K/V generation.  FlowEdge then runs the same seeded ten-step action expert.
This is not native VLM/preprocessor parity or a latency measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = hashlib.sha256()
    value.update(str(array.dtype).encode("ascii"))
    value.update(json.dumps(list(array.shape)).encode("ascii"))
    value.update(array.tobytes(order="C"))
    return value.hexdigest()


def required(capture, name: str, dtype, shape: tuple[int, ...]) -> np.ndarray:
    if name not in capture:
        raise ValueError(f"capture is missing {name!r}")
    value = capture[name]
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(
            f"capture {name!r} must have dtype/shape {dtype}/{shape}, got {value.dtype}/{value.shape}"
        )
    return value


def present_images(capture, image_features, batch_size: int) -> dict[str, np.ndarray]:
    images = {}
    for key, feature in image_features.items():
        if key not in capture:
            continue
        image = capture[key]
        channels = feature.shape[0]
        if (
            image.dtype != np.dtype("float32")
            or image.ndim != 4
            or image.shape[0] != batch_size
            or image.shape[1] != channels
            or image.shape[2] <= 0
            or image.shape[3] <= 0
        ):
            raise ValueError(
                f"capture {key!r} must be float32 [batch, {channels}, height, width], got "
                f"{image.dtype}/{tuple(image.shape)}"
            )
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(f"capture {key!r} must be finite RGB in [0, 1]")
        images[key] = image
    if not images:
        raise ValueError(
            "capture must contain at least one configured observation image"
        )
    return images


def clone_batch(batch: dict, torch) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="FlowEdge SmolVLA action-expert safetensors"
    )
    parser.add_argument("model", type=Path, help="local lerobot/smolvla_base directory")
    parser.add_argument(
        "capture", type=Path, help="recorded source-pipeline .npz capture"
    )
    parser.add_argument(
        "--module-path",
        type=Path,
        required=True,
        help="directory containing flowedge extension",
    )
    parser.add_argument(
        "--plugin-path",
        type=Path,
        default=Path("integrations/lerobot/src"),
        help="directory containing flowedge_lerobot (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="write portable JSON report"
    )
    parser.add_argument(
        "--action-tolerance",
        type=float,
        default=3e-2,
        help="maximum cross-runtime BF16 action error (default: %(default)g)",
    )
    args = parser.parse_args()
    if args.action_tolerance <= 0.0:
        parser.error("--action-tolerance must be positive")

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

        if (
            not args.checkpoint.is_file()
            or not args.model.is_dir()
            or not args.capture.is_file()
        ):
            raise ValueError("checkpoint, model directory, and capture must exist")
        policy = SmolVLAPolicy.from_pretrained(args.model).eval()
        if policy.config.adapt_to_pi_aloha:
            raise ValueError(
                "hybrid verifier does not support SmolVLA Aloha action postprocessing"
            )
        batch_size = 1
        state_dim = policy.config.robot_state_feature.shape[0]
        action_dim = policy.config.action_feature.shape[0]
        capture = np.load(args.capture, allow_pickle=False)
        state = required(
            capture, "observation.state", np.dtype("float32"), (batch_size, state_dim)
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
        if not np.isfinite(state).all() or not np.isfinite(noise).all():
            raise ValueError("capture state and noise must be finite")
        if not np.isin(language_mask, (0, 1)).all():
            raise ValueError("capture language attention mask must contain only 0 or 1")
        batch = {
            "observation.state": torch.from_numpy(state),
            "observation.language.tokens": torch.from_numpy(tokens),
            "observation.language.attention_mask": torch.from_numpy(language_mask),
        }
        for key, image in present_images(
            capture, policy.config.image_features, batch_size
        ).items():
            batch[key] = torch.from_numpy(image)

        # Keep the source policy and hybrid prefix call at the same reset point:
        # both paths therefore exercise its actual image/state/language handling.
        policy.reset()
        with torch.no_grad():
            source_actions = policy.predict_action_chunk(
                clone_batch(batch, torch), noise=torch.from_numpy(noise)
            )
        source_actions = (
            source_actions.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
        )
        expected_shape = (policy.config.n_action_steps, action_dim)
        if (
            source_actions.shape != expected_shape
            or not np.isfinite(source_actions).all()
        ):
            raise RuntimeError("source policy emitted an invalid action chunk")

        # The deployed hybrid retains only the source prefix model. Verify its
        # action chunk against the complete PyTorch source after that release.
        LeRobotSmolVLACacheProvider.release_source_expert(policy)
        policy.reset()
        cache = LeRobotSmolVLACacheProvider(policy)(clone_batch(batch, torch))
        engine = flowedge.Engine(str(args.checkpoint), threads=0)
        hybrid = FlowEdgeSmolVLACachedExpert(
            engine,
            action_dim=action_dim,
            action_steps=policy.config.n_action_steps,
        ).predict_action_chunk(
            cache, noise=np.ascontiguousarray(noise[0]), steps=policy.config.num_steps
        )
        if (
            hybrid.shape != expected_shape
            or hybrid.dtype != np.float32
            or not np.isfinite(hybrid).all()
        ):
            raise RuntimeError("hybrid path emitted an invalid action chunk")
        error = np.abs(hybrid - source_actions)
        maximum = float(error.max())
        if maximum > args.action_tolerance:
            raise RuntimeError(
                f"hybrid action parity exceeded tolerance: actions={maximum}"
            )
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"SmolVLA hybrid verification failed: {error}", file=sys.stderr)
        return 2

    report = {
        "schema_version": 1,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": digest(args.checkpoint),
        "model": "lerobot/smolvla_base",
        "source_capture_sha256": digest(args.capture),
        "source_noise_sha256": array_digest(noise),
        "source_actions_sha256": array_digest(source_actions),
        "hybrid_actions_sha256": array_digest(hybrid),
        "source_cache_keys_sha256": array_digest(cache.keys),
        "source_cache_values_sha256": array_digest(cache.values),
        "prefix_length": int(cache.mask.size),
        "action_shape": list(hybrid.shape),
        "action_max_absolute_error": maximum,
        "action_mean_absolute_error": float(error.mean()),
        "action_tolerance": args.action_tolerance,
        "lerobot_version": lerobot.__version__,
        "transformers_version": transformers.__version__,
        "status": "passed",
        "scope": (
            "source LeRobot image/state/language preprocessing and VLM-prefix execution followed by "
            "native FlowEdge cached action-expert Euler replay; not native VLM/preprocessor parity, "
            "latency, or policy-quality evaluation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
