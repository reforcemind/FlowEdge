#!/usr/bin/env python3
"""Create one provenance-backed SmolVLA capture from local LeRobot v3 data.

The caller supplies already downloaded parquet metadata and camera video files.
This command decodes recorded pixels, reads the recorded state and task, and
uses SmolVLA's tokenizer plus a recorded seed for the initial flow noise.  It
does not synthesize images, duplicate a camera, resize frames, or normalize
state values on the caller's behalf.
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


def camera_argument(value: str) -> tuple[str, Path]:
    key, separator, source = value.partition("=")
    if not separator or not key or not source:
        raise argparse.ArgumentTypeError("camera must be configured-key=local-video-path")
    return key, Path(source)


def decode_rgb_frame(video: Path, frame_index: int) -> np.ndarray:
    import av

    with av.open(video) as container:
        if not container.streams.video:
            raise ValueError(f"camera file has no video stream: {video}")
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                rgb = frame.to_ndarray(format="rgb24")
                return np.ascontiguousarray(np.moveaxis(rgb, -1, 0)[None], dtype=np.float32) / 255.0
    raise ValueError(f"camera file has no decoded frame {frame_index}: {video}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local lerobot/smolvla_base directory")
    parser.add_argument("data", type=Path, help="local LeRobot v3 data parquet shard")
    parser.add_argument(
        "--frame-index",
        type=int,
        required=True,
        help="unique LeRobot data index in the supplied parquet shard",
    )
    parser.add_argument(
        "--video-frame-index",
        type=int,
        help="zero-based frame ordinal in each supplied camera shard (defaults to --frame-index)",
    )
    parser.add_argument("--task", required=True, help="recorded natural-language task")
    parser.add_argument(
        "--camera",
        action="append",
        type=camera_argument,
        required=True,
        metavar="CONFIGURED_KEY=VIDEO",
        help="map one configured observation.images.* key to one recorded camera video",
    )
    parser.add_argument("--noise-seed", type=int, required=True, help="recorded CPU torch seed")
    parser.add_argument("--dataset-id", required=True, help="source dataset repository identifier")
    parser.add_argument("--dataset-revision", required=True, help="immutable source dataset revision")
    parser.add_argument("--dataset-license", required=True, help="source dataset license identifier")
    parser.add_argument("--output", type=Path, required=True, help="write source capture NPZ")
    parser.add_argument("--manifest", type=Path, required=True, help="write capture provenance JSON")
    args = parser.parse_args()
    if args.frame_index < 0 or args.noise_seed < 0 or (
        args.video_frame_index is not None and args.video_frame_index < 0
    ):
        parser.error("--frame-index, --video-frame-index, and --noise-seed must be non-negative")
    video_frame_index = args.frame_index if args.video_frame_index is None else args.video_frame_index

    try:
        import pyarrow.parquet as pq
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from transformers import AutoProcessor

        if not args.model.is_dir() or not args.data.is_file():
            raise ValueError("model directory and data parquet must exist")
        config = PreTrainedConfig.from_pretrained(args.model)
        if not isinstance(config, SmolVLAConfig):
            raise ValueError("checkpoint is not configured as SmolVLA")
        cameras = dict(args.camera)
        if len(cameras) != len(args.camera):
            raise ValueError("each configured camera key may be mapped only once")
        unknown = sorted(set(cameras) - set(config.image_features))
        if unknown:
            raise ValueError(f"camera keys are not configured by this checkpoint: {', '.join(unknown)}")
        for source in cameras.values():
            if not source.is_file():
                raise ValueError(f"camera video does not exist: {source}")

        table = pq.read_table(args.data, filters=[("index", "=", args.frame_index)])
        if table.num_rows != 1:
            raise ValueError(f"expected exactly one recorded index {args.frame_index}, got {table.num_rows}")
        row = table.to_pylist()[0]
        state = np.asarray(row["observation.state"], dtype=np.float32)[None]
        state_dim = config.input_features["observation.state"].shape[0]
        if state.shape != (1, state_dim) or not np.isfinite(state).all():
            raise ValueError(f"recorded state must be finite [1, {state_dim}], got {state.shape}")

        task = args.task if args.task.endswith("\n") else f"{args.task}\n"
        processor = AutoProcessor.from_pretrained(config.vlm_model_name)
        processor.tokenizer.padding_side = "right"
        encoded = processor.tokenizer(
            task,
            max_length=config.tokenizer_max_length,
            truncation=True,
            padding=config.pad_language_to,
            return_tensors="pt",
        )
        torch.manual_seed(args.noise_seed)
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(1, config.chunk_size, config.max_action_dim),
            dtype=torch.float32,
        ).numpy()
        capture = {
            "observation.state": state,
            "observation.language.tokens": encoded["input_ids"].numpy().astype(np.int64, copy=False),
            "observation.language.attention_mask": encoded["attention_mask"].numpy().astype(np.bool_, copy=False),
            "noise": noise,
            "noisy_actions": noise.copy(),
            "timestep": np.ones((1,), dtype=np.float32),
        }
        for key, source in cameras.items():
            capture[key] = decode_rgb_frame(source, video_frame_index)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **capture)
        report = {
            "schema_version": 1,
            "dataset": args.dataset_id,
            "dataset_revision": args.dataset_revision,
            "dataset_license": args.dataset_license,
            "data_parquet": args.data.name,
            "data_parquet_sha256": digest(args.data),
            "frame_index": args.frame_index,
            "video_frame_index": video_frame_index,
            "recorded_state_shape": list(state.shape),
            "task": task,
            "camera_sources": {
                key: {"file": source.name, "sha256": digest(source)} for key, source in cameras.items()
            },
            "camera_shapes": {key: list(value.shape) for key, value in capture.items() if key in cameras},
            "noise_seed": args.noise_seed,
            "noise_sha256": hashlib.sha256(noise.tobytes(order="C")).hexdigest(),
            "capture_sha256": digest(args.output),
            "scope": (
                "recorded LeRobot observation mapped to configured source camera keys; "
                "no image synthesis, camera duplication, or model-quality claim"
            ),
            "status": "passed",
        }
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"LeRobot frame capture failed: {error}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
