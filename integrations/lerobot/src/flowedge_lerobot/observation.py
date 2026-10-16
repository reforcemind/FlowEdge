"""Observation path for the supported LeRobot visual Diffusion Policy.

The source directory supplies the original encoder and normalization statistics.
Native FlowEdge still owns only the converted denoiser and action unnormalization.
The RGB encoder may run on CPU or CUDA; Core still sees a host condition vector.
"""

from collections import deque
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_pair(converted, directory):
    """Reject an encoder from a different training checkpoint before inference."""
    directory = Path(directory)
    with safe_open(converted, framework="numpy") as weights:
        metadata = weights.metadata() or {}
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    source_hash = _sha256(directory / "model.safetensors")
    manifest_path = directory / "flowedge-source.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 1:
            raise ValueError("unsupported FlowEdge source package schema")
        if (
            manifest.get("encoder_model_sha256") != source_hash
            or manifest.get("source_config_sha256") != config_hash
        ):
            raise ValueError("FlowEdge source package integrity mismatch")
        source_hash = manifest.get("original_model_sha256", "")
    if (
        metadata.get("flowedge.source_sha256") != source_hash
        or metadata.get("flowedge.source_config_sha256") != config_hash
    ):
        raise ValueError(
            "source/converted identity mismatch; reconvert from this exact source directory"
        )
    return {
        "source_sha256": source_hash,
        "source_config_sha256": config_hash,
    }


def source_config(directory, device="cpu"):
    if device not in {"cpu", "cuda"} and not str(device).startswith("cuda"):
        raise ValueError("observation encoder device must be cpu or cuda")
    directory = Path(directory)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if raw.get("type") != "diffusion":
        raise ValueError("source checkpoint must be a LeRobot Diffusion Policy")
    args = {
        k: v for k, v in raw.items() if k in {f.name for f in fields(DiffusionConfig)}
    }
    for key in ("input_features", "output_features"):
        args[key] = {
            k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
            for k, v in raw[key].items()
        }
    args["normalization_mapping"] = {
        k: NormalizationMode(v) for k, v in raw["normalization_mapping"].items()
    }
    args.update(
        device=str(device), pretrained_backbone_weights=None, compile_model=False
    )
    config = DiffusionConfig(**args)
    config.validate_features()
    if config.use_separate_rgb_encoder_per_camera:
        raise ValueError("separate camera encoders are not supported by this adapter")
    if not config.image_features:
        raise ValueError("visual adapter requires source image features")
    return config


def source_stats(directory, config):
    """Load legacy embedded statistics or modern processor sidecars, failing closed."""
    directory = Path(directory)
    stats = {}
    with safe_open(directory / "model.safetensors", framework="numpy") as weights:
        for name in config.input_features:
            prefix = "normalize_inputs.buffer_" + name.replace(".", "_") + "."
            values = {
                k[len(prefix) :]: torch.from_numpy(weights.get_tensor(k).copy())
                for k in weights.keys()
                if k.startswith(prefix)
            }
            if values:
                stats[name] = values
    processor = directory / "policy_preprocessor.json"
    if processor.exists():
        for step in json.loads(processor.read_text(encoding="utf-8"))["steps"]:
            if step.get("registry_name") != "normalizer_processor":
                continue
            state = (directory / step["state_file"]).resolve()
            if not state.is_relative_to(directory.resolve()):
                raise ValueError(
                    "processor state must stay inside the source directory"
                )
            with safe_open(state, framework="numpy") as weights:
                for name in config.input_features:
                    values = {
                        k[len(name) + 1 :]: torch.from_numpy(
                            weights.get_tensor(k).copy()
                        )
                        for k in weights.keys()
                        if k.startswith(name + ".")
                    }
                    if values:
                        stats[name] = values
    validate_stats(config, stats)
    return stats


def validate_stats(config, stats):
    for name, feature in config.input_features.items():
        mode = config.normalization_mapping[feature.type.value]
        required = {
            NormalizationMode.MIN_MAX: ("min", "max"),
            NormalizationMode.MEAN_STD: ("mean", "std"),
        }.get(mode, ())
        for key in required:
            if (
                key not in stats.get(name, {})
                or not torch.isfinite(torch.as_tensor(stats[name][key])).all()
            ):
                raise ValueError(
                    f"missing or non-finite normalization statistic {name}.{key}"
                )
        if mode == NormalizationMode.MIN_MAX and torch.any(
            torch.as_tensor(stats[name]["max"]) <= torch.as_tensor(stats[name]["min"])
        ):
            raise ValueError(f"invalid normalization range for {name}")
        if mode == NormalizationMode.MEAN_STD and torch.any(
            torch.as_tensor(stats[name]["std"]) <= 0
        ):
            raise ValueError(f"invalid normalization standard deviation for {name}")


def observation_processor(directory, dataset_stats=None):
    config = source_config(directory)
    stats = source_stats(directory, config) if dataset_stats is None else dataset_stats
    validate_stats(config, stats)
    # Outputs are already unnormalized by the native checkpoint; normalize inputs only.
    config.output_features = {}
    return make_diffusion_pre_post_processors(config, stats)[0]


class DiffusionObservationEncoder:
    def __init__(self, directory, device="cpu"):
        self.device = torch.device(device)
        self.config = source_config(directory, device=str(self.device))
        self.processor = observation_processor(directory)
        self.rgb_encoder = DiffusionRgbEncoder(self.config).eval().to(self.device)
        with safe_open(
            Path(directory) / "model.safetensors", framework="numpy"
        ) as weights:
            prefix = "diffusion.rgb_encoder."
            state = {
                k[len(prefix) :]: torch.from_numpy(weights.get_tensor(k).copy())
                for k in weights.keys()
                if k.startswith(prefix)
            }
        self.rgb_encoder.load_state_dict(state, strict=True)
        self.rgb_encoder.to(self.device)
        self.history = deque(maxlen=self.config.n_obs_steps)

    def reset(self):
        self.history.clear()

    def observe(self, batch, *, normalized=False):
        batch = batch if normalized else self.processor(batch)
        device = getattr(self, "device", torch.device("cpu"))
        frame = {}
        for name, feature in self.config.input_features.items():
            value = batch.get(name)
            if value is None or tuple(value.shape) != (1, *feature.shape):
                raise ValueError(
                    f"{name} must have shape {(1, *feature.shape)} after preprocessing"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            frame[name] = value.detach().to(device, dtype=torch.float32).clone()
        if not self.history:
            self.history.extend([frame] * self.config.n_obs_steps)
        else:
            self.history.append(frame)

    @torch.inference_mode()
    def condition(self):
        if not self.history:
            raise ValueError("observe must precede condition")
        device = getattr(self, "device", torch.device("cpu"))
        features = [
            torch.stack([f["observation.state"] for f in self.history], dim=1).to(
                device
            )
        ]
        images = torch.stack(
            [
                torch.stack([f[k] for k in self.config.image_features], dim=1)
                for f in self.history
            ],
            dim=1,
        ).to(device)
        n = self.config.n_obs_steps
        encoded = self.rgb_encoder(images.flatten(0, 2)).reshape(1, n, -1)
        features.append(encoded)
        if self.config.env_state_feature:
            features.append(
                torch.stack(
                    [f["observation.environment_state"] for f in self.history], dim=1
                ).to(device)
            )
        condition = torch.cat(features, dim=-1).flatten().detach().contiguous()
        if condition.device.type == "cuda":
            return condition
        return np.ascontiguousarray(condition.numpy())
