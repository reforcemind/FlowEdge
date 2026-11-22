"""Independent upstream LeRobot U-Net and diffusers DDIM reference."""

import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d
from safetensors import safe_open

from .observation import source_config


def action_statistics(directory):
    """Read the source policy's MIN_MAX action statistics independently."""
    directory = Path(directory).resolve()
    processor = directory / "policy_postprocessor.json"
    if processor.is_file():
        config = json.loads(processor.read_text(encoding="utf-8"))
        steps = config.get("steps", [])
        matches = [
            step
            for step in steps
            if isinstance(step, dict)
            and step.get("registry_name") == "unnormalizer_processor"
        ]
        if len(matches) != 1:
            raise ValueError("postprocessor must contain one action unnormalizer")
        step = matches[0]
        if step.get("config", {}).get("norm_map", {}).get("ACTION") != "MIN_MAX":
            raise ValueError("reference requires MIN_MAX action normalization")
        filename = step.get("state_file")
        if not isinstance(filename, str) or not filename:
            raise ValueError("postprocessor state_file is required")
        state = (directory / filename).resolve()
        if not state.is_relative_to(directory):
            raise ValueError(
                "postprocessor state must stay inside the source directory"
            )
        names = ("action.min", "action.max")
    else:
        state = directory / "model.safetensors"
        names = (
            "unnormalize_outputs.buffer_action.min",
            "unnormalize_outputs.buffer_action.max",
        )
    with safe_open(state, framework="numpy") as weights:
        if not processor.is_file() and names[0] not in weights.keys():
            names = (
                "normalize_targets.buffer_action.min",
                "normalize_targets.buffer_action.max",
            )
        if any(name not in weights.keys() for name in names):
            raise ValueError("source checkpoint is missing action MIN_MAX statistics")
        minimum, maximum = (
            torch.from_numpy(weights.get_tensor(name).copy()) for name in names
        )
    if (
        minimum.ndim != 1
        or minimum.shape != maximum.shape
        or not torch.isfinite(minimum).all()
        or not torch.isfinite(maximum).all()
        or not torch.all(maximum > minimum)
    ):
        raise ValueError("invalid action MIN_MAX statistics in source checkpoint")
    return minimum, maximum


class TorchDiffusionReference:
    def __init__(self, directory, condition_dim, device="cpu", compile_mode="none"):
        self.device = torch.device(device)
        self.config = source_config(directory, device=str(self.device))
        config = self.config
        # Construct shapes without a second gigabyte-sized random weight allocation.
        with torch.device("meta"):
            self.unet = DiffusionConditionalUnet1d(
                config, global_cond_dim=condition_dim
            ).eval()
        with safe_open(
            Path(directory) / "model.safetensors", framework="numpy"
        ) as weights:
            prefix = "diffusion.unet."
            state = {
                k[len(prefix) :]: torch.from_numpy(weights.get_tensor(k).copy())
                for k in weights.keys()
                if k.startswith(prefix)
            }
        self.minimum, self.maximum = action_statistics(directory)
        self.unet.load_state_dict(state, strict=True, assign=True)
        self.unet.to(self.device)
        if compile_mode != "none":
            self.unet = torch.compile(self.unet, mode=compile_mode, fullgraph=True)
        self.minimum = self.minimum.to(self.device)
        self.maximum = self.maximum.to(self.device)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

    @torch.inference_mode()
    def predict_action_chunk(
        self, condition, noise, *, steps=10, scheduler="ddim", seed=0
    ):
        if scheduler != "ddim":
            raise ValueError("matched reference supports deterministic DDIM only")
        sample = (
            torch.from_numpy(np.asarray(noise, dtype=np.float32))
            .unsqueeze(0)
            .clone()
            .to(self.device)
        )
        if torch.is_tensor(condition):
            condition = condition.to(device=self.device, dtype=torch.float32).reshape(
                1, -1
            )
        else:
            condition = (
                torch.from_numpy(np.asarray(condition, dtype=np.float32))
                .reshape(1, -1)
                .to(self.device)
            )
        self.scheduler.set_timesteps(steps)
        for timestep in self.scheduler.timesteps:
            step = timestep.to(sample.device).reshape(1)
            velocity = self.unet(sample, step, global_cond=condition)
            sample = self.scheduler.step(velocity, timestep, sample, eta=0).prev_sample
        # Match the source policy's MIN_MAX output transformation (not the native implementation).
        sample = (sample + 1) / 2 * (self.maximum - self.minimum) + self.minimum
        start = self.config.n_obs_steps - 1
        return (
            sample[0, start : start + self.config.n_action_steps]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
