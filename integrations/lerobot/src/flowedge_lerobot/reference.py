"""Independent upstream LeRobot U-Net and diffusers DDIM reference."""

from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from safetensors import safe_open
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

from .observation import source_config


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
            self.minimum = torch.from_numpy(
                weights.get_tensor("unnormalize_outputs.buffer_action.min").copy()
            )
            self.maximum = torch.from_numpy(
                weights.get_tensor("unnormalize_outputs.buffer_action.max").copy()
            )
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
