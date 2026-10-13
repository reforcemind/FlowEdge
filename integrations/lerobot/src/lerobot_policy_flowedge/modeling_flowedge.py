"""Inference-only LeRobot policy wrapper around a converted FlowEdge checkpoint."""

import numpy as np
import torch

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE

from flowedge_lerobot import FlowEdgeDiffusionPolicy

from .configuration_flowedge import FlowEdgeConfig


class FlowEdgePolicy(PreTrainedPolicy):
    config_class = FlowEdgeConfig
    name = "flowedge"

    def __init__(self, config: FlowEdgeConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        config.validate_features()
        from flowedge_lerobot.device import config_device

        device = config_device(config)
        self._policy = FlowEdgeDiffusionPolicy.from_checkpoint(
            config.checkpoint_path, device=device
        )
        if config.action_steps is None:
            config.action_steps = self._policy.action_steps
        if config.action_steps > self._policy.action_steps:
            raise ValueError(
                "action_steps exceeds the converted checkpoint's executable horizon"
            )
        if tuple(config.action_feature.shape) != (self._policy.action_dim,):
            raise ValueError("action feature disagrees with converted checkpoint")
        self._encoder = None
        if config.input_mode == "visual":
            from flowedge_lerobot.device import require_native_device
            from flowedge_lerobot.observation import (
                DiffusionObservationEncoder,
                validate_source_pair,
            )

            if str(device).startswith("cuda"):
                import flowedge

                require_native_device("cuda", flowedge, engine=self._policy)
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but torch.cuda is unavailable")
            validate_source_pair(config.checkpoint_path, config.source_checkpoint_path)
            self._encoder = DiffusionObservationEncoder(
                config.source_checkpoint_path, device=device
            )
            source = self._encoder.config
            if (
                source.horizon != self._policy.metadata.horizon
                or source.n_obs_steps != self._policy.metadata.observation_steps
                or source.n_action_steps != self._policy.action_steps
                or source.input_features != config.input_features
            ):
                raise ValueError(
                    "source and converted observation/action contracts disagree"
                )
        elif tuple(config.robot_state_feature.shape) != (
            self._policy.metadata.condition_dim,
        ):
            raise ValueError(
                "encoded observation.state must contain the full condition vector"
            )
        self._noise = np.empty(
            (self._policy.metadata.horizon, self._policy.action_dim), dtype=np.float32
        )
        self._chunk = np.empty(
            (self._policy.action_steps, self._policy.action_dim), dtype=np.float32
        )
        self.reset()

    def get_optim_params(self) -> dict:
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def reset(self) -> None:
        self._policy.reset()
        if self._encoder is not None:
            self._encoder.reset()
        self._rng = np.random.default_rng(self.config.seed)
        self._next_action = self.config.action_steps
        self._condition_value = None

    def forward(self, batch):
        del batch
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def predict_action_chunk(self, batch, **kwargs):
        self._observe(batch)
        self._sample(**kwargs)
        # A direct prediction invalidates any queued actions from a previous prediction.
        self._next_action = self.config.action_steps
        return torch.from_numpy(
            self._chunk[: self.config.action_steps].copy()
        ).unsqueeze(0)

    def _sample(self, *, noise=None, steps=None, scheduler=None, seed=None):
        if noise is None:
            self._rng.standard_normal(
                self._noise.shape, dtype=np.float32, out=self._noise
            )
        else:
            supplied = (
                noise.detach().cpu().numpy()
                if isinstance(noise, torch.Tensor)
                else np.asarray(noise)
            )
            if supplied.shape == (1, *self._noise.shape):
                supplied = supplied[0]
            if supplied.shape != self._noise.shape or not np.isfinite(supplied).all():
                raise ValueError("noise must match the finite full-horizon noise shape")
            np.copyto(self._noise, supplied)
        condition = (
            self._encoder.condition()
            if self._encoder is not None
            else self._condition_value
        )
        self._policy.predict_action_chunk_into(
            condition,
            self._noise,
            self._chunk,
            steps=self.config.inference_steps if steps is None else steps,
            scheduler=self.config.scheduler if scheduler is None else scheduler,
            seed=int(self._rng.integers(0, 2**32)) if seed is None else seed,
        )

    def select_action(self, batch, **kwargs):
        self._observe(batch)
        if self._next_action >= self.config.action_steps:
            self._sample(**kwargs)
            self._next_action = 0
        action = torch.from_numpy(self._chunk[self._next_action].copy()).unsqueeze(0)
        self._next_action += 1
        return action

    def _observe(self, batch):
        if self._encoder is not None:
            self._encoder.observe(batch, normalized=True)
            return
        state = batch.get(OBS_STATE)
        if state is None or tuple(state.shape) != (
            1,
            self._policy.metadata.condition_dim,
        ):
            raise ValueError(
                "FlowEdge plugin requires a batch of one observation.state"
            )
        if not torch.isfinite(state).all():
            raise ValueError("encoded condition contains non-finite values")
        self._condition_value = (
            state.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        )


# LeRobot resolves third-party classes from the configuration module's sibling
# modeling module. Both FlowEdge configs live in configuration_flowedge.py, so
# expose the SmolVLA policy here as well as in its implementation module.
from .modeling_flowedge_smolvla import FlowEdgeSmolVLAPolicy  # noqa: E402,F401
