"""LeRobot policy wrapper for the native SmolVLA action expert."""

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from flowedge_lerobot import FlowEdgeSmolVLACachedExpert, LeRobotSmolVLACacheProvider

from .configuration_flowedge import FlowEdgeSmolVLAConfig


class FlowEdgeSmolVLAPolicy(PreTrainedPolicy):
    config_class = FlowEdgeSmolVLAConfig
    name = "flowedge_smolvla"

    def __init__(self, config: FlowEdgeSmolVLAConfig, *args, **kwargs):
        expert = kwargs.pop("expert", None)
        cache_provider = kwargs.pop("cache_provider", None)
        super().__init__(config, *args, **kwargs)
        config.validate_features()
        from flowedge_lerobot.device import config_device

        device = config_device(config)
        action_dim = int(config.action_feature.shape[0])
        if expert is None:
            import flowedge
            from flowedge_lerobot.device import require_native_device

            require_native_device(device, flowedge)
            engine = flowedge.Engine(config.checkpoint_path)
            require_native_device(device, flowedge, engine=engine)
            expert = FlowEdgeSmolVLACachedExpert(
                engine,
                action_dim=action_dim,
                action_steps=config.action_steps,
                seed=config.seed,
            )
        if config.action_steps is None:
            config.action_steps = expert.action_steps
        if action_dim != expert.action_dim:
            raise ValueError("action feature disagrees with the native expert")
        if cache_provider is None:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            source = SmolVLAPolicy.from_pretrained(config.source_checkpoint_path)
            source.to(device)
            cache_provider = LeRobotSmolVLACacheProvider(
                source, expert_layers=expert.contract.expert_layers
            )
        self._expert = expert
        self._provider = cache_provider
        self.reset()

    def get_optim_params(self) -> dict:
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def reset(self) -> None:
        self._expert.reset()
        self._provider.reset()

    def forward(self, batch):
        del batch
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def _as_noise(self, noise):
        if noise is None:
            return None
        if isinstance(noise, torch.Tensor):
            noise = noise.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(noise, dtype=np.float32))

    def predict_action_chunk(self, batch, **kwargs):
        steps = kwargs.get("steps", self.config.inference_steps)
        cache = self._provider(batch)
        chunk = self._expert.predict_action_chunk(
            cache, noise=self._as_noise(kwargs.get("noise")), steps=steps
        )
        self._expert.reset()
        return torch.from_numpy(chunk.copy()).unsqueeze(0)

    def select_action(self, batch, **kwargs):
        if self._expert.remaining_actions == 0:
            cache = self._provider(batch)
            self._expert.refill(
                cache,
                noise=self._as_noise(kwargs.get("noise")),
                steps=kwargs.get("steps", self.config.inference_steps),
            )
        return torch.from_numpy(self._expert.take_action()).unsqueeze(0)
