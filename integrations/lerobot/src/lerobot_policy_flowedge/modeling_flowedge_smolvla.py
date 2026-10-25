"""LeRobot policy wrapper for the native SmolVLA action expert."""

from pathlib import Path

import numpy as np
import torch
from flowedge_lerobot import FlowEdgeSmolVLACachedExpert, LeRobotSmolVLACacheProvider
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_flowedge import FlowEdgeSmolVLAConfig


class FlowEdgeSmolVLAPolicy(PreTrainedPolicy):
    config_class = FlowEdgeSmolVLAConfig
    name = "flowedge_smolvla"

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Load the native deployment package without dummy Torch weights."""

        from flowedge_lerobot.deployment import load_deployment_policy

        return load_deployment_policy(cls, pretrained_name_or_path, **kwargs)

    def __init__(self, config: FlowEdgeSmolVLAConfig, *args, **kwargs):
        expert = kwargs.pop("expert", None)
        cache_provider = kwargs.pop("cache_provider", None)
        super().__init__(config, *args, **kwargs)
        config.validate_features()
        from flowedge_lerobot.device import config_device, require_torch_cuda

        device = config_device(config)
        require_torch_cuda(device, torch)
        action_dim = int(config.action_feature.shape[0])
        if expert is None:
            import flowedge

            engine = flowedge.Engine(config.checkpoint_path)
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
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            # A portable package contains VLM weights only. The source policy
            # constructs its expert briefly, then the provider releases it.
            source_config = PreTrainedConfig.from_pretrained(
                config.source_checkpoint_path
            )
            if source_config.type != "smolvla":
                raise ValueError("source checkpoint must contain a SmolVLA policy")
            source_config.device = device
            source_config.load_vlm_weights = False
            bundled_vlm = Path(config.source_checkpoint_path) / "vlm_assets"
            if bundled_vlm.is_dir():
                source_config.vlm_model_name = str(bundled_vlm)
            source = SmolVLAPolicy.from_pretrained(
                config.source_checkpoint_path, config=source_config, strict=False
            )
            LeRobotSmolVLACacheProvider.release_source_expert(source)
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
