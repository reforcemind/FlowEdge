"""LeRobot config for the hybrid SmolVLA action expert."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig

from .configuration_flowedge import _fail_closed_device


@PreTrainedConfig.register_subclass("flowedge_smolvla")
@dataclass
class FlowEdgeSmolVLAConfig(PreTrainedConfig):
    checkpoint_path: str = ""
    source_checkpoint_path: str = ""
    action_steps: int | None = None
    seed: int = 0
    inference_steps: int = 10

    def __post_init__(self) -> None:
        _fail_closed_device(self)

    def get_optimizer_preset(self) -> AdamWConfig:
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if self.action_steps is not None and self.action_steps <= 0:
            raise ValueError("action_steps must be positive")
        if self.seed < 0 or self.inference_steps <= 0:
            raise ValueError("invalid seed or inference_steps")
        if self.robot_state_feature is None or self.action_feature is None:
            raise ValueError(
                "FlowEdge SmolVLA requires observation.state input and action output features"
            )
        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path must name a native SmolVLA expert checkpoint"
            )
        if not self.source_checkpoint_path:
            raise ValueError(
                "source_checkpoint_path must name the LeRobot SmolVLA directory that owns the VLM"
            )

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None if self.action_steps is None else list(range(self.action_steps))

    @property
    def reward_delta_indices(self):
        return None
