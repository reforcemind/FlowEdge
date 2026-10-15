"""LeRobot configuration for a converted FlowEdge Diffusion Policy."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig


def _fail_closed_device(config: PreTrainedConfig) -> None:
    """Keep an explicit edge backend from silently falling back to CPU."""

    requested = str(config.device or "cpu")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError("FlowEdge device must be cpu or cuda")
    config.device = requested
    PreTrainedConfig.__post_init__(config)
    if config.device != requested:
        raise RuntimeError(
            f"FlowEdge requested device '{requested}', but LeRobot selected "
            f"'{config.device}'. Install a matching runtime or request device=cpu."
        )


@PreTrainedConfig.register_subclass("flowedge")
@dataclass
class FlowEdgeConfig(PreTrainedConfig):
    checkpoint_path: str = ""
    action_steps: int | None = None
    input_mode: str = "encoded"
    source_checkpoint_path: str = ""
    seed: int = 0
    inference_steps: int = 10
    scheduler: str = "ddim"
    diffusion_precision: str = "exact-f32"

    def __post_init__(self) -> None:
        _fail_closed_device(self)
        if self.diffusion_precision not in {"exact-f32", "approx-fp16-deep"}:
            raise ValueError("unsupported FlowEdge diffusion precision")
        if self.diffusion_precision == "approx-fp16-deep" and not str(
            self.device
        ).startswith("cuda"):
            raise ValueError("approx-fp16-deep precision requires device=cuda")

    def get_optimizer_preset(self) -> AdamWConfig:
        raise RuntimeError(
            "FlowEdge is deployment-only; train the source policy in LeRobot"
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if self.input_mode not in {"encoded", "visual"}:
            raise ValueError("input_mode must be encoded or visual")
        if self.input_mode == "visual" and not self.source_checkpoint_path:
            raise ValueError("visual input requires source_checkpoint_path")
        if self.input_mode == "encoded" and self.image_features:
            raise ValueError("encoded input does not accept images; use visual input")
        if self.action_steps is not None and self.action_steps <= 0:
            raise ValueError("action_steps must be positive")
        if (
            self.seed < 0
            or self.inference_steps <= 0
            or self.scheduler not in {"ddim", "ddpm"}
        ):
            raise ValueError("invalid seed, inference_steps, or scheduler")
        if self.robot_state_feature is None or self.action_feature is None:
            raise ValueError(
                "FlowEdge plugin requires observation.state input and action output features"
            )
        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path must name a converted FlowEdge checkpoint"
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
