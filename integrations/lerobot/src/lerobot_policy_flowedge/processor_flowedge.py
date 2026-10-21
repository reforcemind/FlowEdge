"""LeRobot processor factory for the FlowEdge deployment plugin."""

from flowedge_lerobot.device import config_device
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    IdentityProcessorStep,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)


def make_flowedge_pre_post_processors(config, dataset_stats=None):
    config.validate_features()
    if config.input_mode == "visual":
        from flowedge_lerobot.observation import observation_processor

        preprocessor = observation_processor(
            config.source_checkpoint_path, dataset_stats
        )
    else:
        if dataset_stats:
            raise ValueError(
                "encoded conditions must already be normalized; dataset_stats requires visual input"
            )
        preprocessor = PolicyProcessorPipeline(
            steps=[
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=config_device(config)),
            ],
            name="flowedge_preprocessor",
        )
    return (
        preprocessor,
        PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device="cpu"), IdentityProcessorStep()],
            name="flowedge_postprocessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def make_flowedge_smolvla_pre_post_processors(config, dataset_stats=None):
    """Use the paired source SmolVLA preprocessing contract for the hybrid."""

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    source = PreTrainedConfig.from_pretrained(config.source_checkpoint_path)
    if not isinstance(source, SmolVLAConfig):
        raise ValueError("source_checkpoint_path must contain a SmolVLA policy")
    source.device = config_device(config)
    if config.input_features:
        source.input_features = config.input_features
    if config.output_features:
        source.output_features = config.output_features
    from pathlib import Path

    bundled_vlm = Path(config.source_checkpoint_path) / "vlm_assets"
    if not bundled_vlm.is_dir():
        if dataset_stats is not None:
            from lerobot.policies.smolvla.processor_smolvla import (
                make_smolvla_pre_post_processors,
            )

            return make_smolvla_pre_post_processors(source, dataset_stats)
        raise FileNotFoundError("SmolVLA package is missing bundled VLM assets")
    source.vlm_model_name = str(bundled_vlm)
    return make_pre_post_processors(
        source,
        pretrained_path=config.source_checkpoint_path,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": config_device(config)},
            "tokenizer_processor": {"tokenizer_name": str(bundled_vlm)},
        },
    )
