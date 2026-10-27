"""Small inference API for packaged FlowEdge LeRobot policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedPolicy:
    """Own the registered policy and its matching observation processors."""

    policy: Any
    preprocessor: Any
    postprocessor: Any

    @property
    def input_features(self) -> dict:
        return self.policy.config.input_features

    def reset(self) -> None:
        """Start a fresh episode and clear policy and processor state."""
        self.policy.reset()
        for processor in (self.preprocessor, self.postprocessor):
            reset = getattr(processor, "reset", None)
            if callable(reset):
                reset()

    def select_action(self, observation: dict, *, task: str | None = None):
        """Return a postprocessed LeRobot action tensor shaped ``(1, action_dim)``.

        The result is not the named joint mapping accepted by a robot driver;
        LeRobot's rollout path performs that final mapping.
        """
        import torch

        raw = dict(observation)
        if self.policy.config.type == "flowedge_smolvla":
            if not task:
                raise ValueError("SmolVLA requires a non-empty task instruction")
            raw["task"] = task
        with torch.inference_mode():
            batch = self.preprocessor(raw)
            action = self.policy.select_action(batch)
            return self.postprocessor(action)


def load_policy(
    reference: str | Path,
    *,
    device: str = "cpu",
    verify_integrity: bool = False,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> LoadedPolicy:
    """Load a portable Diffusion Policy or hybrid SmolVLA package.

    ``reference`` may be a local package directory or a Hugging Face repo ID.
    Pass ``revision`` to pin a remote package. For Diffusion Policy, ``cuda``
    selects native FlowEdge CUDA and fails rather than falling back to CPU. For
    SmolVLA, ``cuda`` selects a hybrid CUDA VLM plus CPU action expert. By
    default, loading checks package structure and file presence without a second
    full read to hash model weights. Set ``verify_integrity=True`` for full
    manifest hashing, or run ``flowedge-lerobot-preflight`` before deployment.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    from flowedge_lerobot.deployment import validate_deployment_package

    package = validate_deployment_package(
        reference,
        verify_integrity=verify_integrity,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
    )
    policy_type = package["manifest"]["policy_type"]
    if policy_type == "flowedge_smolvla" and device == "cuda":
        from flowedge_lerobot.device import require_torch_cuda

        require_torch_cuda(device)

    import lerobot_policy_flowedge  # noqa: F401  Registers the plugin.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot_policy_flowedge.processor_flowedge import (
        make_flowedge_pre_post_processors,
        make_flowedge_smolvla_pre_post_processors,
    )

    config = PreTrainedConfig.from_pretrained(package["root"])
    if config.type != policy_type:
        raise ValueError("policy type disagrees with the deployment package")
    config.device = device
    policy = get_policy_class(policy_type).from_pretrained(
        package["root"], config=config
    )
    processor_factory = (
        make_flowedge_pre_post_processors
        if policy_type == "flowedge"
        else make_flowedge_smolvla_pre_post_processors
    )
    preprocessor, postprocessor = processor_factory(policy.config)
    return LoadedPolicy(policy, preprocessor, postprocessor)
