"""Portable LeRobot deployment packages for FlowEdge policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

PACKAGE_SCHEMA = 1
NATIVE_CHECKPOINT = "flowedge_model.safetensors"
SOURCE_DIRECTORY = "source"
SOURCE_MANIFEST = "flowedge-source.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _inside(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} must stay inside the deployment package")
    return path


def _resolve_bundle_root(
    reference: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    force_download: bool,
    token: str | bool | None,
    local_files_only: bool,
) -> Path:
    local = Path(reference)
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(reference),
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
    ).resolve()


def _resolve_native_checkpoint(root: Path, value: str) -> str:
    configured = Path(value)
    if configured.is_absolute():
        if not configured.is_file():
            raise FileNotFoundError(f"FlowEdge checkpoint not found: {configured}")
        return str(configured.resolve())
    checkpoint = _inside(root, root / configured, "checkpoint_path")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FlowEdge deployment package is missing {configured}")
    return str(checkpoint)


def _resolve_source_directory(
    root: Path,
    value: str,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    force_download: bool,
    token: str | bool | None,
    local_files_only: bool,
) -> str:
    configured = Path(value)
    if configured.is_absolute() and configured.is_dir():
        return str(configured.resolve())
    bundled = _inside(root, root / configured, "source_checkpoint_path")
    if bundled.is_dir():
        return str(bundled)
    if configured.is_dir():
        return str(configured.resolve())
    return str(
        _resolve_bundle_root(
            value,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
    )


def load_deployment_policy(
    policy_class,
    pretrained_name_or_path: str | Path,
    *,
    config=None,
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    revision: str | None = None,
    strict: bool = False,
    **kwargs,
):
    """Load a native policy package without requiring dummy PyTorch weights."""

    del resume_download, proxies, strict
    constructor_kwargs = kwargs
    if config is None:
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **kwargs,
        )
        constructor_kwargs = {}
    config = deepcopy(config)
    root = _resolve_bundle_root(
        pretrained_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
    )
    config.checkpoint_path = _resolve_native_checkpoint(root, config.checkpoint_path)
    if getattr(config, "input_mode", None) == "visual":
        config.source_checkpoint_path = _resolve_source_directory(
            root,
            config.source_checkpoint_path,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
    elif config.source_checkpoint_path:
        bundled = root / config.source_checkpoint_path
        if bundled.is_dir():
            config.source_checkpoint_path = str(bundled.resolve())
    instance = policy_class(config, **constructor_kwargs)
    instance.to(config.device)
    instance.eval()
    return instance


def _copy_processor_state(source: Path, destination: Path) -> list[str]:
    processor = source / "policy_preprocessor.json"
    if not processor.is_file():
        return []
    raw = json.loads(processor.read_text(encoding="utf-8"))
    copied = [processor.name]
    shutil.copy2(processor, destination / processor.name)
    for step in raw.get("steps", []):
        state_file = step.get("state_file")
        if not state_file:
            continue
        state = _inside(source, source / state_file, "processor state_file")
        if not state.is_file():
            raise FileNotFoundError(f"processor state file not found: {state_file}")
        target = _inside(destination, destination / state_file, "processor state_file")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(state, target)
        copied.append(state_file)
    return copied


def _write_encoder_checkpoint(source: Path, destination: Path) -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(
        source / "model.safetensors", framework="pt", device="cpu"
    ) as weights:
        for name in weights.keys():
            if name.startswith("diffusion.rgb_encoder.") or name.startswith(
                "normalize_inputs.buffer_"
            ):
                tensors[name] = weights.get_tensor(name)
    if not any(name.startswith("diffusion.rgb_encoder.") for name in tensors):
        raise ValueError("source checkpoint has no Diffusion Policy RGB encoder")
    save_file(tensors, destination)
    return len(tensors)


def create_deployment_package(
    converted: str | Path,
    source: str | Path,
    output: str | Path,
    *,
    action_steps: int | None = None,
    inference_steps: int = 10,
    scheduler: str = "ddim",
    seed: int = 0,
) -> dict:
    """Create a relocatable visual Diffusion Policy package for ``lerobot-rollout``."""

    from flowedge_lerobot.observation import source_config, validate_source_pair
    from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig

    converted = Path(converted).resolve()
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not converted.is_file():
        raise FileNotFoundError(f"converted checkpoint not found: {converted}")
    if not source.is_dir():
        raise FileNotFoundError(f"source policy directory not found: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError(
            "output and source policy directories must not contain each other"
        )
    if inference_steps <= 0 or seed < 0 or scheduler not in {"ddim", "ddpm"}:
        raise ValueError("invalid inference_steps, seed, or scheduler")

    identity = validate_source_pair(converted, source)
    source_policy = source_config(source)
    selected_action_steps = (
        source_policy.n_action_steps if action_steps is None else action_steps
    )
    if selected_action_steps <= 0 or selected_action_steps > source_policy.horizon:
        raise ValueError("action_steps must fit the source policy horizon")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="flowedge-package-", dir=output.parent
    ) as tmp:
        package = Path(tmp) / "package"
        slim_source = package / SOURCE_DIRECTORY
        slim_source.mkdir(parents=True)
        shutil.copy2(converted, package / NATIVE_CHECKPOINT)
        shutil.copy2(source / "config.json", slim_source / "config.json")
        processor_files = _copy_processor_state(source, slim_source)
        encoder_tensors = _write_encoder_checkpoint(
            source, slim_source / "model.safetensors"
        )

        source_model_sha256 = _sha256(slim_source / "model.safetensors")
        source_manifest = {
            "schema": PACKAGE_SCHEMA,
            "original_model_sha256": identity["source_sha256"],
            "source_config_sha256": identity["source_config_sha256"],
            "encoder_model_sha256": source_model_sha256,
            "encoder_tensors": encoder_tensors,
            "processor_files": processor_files,
        }
        (slim_source / SOURCE_MANIFEST).write_text(
            json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
        )
        validate_source_pair(package / NATIVE_CHECKPOINT, slim_source)

        config = FlowEdgeConfig(
            n_obs_steps=source_policy.n_obs_steps,
            input_features=source_policy.input_features,
            output_features=source_policy.output_features,
            device="cpu",
            push_to_hub=False,
            checkpoint_path=NATIVE_CHECKPOINT,
            source_checkpoint_path=SOURCE_DIRECTORY,
            input_mode="visual",
            action_steps=selected_action_steps,
            seed=seed,
            inference_steps=inference_steps,
            scheduler=scheduler,
        )
        config.save_pretrained(package)
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "policy_type": "flowedge",
            "converted_sha256": _sha256(package / NATIVE_CHECKPOINT),
            "source_original_sha256": identity["source_sha256"],
            "source_config_sha256": identity["source_config_sha256"],
            "source_encoder_sha256": source_model_sha256,
            "action_steps": selected_action_steps,
            "inference_steps": inference_steps,
            "scheduler": scheduler,
            "default_device": "cpu",
            "default_precision": "exact-f32",
        }
        (package / "flowedge-package.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (package / "README.md").write_text(
            "# FlowEdge LeRobot policy\n\n"
            "This package is loaded directly by LeRobot 0.6.1 through the installed "
            "`lerobot_policy_flowedge` plugin.\n\n"
            "```bash\n"
            "lerobot-rollout \\\n"
            "  --strategy.type=base \\\n"
            "  --policy.path=/path/to/this/package \\\n"
            "  --robot.type=so101_follower \\\n"
            "  --robot.port=/dev/ttyACM0 \\\n"
            '  --task="describe the trained task" \\\n'
            "  --duration=60\n"
            "```\n\n"
            "LeRobot owns the robot driver, joint limits, cameras, and emergency stop. "
            "Start with the robot unloaded and a short duration.\n",
            encoding="utf-8",
        )
        package.replace(output)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowedge-lerobot-package",
        description=(
            "Create a relocatable FlowEdge Diffusion Policy package for LeRobot rollout."
        ),
    )
    parser.add_argument("converted", help="converted FlowEdge Diffusion Policy")
    parser.add_argument("source", help="matching LeRobot Diffusion Policy directory")
    parser.add_argument("output", help="new deployment package directory")
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--scheduler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = create_deployment_package(
        args.converted,
        args.source,
        args.output,
        action_steps=args.action_steps,
        inference_steps=args.inference_steps,
        scheduler=args.scheduler,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
