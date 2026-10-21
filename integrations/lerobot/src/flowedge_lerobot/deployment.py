"""Portable LeRobot deployment packages for FlowEdge policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

PACKAGE_SCHEMA = 1
NATIVE_CHECKPOINT = "flowedge_model.safetensors"
SOURCE_DIRECTORY = "source"
SOURCE_MANIFEST = "flowedge-source.json"
PACKAGE_MANIFEST = "flowedge-package.json"
VLM_ASSETS = "vlm_assets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hub_blob_directory(snapshot_root: Path) -> Path | None:
    """Return the blob store for a standard Hugging Face model snapshot."""
    snapshot_root = snapshot_root.resolve()
    cache_repo = snapshot_root.parent.parent
    if snapshot_root.parent.name != "snapshots" or not cache_repo.name.startswith(
        "models--"
    ):
        return None
    blobs = cache_repo / "blobs"
    return blobs.resolve() if blobs.is_dir() else None


def _asset_file_hashes(
    directory: Path, *, allowed_blob_root: Path | None = None
) -> dict[str, str]:
    directory = directory.resolve()
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            resolved = path.resolve()
            if (
                allowed_blob_root is None
                or not resolved.is_relative_to(allowed_blob_root.resolve())
                or not resolved.is_file()
            ):
                raise ValueError(
                    "VLM asset symlinks must target files in the Hugging Face blob store"
                )
        if path.is_file():
            files[str(path.relative_to(directory)).replace("\\", "/")] = _sha256(
                path
            )
    return files


def _asset_bundle_sha256(files: dict[str, str]) -> str:
    encoded = json.dumps(
        sorted(files.items()), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_sha256(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _inside(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    candidate = Path(os.path.abspath(path))
    if not candidate.is_relative_to(root):
        raise ValueError(f"{label} must stay inside the deployment package")
    resolved = candidate.resolve()
    blob_root = _hub_blob_directory(root)
    if not resolved.is_relative_to(root) and not (
        blob_root is not None and resolved.is_relative_to(blob_root)
    ):
        raise ValueError(f"{label} must stay inside the deployment package")
    return resolved


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


def validate_deployment_package(
    reference: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> dict:
    """Resolve and verify a portable FlowEdge policy package.

    This intentionally hashes the large native checkpoint. Keep it on explicit
    preflight and packaging paths rather than every policy load.
    """

    from flowedge_lerobot.observation import validate_source_pair

    root = _resolve_bundle_root(
        reference,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
    )
    manifest_path = _inside(root, root / PACKAGE_MANIFEST, "package manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"deployment package is missing {PACKAGE_MANIFEST}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("unsupported FlowEdge deployment package schema")
    if manifest.get("policy_type") == "flowedge_smolvla":
        return _validate_smolvla_package(root, manifest)
    if manifest.get("policy_type") != "flowedge":
        raise ValueError("deployment package must contain a FlowEdge policy")

    config_path = _inside(root, root / "config.json", "config.json")
    if not config_path.is_file():
        raise FileNotFoundError("deployment package is missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") != "flowedge" or config.get("input_mode") != "visual":
        raise ValueError("deployment config must describe a visual FlowEdge policy")

    checkpoint = Path(_resolve_native_checkpoint(root, config["checkpoint_path"]))
    if _sha256(checkpoint) != manifest.get("converted_sha256"):
        raise ValueError("native checkpoint integrity mismatch")
    source = Path(
        _resolve_source_directory(
            root,
            config["source_checkpoint_path"],
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
    )
    if not source.is_relative_to(root):
        raise ValueError("deployment source directory must stay inside the package")
    identity = validate_source_pair(checkpoint, source)
    if (
        identity["source_sha256"] != manifest.get("source_original_sha256")
        or identity["source_config_sha256"] != manifest.get("source_config_sha256")
        or _sha256(source / "model.safetensors")
        != manifest.get("source_encoder_sha256")
    ):
        raise ValueError("deployment source integrity mismatch")
    return {
        "root": root,
        "manifest": manifest,
        "config": config,
        "checkpoint": checkpoint,
        "source": source,
    }


def _validate_smolvla_package(root: Path, manifest: dict) -> dict:
    config_path = _inside(root, root / "config.json", "config.json")
    if not config_path.is_file():
        raise FileNotFoundError("deployment package is missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") != "flowedge_smolvla":
        raise ValueError("deployment config must describe FlowEdge SmolVLA")
    checkpoint = Path(_resolve_native_checkpoint(root, config["checkpoint_path"]))
    source = _inside(
        root, root / config["source_checkpoint_path"], "source_checkpoint_path"
    )
    if not source.is_dir():
        raise FileNotFoundError("deployment package is missing the SmolVLA source")
    asset_identity = manifest.get("vlm_assets")
    if (
        not isinstance(asset_identity, dict)
        or not isinstance(asset_identity.get("model"), str)
        or not asset_identity["model"]
    ):
        raise ValueError("SmolVLA package is missing its VLM asset identity")
    asset_files = _asset_file_hashes(
        source / VLM_ASSETS, allowed_blob_root=_hub_blob_directory(root)
    )
    if (
        asset_files != asset_identity.get("files")
        or _asset_bundle_sha256(asset_files) != asset_identity.get("sha256")
    ):
        raise ValueError("SmolVLA VLM asset identity mismatch")
    for required in (
        source / "model.safetensors",
        source / "config.json",
        source / "policy_preprocessor.json",
        source / "policy_postprocessor.json",
        source / VLM_ASSETS / "config.json",
        source / VLM_ASSETS / "tokenizer_config.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"SmolVLA package is missing {required.relative_to(root)}"
            )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SmolVLA package has no integrity manifest")
    for relative, expected in files.items():
        path = _inside(root, root / relative, "manifest file")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"SmolVLA package integrity mismatch: {relative}")
    if (
        Path(config["checkpoint_path"]) != Path(NATIVE_CHECKPOINT)
        or Path(config["source_checkpoint_path"]) != Path(SOURCE_DIRECTORY)
    ):
        raise ValueError("SmolVLA package paths disagree with the manifest")
    return {
        "root": root,
        "manifest": manifest,
        "config": config,
        "checkpoint": checkpoint,
        "source": source,
    }


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


def _copy_processor_state(
    source: Path, destination: Path, name: str = "policy_preprocessor.json"
) -> list[str]:
    processor = source / name
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
            if name.startswith(("diffusion.rgb_encoder.", "normalize_inputs.buffer_")):
                tensors[name] = weights.get_tensor(name)
    if not any(name.startswith("diffusion.rgb_encoder.") for name in tensors):
        raise ValueError("source checkpoint has no Diffusion Policy RGB encoder")
    save_file(tensors, destination)
    return len(tensors)


def _split_smolvla_weights(source: Path, native: Path, vlm: Path) -> dict[str, int]:
    """Split the source checkpoint into the native expert and PyTorch prefix."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    native_weights = {}
    vlm_weights = {}
    with safe_open(source, framework="pt", device="cpu") as weights:
        for name in weights.keys():
            expert = name.startswith(
                ("model.vlm_with_expert.lm_expert.", "model.action_")
            )
            state = name.startswith("model.state_proj.")
            if expert or state:
                native_weights[name] = weights.get_tensor(name)
            if not expert or state:
                vlm_weights[name] = weights.get_tensor(name)
    required = {
        "model.action_in_proj.weight",
        "model.action_out_proj.weight",
        "model.state_proj.weight",
        "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight",
    }
    if not required.issubset(native_weights):
        raise ValueError("source checkpoint has no compatible SmolVLA action expert")
    if not any(name.startswith("model.vlm_with_expert.vlm.") for name in vlm_weights):
        raise ValueError("source checkpoint has no SmolVLA vision-language model")
    save_file(native_weights, native)
    save_file(vlm_weights, vlm)
    return {"native_tensors": len(native_weights), "source_tensors": len(vlm_weights)}


def _copy_vlm_assets(
    model_name: str, destination: Path, *, revision: str | None = None
) -> dict:
    """Bundle and identify the VLM tokenizer/config without downloading weights."""
    asset_source = Path(model_name).expanduser()
    resolved_revision = None
    if asset_source.is_dir():
        if revision is not None:
            raise ValueError("vlm_revision applies only to a remote VLM model ID")
    else:
        from huggingface_hub import snapshot_download

        asset_source = Path(
            snapshot_download(
                repo_id=model_name,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.model",
                    "*.txt",
                    "*.jinja",
                    "*.tiktoken",
                ],
                ignore_patterns=["*.safetensors", "*.bin", "*.gguf"],
            )
        )
        if asset_source.parent.name != "snapshots":
            raise ValueError("could not resolve an immutable VLM asset revision")
        resolved_revision = asset_source.name
    for name in ("config.json", "tokenizer_config.json"):
        if not (asset_source / name).is_file():
            raise FileNotFoundError(f"SmolVLM assets are missing {name}")
    destination.mkdir()
    for path in asset_source.rglob("*"):
        if not path.is_file() or path.suffix in {".safetensors", ".bin", ".gguf"}:
            continue
        target = destination / path.relative_to(asset_source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    files = _asset_file_hashes(destination)
    return {
        "model": model_name,
        "revision": resolved_revision,
        "sha256": _asset_bundle_sha256(files),
        "files": files,
    }


def create_smolvla_deployment_package(
    source: str | Path,
    output: str | Path,
    *,
    action_steps: int | None = None,
    inference_steps: int | None = None,
    seed: int = 0,
    vlm_revision: str | None = None,
) -> dict:
    """Build a portable hybrid SmolVLA policy from a trained LeRobot checkpoint."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeSmolVLAConfig

    source = Path(source).resolve()
    output = Path(output).resolve()
    if not (source / "model.safetensors").is_file():
        raise FileNotFoundError("SmolVLA source must contain model.safetensors")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output directories must not contain each other")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    source_config = PreTrainedConfig.from_pretrained(source)
    if source_config.type != "smolvla":
        raise ValueError("source must be a LeRobot SmolVLA policy")
    if getattr(source_config, "adapt_to_pi_aloha", False) or getattr(
        source_config, "use_delta_joint_actions_aloha", False
    ):
        raise ValueError("SmolVLA Aloha action transforms are not supported")
    if getattr(source_config, "use_peft", False):
        raise ValueError("SmolVLA PEFT adapters must be merged before packaging")
    steps = source_config.n_action_steps if action_steps is None else action_steps
    denoising = source_config.num_steps if inference_steps is None else inference_steps
    if steps <= 0 or steps > source_config.chunk_size or denoising <= 0:
        raise ValueError(
            "action_steps or inference_steps are outside the source contract"
        )
    # The normalizer and tokenizer must travel with a real trained policy.
    if (
        not (source / "policy_preprocessor.json").is_file()
        or not (source / "policy_postprocessor.json").is_file()
    ):
        raise FileNotFoundError("SmolVLA source must include saved pre/post processors")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="flowedge-smolvla-", dir=output.parent
    ) as tmp:
        package = Path(tmp) / "package"
        slim_source = package / SOURCE_DIRECTORY
        slim_source.mkdir(parents=True)
        shutil.copy2(source / "config.json", slim_source / "config.json")
        vlm_assets = _copy_vlm_assets(
            source_config.vlm_model_name,
            slim_source / VLM_ASSETS,
            revision=vlm_revision,
        )
        _copy_processor_state(source, slim_source)
        _copy_processor_state(source, slim_source, "policy_postprocessor.json")
        # LeRobot 0.6's rollout factory reads processors beside config.json.
        # Keep the source copies for the direct Python API as well.
        _copy_processor_state(slim_source, package)
        _copy_processor_state(slim_source, package, "policy_postprocessor.json")
        split = _split_smolvla_weights(
            source / "model.safetensors",
            package / NATIVE_CHECKPOINT,
            slim_source / "model.safetensors",
        )
        config = FlowEdgeSmolVLAConfig(
            n_obs_steps=source_config.n_obs_steps,
            input_features=source_config.input_features,
            output_features=source_config.output_features,
            device="cpu",
            push_to_hub=False,
            checkpoint_path=NATIVE_CHECKPOINT,
            source_checkpoint_path=SOURCE_DIRECTORY,
            action_steps=steps,
            inference_steps=denoising,
            seed=seed,
        )
        config.save_pretrained(package)
        files = {
            str(path.relative_to(package)).replace("\\", "/"): _sha256(path)
            for path in package.rglob("*")
            if path.is_file()
        }
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "policy_type": "flowedge_smolvla",
            "source_original_sha256": _sha256(source / "model.safetensors"),
            "vlm_assets": vlm_assets,
            "files": files,
            "split": split,
            "default_device": "cpu",
        }
        (package / PACKAGE_MANIFEST).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _validate_smolvla_package(package.resolve(), manifest)
        package.replace(output)
    return manifest


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

    from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig

    from flowedge_lerobot.observation import source_config, validate_source_pair

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
        (package / PACKAGE_MANIFEST).write_text(
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
            "  --device=cpu \\\n"
            "  --robot.type=so101_follower \\\n"
            "  --robot.port=/dev/ttyACM0 \\\n"
            '  --task="describe the trained task" \\\n'
            "  --duration=60\n"
            "```\n\n"
            "The package defaults to CPU. Use --device=cuda only with a CUDA "
            "FlowEdge runtime and CUDA-enabled PyTorch. LeRobot owns the robot "
            "driver, joint limits, cameras, and emergency stop. Start with the "
            "robot unloaded and a short duration.\n",
            encoding="utf-8",
        )
        validate_deployment_package(package)
        package.replace(output)
    return manifest


def package_policy(
    source: str | Path,
    output: str | Path,
    *,
    converted_checkpoint: str | Path | None = None,
    action_steps: int | None = None,
    inference_steps: int | None = None,
    scheduler: str = "ddim",
    seed: int = 0,
    vlm_revision: str | None = None,
) -> Path:
    """Create a deployable package for a supported local LeRobot policy.

    The policy family is read from its ``config.json``. SmolVLA can be
    packaged directly. Diffusion Policy additionally needs its matching
    FlowEdge-converted checkpoint. ``vlm_revision`` pins remote SmolVLA
    tokenizer/config assets to a Hugging Face revision.
    """
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    converted_path = (
        Path(converted_checkpoint).expanduser().resolve()
        if converted_checkpoint is not None
        else None
    )
    config_path = source_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"policy directory is missing {config_path.name}")
    try:
        policy_type = json.loads(config_path.read_text(encoding="utf-8")).get("type")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError(f"invalid policy config: {config_path}") from exc

    if policy_type == "smolvla":
        if converted_checkpoint is not None:
            raise ValueError("SmolVLA packaging does not use converted_checkpoint")
        if scheduler != "ddim":
            raise ValueError("scheduler is only configurable for Diffusion Policy")
        create_smolvla_deployment_package(
            source_path,
            output_path,
            action_steps=action_steps,
            inference_steps=inference_steps,
            seed=seed,
            vlm_revision=vlm_revision,
        )
        return output_path

    if policy_type == "diffusion":
        if vlm_revision is not None:
            raise ValueError("vlm_revision is only supported for SmolVLA")
        if converted_path is None:
            raise ValueError(
                "Diffusion Policy needs converted_checkpoint; convert the trained "
                "policy with FlowEdge first"
            )
        create_deployment_package(
            converted_path,
            source_path,
            output_path,
            action_steps=action_steps,
            inference_steps=10 if inference_steps is None else inference_steps,
            scheduler=scheduler,
            seed=seed,
        )
        return output_path

    raise ValueError(
        f"unsupported LeRobot policy type {policy_type!r}; supported types are "
        "'diffusion' and 'smolvla'"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowedge-lerobot-package",
        description="Package a supported LeRobot policy for FlowEdge deployment.",
    )
    parser.add_argument("source", help="trained LeRobot policy directory")
    parser.add_argument("output", help="new deployment package directory")
    parser.add_argument(
        "--converted-checkpoint",
        help="matching FlowEdge checkpoint (required for Diffusion Policy)",
    )
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--scheduler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vlm-revision",
        help="Hugging Face revision for SmolVLA tokenizer/config assets",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = package_policy(
        args.source,
        args.output,
        converted_checkpoint=args.converted_checkpoint,
        action_steps=args.action_steps,
        inference_steps=args.inference_steps,
        scheduler=args.scheduler,
        seed=args.seed,
        vlm_revision=args.vlm_revision,
    )
    manifest = validate_deployment_package(output)["manifest"]
    print(json.dumps({"output": str(output), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
