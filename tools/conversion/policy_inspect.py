#!/usr/bin/env python3
"""Validate a LeRobot policy directory before FlowEdge conversion."""

import argparse
import json
from pathlib import Path


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def read_json(path: Path, errors: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(errors, f"cannot read {path.name}: {error}")
        return {}
    if not isinstance(value, dict):
        fail(errors, f"{path.name} must contain an object")
        return {}
    return value


def action_contract(config: dict, errors: list[str]) -> dict:
    action = config.get("output_features", {}).get("action", {})
    shape = action.get("shape") if isinstance(action, dict) else None
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int):
        fail(errors, "config output_features.action.shape must be a one-dimensional integer shape")
        return {}
    fields = {"action_dim": shape[0], "action_horizon": config.get("horizon"),
              "observation_steps": config.get("n_obs_steps"), "action_steps": config.get("n_action_steps")}
    for name, value in fields.items():
        if not isinstance(value, int) or value <= 0:
            fail(errors, f"config {name} must be a positive integer")
    return fields


def processor_contract(directory: Path, errors: list[str]) -> dict:
    path = directory / "policy_postprocessor.json"
    if not path.is_file():
        return {"processor": "legacy-config", "normalization": None}
    processor = read_json(path, errors)
    steps = processor.get("steps")
    step = next((item for item in steps if isinstance(item, dict)
                 and item.get("registry_name") == "unnormalizer_processor"), None) if isinstance(steps, list) else None
    if step is None:
        fail(errors, "policy_postprocessor.json has no unnormalizer_processor step")
        return {"processor": "invalid", "normalization": None}
    config = step.get("config", {})
    normalization = config.get("norm_map", {}).get("ACTION") if isinstance(config, dict) else None
    state_file = step.get("state_file")
    state = Path(state_file) if isinstance(state_file, str) else Path()
    if normalization != "MIN_MAX":
        fail(errors, f"processor ACTION normalization must be MIN_MAX, got {normalization!r}")
    if not state_file or state.is_absolute() or ".." in state.parts or not (directory / state).is_file():
        fail(errors, "processor state_file must name an existing file beside policy_postprocessor.json")
    return {"processor": "policy_postprocessor", "normalization": normalization, "state_file": state_file}


def inspect(directory: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not directory.is_dir():
        return {}, [f"policy directory does not exist: {directory}"]
    model, config_path = directory / "model.safetensors", directory / "config.json"
    if not model.is_file():
        fail(errors, "model.safetensors is required")
    config = read_json(config_path, errors) if config_path.is_file() else {}
    if not config_path.is_file():
        fail(errors, "config.json is required")
    contract = action_contract(config, errors) if config else {}
    legacy_normalization = config.get("normalization_mapping", {}).get("ACTION") if isinstance(config.get("normalization_mapping"), dict) else None
    processor = processor_contract(directory, errors)
    normalization = processor["normalization"] or legacy_normalization
    if normalization != "MIN_MAX":
        fail(errors, f"ACTION normalization must be MIN_MAX, got {normalization!r}")
    manifest = {
        "manifest_version": 1,
        "source": "lerobot-directory",
        "architecture": config.get("type"),
        "artifacts": {"model": model.name, "config": config_path.name, **processor},
        "action_contract": contract,
        "backend": {"native": "flowedge-safetensors", "external_runtime": None},
    }
    return manifest, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable manifest")
    args = parser.parse_args()
    manifest, errors = inspect(args.directory)
    payload = {"manifest": manifest, "errors": errors, "supported": not errors}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("supported" if not errors else "unsupported")
        for error in errors:
            print(f"error: {error}")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
