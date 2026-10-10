"""Build or verify a local review bundle using explicit include rules and hashes."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from tools.verification.anonymize_artifact import (
    anonymize_json_bytes,
    anonymize_text_bytes,
    identity_path_matches_json_bytes,
    identity_token_matches,
)

ROOT = Path(__file__).resolve().parents[2]

# Source path -> archive path (root-facing docs for the review zip).
ARCHIVE_ALIASES = {
    "papers/action-delivery/artifact-root-README.md": "README.md",
    "papers/action-delivery/AI_USE.md": "AI_USE.md",
}

INCLUDE = (
    "tools/delivery_check/__init__.py",
    "tools/delivery_check/artifact.py",
    "tools/delivery_check/async_overhead.py",
    "tools/delivery_check/baseline.py",
    "tools/delivery_check/controlled.py",
    "tools/delivery_check/corruptions.py",
    "tools/delivery_check/erasure.py",
    "tools/delivery_check/experiments.py",
    "tools/delivery_check/extension.py",
    "tools/delivery_check/heldout.py",
    "tools/delivery_check/lerobot_case.py",
    "tools/delivery_check/model.py",
    "tools/delivery_check/monitor.py",
    "tools/delivery_check/native.py",
    "tools/delivery_check/observation.py",
    "tools/delivery_check/overhead.py",
    "tools/delivery_check/payload_cost.py",
    "tools/delivery_check/source_audit.py",
    "tools/delivery_check/tlc.py",
    "tools/verification/verify_delivery.py",
    "tools/verification/anonymize_artifact.py",
    "integrations/lerobot/src/flowedge_lerobot/__init__.py",
    "integrations/lerobot/src/flowedge_lerobot/async_rollout.py",
    "integrations/lerobot/src/flowedge_lerobot/contracts.py",
    "integrations/lerobot/src/flowedge_lerobot/delivery_trace.py",
    "integrations/lerobot/src/flowedge_lerobot/diffusion.py",
    "integrations/lerobot/src/flowedge_lerobot/rollout.py",
    "integrations/lerobot/src/flowedge_lerobot/smolvla.py",
    "integrations/lerobot/tests/test_delivery_check.py",
    "integrations/lerobot/tests/test_delivery_corruptions.py",
    "integrations/lerobot/tests/test_delivery_extension.py",
    "integrations/lerobot/tests/test_delivery_heldout.py",
    "papers/action-delivery/artifact-root-README.md",
    "papers/action-delivery/AI_USE.md",
    "papers/action-delivery/correspondence.md",
    "papers/action-delivery/heldout.md",
    "papers/action-delivery/heldout-protocol.json",
    "papers/action-delivery/prospective-protocol.json",
    "papers/action-delivery/prospective-SHA256SUMS.txt",
    "papers/action-delivery/prospective-round-b.json",
    "papers/action-delivery/prospective-round-b-SHA256SUMS.txt",
    "papers/action-delivery/heldout-protocol-exercising.json",
    "papers/action-delivery/reproduce.md",
    "papers/action-delivery/trace-schema.md",
    "papers/action-delivery/verification.md",
    "papers/action-delivery/extension-protocol.json",
    "papers/action-delivery/results/async-overhead-linux.json",
    "papers/action-delivery/results/async_slice_minus_one.json",
    "papers/action-delivery/results/async_wrong_request.json",
    "papers/action-delivery/results/development.json",
    "papers/action-delivery/results/development-posthoc-windows.json",
    "papers/action-delivery/results/extension-linux.json",
    "papers/action-delivery/results/extension-posthoc-windows.json",
    "papers/action-delivery/results/general-tool-baseline.json",
    "papers/action-delivery/results/tla-grid.json",
    "papers/action-delivery/results/heldout.json",
    "papers/action-delivery/results/heldout-exercising.json",
    "papers/action-delivery/results/heldout-posthoc-windows.json",
    "papers/action-delivery/results/monitor-corruptions.json",
    "papers/action-delivery/results/monitor-corruptions-posthoc-windows.json",
    "papers/action-delivery/results/native-windows.json",
    "papers/action-delivery/results/observation-adequacy.json",
    "papers/action-delivery/results/observation-adequacy-posthoc-windows.json",
    "papers/action-delivery/results/overhead-windows.json",
    "papers/action-delivery/results/payload-cost-linux.json",
    "papers/action-delivery/results/prospective.json",
    "papers/action-delivery/results/prospective-round-b.json",
    "papers/action-delivery/results/inference-latency-windows.json",
    "papers/action-delivery/results/sync_rejected_hold.json",
    "papers/action-delivery/results/sync_send_after_stop.json",
    "papers/action-delivery/results/tlc.json",
    "papers/action-delivery/tla/Delivery.cfg",
    "papers/action-delivery/tla/Delivery.tla",
    "papers/action-delivery/tla/Mutant.cfg",
    "reproduction/action-delivery/README.md",
    "reproduction/action-delivery/.python-version",
    "reproduction/action-delivery/THIRD_PARTY.md",
    "reproduction/action-delivery/expected.json",
    "reproduction/action-delivery/requirements.txt",
    "reproduction/action-delivery/run.py",
    "LICENSE",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def included(root):
    return [resolve_source(root, relative) for relative in INCLUDE]


def resolve_source(root, relative):
    """Locate an INCLUDE path in a full checkout or an extracted review zip."""
    root = Path(root)
    path = root / relative
    if path.is_file():
        return path
    alias = ARCHIVE_ALIASES.get(relative)
    if alias is not None:
        aliased = root / alias
        if aliased.is_file():
            return aliased
    return path


def archive_name(relative):
    return ARCHIVE_ALIASES.get(relative, relative)


def _git_metadata(root):
    root = Path(root)
    inherited = root / "artifact-manifest.json"
    try:
        artifact_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root,
                text=True,
            ).strip()
        )
        if inherited.is_file():
            prior = json.loads(inherited.read_text(encoding="utf-8"))
            base_revision = prior.get(
                "flowedge_base_revision", prior.get("base_revision")
            )
        else:
            base_revision = artifact_revision
            artifact_revision = None
        return base_revision, artifact_revision, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        if inherited.is_file():
            prior = json.loads(inherited.read_text(encoding="utf-8"))
            return (
                prior.get("flowedge_base_revision", prior.get("base_revision")),
                prior.get("artifact_revision"),
                prior.get("working_tree_dirty"),
            )
        return None, None, None


def _entry(name):
    entry = zipfile.ZipInfo(name, (2026, 9, 23, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    return entry


def _prepare_payload(name, original, *, anonymized):
    if not anonymized:
        return original
    if name.endswith(".json"):
        return anonymize_json_bytes(original)
    if name.endswith((".py", ".md", ".txt", ".cfg", ".tla")):
        return anonymize_text_bytes(original)
    return original


def build(root, output, *, anonymized=False):
    root = Path(root)
    resolved = []
    missing = []
    for relative in INCLUDE:
        path = resolve_source(root, relative)
        if path.is_file():
            resolved.append((relative, path))
        else:
            missing.append(relative)
    if missing:
        raise FileNotFoundError(f"required artifact files are missing: {missing}")
    payloads = {}
    anonymized_files = []
    for relative, path in resolved:
        name = archive_name(relative)
        if name in payloads:
            raise ValueError(f"duplicate archive path: {name}")
        original = path.read_bytes()
        payload = _prepare_payload(relative, original, anonymized=anonymized)
        if payload != original:
            anonymized_files.append(name)
        payloads[name] = payload
    if anonymized:
        path_leaks = {
            name: matches
            for name, payload in payloads.items()
            if name.endswith(".json")
            and (matches := identity_path_matches_json_bytes(payload))
        }
        if path_leaks:
            raise ValueError(
                f"identity paths remain after anonymization: {path_leaks}"
            )
        token_leaks = {
            name: matches
            for name, payload in payloads.items()
            if (matches := identity_token_matches(payload))
        }
        if token_leaks:
            raise ValueError(
                f"identity tokens remain after anonymization: {token_leaks}"
            )
    files = {name: digest_bytes(payload) for name, payload in payloads.items()}
    source_tree_sha256 = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if anonymized:
        manifest = {
            "schema": 2,
            "anonymized": True,
            "scope": (
                "anonymized derived review bundle; not a public archival release"
            ),
            "source_tree_sha256": source_tree_sha256,
            "anonymization": {
                "json_string_path_replacement": True,
                "public_repo_revision_redaction": True,
                "files_transformed": anonymized_files,
                "identity_path_scan": "pass",
                "identity_token_scan": "pass",
            },
            "files": files,
        }
    else:
        base_revision, artifact_revision, dirty = _git_metadata(root)
        manifest = {
            "schema": 2,
            "flowedge_base_revision": base_revision,
            "artifact_revision": artifact_revision,
            "working_tree_dirty": dirty,
            "anonymized": False,
            "scope": (
                "committed local reproduction snapshot; not a public archival release"
                if artifact_revision is not None and dirty is False
                else "non-anonymous local review bundle; not an archival release"
            ),
            "source_tree_sha256": source_tree_sha256,
            "anonymization": {
                "json_string_path_replacement": False,
                "files_transformed": [],
                "identity_path_scan": "not_run",
                "identity_token_scan": "not_run",
            },
            "files": files,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for name, payload in payloads.items():
            bundle.writestr(_entry(name), payload)
        bundle.writestr(
            _entry("artifact-manifest.json"),
            json.dumps(manifest, indent=2) + "\n",
        )
    return {
        "files": len(resolved),
        "anonymized_files": len(anonymized_files),
        "sha256": digest(output),
        "output": str(output),
    }


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") not in (1, 2) or not isinstance(
        manifest.get("files"), dict
    ):
        return {
            "verdict": "fail",
            "files": 0,
            "failures": ["artifact-manifest.json"],
            "reason": "unsupported_manifest_schema",
        }
    missing, modified = [], []
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            missing.append(name)
        elif digest(path) != expected:
            modified.append(name)
    expected_names = set(manifest["files"]) | {"artifact-manifest.json"}
    actual_names = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    unexpected = sorted(actual_names - expected_names)
    metadata_failures = []
    expected_tree_hash = hashlib.sha256(
        json.dumps(
            manifest["files"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if (
        manifest.get("schema") == 2
        and manifest.get("source_tree_sha256") != expected_tree_hash
    ):
        metadata_failures.append("source_tree_sha256")
    identity_leaks = {}
    if manifest.get("anonymized") is True:
        for name in manifest["files"]:
            path = root / name
            if not path.is_file():
                continue
            data = path.read_bytes()
            matches = []
            if path.suffix.lower() == ".json":
                try:
                    matches.extend(identity_path_matches_json_bytes(data))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    matches.append("invalid_json")
            matches.extend(identity_token_matches(data))
            if matches:
                identity_leaks[name] = sorted(set(matches))
    failures = sorted(
        set(
            missing
            + modified
            + unexpected
            + [f"manifest:{name}" for name in metadata_failures]
            + [f"identity:{name}" for name in identity_leaks]
        )
    )
    return {
        "verdict": "fail" if failures else "pass",
        "files": len(manifest["files"]),
        "failures": failures,
        "missing": sorted(missing),
        "modified": sorted(modified),
        "unexpected": unexpected,
        "metadata_failures": metadata_failures,
        "identity_leaks": identity_leaks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--anonymized", action="store_true")
    args = parser.parse_args()
    if args.command == "build" and args.output is None:
        parser.error("build requires --output")
    result = (
        build(args.root, args.output, anonymized=args.anonymized)
        if args.command == "build"
        else verify(args.root)
    )
    print(json.dumps(result, indent=2))
    return 1 if result.get("verdict") == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
