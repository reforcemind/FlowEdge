import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from flowedge_lerobot.deployment import (
    NATIVE_CHECKPOINT,
    SOURCE_DIRECTORY,
    _asset_bundle_sha256,
    _asset_file_hashes,
    _config_sha256,
    _copy_vlm_assets,
    _hub_blob_directory,
    _inside,
    _package_relative_path,
    _resolve_bundle_root,
    _resolve_native_checkpoint,
    _sha256,
    create_deployment_package,
    create_smolvla_deployment_package,
    package_policy,
    validate_deployment_package,
)


class DeploymentPathTests(unittest.TestCase):
    def test_local_bundle_paths_expand_home_and_do_not_fall_through_to_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            package = home / "deploy"
            package.mkdir()
            options = {
                "revision": None,
                "cache_dir": None,
                "force_download": False,
                "token": None,
                "local_files_only": False,
            }
            with patch.dict(os.environ, {"HOME": tmp, "USERPROFILE": tmp}):
                self.assertEqual(
                    _resolve_bundle_root("~/deploy", **options), package.resolve()
                )
                with self.assertRaisesRegex(FileNotFoundError, "package directory"):
                    _resolve_bundle_root("~/missing", **options)

    def test_fast_smolvla_validation_checks_layout_without_hashing_weights(self):
        from flowedge_lerobot.deployment import PACKAGE_MANIFEST

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "package")
            source = root / SOURCE_DIRECTORY
            assets = source / "vlm_assets"
            assets.mkdir(parents=True)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "type": "flowedge_smolvla",
                        "checkpoint_path": NATIVE_CHECKPOINT,
                        "source_checkpoint_path": SOURCE_DIRECTORY,
                    }
                ),
                encoding="utf-8",
            )
            relative_files = (
                "flowedge_model.safetensors",
                "source/model.safetensors",
                "source/config.json",
                "source/policy_preprocessor.json",
                "source/policy_postprocessor.json",
                "source/vlm_assets/config.json",
                "source/vlm_assets/tokenizer_config.json",
            )
            for relative in relative_files:
                path = root.joinpath(*relative.split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

            file_hashes = {
                relative: _sha256(root / relative) for relative in relative_files
            }
            asset_hashes = {
                name.removeprefix("source/vlm_assets/"): digest
                for name, digest in file_hashes.items()
                if name.startswith("source/vlm_assets/")
            }
            manifest = {
                "schema": 1,
                "policy_type": "flowedge_smolvla",
                "files": file_hashes,
                "vlm_assets": {
                    "model": "org/test-vlm",
                    "files": asset_hashes,
                    "sha256": _asset_bundle_sha256(asset_hashes),
                },
            }
            (root / PACKAGE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with patch(
                "flowedge_lerobot.deployment._sha256",
                side_effect=AssertionError("fast load must not hash model files"),
            ):
                validated = validate_deployment_package(
                    root, verify_integrity=False
                )

            self.assertFalse(validated["integrity_verified"])

    def test_package_paths_reject_absolute_and_parent_paths_on_both_platforms(self):
        for value in (
            "/outside/model.safetensors",
            "C:\\outside\\model.safetensors",
            "../outside/model.safetensors",
            "source/../../outside/model.safetensors",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "must stay inside"
            ):
                _package_relative_path(value, "checkpoint_path")

    def test_package_validator_rejects_absolute_native_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "package")
            root.mkdir()
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "type": "flowedge",
                        "input_mode": "visual",
                        "checkpoint_path": str(Path(tmp, "outside.safetensors")),
                        "source_checkpoint_path": "source",
                    }
                ),
                encoding="utf-8",
            )
            (root / "flowedge-package.json").write_text(
                json.dumps({"schema": 1, "policy_type": "flowedge"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint_path must stay inside"):
                validate_deployment_package(root)

    def test_package_validator_rejects_absolute_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "package")
            root.mkdir()
            checkpoint = root / NATIVE_CHECKPOINT
            checkpoint.write_bytes(b"fixture")
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "type": "flowedge",
                        "input_mode": "visual",
                        "checkpoint_path": NATIVE_CHECKPOINT,
                        "source_checkpoint_path": str(Path(tmp, "outside-source")),
                    }
                ),
                encoding="utf-8",
            )
            (root / "flowedge-package.json").write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "policy_type": "flowedge",
                        "converted_sha256": _sha256(checkpoint),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "source_checkpoint_path must stay inside"
            ):
                validate_deployment_package(root)

    def test_native_checkpoint_cannot_escape_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "package")
            root.mkdir()
            Path(tmp, "outside.safetensors").write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "must stay inside"):
                _resolve_native_checkpoint(root, "../outside.safetensors")

    def test_hugging_face_snapshot_symlinks_are_confined_to_blob_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_repo = Path(tmp, "models--org--policy")
            snapshot = cache_repo / "snapshots" / ("a" * 40)
            blobs = cache_repo / "blobs"
            assets = snapshot / "source" / "vlm_assets"
            assets.mkdir(parents=True)
            blobs.mkdir()
            blob = blobs / "asset-hash"
            blob.write_bytes(b"hugging face cache payload")
            try:
                os.symlink(blob, snapshot / "config.json")
                os.symlink(blob, assets / "tokenizer_config.json")
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            self.assertEqual(
                _inside(snapshot, snapshot / "config.json", "config.json"),
                blob.resolve(),
            )
            self.assertEqual(
                _asset_file_hashes(
                    assets, allowed_blob_root=_hub_blob_directory(snapshot)
                ),
                {"tokenizer_config.json": _sha256(blob)},
            )

            outside = Path(tmp, "outside.json")
            outside.write_bytes(b"outside the Hub cache")
            try:
                os.symlink(outside, assets / "escape.json")
            except OSError as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "blob store"):
                _asset_file_hashes(
                    assets, allowed_blob_root=_hub_blob_directory(snapshot)
                )


class PackagePolicyTests(unittest.TestCase):
    def test_dispatches_smolvla_from_source_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "source")
            source.mkdir()
            (source / "config.json").write_text('{"type":"smolvla"}')
            with patch(
                "flowedge_lerobot.deployment.create_smolvla_deployment_package",
                return_value={"policy_type": "flowedge_smolvla"},
            ) as package:
                result = package_policy(source, Path(tmp, "output"))
            self.assertEqual(result, Path(tmp, "output").resolve())
            package.assert_called_once_with(
                source.resolve(),
                Path(tmp, "output").resolve(),
                action_steps=None,
                inference_steps=None,
                seed=0,
                vlm_revision=None,
            )

    def test_smolvla_accepts_an_explicit_vlm_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "source")
            source.mkdir()
            (source / "config.json").write_text('{"type":"smolvla"}')
            with patch(
                "flowedge_lerobot.deployment.create_smolvla_deployment_package"
            ) as package:
                package_policy(
                    source,
                    Path(tmp, "output"),
                    vlm_revision="0123456789abcdef",
                )
            self.assertEqual(package.call_args.kwargs["vlm_revision"], "0123456789abcdef")

    def test_smolvla_vlm_asset_bundle_records_resolved_hub_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "models--org--smolvlm" / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            destination = root / "bundle"
            with patch(
                "huggingface_hub.snapshot_download", return_value=snapshot
            ) as download:
                identity = _copy_vlm_assets(
                    "org/smolvlm", destination, revision="main"
                )
            self.assertEqual(identity["revision"], "a" * 40)
            self.assertEqual(set(identity["files"]), {"config.json", "tokenizer_config.json"})
            download.assert_called_once()

    def test_dispatches_diffusion_with_explicit_converted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "source")
            source.mkdir()
            (source / "config.json").write_text('{"type":"diffusion"}')
            converted = Path(tmp, "converted.safetensors")
            output = Path(tmp, "output")
            with patch(
                "flowedge_lerobot.deployment.create_deployment_package",
                return_value={"policy_type": "flowedge"},
            ) as package:
                result = package_policy(
                    source,
                    output,
                    converted_checkpoint=converted,
                    scheduler="ddpm",
                )
            self.assertEqual(result, output.resolve())
            package.assert_called_once_with(
                converted.resolve(),
                source.resolve(),
                output.resolve(),
                action_steps=None,
                inference_steps=10,
                scheduler="ddpm",
                seed=0,
            )

    def test_expands_home_for_output_and_converted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "source")
            source.mkdir()
            (source / "config.json").write_text('{"type":"diffusion"}')
            with patch.dict(os.environ, {"HOME": tmp, "USERPROFILE": tmp}):
                output = Path("~/deployment")
                converted = Path("~/converted.safetensors")
                expected_output = output.expanduser().resolve()
                expected_converted = converted.expanduser().resolve()
                with patch(
                    "flowedge_lerobot.deployment.create_deployment_package",
                    return_value={"policy_type": "flowedge"},
                ) as package:
                    result = package_policy(
                        source,
                        output,
                        converted_checkpoint=converted,
                    )
            self.assertEqual(result, expected_output)
            self.assertEqual(package.call_args.args[0], expected_converted)
            self.assertEqual(package.call_args.args[2], expected_output)

    def test_diffusion_requires_its_converted_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "source")
            source.mkdir()
            (source / "config.json").write_text('{"type":"diffusion"}')
            with self.assertRaisesRegex(ValueError, "converted_checkpoint"):
                package_policy(source, Path(tmp, "output"))

    def test_unknown_policy_type_fails_with_supported_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "source")
            source.mkdir()
            (source / "config.json").write_text('{"type":"act"}')
            with self.assertRaisesRegex(ValueError, "diffusion.*smolvla"):
                package_policy(source, Path(tmp, "output"))


@unittest.skipUnless(
    importlib.util.find_spec("lerobot") and importlib.util.find_spec("torchvision"),
    "requires the full optional LeRobot runtime",
)
class DeploymentPackageTests(unittest.TestCase):
    def test_lerobot_discovers_smolvla_policy_and_processors(self):
        import lerobot_policy_flowedge  # noqa: F401
        from lerobot.policies.factory import get_policy_class
        from lerobot_policy_flowedge.modeling_flowedge_smolvla import (
            FlowEdgeSmolVLAPolicy,
        )

        self.assertIs(get_policy_class("flowedge_smolvla"), FlowEdgeSmolVLAPolicy)

    def _features(self):
        from lerobot.configs.types import FeatureType, PolicyFeature

        return (
            {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
                "observation.images.front": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 8, 8)
                ),
            },
            {
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            },
        )

    def test_standard_pretrained_path_needs_no_dummy_torch_weights(self):
        import torch
        from flowedge_lerobot import FlowEdgeDiffusionPolicy
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig
        from lerobot_policy_flowedge.modeling_flowedge import FlowEdgePolicy

        class Engine:
            diffusion_metadata: ClassVar[dict] = {
                "condition_dim": 3,
                "action_dim": 2,
                "horizon": 4,
                "action_steps": 3,
                "observation_steps": 2,
            }

            def sample_diffusion(self, *args, **kwargs):
                del args, kwargs
                return torch.zeros(4, 2).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / NATIVE_CHECKPOINT
            native.write_bytes(b"native")
            inputs, outputs = self._features()
            inputs.pop("observation.images.front")
            config = FlowEdgeConfig(
                checkpoint_path=NATIVE_CHECKPOINT,
                input_features=inputs,
                output_features=outputs,
                device="cpu",
            )
            config.save_pretrained(root)
            with patch(
                "flowedge_lerobot.FlowEdgeDiffusionPolicy.from_checkpoint",
                return_value=FlowEdgeDiffusionPolicy(Engine()),
            ) as load:
                policy = FlowEdgePolicy.from_pretrained(root)
            has_dummy_weights = (root / "model.safetensors").exists()

        self.assertFalse(has_dummy_weights)
        self.assertFalse(policy.training)
        self.assertEqual(Path(load.call_args.args[0]), native.resolve())

    def test_package_contains_relative_native_and_slim_visual_source(self):
        import flowedge_lerobot.observation  # noqa: F401  Patch the loaded module.
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        inputs, outputs = self._features()
        source_policy = SimpleNamespace(
            n_obs_steps=2,
            n_action_steps=3,
            horizon=4,
            input_features=inputs,
            output_features=outputs,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source-policy"
            source.mkdir()
            source_config = {"type": "diffusion", "fixture": True}
            (source / "config.json").write_text(
                json.dumps(source_config), encoding="utf-8"
            )
            save_file(
                {
                    "diffusion.rgb_encoder.fixture": torch.ones(2, 2),
                    "diffusion.unet.fixture": torch.ones(8, 8),
                    "normalize_inputs.buffer_observation_state.min": torch.zeros(3),
                },
                source / "model.safetensors",
            )
            converted = root / "converted.safetensors"
            identity = {
                "source_sha256": _sha256(source / "model.safetensors"),
                "source_config_sha256": _config_sha256(source_config),
            }
            save_file(
                {"dp.fixture": torch.ones(1)},
                converted,
                metadata={
                    "flowedge.source_sha256": identity["source_sha256"],
                    "flowedge.source_config_sha256": identity["source_config_sha256"],
                },
            )
            output = root / "deployment"
            with patch(
                "flowedge_lerobot.observation.source_config",
                return_value=source_policy,
            ):
                manifest = create_deployment_package(converted, source, output)

            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["type"], "flowedge")
            self.assertEqual(config["checkpoint_path"], NATIVE_CHECKPOINT)
            self.assertEqual(config["source_checkpoint_path"], SOURCE_DIRECTORY)
            self.assertEqual(
                manifest["source_original_sha256"], identity["source_sha256"]
            )
            self.assertFalse((output / "model.safetensors").exists())
            with safe_open(
                output / SOURCE_DIRECTORY / "model.safetensors", framework="pt"
            ) as weights:
                self.assertIn("diffusion.rgb_encoder.fixture", weights.keys())
                self.assertNotIn("diffusion.unet.fixture", weights.keys())
            validated = validate_deployment_package(output)
            self.assertEqual(validated["root"], output.resolve())
            self.assertEqual(validated["manifest"], manifest)

            with (output / NATIVE_CHECKPOINT).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint integrity"):
                validate_deployment_package(output)

    def test_smolvla_package_splits_expert_and_prefix_and_checks_integrity(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        inputs, outputs = self._features()
        source_policy = SimpleNamespace(
            type="smolvla",
            n_obs_steps=1,
            n_action_steps=2,
            chunk_size=3,
            num_steps=2,
            input_features=inputs,
            output_features=outputs,
            adapt_to_pi_aloha=False,
            use_delta_joint_actions_aloha=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "trained"
            source.mkdir()
            assets = root / "vlm-assets"
            assets.mkdir()
            (assets / "config.json").write_text("{}", encoding="utf-8")
            (assets / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            source_policy.vlm_model_name = str(assets)
            (source / "config.json").write_text('{"type":"smolvla"}', encoding="utf-8")
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                (source / name).write_text('{"steps":[]}', encoding="utf-8")
            weights = {
                "model.action_in_proj.weight": torch.ones(2, 2),
                "model.action_out_proj.weight": torch.ones(2, 2),
                "model.state_proj.weight": torch.ones(2, 2),
                "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight": torch.ones(
                    2, 2
                ),
                "model.vlm_with_expert.vlm.fixture": torch.ones(2, 2),
            }
            save_file(weights, source / "model.safetensors")
            output = root / "portable"
            with patch(
                "lerobot.configs.policies.PreTrainedConfig.from_pretrained",
                return_value=source_policy,
            ):
                manifest = create_smolvla_deployment_package(source, output)
            self.assertEqual(manifest["policy_type"], "flowedge_smolvla")
            with safe_open(output / NATIVE_CHECKPOINT, framework="pt") as native:
                self.assertIn("model.action_in_proj.weight", native.keys())
                self.assertNotIn("model.vlm_with_expert.vlm.fixture", native.keys())
            with safe_open(
                output / SOURCE_DIRECTORY / "model.safetensors", framework="pt"
            ) as prefix:
                self.assertIn("model.vlm_with_expert.vlm.fixture", prefix.keys())
                self.assertNotIn("model.action_in_proj.weight", prefix.keys())
            self.assertTrue(
                (output / SOURCE_DIRECTORY / "vlm_assets" / "config.json").is_file()
            )
            self.assertEqual(manifest["vlm_assets"]["revision"], None)
            self.assertRegex(manifest["vlm_assets"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(validate_deployment_package(output)["manifest"], manifest)
            with (output / SOURCE_DIRECTORY / "model.safetensors").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                validate_deployment_package(output)


if __name__ == "__main__":
    unittest.main()
