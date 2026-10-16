import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flowedge_lerobot.deployment import (
    NATIVE_CHECKPOINT,
    SOURCE_DIRECTORY,
    _config_sha256,
    _resolve_native_checkpoint,
    _sha256,
    create_deployment_package,
)


class DeploymentPathTests(unittest.TestCase):
    def test_native_checkpoint_cannot_escape_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "package")
            root.mkdir()
            Path(tmp, "outside.safetensors").write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "must stay inside"):
                _resolve_native_checkpoint(root, "../outside.safetensors")


@unittest.skipUnless(
    importlib.util.find_spec("lerobot") and importlib.util.find_spec("torchvision"),
    "requires the full optional LeRobot runtime",
)
class DeploymentPackageTests(unittest.TestCase):
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
            diffusion_metadata = {
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


if __name__ == "__main__":
    unittest.main()
