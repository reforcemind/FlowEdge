import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flowedge_lerobot.policy import LoadedPolicy


class PolicyAPITests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_select_action_uses_matching_processors_and_task(self):
        import torch

        class Policy:
            config = SimpleNamespace(
                type="flowedge_smolvla", input_features={"observation.state": None}
            )

            def reset(self):
                self.reset_called = True

            def select_action(self, batch):
                self.batch = batch
                return torch.ones(1, 2)

        policy = Policy()
        events = []

        class Processor:
            def __init__(self, transform):
                self.transform = transform

            def __call__(self, value):
                return self.transform(value)

            def reset(self):
                events.append("processor")

        runner = LoadedPolicy(
            policy,
            Processor(lambda raw: raw),
            Processor(lambda action: action[0] * 2),
        )
        with self.assertRaisesRegex(ValueError, "task instruction"):
            runner.select_action({"observation.state": torch.zeros(2)})
        action = runner.select_action(
            {"observation.state": torch.zeros(2)}, task="pick"
        )
        self.assertEqual(policy.batch["task"], "pick")
        self.assertEqual(action.tolist(), [2.0, 2.0])
        runner.reset()
        self.assertTrue(policy.reset_called)
        self.assertEqual(events, ["processor", "processor"])

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "requires LeRobot")
    def test_real_processors_return_a_batch_one_lerobot_action(self):
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig
        from lerobot_policy_flowedge.processor_flowedge import (
            make_flowedge_pre_post_processors,
        )

        config = FlowEdgeConfig(
            checkpoint_path="fixture.safetensors",
            input_features={
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))
            },
            device="cpu",
        )
        preprocessor, postprocessor = make_flowedge_pre_post_processors(config)

        class Policy:
            def __init__(self):
                self.config = config

            def reset(self):
                pass

            def select_action(self, batch):
                self.asserted_batch = batch
                return torch.ones(1, 2)

        policy = Policy()
        runner = LoadedPolicy(policy, preprocessor, postprocessor)
        action = runner.select_action({"observation.state": torch.zeros(3)})

        self.assertEqual(
            tuple(policy.asserted_batch["observation.state"].shape), (1, 3)
        )
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(tuple(action.shape), (1, 2))

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "requires LeRobot")
    def test_load_policy_forwards_hub_resolution_options(self):
        from flowedge_lerobot.policy import load_policy

        config = SimpleNamespace(type="flowedge_smolvla", device=None)
        policy = SimpleNamespace(config=config)

        class PolicyClass:
            @classmethod
            def from_pretrained(cls, reference, *, config):
                del cls, reference, config
                return policy

        package = {
            "manifest": {"policy_type": "flowedge_smolvla"},
            "root": "cached-package",
        }
        options = {
            "verify_integrity": True,
            "revision": "0123456789abcdef",
            "cache_dir": "cache",
            "force_download": True,
            "token": "hf-test-token",
            "local_files_only": False,
        }
        with (
            patch(
                "flowedge_lerobot.deployment.validate_deployment_package",
                return_value=package,
            ) as validate,
            patch(
                "lerobot.configs.policies.PreTrainedConfig.from_pretrained",
                return_value=config,
            ),
            patch(
                "lerobot.policies.factory.get_policy_class", return_value=PolicyClass
            ),
            patch(
                "lerobot_policy_flowedge.processor_flowedge.make_flowedge_smolvla_pre_post_processors",
                return_value=(lambda value: value, lambda value: value),
            ),
        ):
            loaded = load_policy("org/policy", **options)

        self.assertIs(loaded.policy, policy)
        validate.assert_called_once_with("org/policy", **options)

    def test_smolvla_cuda_load_selects_hybrid_source_device(self):
        from flowedge_lerobot.policy import load_policy

        config = SimpleNamespace(type="flowedge_smolvla", device="cpu")

        class PolicyClass:
            @classmethod
            def from_pretrained(cls, reference, *, config):
                del cls, reference
                return SimpleNamespace(config=config)

        def fake_module(name, **attributes):
            value = types.ModuleType(name)
            value.__dict__.update(attributes)
            value.__path__ = []
            return value

        config_module = fake_module(
            "lerobot.configs.policies",
            PreTrainedConfig=SimpleNamespace(from_pretrained=lambda _: config),
        )
        factory_module = fake_module(
            "lerobot.policies.factory", get_policy_class=lambda _: PolicyClass
        )
        processor_module = fake_module(
            "lerobot_policy_flowedge.processor_flowedge",
            make_flowedge_pre_post_processors=lambda _: (
                lambda value: value,
                lambda value: value,
            ),
            make_flowedge_smolvla_pre_post_processors=lambda _: (
                lambda value: value,
                lambda value: value,
            ),
        )
        mocked_modules = {
            "lerobot": fake_module("lerobot"),
            "lerobot.configs": fake_module("lerobot.configs"),
            "lerobot.configs.policies": config_module,
            "lerobot.policies": fake_module("lerobot.policies"),
            "lerobot.policies.factory": factory_module,
            "lerobot_policy_flowedge": fake_module("lerobot_policy_flowedge"),
            "lerobot_policy_flowedge.processor_flowedge": processor_module,
        }

        package = {
            "manifest": {"policy_type": "flowedge_smolvla"},
            "root": "cached-package",
        }
        with (
            patch.dict(sys.modules, mocked_modules),
            patch(
                "flowedge_lerobot.deployment.validate_deployment_package",
                return_value=package,
            ),
            patch("flowedge_lerobot.device.require_torch_cuda") as require_cuda,
        ):
            loaded = load_policy("org/smolvla", device="cuda")

        self.assertEqual(loaded.policy.config.type, "flowedge_smolvla")
        self.assertEqual(config.device, "cuda")
        require_cuda.assert_called_once_with("cuda")


if __name__ == "__main__":
    unittest.main()
