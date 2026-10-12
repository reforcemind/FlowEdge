import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flowedge_lerobot.device import config_device, require_native_device


class NativeDeviceTests(unittest.TestCase):
    def test_config_device_defaults_to_cpu(self):
        self.assertEqual(config_device(SimpleNamespace()), "cpu")
        self.assertEqual(config_device(SimpleNamespace(device="cuda")), "cuda")

    def test_cpu_is_always_allowed(self):
        require_native_device("cpu", SimpleNamespace(cuda=False))
        require_native_device(None, SimpleNamespace(cuda=False))

    def test_cuda_requires_native_backend(self):
        with self.assertRaisesRegex(RuntimeError, "FLOWEDGE_BACKEND=cuda"):
            require_native_device("cuda", SimpleNamespace(cuda=False))
        require_native_device("cuda", SimpleNamespace(cuda=True))
        require_native_device("cuda:0", SimpleNamespace(cuda=True))
        require_native_device(
            "cuda", SimpleNamespace(cuda=True), engine=SimpleNamespace(cuda_resident=True)
        )

    def test_cuda_requires_resident_engine(self):
        with self.assertRaisesRegex(RuntimeError, "CPU kernels"):
            require_native_device(
                "cuda",
                SimpleNamespace(cuda=True),
                engine=SimpleNamespace(cuda_resident=False),
            )

    def test_unknown_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cpu or cuda"):
            require_native_device("metal", SimpleNamespace(cuda=True))

    def test_plugin_config_does_not_silently_replace_cuda_with_cpu(self):
        import torch
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig

        with patch(
            "lerobot.configs.policies.is_torch_device_available", return_value=False
        ):
            with patch(
                "lerobot.configs.policies.auto_select_torch_device",
                return_value=torch.device("cpu"),
            ):
                with self.assertRaisesRegex(RuntimeError, "selected 'cpu'"):
                    FlowEdgeConfig(device="cuda")

    def test_plugin_config_defaults_to_cpu(self):
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig

        self.assertEqual(FlowEdgeConfig().device, "cpu")


if __name__ == "__main__":
    unittest.main()
