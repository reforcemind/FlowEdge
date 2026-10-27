import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flowedge_lerobot.device import (
    config_device,
    require_native_device,
    require_torch_cuda,
)


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
            "cuda",
            SimpleNamespace(cuda=True),
            engine=SimpleNamespace(cuda_resident=True),
        )

    def test_cuda_requires_resident_engine(self):
        with self.assertRaisesRegex(RuntimeError, "CPU kernels"):
            require_native_device(
                "cuda",
                SimpleNamespace(cuda=True),
                engine=SimpleNamespace(cuda_resident=False),
            )

    def test_source_cuda_only_requires_torch_device(self):
        available_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
            device=lambda _: SimpleNamespace(index=None),
        )
        require_torch_cuda("cuda", available_torch)
        with self.assertRaisesRegex(RuntimeError, "PyTorch CUDA is unavailable"):
            require_torch_cuda(
                "cuda",
                SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            )

    def test_source_cuda_validates_explicit_device_index(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
            ),
            device=lambda _: SimpleNamespace(index=1),
        )
        with self.assertRaisesRegex(RuntimeError, "cuda:1"):
            require_torch_cuda("cuda:1", fake_torch)

    def test_unknown_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cpu or cuda"):
            require_native_device("metal", SimpleNamespace(cuda=True))

    @unittest.skipUnless(
        importlib.util.find_spec("lerobot"), "requires the optional LeRobot runtime"
    )
    def test_plugin_config_does_not_silently_replace_cuda_with_cpu(self):
        import torch
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig

        with (
            patch(
                "lerobot.configs.policies.is_torch_device_available", return_value=False
            ),
            patch(
                "lerobot.configs.policies.auto_select_torch_device",
                return_value=torch.device("cpu"),
            ),
            self.assertRaisesRegex(RuntimeError, "selected 'cpu'"),
        ):
            FlowEdgeConfig(device="cuda")

    @unittest.skipUnless(
        importlib.util.find_spec("lerobot"), "requires the optional LeRobot runtime"
    )
    def test_plugin_config_defaults_to_cpu(self):
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig

        self.assertEqual(FlowEdgeConfig().device, "cpu")

    @unittest.skipUnless(
        importlib.util.find_spec("lerobot"), "requires the optional LeRobot runtime"
    )
    def test_approximate_plugin_precision_requires_cuda(self):
        from lerobot_policy_flowedge.configuration_flowedge import FlowEdgeConfig

        with self.assertRaisesRegex(ValueError, "requires device=cuda"):
            FlowEdgeConfig(diffusion_precision="approx-fp16-deep")


if __name__ == "__main__":
    unittest.main()
