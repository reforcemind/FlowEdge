import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class ReplayDeviceTests(unittest.TestCase):
    def test_compiled_reference_requires_warmup_before_loading_model(self):
        from flowedge_lerobot.benchmark import main

        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "missing.safetensors",
                    "--source",
                    "missing",
                    "--revision",
                    "test",
                    "--output",
                    "unused.json",
                    "--reference-compile",
                    "default",
                    "--warmup",
                    "0",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_cpu_is_always_allowed(self):
        from flowedge_lerobot.benchmark import require_replay_device

        require_replay_device("cpu", SimpleNamespace(cuda=False))

    def test_cuda_fails_closed_without_torch_or_native(self):
        from flowedge_lerobot.benchmark import require_replay_device

        with self.assertRaises(ValueError):
            require_replay_device("metal", SimpleNamespace(cuda=True))
        with patch(
            "flowedge_lerobot.benchmark.torch.cuda.is_available", return_value=False
        ):
            with self.assertRaises(RuntimeError):
                require_replay_device("cuda", SimpleNamespace(cuda=True))
        with patch(
            "flowedge_lerobot.benchmark.torch.cuda.is_available", return_value=True
        ):
            with self.assertRaises(RuntimeError):
                require_replay_device("cuda", SimpleNamespace(cuda=False))
            require_replay_device("cuda", SimpleNamespace(cuda=True))

    def test_observation_fixture_round_trips_stacked_frames(self):
        import torch
        from flowedge_lerobot.benchmark import load_observation_frames

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obs.npz"
            np.savez_compressed(
                path,
                **{
                    "observation.state": np.array(
                        [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
                    ),
                    "observation.image": np.ones((2, 1, 3, 2, 2), dtype=np.float32),
                },
            )
            frames = load_observation_frames(path)
            self.assertEqual(len(frames), 2)
            torch.testing.assert_close(
                frames[1]["observation.state"], torch.tensor([3.0, 4.0])
            )

    def test_max_abs_error_is_the_largest_elementwise_gap(self):
        from flowedge_lerobot.benchmark import max_abs_error

        left = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        right = np.array([[0.0, 1.5], [1.0, 3.0]], dtype=np.float32)
        self.assertEqual(max_abs_error(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()
