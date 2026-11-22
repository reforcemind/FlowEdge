import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class ReplayDeviceTests(unittest.TestCase):
    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name)
            for name in ("lerobot", "diffusers", "safetensors")
        ),
        "requires LeRobot reference dependencies",
    )
    def test_reference_action_statistics_supports_processor_sidecars_and_legacy(self):
        import torch
        from flowedge_lerobot.reference import action_statistics
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(
                {
                    "unnormalize_outputs.buffer_action.min": torch.tensor([-1.0, -2.0]),
                    "unnormalize_outputs.buffer_action.max": torch.tensor([1.0, 2.0]),
                },
                root / "model.safetensors",
            )
            minimum, maximum = action_statistics(root)
            self.assertEqual(minimum.tolist(), [-1.0, -2.0])
            self.assertEqual(maximum.tolist(), [1.0, 2.0])
            save_file(
                {
                    "action.min": torch.tensor([0.0, 1.0]),
                    "action.max": torch.tensor([3.0, 4.0]),
                },
                root / "stats.safetensors",
            )
            processor = {
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {"norm_map": {"ACTION": "MIN_MAX"}},
                        "state_file": "stats.safetensors",
                    }
                ]
            }
            (root / "policy_postprocessor.json").write_text(
                json.dumps(processor), encoding="utf-8"
            )
            minimum, maximum = action_statistics(root)
            self.assertEqual(minimum.tolist(), [0.0, 1.0])
            self.assertEqual(maximum.tolist(), [3.0, 4.0])
            processor["steps"][0]["state_file"] = "../outside.safetensors"
            (root / "policy_postprocessor.json").write_text(
                json.dumps(processor), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "inside the source directory"):
                action_statistics(root)

    def test_cuda_action_copy_is_inside_timed_region(self):
        from flowedge_lerobot.benchmark import time_policy_loop

        events = []

        class TensorResult:
            def detach(self):
                return self

            def cpu(self):
                events.append("host_copy")
                return self

            def numpy(self):
                return np.zeros((1, 2), dtype=np.float32)

        class Encoder:
            def reset(self):
                pass

            def observe(self, frame):
                pass

            def condition(self):
                return None

        class Policy:
            def predict_action_chunk(self, condition, noise, *, steps):
                return TensorResult()

        def clock():
            events.append("clock")
            return len(events) * 1_000_000

        samples = {
            "flowedge": {
                name: [] for name in ("encoder_ms", "policy_ms", "end_to_end_ms")
            }
        }
        with (
            patch("flowedge_lerobot.benchmark.perf_counter_ns", side_effect=clock),
            patch("flowedge_lerobot.benchmark.synchronize"),
            patch("flowedge_lerobot.benchmark.torch.is_tensor", return_value=True),
        ):
            time_policy_loop(
                "flowedge", Policy(), Encoder(), [{}], None, 1, "cuda", 0, 1, samples
            )
        self.assertEqual(events, ["clock", "clock", "host_copy", "clock"])
        self.assertEqual(samples["flowedge"]["end_to_end_ms"], [3.0])

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
        with (
            patch(
                "flowedge_lerobot.benchmark.torch.cuda.is_available", return_value=False
            ),
            self.assertRaises(RuntimeError),
        ):
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
