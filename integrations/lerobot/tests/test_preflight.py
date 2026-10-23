import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flowedge_lerobot.preflight import _exercise, _latency_summary, main, run_preflight


class PreflightTests(unittest.TestCase):
    def test_smolvla_cuda_request_is_explicitly_rejected(self):
        with patch(
            "flowedge_lerobot.deployment.validate_deployment_package",
            return_value={"manifest": {"policy_type": "flowedge_smolvla"}},
        ):
            with self.assertRaisesRegex(ValueError, "requires device=cpu"):
                run_preflight("fixture", device="cuda")

    def test_latency_summary_interpolates_percentiles(self):
        summary = _latency_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["p50_ms"], 2.0)
        self.assertAlmostEqual(summary["p95_ms"], 2.9)
        self.assertAlmostEqual(summary["p99_ms"], 2.98)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_exercise_uses_fresh_finite_action_chunks(self):
        import torch

        feature = SimpleNamespace(shape=(2,))
        config = SimpleNamespace(
            device="cpu",
            action_steps=3,
            action_feature=SimpleNamespace(shape=(1,)),
            input_features={"observation.state": feature},
        )

        class Policy:
            def __init__(self):
                self.config = config
                self.resets = 0

            def reset(self):
                self.resets += 1

            def predict_action_chunk(self, batch):
                self.assert_batch = tuple(batch["observation.state"].shape)
                return torch.ones(1, 3, 1)

        policy = Policy()
        result = _exercise(
            policy,
            lambda raw: {name: value.unsqueeze(0) for name, value in raw.items()},
            warmup=1,
            iterations=2,
        )
        self.assertEqual(policy.resets, 3)
        self.assertEqual(policy.assert_batch, (1, 2))
        self.assertEqual(result["iterations"], 2)
        self.assertGreaterEqual(result["end_to_end"]["p50_ms"], 0)

    def test_failure_is_written_as_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, "report.json")
            with patch(
                "flowedge_lerobot.preflight.run_preflight",
                side_effect=ValueError("bad package"),
            ):
                with redirect_stdout(io.StringIO()):
                    result = main(["missing", "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result, 1)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["type"], "ValueError")
        self.assertEqual(report["error"]["message"], "bad package")


if __name__ == "__main__":
    unittest.main()
