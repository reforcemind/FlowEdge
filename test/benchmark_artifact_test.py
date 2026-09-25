#!/usr/bin/env python3
"""Unit tests for the dependency-free benchmark artifact contract."""

import copy
import json
import unittest
from pathlib import Path

from tools.benchmark import benchmark_artifact

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "bench" / "artifacts" / "policy" / "lerobot-vs-flowedge.template.json"


def measured():
    document = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    sample = {
        "status": "measured",
        "startup_ms": 3.0,
        "encoder_ms": {"p50": 1.0, "p95": 1.5, "p99": 2.0},
        "policy_ms": {"p50": 4.0, "p95": 5.0, "p99": 6.0},
        "end_to_end_ms": {"p50": 5.0, "p95": 6.5, "p99": 8.0},
        "throughput_hz": 200.0,
        "rss_mb": 64.0,
        "allocations": {"setup": 13, "hot_path": 0},
    }
    document["measurements"] = {"flowedge": sample, "lerobot": copy.deepcopy(sample)}
    return document


class BenchmarkArtifactTest(unittest.TestCase):
    def test_size_budget_rebaseline_is_auditable(self):
        budgets = json.loads((ROOT / "bench/config/budgets.json").read_text())
        baseline = json.loads((ROOT / "bench/config/budget-baseline.json").read_text())
        self.assertEqual(budgets["version"], 1)
        self.assertEqual(baseline["schema"], 1)
        self.assertEqual(
            budgets["max_library_bytes"]["core"],
            baseline["current"]["core_budget"],
        )
        self.assertEqual(budgets["max_library_bytes"]["relay"], 450000)
        self.assertEqual(
            baseline["previous"]["relay_bytes"], baseline["current"]["relay_bytes"]
        )
        self.assertLess(
            baseline["current"]["core_bytes"],
            baseline["current"]["core_budget"],
        )
        self.assertEqual(
            baseline["current"]["core_bytes"] - baseline["previous"]["core_bytes"],
            85592,
        )

    def test_template_is_valid_and_explicitly_unmeasured(self):
        document = benchmark_artifact.validate(
            json.loads(TEMPLATE.read_text(encoding="utf-8"))
        )
        report = benchmark_artifact.render(document)
        self.assertIn("not measured", report)
        self.assertNotIn("x |", report)

    def test_measured_report_separates_encoder_and_policy(self):
        document = measured()
        document["comparison"] = {
            "candidate": "FlowEdge native runtime",
            "reference": "LeRobot policy executed with PyTorch",
            "reference_framework": "PyTorch",
        }
        report = benchmark_artifact.render(benchmark_artifact.validate(document))
        self.assertIn("Encoder p50 / p95 / p99", report)
        self.assertIn("Policy p50 / p95 / p99", report)
        self.assertIn("Relative comparison", report)
        self.assertIn("13 / 0", report)
        self.assertIn("Reference framework | `PyTorch`", report)

    def test_quantiles_must_be_monotonic(self):
        document = measured()
        document["measurements"]["flowedge"]["policy_ms"]["p95"] = 3.0
        with self.assertRaises(benchmark_artifact.ArtifactError):
            benchmark_artifact.validate(document)

    def test_unknown_allocations_need_a_reason_and_are_not_rendered_as_zero(self):
        document = measured()
        document["measurements"]["flowedge"]["allocations"] = {
            "setup": None,
            "hot_path": None,
        }
        with self.assertRaises(benchmark_artifact.ArtifactError):
            benchmark_artifact.validate(document)
        document["measurements"]["flowedge"]["allocations"]["reason"] = (
            "not instrumented"
        )
        report = benchmark_artifact.render(benchmark_artifact.validate(document))
        self.assertIn("not measured / not measured", report)

    def test_missing_reference_is_allowed_but_marked(self):
        document = measured()
        document["measurements"]["lerobot"] = None
        report = benchmark_artifact.render(benchmark_artifact.validate(document))
        self.assertIn("| lerobot | not measured", report)
        self.assertNotIn("Relative comparison", report)

    def test_cuda_device_is_rendered_and_requires_cuda_accelerator(self):
        document = measured()
        document["hardware"]["accelerator"] = "cuda"
        document["hardware"]["cuda_device"] = "NVIDIA GeForce GTX 1650"
        document["comparison"] = {
            "candidate": "FlowEdge native runtime",
            "reference": "LeRobot policy executed with PyTorch",
            "reference_framework": "PyTorch",
            "scope": (
                "matched observation encoder, history, noise, and DDIM schedule on CUDA; "
                "not TensorRT/ONNX; not Jetson/ARM"
            ),
        }
        document["parity"] = {"max_abs_error": 3.05e-5, "atol": 1e-3, "rtol": 1e-4}
        report = benchmark_artifact.render(benchmark_artifact.validate(document))
        self.assertIn("CUDA `NVIDIA GeForce GTX 1650`", report)
        self.assertIn("not TensorRT/ONNX; not Jetson/ARM", report)
        self.assertIn("Max abs error | 3.05e-05", report)
        document["hardware"]["accelerator"] = "cpu"
        with self.assertRaises(benchmark_artifact.ArtifactError):
            benchmark_artifact.validate(document)


if __name__ == "__main__":
    unittest.main()
