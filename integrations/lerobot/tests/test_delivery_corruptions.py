import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.delivery_check.corruptions import CORRUPTIONS, evaluate  # noqa: E402


class DeliveryCorruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = evaluate(ROOT)

    def test_complete_protocol_domain_and_named_classes(self):
        result = self.result
        self.assertEqual(result["base_cases"]["total"], 513)
        self.assertEqual(result["base_cases"]["valid"], 513)
        self.assertEqual(result["base_cases"]["invalid"], 0)
        self.assertEqual(sum(result["base_cases"]["profiles"].values()), 513)
        self.assertEqual(
            [row["name"] for row in result["classes"]],
            [name for name, _, _ in CORRUPTIONS],
        )
        self.assertTrue(all(row["applicable_cases"] > 0 for row in result["classes"]))

    def test_every_actual_verdict_matches_its_expectation(self):
        for row in self.result["classes"]:
            with self.subTest(corruption=row["name"]):
                self.assertEqual(row["unexpected_or_mismatched"], 0)
                self.assertEqual(row["actual_verdict_counts"].get("pass", 0), 0)
                self.assertEqual(
                    row["actual_verdict_counts"],
                    {row["expected_verdict"]: row["applicable_cases"]},
                )
                self.assertEqual(
                    sum(row["actual_reason_counts"].values()),
                    row["applicable_cases"],
                )
                self.assertIsNotNone(row["minimized_witness"])
                self.assertEqual(
                    row["minimized_witness"]["report"]["verdict"],
                    row["expected_verdict"],
                )

    def test_top_level_accounting_is_exact_and_successful(self):
        result = self.result
        generated = sum(row["applicable_cases"] for row in result["classes"])
        mismatches = sum(row["unexpected_or_mismatched"] for row in result["classes"])
        self.assertEqual(result["generated_corruptions"], generated)
        self.assertEqual(result["summary"]["classes"], len(CORRUPTIONS))
        self.assertEqual(result["summary"]["mismatches"], mismatches)
        self.assertEqual(result["summary"]["unexpected_passes"], 0)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["expectations_met"])
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
