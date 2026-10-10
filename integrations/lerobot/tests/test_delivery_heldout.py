import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.delivery_check.artifact import INCLUDE
from tools.delivery_check.baseline import run as run_baseline
from tools.delivery_check.heldout import (
    DEVELOPMENT_NAMES,
    CANDIDATES,
    candidate_pool_sha256,
    evaluate_protocol,
    exercising_protocol,
    generate_protocol,
    select_candidates,
)
from tools.delivery_check.observation import analyze as analyze_observations

EVIDENCE = ROOT / "papers/action-delivery"
HELDOUT_PROTOCOL = EVIDENCE / "heldout-protocol.json"
EXERCISING_PROTOCOL = EVIDENCE / "heldout-protocol-exercising.json"
BASELINE_EVIDENCE = all(
    path.is_file()
    for path in (
        EVIDENCE / "results/tlc.json",
        EVIDENCE / "tla/Delivery.tla",
        EVIDENCE / "tla/Delivery.cfg",
        EVIDENCE / "tla/Mutant.cfg",
    )
)
OBSERVATION_EVIDENCE = all(
    path.is_file()
    for path in (
        EVIDENCE / "results/development.json",
        EVIDENCE / "results/extension-linux.json",
        EVIDENCE / "results/heldout-posthoc-windows.json",
    )
)


class HeldOutProtocolTests(unittest.TestCase):
    def test_selection_is_deterministic_and_disjoint(self):
        first = generate_protocol()
        second = generate_protocol()
        self.assertEqual(first, second)
        names = {item["name"] for item in first["mutations"]}
        self.assertEqual(len(names), 4)
        self.assertTrue(names.isdisjoint(DEVELOPMENT_NAMES))
        self.assertEqual(first["candidate_pool_sha256"], candidate_pool_sha256())

    def test_pool_has_unique_sites(self):
        self.assertEqual(len({item["name"] for item in CANDIDATES}), len(CANDIDATES))
        self.assertEqual(
            {item["profile"] for item in select_candidates(240924, 2)},
            {"async", "sync"},
        )

    @unittest.skipUnless(
        HELDOUT_PROTOCOL.is_file(), "requires frozen held-out protocol"
    )
    def test_frozen_protocol_evaluation_retains_survivor(self):
        result = evaluate_protocol(HELDOUT_PROTOCOL)
        self.assertTrue(result["expectations_met"])
        self.assertEqual(
            result["outcome_counts"],
            {
                "selected_mutations": 4,
                "observable_mutations": 4,
                "observable_detected": 3,
                "projection_rejected_mutations": 2,
                "baseline_projection_rejections": 0,
                "execution_errors": 0,
                "cases": 378,
            },
        )
        survivor = next(
            row
            for row in result["mutations"]
            if row["name"] == "async_refill_strict"
        )
        self.assertEqual(survivor["detections"]["either"], 0)
        self.assertEqual(survivor["detections"]["trace_differences"], 108)
        self.assertEqual(survivor["detections"]["projection"], 108)
        self.assertEqual(
            result["domain_characterization"]["async"],
            {
                "schedule_cases": 108,
                "distinct_observed_executions": 5,
                "live_accept_cases": 0,
                "baseline_projection": {
                    "passes": 108,
                    "rejections": 0,
                    "reason_counts": {},
                },
            },
        )

    @unittest.skipUnless(
        HELDOUT_PROTOCOL.is_file() and EXERCISING_PROTOCOL.is_file(),
        "requires frozen held-out protocols",
    )
    def test_exercising_domain_accepts_live_chunks_and_keeps_refill_gap(self):
        frozen = json.loads(
            HELDOUT_PROTOCOL.read_text(encoding="utf-8")
        )
        derived = exercising_protocol(frozen)
        self.assertEqual(
            [item["name"] for item in derived["mutations"]],
            [item["name"] for item in frozen["mutations"]],
        )
        self.assertEqual(derived["async_domain"]["horizons"], [8, 12])
        result = evaluate_protocol(EXERCISING_PROTOCOL)
        async_domain = result["domain_characterization"]["async"]
        self.assertGreater(async_domain["live_accept_cases"], 0)
        self.assertGreater(async_domain["distinct_observed_executions"], 5)
        self.assertEqual(async_domain["schedule_cases"], 72)
        self.assertEqual(async_domain["live_accept_cases"], 36)
        self.assertEqual(async_domain["distinct_observed_executions"], 19)
        survivor = next(
            row
            for row in result["mutations"]
            if row["name"] == "async_refill_strict"
        )
        self.assertEqual(survivor["detections"]["either"], 0)
        self.assertEqual(survivor["detections"]["trace_differences"], 72)
        self.assertEqual(survivor["detections"]["action_differences"], 36)
        self.assertEqual(survivor["detections"]["projection"], 72)
        self.assertEqual(result["outcome_counts"]["observable_detected"], 3)
        self.assertEqual(result["outcome_counts"]["cases"], 306)

    @unittest.skipUnless(BASELINE_EVIDENCE, "requires frozen TLA+ evidence")
    def test_matched_baseline_uses_verified_stored_tlc_when_jar_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_baseline(
                output=root / "baseline.json",
                jar=root / "missing-tla2tools.jar",
                repeat=1,
                skip_tlc=False,
                stored=EVIDENCE / "results/tlc.json",
            )
        self.assertTrue(result["expectations_met"])
        self.assertEqual(result["general_tool"]["availability"], "stored")
        self.assertEqual(result["comparison"]["python_valid_states"], 130)
        self.assertEqual(result["comparison"]["tla_valid_distinct_states"], 130)
        self.assertTrue(result["comparison"]["both_detect_offset_mutation"])
        self.assertFalse(result["effort_evidence"]["independent_authors"])
        self.assertIsNone(result["effort_evidence"]["recorded_person_hours"])
        self.assertIn(
            "No specification-effort advantage",
            result["effort_evidence"]["conclusion"],
        )

    @unittest.skipUnless(
        OBSERVATION_EVIDENCE, "requires frozen observation evidence"
    )
    def test_observation_adequacy_retains_the_heldout_gap(self):
        results = EVIDENCE / "results"
        result = analyze_observations(
            results / "development.json",
            results / "extension-linux.json",
            results / "heldout-posthoc-windows.json",
        )
        self.assertTrue(result["expectations_met"])
        self.assertEqual(result["summary"]["observable_mutations"], 17)
        self.assertEqual(result["summary"]["detected_mutations"], 16)
        self.assertEqual(
            result["summary"]["survivor_ids"],
            ["heldout:async_refill_strict"],
        )
        self.assertEqual(
            result["summary"]["minimal_channel_sets_for_detectable_mutations"],
            [["monitor", "tensor", "lifecycle"]],
        )
        self.assertFalse(result["summary"]["all_observable_covered"])
        self.assertEqual(
            result["summary"]["offline_projection_rejected_survivors"],
            ["heldout:async_refill_strict"],
        )

    def test_review_bundle_contains_frozen_inputs_and_results(self):
        required = {
            "tools/delivery_check/heldout.py",
            "integrations/lerobot/tests/test_delivery_heldout.py",
            "papers/action-delivery/heldout.md",
            "papers/action-delivery/heldout-protocol.json",
            "papers/action-delivery/AI_USE.md",
            "papers/action-delivery/artifact-root-README.md",
            "papers/action-delivery/results/heldout.json",
            "tools/delivery_check/baseline.py",
            "tools/delivery_check/observation.py",
            "papers/action-delivery/results/general-tool-baseline.json",
            "papers/action-delivery/results/observation-adequacy.json",
            "papers/action-delivery/results/development-posthoc-windows.json",
            "papers/action-delivery/results/extension-posthoc-windows.json",
            "papers/action-delivery/results/heldout-posthoc-windows.json",
            "papers/action-delivery/results/monitor-corruptions-posthoc-windows.json",
            "papers/action-delivery/results/observation-adequacy-posthoc-windows.json",
        }
        self.assertTrue(required.issubset(INCLUDE))
        self.assertNotIn("papers/action-delivery/paper.pdf", INCLUDE)
        self.assertNotIn("papers/action-delivery/human-review-packet.md", INCLUDE)
        self.assertNotIn(
            "papers/action-delivery/archival-release-checklist.md", INCLUDE
        )


if __name__ == "__main__":
    unittest.main()
