import copy
import sys
import unittest
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.delivery_check.controlled import Completion, execute_async, execute_sync
from tools.delivery_check.model import explore, replay
from tools.delivery_check.monitor import check_trace
from tools.delivery_check.erasure import evaluate as evaluate_erasure
from tools.delivery_check.experiments import account_verdicts
from tools.delivery_check.source_audit import TARGETS, audit_sources


def renumber(events):
    for i, event in enumerate(events):
        event["seq"] = i
    return events


class DeliveryContractTests(unittest.TestCase):
    def assert_conforms(self, case):
        self.assertTrue(case["stopped"])
        self.assertEqual(case["monitor"]["verdict"], "pass", case["monitor"])
        self.assertEqual(case["boundary"]["verdict"], "pass", case["boundary"])

    def test_sync_deadline_boundary_fallbacks_and_initial_hold(self):
        for mode, durations in product(
            ("hold", "drop", "raise"), product((0, 1_000_000, 1_000_001), repeat=3)
        ):
            with self.subTest(mode=mode, durations=durations):
                self.assert_conforms(execute_sync(durations, mode=mode))
        self.assertEqual(execute_sync((2_000_000,))["actions"], [[0.0]])
        self.assertEqual(execute_sync((0, 2_000_000))["actions"], [[1001.0], [1001.0]])

    def test_async_delay_phase_expiry_and_pending_stop(self):
        for horizon, delay, phase in product(
            (1, 2, 4), range(6), ("before_check", "after_check")
        ):
            with self.subTest(horizon=horizon, delay=delay, phase=phase):
                self.assert_conforms(
                    execute_async((Completion(delay, phase),), horizon=horizon)
                )

    def test_readiness_phases_change_selected_request(self):
        before = execute_async((Completion(1, "before_check"),))
        after = execute_async((Completion(1, "after_check"),))
        self.assertEqual(before["actions"][1], [2002.0])
        self.assertEqual(after["actions"][1], [1002.0])

    def test_faults_and_early_termination_are_separate_from_skips(self):
        for case in (
            execute_sync(fail_send=0),
            execute_async(fail_send=1),
            execute_async(end_after=1),
        ):
            self.assert_conforms(case)
        for case in (
            execute_async(invalid_request=1),
            execute_async(invalid_request=0),
        ):
            self.assertTrue(case["stopped"])
            self.assertEqual(case["monitor"]["verdict"], "unknown")
            self.assertEqual(case["monitor"]["reason"], "unexplained_fault")
            self.assertEqual(case["boundary"]["verdict"], "pass")
        events = execute_sync((0,))["events"]
        self.assertEqual(check_trace(events[:-1]).verdict, "unknown")

    def test_error_terminal_requires_observed_fault(self):
        relabeled = copy.deepcopy(execute_sync((0,))["events"])
        relabeled[-1]["status"] = "error"
        report = check_trace(relabeled)
        self.assertEqual(report.verdict, "unknown")
        self.assertEqual(report.reason, "error_without_fault_observation")

        missing_fault = [
            event
            for event in copy.deepcopy(execute_sync((0,), fail_send=0)["events"])
            if event["kind"] != "fault"
        ]
        report = check_trace(renumber(missing_fault))
        self.assertEqual(report.verdict, "unknown")
        self.assertEqual(report.reason, "error_without_fault_observation")

        inserted_fault = copy.deepcopy(execute_sync((0,))["events"])
        stop = next(
            index
            for index, event in enumerate(inserted_fault)
            if event["kind"] == "stop_begin"
        )
        inserted_fault.insert(
            stop,
            {
                "schema": 1,
                "run": inserted_fault[0]["run"],
                "kind": "fault",
                "step": 1,
            },
        )
        inserted_fault[-1]["status"] = "error"
        report = check_trace(renumber(inserted_fault))
        self.assertEqual(report.verdict, "unknown")
        self.assertEqual(report.reason, "unexplained_fault")

    def test_bootstrap_exemption_cannot_be_reused_midrun(self):
        events = copy.deepcopy(execute_async()["events"])
        complete = next(
            event
            for event in events
            if event["kind"] == "complete" and event["step"] != -1
        )
        request = complete["request"]
        complete["step"] = -1
        disposition = next(
            event
            for event in events
            if event["kind"] == "accept"
            and event["step"] != -1
            and event["request"] == request
        )
        disposition["step"] = -1
        report = check_trace(events)
        self.assertEqual(report.verdict, "fail")
        self.assertEqual(report.reason, "illegal_bootstrap_completion")

    def test_optional_tracing_does_not_change_delivery(self):
        for traced, untraced in (
            (execute_sync((0, 2_000_000)), execute_sync((0, 2_000_000), trace=False)),
            (execute_async(), execute_async(trace=False)),
        ):
            for key in ("actions", "attempts", "error", "stopped"):
                self.assertEqual(traced[key], untraced[key], key)
            self.assertTrue(traced["events"])
            self.assertEqual(untraced["events"], [])

    def test_integrity_and_observation_ablations_fail_closed(self):
        for invalid in (None, "bad", [1], [{"schema": 1}]):
            self.assertEqual(check_trace(invalid).verdict, "unknown")
        original = execute_async()["events"]
        for events in (
            original[1:],
            original[:-1],
            original[:5] + original[6:],
            original[:5] + [original[4]] + original[5:],
        ):
            self.assertEqual(check_trace(events).verdict, "unknown")
        no_origins = copy.deepcopy(original)
        for event in no_origins:
            event.pop("origin", None)
        self.assertEqual(check_trace(no_origins).verdict, "unknown")
        no_phase = [e for e in original if e["kind"] != "ready_check"]
        # Unrenumbered removal is an observation gap, not a valid execution.
        self.assertEqual(check_trace(no_phase).verdict, "unknown")

    def test_complete_step_without_send_or_skip_is_a_violation(self):
        events = execute_sync((2_000_000,), mode="drop")["events"]
        events = renumber([e for e in events if e["kind"] != "skip"])
        result = check_trace(events)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.reason, "missing_or_wrong_step_outcome")

    def test_hand_derived_wrong_origin_and_missing_held_commit(self):
        events = execute_sync((0, 2_000_000))["events"]
        wrong = copy.deepcopy(events)
        send = next(e for e in wrong if e["kind"] == "send_begin" and e["step"] == 1)
        send["origin"] = {"kind": "policy", "request": 1, "observation": 1, "slot": 0}
        self.assertEqual(check_trace(wrong).reason, "sent_origin")
        missing = renumber([e for e in events if e["kind"] != "held_commit"])
        self.assertEqual(check_trace(missing).reason, "missing_held_commit")

    def test_sink_failure_does_not_prevent_stop(self):
        from flowedge_lerobot.delivery_trace import DeliveryTrace

        from tools.delivery_check.controlled import Robot

        robot = Robot()

        def broken(event):
            raise OSError("disk full")

        trace = DeliveryTrace(broken)
        trace.stop(robot)
        self.assertTrue(robot.stopped)
        self.assertIsInstance(trace.error, OSError)

    def test_bounded_model_and_replay(self):
        result = explore()
        self.assertEqual(result["verdict"], "pass", result)
        for mutation in ("offset_minus_one", "send_after_stop"):
            result = explore(mutation=mutation)
            self.assertEqual(result["verdict"], "fail", result)
            self.assertEqual(
                replay(result["witness"]["actions"], mutation=mutation)["verdict"],
                "fail",
            )

    def test_analysis_bounds_are_not_runtime_rules(self):
        for result in (
            explore(max_states=1),
            explore(queue_bound=1),
            explore(seconds=1e-12),
        ):
            self.assertEqual(result["verdict"], "unknown", result)
        short = explore(events=1)
        self.assertEqual(short["verdict"], "pass")
        self.assertGreater(short["event_frontier"], 0)

    def test_real_code_mutations_are_detected(self):
        from tools.delivery_check.experiments import mutation_cases

        for case in mutation_cases():
            self.assertTrue(case["detected"], case["name"])
        held = next(c for c in mutation_cases() if c["name"] == "sync_rejected_hold")
        self.assertEqual(held["witness"]["monitor"]["verdict"], "pass")
        self.assertEqual(
            held["witness"]["boundary"]["reason"], "tensor_origin_disagreement"
        )

    def test_declared_missing_phase_is_unknown_even_with_contiguous_sequence(self):
        events = execute_async()["events"]
        events[0]["observations"].remove("phase")
        self.assertEqual(
            check_trace(events).reason, "insufficient_observation_coverage"
        )

    def test_duplicate_chunk_acceptance_is_rejected(self):
        events = execute_async()["events"]
        index = next(i for i, e in enumerate(events) if e["kind"] == "accept")
        events.insert(index, copy.deepcopy(events[index]))
        self.assertEqual(
            check_trace(renumber(events)).reason, "duplicate_chunk_disposition"
        )

    def test_ready_chunk_cannot_be_ignored_without_disposition(self):
        events = execute_async()["events"]
        index = next(
            i for i, e in enumerate(events) if e["kind"] == "accept" and e["step"] != -1
        )
        del events[index]
        self.assertEqual(
            check_trace(renumber(events)).reason, "missing_chunk_disposition"
        )

    def test_true_readiness_requires_matching_completion(self):
        original = execute_async((Completion(5),), steps=6)["events"]
        events = copy.deepcopy(original)
        ready = next(
            event
            for event in events
            if event["kind"] == "ready_check"
            and event["ready"] is False
            and event["request"] is not None
        )
        ready["ready"] = True
        result = check_trace(events)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.reason, "missing_ready_completion")

        missing_request = copy.deepcopy(original)
        ready = next(
            event
            for event in missing_request
            if event["kind"] == "ready_check"
            and event["ready"] is False
            and event["request"] is not None
        )
        ready["request"] = None
        result = check_trace(missing_request)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.reason, "pending_request_mismatch")

    def test_payload_erasure_finite_sanity_check_and_negative_control(self):
        result = evaluate_erasure()
        self.assertEqual(result["verdict"], "pass", result)
        self.assertEqual(result["negative_controls_detected"], 5)
        self.assertTrue(all(c["detected"] for c in result["negative_controls"]))

    def test_source_observation_inventory_and_added_send_negative_control(self):
        root = Path(__file__).resolve().parents[3]
        result = audit_sources(root)
        self.assertEqual(result["verdict"], "pass", result)
        path = "integrations/lerobot/src/flowedge_lerobot/rollout.py"
        source = (root / path).read_text(encoding="utf-8")
        needle = "robot.send_action(action)"
        self.assertGreaterEqual(source.count(needle), 1)
        mutated = source.replace(
            needle, needle + "\n                robot.send_action(action)", 1
        )
        result = audit_sources(root, overrides={path: mutated})
        self.assertEqual(result["verdict"], "fail")
        self.assertTrue(result["failures"])
        alias = source.replace(
            needle,
            "send = robot.send_action\n                send(action)",
            1,
        )
        result = audit_sources(root, overrides={path: alias})
        self.assertEqual(result["verdict"], "fail")
        self.assertTrue(
            any("alias" in failure.get("reason", "") for failure in result["failures"])
        )
        self.assertIn(path, TARGETS)

    def test_evaluation_accounting_separates_expected_unknowns(self):
        counts = account_verdicts(
            [
                {"verdict": "pass", "reason": "bounded_product_exhausted"},
                {"verdict": "unknown", "reason": "state_budget_exhausted"},
                {"verdict": "unknown", "reason": "unexpected_reason"},
            ],
            {"missing": {"verdict": "fail", "reason": "violation"}},
        )
        self.assertEqual(
            counts,
            {
                "pass": 1,
                "fail": 1,
                "unknown_expected": 1,
                "unknown_unexpected": 1,
            },
        )

    def test_bounded_history_generator_has_unique_requests(self):
        from tools.delivery_check.lerobot_case import history_operations

        operations = history_operations(("m0", "get", "m1", "clear", "m2"))
        requests = [op["request"] for op in operations if op["kind"] == "merge"]
        self.assertEqual(requests, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
