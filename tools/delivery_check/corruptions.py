"""Systematically corrupt valid delivery traces and measure monitor responses.

The experiment is deliberately trace-local: it reuses the protocol-fixed 513-case
extension domain, changes one observation (or silently omits one event), and
checks the independent finite-trace monitor.  It does not mutate the runner.
"""

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .experiments import provenance
from .extension import domains, execute
from .monitor import check_trace

ROOT = Path(__file__).resolve().parents[2]


def _changed(events, index, field, value, description):
    changed = copy.deepcopy(events)
    before = copy.deepcopy(changed[index])
    changed[index][field] = value
    return changed, {
        "operation": description,
        "event_index": index,
        "before": before,
        "after": copy.deepcopy(changed[index]),
    }


def _omitted(events, index, *, renumber, description):
    changed = copy.deepcopy(events)
    removed = changed.pop(index)
    if renumber:
        for sequence, event in enumerate(changed):
            event["seq"] = sequence
    return changed, {
        "operation": description,
        "event_index": index,
        "removed": removed,
        "renumbered_suffix": renumber,
    }


def _find(events, predicate):
    return next((i for i, event in enumerate(events) if predicate(event)), None)


def _unrenumbered_interior_omission(events):
    if len(events) < 3:
        return None
    return _omitted(
        events,
        1,
        renumber=False,
        description="omit one interior event while preserving sequence numbers",
    )


def _declared_capability_removal(events):
    observations = events[0].get("observations")
    if not isinstance(observations, list) or "origin" not in observations:
        return None
    changed = copy.deepcopy(events)
    before = copy.deepcopy(changed[0])
    changed[0]["observations"].remove("origin")
    return changed, {
        "operation": "remove declared origin-observation capability",
        "event_index": 0,
        "before": before,
        "after": copy.deepcopy(changed[0]),
    }


def _true_readiness_without_completion(events):
    index = _find(
        events,
        lambda event: event.get("kind") == "ready_check"
        and event.get("ready") is False
        and event.get("request") is not None,
    )
    if index is None:
        return None
    return _changed(
        events,
        index,
        "ready",
        True,
        "report a pending request ready without a completion event",
    )


def _hidden_pending_request(events):
    index = _find(
        events,
        lambda event: event.get("kind") == "ready_check"
        and event.get("ready") is False
        and event.get("request") is not None,
    )
    if index is None:
        return None
    return _changed(
        events,
        index,
        "request",
        None,
        "hide the pending request at its readiness check",
    )


def _omitted_async_completion(events):
    if events[0].get("profile") != "async":
        return None
    index = _find(events, lambda event: event.get("kind") == "complete")
    if index is None:
        return None
    return _omitted(
        events,
        index,
        renumber=True,
        description="silently omit an asynchronous completion",
    )


def _omitted_disposition(events):
    if events[0].get("profile") != "async":
        return None
    index = _find(events, lambda event: event.get("kind") in ("accept", "discard"))
    if index is None:
        return None
    return _omitted(
        events,
        index,
        renumber=True,
        description="silently omit an asynchronous chunk disposition",
    )


def _corrupted_request_association(events):
    index = _find(events, lambda event: event.get("kind") == "complete")
    if index is None:
        return None
    request = events[index].get("request")
    if type(request) is not int:
        return None
    return _changed(
        events,
        index,
        "request",
        request + 10_000,
        "associate completion with an unknown request",
    )


def _corrupted_base_association(events):
    index = _find(events, lambda event: event.get("kind") == "submit")
    if index is None:
        return None
    base = events[index].get("base")
    if type(base) is not int:
        return None
    return _changed(
        events,
        index,
        "base",
        base + 1,
        "associate a request with the wrong observation base",
    )


def _corrupted_slot_association(events):
    index = _find(
        events,
        lambda event: event.get("kind") in ("dequeue", "send_begin")
        and isinstance(event.get("origin"), dict)
        and event["origin"].get("kind") == "policy",
    )
    if index is None:
        return None
    changed = copy.deepcopy(events)
    before = copy.deepcopy(changed[index])
    changed[index]["origin"]["slot"] += 1
    return changed, {
        "operation": "associate a selected policy action with the wrong slot",
        "event_index": index,
        "before": before,
        "after": copy.deepcopy(changed[index]),
    }


def _omitted_sync_held_commit(events):
    if events[0].get("profile") != "sync":
        return None
    index = _find(events, lambda event: event.get("kind") == "held_commit")
    if index is None:
        return None
    return _omitted(
        events,
        index,
        renumber=True,
        description="silently omit a successful synchronous held commit",
    )


def _omitted_stop_completion(events):
    index = _find(events, lambda event: event.get("kind") == "stop_end")
    if index is None:
        return None
    return _omitted(
        events,
        index,
        renumber=True,
        description="silently omit stop completion",
    )


def _wrong_terminal_completed_count(events):
    index = _find(events, lambda event: event.get("kind") == "run_end")
    if index is None:
        return None
    completed = events[index].get("completed")
    if type(completed) is not int:
        return None
    return _changed(
        events,
        index,
        "completed",
        completed + 1,
        "report the wrong terminal completed-step count",
    )


def _wrong_step_outcome(events):
    index = _find(events, lambda event: event.get("kind") == "step_end")
    if index is None:
        return None
    outcome = events[index].get("outcome")
    if outcome not in ("send", "skip"):
        return None
    return _changed(
        events,
        index,
        "outcome",
        "skip" if outcome == "send" else "send",
        "report the opposite terminal outcome for one completed step",
    )


# Ordered for stable output and review.  Each function changes one observation;
# silent event omissions renumber the suffix so integrity checking cannot alone
# explain the semantic failure.
CORRUPTIONS = (
    ("unrenumbered_interior_omission", "unknown", _unrenumbered_interior_omission),
    (
        "declared_observation_capability_removal",
        "unknown",
        _declared_capability_removal,
    ),
    ("true_readiness_without_completion", "fail", _true_readiness_without_completion),
    ("hidden_pending_request", "fail", _hidden_pending_request),
    ("omitted_async_completion", "fail", _omitted_async_completion),
    ("omitted_disposition", "fail", _omitted_disposition),
    ("corrupted_request_association", "fail", _corrupted_request_association),
    ("corrupted_base_association", "fail", _corrupted_base_association),
    ("corrupted_slot_association", "fail", _corrupted_slot_association),
    ("omitted_sync_held_commit", "fail", _omitted_sync_held_commit),
    ("omitted_stop_completion", "fail", _omitted_stop_completion),
    ("wrong_terminal_completed_count", "fail", _wrong_terminal_completed_count),
    ("wrong_step_outcome", "fail", _wrong_step_outcome),
)


def evaluate(root=ROOT, protocol_path=None):
    """Run all applicable single-corruption checks over the protocol domain."""
    protocol_path = (
        Path(protocol_path)
        if protocol_path is not None
        else Path(root) / "papers/action-delivery/extension-protocol.json"
    )
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    grids = domains(protocol)
    cases = []
    profile_counts = {}
    for profile, grid in grids.items():
        profile_counts[profile] = len(grid)
        cases.extend(
            (profile, index, schedule, execute(schedule))
            for index, schedule in enumerate(grid)
        )

    valid_cases = [
        row
        for row in cases
        if row[3]["stopped"]
        and row[3]["monitor"]["verdict"] == "pass"
        and row[3]["boundary"]["verdict"] == "pass"
    ]
    class_results = []
    for name, expected, corrupt in CORRUPTIONS:
        verdicts = Counter()
        reasons = Counter()
        mismatches = 0
        witness = None
        witness_key = None
        for profile, index, schedule, case in valid_cases:
            corrupted = corrupt(case["events"])
            if corrupted is None:
                continue
            events, change = corrupted
            report = check_trace(events).record()
            verdicts[report["verdict"]] += 1
            reasons[report["reason"]] += 1
            mismatches += report["verdict"] != expected
            key = (len(case["events"]), profile, index)
            if witness is None or key < witness_key:
                witness_key = key
                witness = {
                    "base_case": {"profile": profile, "index": index},
                    "schedule": schedule,
                    "base_event_count": len(case["events"]),
                    "change": change,
                    "report": report,
                }
        applicable = sum(verdicts.values())
        class_results.append(
            {
                "name": name,
                "expected_verdict": expected,
                "applicable_cases": applicable,
                "actual_verdict_counts": dict(sorted(verdicts.items())),
                "actual_reason_counts": dict(sorted(reasons.items())),
                "unexpected_or_mismatched": mismatches,
                "minimized_witness": witness,
            }
        )

    generated = sum(row["applicable_cases"] for row in class_results)
    mismatches = sum(row["unexpected_or_mismatched"] for row in class_results)
    unexpected_passes = sum(
        row["actual_verdict_counts"].get("pass", 0)
        for row in class_results
        if row["expected_verdict"] != "pass"
    )
    complete = (
        len(cases) == 513
        and len(valid_cases) == len(cases)
        and all(row["applicable_cases"] > 0 for row in class_results)
        and generated > 0
        and mismatches == 0
        and unexpected_passes == 0
    )
    return {
        "schema": 1,
        "scope": "systematic_monitor_corruption_robustness",
        "status": "completed",
        "expectations_met": complete,
        "provenance": provenance(root),
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generation_command": [sys.executable, *sys.argv],
        "base_cases": {
            "total": len(cases),
            "valid": len(valid_cases),
            "invalid": len(cases) - len(valid_cases),
            "profiles": profile_counts,
        },
        "generated_corruptions": generated,
        "classes": class_results,
        "summary": {
            "classes": len(class_results),
            "expected_unknown_classes": sum(
                row["expected_verdict"] == "unknown" for row in class_results
            ),
            "expected_fail_classes": sum(
                row["expected_verdict"] == "fail" for row in class_results
            ),
            "mismatches": mismatches,
            "unexpected_passes": unexpected_passes,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "papers/action-delivery/extension-protocol.json",
    )
    args = parser.parse_args()
    result = evaluate(ROOT, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "expectations_met": result["expectations_met"],
                "base_cases": result["base_cases"],
                "generated_corruptions": result["generated_corruptions"],
                "summary": result["summary"],
            }
        )
    )
    return 0 if result["expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
