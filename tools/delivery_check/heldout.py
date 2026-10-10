"""Generate and execute a frozen, development-disjoint mutation study.

This module deliberately keeps protocol generation separate from execution.  A
protocol must be generated once, inspected or archived, and then passed to
``evaluate``.  The evaluator never samples or changes the mutation set.  The
study is held out from the repository's development and extension mutations;
it is not evidence about natural bug prevalence or an independently authored
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/lerobot/src"))

from .experiments import MUTATIONS, provenance
from .controlled import observed_execution_signature
from .extension import compile_mutation, detections, domains, execute, project_trace


# These patches were not used by either the four-case development study or the
# ten-case protocol-fixed extension.  The pool is fixed in source and hashed in
# the generated protocol before any held-out result is produced.
CANDIDATES = (
    {
        "name": "async_expiration_inclusive",
        "profile": "async",
        "old": "if elapsed_actions < action_steps:",
        "new": "if elapsed_actions <= action_steps:",
        "classification": "observable",
        "obligation": "expired chunks must not be accepted",
    },
    {
        "name": "async_fallback_as_policy",
        "profile": "async",
        "old": 'Origin("fallback", -1, tick, 0)',
        "new": 'Origin("policy", -1, tick, 0)',
        "classification": "observable",
        "obligation": "fallback actions retain their fallback origin",
    },
    {
        "name": "async_send_tick_plus_one",
        "profile": "async",
        "old": "trace.send(robot, action, tick, origin)",
        "new": "trace.send(robot, action, tick + 1, origin)",
        "classification": "observable",
        "obligation": "send observation uses the current control tick",
    },
    {
        "name": "async_refill_strict",
        "profile": "async",
        "old": "len(actions) <= int(action_steps * refill_threshold)",
        "new": "len(actions) < int(action_steps * refill_threshold)",
        "classification": "observable",
        "obligation": "refill at the declared threshold",
    },
    {
        "name": "sync_hold_origin_policy",
        "profile": "sync",
        "old": "trace.send(robot, held, tick, held_origin)",
        "new": "trace.send(robot, held, tick, origin)",
        "classification": "observable",
        "obligation": "held payload retains the prior committed origin",
    },
    {
        "name": "sync_zero_hold_after_commit",
        "profile": "sync",
        "old": "np.copyto(held, action)",
        "new": "np.copyto(held, action * 0)",
        "classification": "observable",
        "obligation": "successful sends commit the sent payload",
    },
    {
        "name": "sync_drop_deadline_action",
        "profile": "sync",
        "old": "trace.send(robot, action, tick, origin)",
        "new": "trace.send(robot, held, tick, held_origin)",
        "classification": "observable",
        "obligation": "on-time sends use the newly computed action",
    },
    {
        "name": "sync_late_hold_is_drop",
        "profile": "sync",
        "old": 'if on_miss == "hold":',
        "new": 'if on_miss == "drop":',
        "classification": "observable",
        "obligation": "the hold policy sends the previous action on a miss",
    },
)

DEVELOPMENT_NAMES = frozenset(MUTATIONS) | frozenset(
    {
        "async_slice_plus_one",
        "async_pop_tail",
        "async_keep_old_queue",
        "async_stale_observation",
        "async_duplicate_send",
        "async_suppress_actual_stop",
        "sync_equal_is_late",
        "sync_uncommitted_hold",
        "sync_nonzero_initial",
        "sync_early_private_commit",
    }
)


def canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def candidate_pool_sha256() -> str:
    return hashlib.sha256(canonical_json(list(CANDIDATES))).hexdigest()


def select_candidates(seed: int, per_profile: int) -> list[dict]:
    if per_profile <= 0:
        raise ValueError("per_profile must be positive")
    chosen = []
    rng = random.Random(seed)
    for profile in ("async", "sync"):
        pool = [item for item in CANDIDATES if item["profile"] == profile]
        if per_profile > len(pool):
            raise ValueError(f"per_profile exceeds {profile} candidate pool")
        chosen.extend(rng.sample(pool, per_profile))
    return chosen


def _validate_patch(description: dict) -> str:
    function_name = "run_async_rollout" if description["profile"] == "async" else "run_rollout"
    module_name = (
        "flowedge_lerobot.async_rollout"
        if description["profile"] == "async"
        else "flowedge_lerobot.rollout"
    )
    module = __import__(module_name, fromlist=[function_name])
    source = inspect.getsource(getattr(module, function_name))
    if source.count(description["old"]) != 1:
        raise ValueError(f"mutation site drift: {description['name']}")
    return hashlib.sha256(source.encode()).hexdigest()


# Delays and horizons are disjoint from the extension domain (delays 0--5,
# horizons 1, 2, and 4). Each horizon is paired with a shorter delay, so a
# non-bootstrap suffix is eligible, and with a longer delay, so the chunk is
# expired. Steps extend past the longest delay, so a completion can arrive
# after the queue has been consumed below the refill threshold. Both completion
# phases are included. This domain was specified from those coverage
# requirements, not from a mutation outcome.
EXERCISING_ASYNC_DOMAIN = {
    "horizons": [8, 12],
    "delays": [6, 9, 13],
    "phases": ["before_check", "after_check"],
    "positions": 2,
    "steps": 16,
    "domain_id": "live-accept-v1",
}


def exercising_protocol(frozen: dict) -> dict:
    """Same frozen mutations, on a domain that accepts live chunks."""
    if frozen.get("scope") != "development_disjoint_heldout":
        raise ValueError("exercising domain must start from the frozen protocol")
    protocol = json.loads(json.dumps(frozen))
    protocol["label"] = (
        "same frozen mutations on a live-accept domain; "
        "not natural-bug prevalence"
    )
    protocol["async_domain"] = dict(EXERCISING_ASYNC_DOMAIN)
    protocol["domain_id"] = EXERCISING_ASYNC_DOMAIN["domain_id"]
    protocol["domain_rationale"] = (
        "Horizons 8 and 12 and delays 6, 9, and 13 are disjoint from the "
        "extension grid. Delay 6 is below both horizons, delay 9 is below only "
        "12, and delay 13 is above both. Both phases are used, and 16 steps "
        "pass the longest delay."
    )
    return protocol


def generate_protocol(*, seed: int = 240924, per_profile: int = 2) -> dict:
    selected = select_candidates(seed, per_profile)
    names = {item["name"] for item in selected}
    if names & DEVELOPMENT_NAMES:
        raise ValueError("held-out mutation overlaps development/extension set")
    source_hashes = {item["name"]: _validate_patch(item) for item in selected}
    return {
        "schema": 1,
        "scope": "development_disjoint_heldout",
        "label": "frozen held-out mutation study; not natural-bug prevalence",
        "freeze_policy": (
            "Generate once before evaluation. Archive this exact JSON and its hash. "
            "The evaluator does not select, remove, or rewrite mutations after execution."
        ),
        "generated_date": "2026-09-24",
        "selection_seed": seed,
        "per_profile": per_profile,
        "candidate_pool_sha256": candidate_pool_sha256(),
        "development_exclusion": sorted(DEVELOPMENT_NAMES),
        "domain_order_seed": 240925,
        "async_domain": {
            "horizons": [2, 3, 5],
            "delays": [6, 7, 8],
            "phases": ["before_check", "after_check"],
            "positions": 2,
            "steps": 7,
        },
        "sync_domain": {
            "durations": [500000, 1500000, 2000000],
            "positions": 3,
            "modes": ["hold", "drop", "raise"],
        },
        "mutations": [dict(item, source_sha256=source_hashes[item["name"]]) for item in selected],
    }


def _ordered_domains(protocol: dict) -> dict[str, list[dict]]:
    grids = domains(protocol)
    rng = random.Random(protocol["domain_order_seed"])
    for grid in grids.values():
        rng.shuffle(grid)
    return grids


def evaluate_protocol(protocol_path: Path) -> dict:
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("scope") != "development_disjoint_heldout":
        raise ValueError("wrong protocol scope")
    if protocol.get("candidate_pool_sha256") != candidate_pool_sha256():
        raise ValueError("candidate pool changed after protocol freeze")
    selected_names = {item["name"] for item in protocol["mutations"]}
    if selected_names & DEVELOPMENT_NAMES:
        raise ValueError("protocol overlaps development/extension mutations")
    if len(selected_names) != len(protocol["mutations"]):
        raise ValueError("protocol contains duplicate mutation names")
    expected_sources = {
        item["name"]: _validate_patch(item) for item in protocol["mutations"]
    }
    for item in protocol["mutations"]:
        if item.get("source_sha256") != expected_sources[item["name"]]:
            raise ValueError(f"source drift: {item['name']}")

    grids = _ordered_domains(protocol)
    baselines = {
        profile: [execute(schedule) for schedule in grid]
        for profile, grid in grids.items()
    }
    domain_characterization = {}
    for profile, cases in baselines.items():
        row = {
            "schedule_cases": len(cases),
            "distinct_observed_executions": len(
                {observed_execution_signature(case) for case in cases}
            ),
        }
        if profile == "async":
            row["live_accept_cases"] = sum(
                any(
                    event["kind"] == "accept" and event["step"] != -1
                    for event in case["events"]
                )
                for case in cases
            )
            projection_reasons = {}
            projection_passes = 0
            for schedule, case in zip(grids[profile], cases):
                try:
                    project_trace(case, schedule["horizon"], schedule["steps"])
                    projection_passes += 1
                except (AssertionError, ValueError) as error:
                    reason = str(error)
                    projection_reasons[reason] = (
                        projection_reasons.get(reason, 0) + 1
                    )
            row["baseline_projection"] = {
                "passes": projection_passes,
                "rejections": len(cases) - projection_passes,
                "reason_counts": projection_reasons,
            }
        domain_characterization[profile] = row
    mutation_rows = []
    for description in protocol["mutations"]:
        runner = compile_mutation(description)
        counters = {
            "monitor": 0,
            "tensor": 0,
            "lifecycle": 0,
            "either": 0,
            "unknown": 0,
            "trace_differences": 0,
            "action_differences": 0,
            "projection": 0,
        }
        errors = []
        projection_reasons = {}
        grid = grids[description["profile"]]
        for index, schedule in enumerate(grid):
            try:
                baseline = baselines[description["profile"]][index]
                case = execute(schedule, runner)
                found = detections(case)
                for key, value in found.items():
                    counters[key] += int(value)
                counters["either"] += int(any(found.values()))
                counters["unknown"] += int(any(case[k]["verdict"] == "unknown" for k in ("monitor", "boundary")))
                counters["trace_differences"] += int(case["events"] != baseline["events"])
                counters["action_differences"] += int(
                    case["actions"] != baseline["actions"]
                    or case["attempts"] != baseline["attempts"]
                )
                if description["profile"] == "async":
                    try:
                        project_trace(case, schedule["horizon"], schedule["steps"])
                    except (AssertionError, ValueError) as error:
                        counters["projection"] += 1
                        reason = str(error)
                        projection_reasons[reason] = (
                            projection_reasons.get(reason, 0) + 1
                        )
            except BaseException as error:  # record all outcomes, including faults
                errors.append({"index": index, "schedule": schedule, "error": type(error).__name__, "message": str(error)})
        if counters["either"]:
            interpretation = "detected by at least one declared observation channel"
        elif counters["projection"]:
            interpretation = (
                "online channels accept, but the offline phase-model projection "
                "rejects the changed event history"
            )
        elif counters["trace_differences"]:
            interpretation = "trace differs, but the declared monitor/tensor/lifecycle channels do not reject it"
        else:
            interpretation = "no difference in the controlled observations"
        mutation_rows.append(
            {
                "name": description["name"],
                "profile": description["profile"],
                "classification": description["classification"],
                "cases": len(grid),
                "detections": counters,
                "execution_errors": errors,
                "projection_applicable": description["profile"] == "async",
                "projection_reason_counts": projection_reasons,
                "interpretation": interpretation,
            }
        )
    observable = [row for row in mutation_rows if row["classification"] == "observable"]
    result = {
        "schema": 1,
        "scope": "development_disjoint_heldout",
        "status": "completed",
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_path": str(protocol_path),
        "protocol": protocol,
        "provenance": provenance(ROOT),
        "domains": {name: len(grid) for name, grid in grids.items()},
        "domain_characterization": domain_characterization,
        "mutations": mutation_rows,
        "outcome_counts": {
            "selected_mutations": len(mutation_rows),
            "observable_mutations": len(observable),
            "observable_detected": sum(row["detections"]["either"] > 0 for row in observable),
            "projection_rejected_mutations": sum(
                row["detections"]["projection"] > 0 for row in observable
            ),
            "baseline_projection_rejections": domain_characterization.get(
                "async", {}
            )
            .get("baseline_projection", {})
            .get("rejections", 0),
            "execution_errors": sum(len(row["execution_errors"]) for row in mutation_rows),
            "cases": sum(row["cases"] for row in mutation_rows),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    counts = result["outcome_counts"]
    # A survivor is a valid scientific outcome, not a harness failure.  The
    # runner succeeds only when every frozen case was executed and recorded;
    # coverage is reported separately so it cannot be silently converted into
    # an all-mutants-detected claim.
    result["expectations_met"] = (
        counts["execution_errors"] == 0
        and counts["baseline_projection_rejections"] == 0
    )
    result["detection_coverage"] = (
        counts["observable_detected"] / counts["observable_mutations"]
        if counts["observable_mutations"]
        else 0.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="write the frozen protocol")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=240924)
    generate.add_argument("--per-profile", type=int, default=2)
    evaluate = sub.add_parser("evaluate", help="execute an existing frozen protocol")
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    derive = sub.add_parser(
        "derive-domain",
        help="copy frozen mutations onto the live-accept domain",
    )
    derive.add_argument("--protocol", type=Path, required=True)
    derive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "derive-domain":
        frozen = json.loads(args.protocol.read_bytes())
        protocol = exercising_protocol(frozen)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(protocol))
        print(json.dumps({
            "protocol": str(args.output),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "mutations": [item["name"] for item in protocol["mutations"]],
            "async_domain": protocol["async_domain"],
        }))
        return 0
    if args.command == "generate":
        protocol = generate_protocol(seed=args.seed, per_profile=args.per_profile)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(protocol))
        print(json.dumps({"protocol": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "mutations": [m["name"] for m in protocol["mutations"]]}))
        return 0
    result = evaluate_protocol(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"expectations_met": result["expectations_met"], "outcome_counts": result["outcome_counts"]}))
    return 0 if result["expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
