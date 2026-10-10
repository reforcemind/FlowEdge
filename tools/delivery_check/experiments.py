"""Measured, reproducible experiments. Development mutants are not natural bugs."""

import copy
import hashlib
import importlib.metadata
import inspect
import platform
import random
import subprocess
from itertools import product
from pathlib import Path
from time import perf_counter

from .controlled import (
    Completion,
    execute_async,
    execute_sync,
    observed_execution_signature,
)
from .erasure import evaluate as evaluate_erasure
from .model import explore
from .monitor import check_trace
from .source_audit import audit_sources

MUTATIONS = {
    "async_slice_minus_one": (
        "async",
        "chunk[elapsed_actions:]",
        "chunk[max(0, elapsed_actions - 1):]",
    ),
    "async_wrong_request": (
        "async",
        'Origin("policy", pending_request, pending_tick, slot)',
        'Origin("policy", pending_request - 1, pending_tick, slot)',
    ),
    "sync_rejected_hold": (
        "sync",
        "trace.send(robot, held, tick, held_origin)",
        "trace.send(robot, action, tick, held_origin)",
    ),
    "sync_send_after_stop": (
        "sync",
        "trace.stop(robot)",
        "trace.stop(robot)\n            trace.send(robot, action, completed, origin)",
    ),
}


def mutated_runner(name):
    from flowedge_lerobot.async_rollout import run_async_rollout
    from flowedge_lerobot.rollout import run_rollout

    profile, old, new = MUTATIONS[name]
    function = run_async_rollout if profile == "async" else run_rollout
    source = inspect.getsource(function)
    if source.count(old) != 1:
        raise ValueError(f"mutation {name} no longer matches exactly one source site")
    changed = source.replace(old, new)
    namespace = dict(function.__globals__)
    exec(compile(changed, f"<mutation:{name}>", "exec"), namespace)  # noqa: S102 - fixed local mutations only.
    return namespace[function.__name__], {
        "name": name,
        "profile": profile,
        "original": old,
        "replacement": new,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def violation(case):
    return case["monitor"]["verdict"] == "fail" or case["boundary"]["verdict"] == "fail"


def mutation_cases():
    results = []
    for name in MUTATIONS:
        runner, description = mutated_runner(name)
        if description["profile"] == "async":
            case = execute_async((Completion(1),), runner=runner)
            schedule = {
                "profile": "async",
                "completions": [{"delay": 1, "phase": "before_check"}],
                "steps": 6,
                "horizon": 4,
            }
        else:
            case = execute_sync((0, 2_000_000), runner=runner)
            schedule = {"profile": "sync", "durations": [0, 2_000_000], "mode": "hold"}
        results.append(
            dict(
                **description, detected=violation(case), schedule=schedule, witness=case
            )
        )
    return results


def baseline_comparison(*, count=36, seconds=0.1, seeds=(0, 1, 2)):
    """Both methods invoke the SAME mutant, monitor, tensor observer and domain."""
    runner, _ = mutated_runner("async_slice_minus_one")
    domain = [
        Completion(d, p) for d, p in product(range(6), ("before_check", "after_check"))
    ]
    rows = []
    for regime, method, seed in product(
        ("count", "time"), ("systematic", "random"), seeds
    ):
        rng = random.Random(seed)
        started = perf_counter()
        executions = detections = 0
        unique = set()
        while (
            executions < count
            if regime == "count"
            else perf_counter() - started < seconds
        ):
            index = (
                executions % (len(domain) ** 2)
                if method == "systematic"
                else rng.randrange(len(domain) ** 2)
            )
            first, second = divmod(index, len(domain))
            schedules = (domain[first], domain[second])
            case = execute_async(schedules, runner=runner)
            detections += violation(case)
            unique.add(index)
            executions += 1
        rows.append(
            {
                "regime": regime,
                "method": method,
                "seed": seed,
                "executions": executions,
                "detecting_executions": detections,
                "unique_schedules": len(unique),
                "seconds": perf_counter() - started,
                "count_budget": count if regime == "count" else None,
                "time_budget": seconds if regime == "time" else None,
            }
        )
    return {
        "domain": {
            "delays": list(range(6)),
            "phases": ["before_check", "after_check"],
            "request_positions": 2,
            "repeat_last": True,
            "steps": 6,
            "horizon": 4,
        },
        "mutation": "async_slice_minus_one",
        "rows": rows,
        "interpretation": "development sensitivity experiment; not an unbiased method comparison",
    }


def provenance(root):
    paths = [
        *Path(root, "tools/delivery_check").glob("*.py"),
        *[
            Path(root, "integrations/lerobot/src/flowedge_lerobot", file)
            for file in ("rollout.py", "async_rollout.py", "delivery_trace.py")
        ],
    ]
    bundle_manifest = Path(root, "artifact-manifest.json")
    artifact_revision = None
    if bundle_manifest.is_file():
        import json

        manifest = json.loads(bundle_manifest.read_text(encoding="utf-8"))
        revision = manifest.get("flowedge_base_revision", manifest.get("base_revision"))
        if Path(root, ".git").exists():
            artifact_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=all"],
                    cwd=root,
                    text=True,
                ).strip()
            )
        else:
            artifact_revision = manifest.get("artifact_revision")
            dirty = manifest.get("working_tree_dirty")
    else:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root,
                text=True,
            ).strip()
        )
    sources = {
        str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted(paths)
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "flowedge_base_revision": revision,
        "artifact_revision": artifact_revision,
        "working_tree_dirty": dirty,
        "artifact_source_set_sha256": hashlib.sha256(
            repr(sorted(sources.items())).encode()
        ).hexdigest(),
        "dependencies": {name: _package_version(name) for name in ("numpy", "torch")},
        "sources": sources,
    }


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


EXPECTED_UNKNOWN_REASONS = {
    "state_budget_exhausted",
    "initial_queue_exceeds_analysis_bound",
    "insufficient_observation_coverage",
    "truncated_trace",
}


def account_verdicts(models, ablations):
    records = [*models, *ablations.values()]
    passed = sum(row["verdict"] == "pass" for row in records)
    failed = sum(row["verdict"] == "fail" for row in records)
    expected_unknown = sum(
        row["verdict"] == "unknown" and row.get("reason") in EXPECTED_UNKNOWN_REASONS
        for row in records
    )
    unexpected_unknown = sum(
        row["verdict"] == "unknown"
        and row.get("reason") not in EXPECTED_UNKNOWN_REASONS
        for row in records
    )
    return {
        "pass": passed,
        "fail": failed,
        "unknown_expected": expected_unknown,
        "unknown_unexpected": unexpected_unknown,
    }


def evaluate(root, *, upstream=False, fetch=False, cache="build-delivery/upstream"):
    from .lerobot_case import evaluate as evaluate_lerobot

    started = perf_counter()
    sync_rows, async_rows = [], []
    sync_signatures, async_signatures = set(), set()
    async_live_accept_cases = 0
    for mode, durations in product(
        ("hold", "drop", "raise"), product((0, 1_000_000, 1_000_001), repeat=3)
    ):
        case = execute_sync(durations, mode=mode)
        sync_signatures.add(observed_execution_signature(case))
        sync_rows.append(
            {
                "mode": mode,
                "durations": durations,
                "monitor": case["monitor"],
                "boundary": case["boundary"],
            }
        )
    domain = [
        Completion(d, p) for d, p in product(range(6), ("before_check", "after_check"))
    ]
    for horizon, first, second in product((1, 2, 4), domain, domain):
        case = execute_async((first, second), horizon=horizon, steps=8)
        async_signatures.add(observed_execution_signature(case))
        async_live_accept_cases += any(
            event["kind"] == "accept" and event["step"] != -1
            for event in case["events"]
        )
        async_rows.append(
            {
                "horizon": horizon,
                "completions": [vars(first), vars(second)],
                "monitor": case["monitor"],
                "boundary": case["boundary"],
            }
        )
    models = [
        explore(events=30, horizon=h, queue_bound=h, steps=n)
        for h, n in product((1, 2, 4), (2, 4, 6))
    ]
    models.extend([explore(max_states=1), explore(queue_bound=1)])
    trace = execute_async()["events"]
    no_origin = copy.deepcopy(trace)
    for event in no_origin:
        event.pop("origin", None)
    no_origin[0]["observations"].remove("origin")
    no_phase = copy.deepcopy(trace)
    no_phase[0]["observations"].remove("phase")
    delayed_trace = execute_async((Completion(5),), steps=6)["events"]
    omitted_completion = copy.deepcopy(delayed_trace)
    pending_false = next(
        event
        for event in omitted_completion
        if event["kind"] == "ready_check"
        and event["ready"] is False
        and event["request"] is not None
    )
    pending_false["ready"] = True
    omitted_pending_request = copy.deepcopy(delayed_trace)
    pending_false = next(
        event
        for event in omitted_pending_request
        if event["kind"] == "ready_check"
        and event["ready"] is False
        and event["request"] is not None
    )
    pending_false["request"] = None
    monitor_negative_controls = {
        "ready_without_completion": check_trace(omitted_completion).record(),
        "missing_pending_request": check_trace(omitted_pending_request).record(),
    }
    ablations = {
        "full": check_trace(trace).record(),
        "no_origin": check_trace(no_origin).record(),
        "no_phase": check_trace(no_phase).record(),
        "truncated": check_trace(trace[:-1]).record(),
    }
    upstream_result = (
        evaluate_lerobot(cache, fetch=fetch)
        if upstream
        else {"verdict": "unknown", "reason": "independent_component_not_requested"}
    )
    result = {
        "schema": 2,
        "scope": "development_evaluation",
        "status": "completed",
        "provenance": provenance(root),
        "sync": sync_rows,
        "async_cases": async_rows,
        "models": models,
        "mutations": mutation_cases(),
        "ablations": ablations,
        "monitor_negative_controls": monitor_negative_controls,
        "baselines": baseline_comparison(),
        "lerobot": upstream_result,
        "source_audit": audit_sources(root),
        "erasure": evaluate_erasure(),
    }
    result["seconds"] = perf_counter() - started
    result["summary"] = {
        "sync_cases": len(sync_rows),
        "async_cases": len(async_rows),
        "sync_distinct_observed_executions": len(sync_signatures),
        "async_distinct_observed_executions": len(async_signatures),
        "async_live_accept_cases": async_live_accept_cases,
        "routing_failures": sum(
            row[k]["verdict"] != "pass"
            for row in sync_rows + async_rows
            for k in ("monitor", "boundary")
        ),
        "mutants_detected": sum(m["detected"] for m in result["mutations"]),
        "source_audit_failures": len(result["source_audit"]["failures"]),
        "erasure_state_operation_pairs": result["erasure"][
            "state_operation_pairs_checked"
        ],
        "monitor_negative_controls_detected": sum(
            row["verdict"] == "fail"
            for row in result["monitor_negative_controls"].values()
        ),
        "upstream_parameter_cases": len(upstream_result.get("cases", []))
        if upstream
        else None,
        "upstream_histories": upstream_result.get("histories", {}).get("histories")
        if upstream
        else None,
    }
    result["verdict_counts"] = account_verdicts(models, ablations)
    result["expectations_met"] = (
        result["summary"]["routing_failures"] == 0
        and result["summary"]["mutants_detected"] == 4
        and result["source_audit"]["verdict"] == "pass"
        and result["erasure"]["verdict"] == "pass"
        and result["summary"]["monitor_negative_controls_detected"]
        == len(result["monitor_negative_controls"])
        and result["verdict_counts"]["fail"] == 0
        and result["verdict_counts"]["unknown_unexpected"] == 0
        and (not upstream or upstream_result["verdict"] == "pass")
    )
    return result
