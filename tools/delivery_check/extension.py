"""Execute the protocol-fixed follow-up and retain diagnostic survivors."""

import argparse
import gc
import hashlib
import inspect
import json
import random
import statistics
import sys
import tracemalloc
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/lerobot/src"))

from flowedge_lerobot.async_rollout import run_async_rollout
from flowedge_lerobot.rollout import run_rollout

from .controlled import Completion, execute_async, execute_sync
from .experiments import provenance
from .model import Contract, State, explore, successors


def compile_mutation(description):
    function = run_async_rollout if description["profile"] == "async" else run_rollout
    source = inspect.getsource(function)
    if source.count(description["old"]) != 1:
        raise ValueError(f"mutation site drift: {description['name']}")
    namespace = dict(function.__globals__)
    changed = source.replace(description["old"], description["new"])
    exec(compile(changed, "<frozen-local-mutation>", "exec"), namespace)  # noqa: S102 - trusted local protocol, never remote input.
    return namespace[function.__name__]


def domains(protocol):
    a, s = protocol["async_domain"], protocol["sync_domain"]
    completions = [
        {"delay": d, "phase": p} for d, p in product(a["delays"], a["phases"])
    ]
    return {
        "async": [
            {"profile": "async", "horizon": h, "steps": a["steps"], "completions": cs}
            for h, cs in product(
                a["horizons"], product(completions, repeat=a["positions"])
            )
        ],
        "sync": [
            {"profile": "sync", "durations": ds, "mode": m}
            for m, ds in product(
                s["modes"], product(s["durations"], repeat=s["positions"])
            )
        ],
    }


def execute(schedule, runner=None, **kwargs):
    if runner is not None:
        kwargs["runner"] = runner
    if schedule["profile"] == "sync":
        return execute_sync(schedule["durations"], mode=schedule["mode"], **kwargs)
    return execute_async(
        tuple(Completion(**c) for c in schedule["completions"]),
        steps=schedule["steps"],
        horizon=schedule["horizon"],
        **kwargs,
    )


def detections(case):
    return {
        "monitor": case["monitor"]["verdict"] == "fail",
        "tensor": case["boundary"]["verdict"] == "fail",
        "lifecycle": not case["stopped"],
    }


def project_trace(case, horizon, steps):
    """Project one fault-free async trace onto the finite phase model.

    The model hides worker completion until a true readiness check. Source-only
    records (submission, disposition, send return, and lifecycle events) are
    checked as correspondence obligations even when they do not advance the
    model. This remains a finite trace check, not a source-level proof.
    """
    state = State(
        0, "before_check", tuple((0, 0, j) for j in range(horizon)), None, False, 1
    )
    contract = Contract()
    projected = []
    refill_threshold = 0.75

    def take(action):
        nonlocal state, contract
        choices = [
            x
            for x in successors(
                state,
                contract,
                horizon,
                steps,
                refill_threshold=refill_threshold,
            )
            if x[0] == action
        ]
        if len(choices) != 1 or choices[0][3]:
            raise AssertionError(f"invalid projection: {action} at {state}")
        _, state, contract, _ = choices[0]
        projected.append(action)

    if case["error"] is not None:
        raise ValueError("projection supports only fault-free runs")
    events = case["events"]
    if not events:
        raise AssertionError("missing source trace")
    begin = events[0]
    if (
        begin.get("kind") != "run_begin"
        or begin.get("profile") != "async"
        or begin.get("horizon") != horizon
        or begin.get("steps") != steps
        or begin.get("fixed_plan") is not True
        or type(begin.get("refill_threshold")) not in (int, float)
        or not 0 <= begin["refill_threshold"] <= 1
    ):
        raise AssertionError("run configuration correspondence")
    refill_threshold = begin["refill_threshold"]

    bootstrap = 0
    requests = {}
    source_pending = None
    awaiting_completion = None
    awaiting_disposition = None
    current = None
    selected = None
    send_open = None
    send_complete = False
    completed = 0
    stop_started = stop_complete = ended = False

    def label(event):
        value = event.get("origin")
        if not isinstance(value, dict):
            raise AssertionError("missing source origin")
        return (
            value.get("kind"),
            value.get("request"),
            value.get("observation"),
            value.get("slot"),
        )

    def expected_origin():
        if state.queue:
            return ("policy", *state.queue[0])
        return ("fallback", -1, state.tick, 0)

    for event in events[1:]:
        kind = event["kind"]
        if ended:
            raise AssertionError("event after source run end")
        if kind == "submit":
            request = event.get("request")
            record = (event.get("base"), event.get("length"))
            if request in requests or record[1] != horizon:
                raise AssertionError("submission correspondence")
            requests[request] = record
            if event.get("step") == -1:
                if bootstrap != 0 or request != 0 or record != (0, horizon):
                    raise AssertionError("bootstrap submission correspondence")
                bootstrap = 1
            else:
                if (
                    bootstrap != 3
                    or source_pending is not None
                    or state.pending != (request, record[0])
                    or event.get("step") != record[0]
                ):
                    raise AssertionError("dispatch correspondence")
                source_pending = request
        elif kind == "complete":
            request = event.get("request")
            if event.get("step") == -1:
                if bootstrap != 1 or request != 0:
                    raise AssertionError("bootstrap completion correspondence")
                bootstrap = 2
            else:
                if request != awaiting_completion or request != source_pending:
                    raise AssertionError("completion correspondence")
                awaiting_completion = None
                awaiting_disposition = request
        elif kind in ("accept", "discard"):
            request = event.get("request")
            if event.get("step") == -1:
                if (
                    kind != "accept"
                    or bootstrap != 2
                    or request != 0
                    or event.get("offset") != 0
                    or event.get("length") != horizon
                ):
                    raise AssertionError("bootstrap disposition correspondence")
                bootstrap = 3
                continue
            if request != awaiting_disposition or request != source_pending:
                raise AssertionError("disposition request correspondence")
            base, length = requests[request]
            offset = state.tick - base
            expected_kind = "accept" if offset < horizon else "discard"
            if kind != expected_kind:
                raise AssertionError("disposition kind correspondence")
            if kind == "accept" and (
                event.get("offset") != offset
                or event.get("length") != length - offset
            ):
                raise AssertionError("disposition range correspondence")
            take("check")
            awaiting_disposition = None
            source_pending = None
        elif kind == "step_begin":
            if (
                bootstrap != 3
                or current is not None
                or event.get("step") != state.tick
                or state.phase != "before_check"
                or (state.pending is None) != (source_pending is None)
                or (
                    state.pending is not None
                    and state.pending[0] != source_pending
                )
            ):
                raise AssertionError("step boundary correspondence")
            current = state.tick
            selected = None
            send_complete = False
        elif kind == "ready_check":
            if current != state.tick or state.phase != "before_check":
                raise AssertionError("readiness boundary correspondence")
            expected_request = None if state.pending is None else state.pending[0]
            if event["request"] != expected_request:
                raise AssertionError("pending request correspondence")
            if event["ready"]:
                if source_pending != expected_request:
                    raise AssertionError("ready source request correspondence")
                take("complete")
                awaiting_completion = expected_request
            else:
                take("check")
        elif kind in ("dequeue", "fallback"):
            if state.phase != "deliver" or current != state.tick or selected is not None:
                raise AssertionError("selection boundary correspondence")
            expected = expected_origin()
            if kind != ("dequeue" if state.queue else "fallback"):
                raise AssertionError("selection kind correspondence")
            if label(event) != expected:
                raise AssertionError("selection origin correspondence")
            selected = expected
        elif kind == "send_begin":
            actual = label(event)
            expected = expected_origin()
            if selected != expected:
                raise AssertionError("missing source selection")
            if actual != expected:
                raise AssertionError(
                    f"source/model origin mismatch: {actual} != {expected}"
                )
            take("send")
            send_open = actual
        elif kind == "send_end":
            if (
                send_open is None
                or label(event) != send_open
                or event.get("status") != "ok"
            ):
                raise AssertionError("send return correspondence")
            send_open = None
            send_complete = True
        elif kind == "step_end":
            if (
                current != event.get("step")
                or send_open is not None
                or not send_complete
                or event.get("outcome") != "send"
            ):
                raise AssertionError("step outcome correspondence")
            take("advance")
            completed += 1
            current = None
        elif kind == "stop_begin":
            if stop_started or current is not None or send_open is not None:
                raise AssertionError("stop boundary correspondence")
            if state.phase != "stopped":
                take("stop")
            stop_started = True
        elif kind == "stop_end":
            if (
                not stop_started
                or stop_complete
                or event.get("status") != "ok"
            ):
                raise AssertionError("stop outcome correspondence")
            stop_complete = True
        elif kind == "drain":
            if (
                not stop_complete
                or source_pending is None
                or event.get("request") != source_pending
            ):
                raise AssertionError("drain correspondence")
            source_pending = None
        elif kind == "run_end":
            if (
                not stop_complete
                or state.phase != "stopped"
                or event.get("status") != "ok"
                or event.get("completed") != completed
                or awaiting_completion is not None
                or awaiting_disposition is not None
            ):
                raise AssertionError("terminal correspondence")
            ended = True
        else:
            raise AssertionError(f"unsupported source event: {kind}")
    if not ended or state.phase != "stopped":
        raise AssertionError("missing model termination")
    return len(projected)


def mutation_study(protocol, grids):
    rows, runners = [], {}
    for description in protocol["mutations"]:
        runner = runners[description["name"]] = compile_mutation(description)
        counters = {
            "monitor": 0,
            "tensor": 0,
            "lifecycle": 0,
            "either": 0,
            "unknown": 0,
        }
        witness = None
        grid = grids[description["profile"]]
        for index, schedule in enumerate(grid):
            case = execute(schedule, runner)
            found = detections(case)
            for key, value in found.items():
                counters[key] += value
            counters["either"] += any(found.values())
            counters["unknown"] += any(
                case[k]["verdict"] == "unknown" for k in ("monitor", "boundary")
            )
            if any(found.values()) and witness is None:
                witness = {"index": index, "schedule": schedule, "case": case}
        rows.append(
            {
                "name": description["name"],
                "classification": description["classification"],
                "cases": len(grid),
                "detections": counters,
                "witness": witness,
            }
        )
    # The delivery-equivalent mutation remains unobservable when send fails.
    equivalent = runners["sync_early_private_commit"]
    fault_controls = []
    for fail_send in (0, 1, 2):
        original = execute_sync((0, 0, 0), fail_send=fail_send)
        changed = execute_sync((0, 0, 0), fail_send=fail_send, runner=equivalent)
        fault_controls.append(
            {"fail_send": fail_send, "identical": original == changed}
        )
    return rows, fault_controls, runners


def searches(protocol, grids, runners):
    rows = []
    config = protocol["search"]
    for description in protocol["mutations"]:
        if description["classification"] != "observable":
            continue
        grid, runner = grids[description["profile"]], runners[description["name"]]
        for regime, seed in product(("count", "time"), config["seeds"]):
            # Rotate method order; exclude compilation and import from both budgets.
            methods = ("systematic", "random") if seed % 2 else ("random", "systematic")
            for method in methods:
                rng = random.Random(seed)
                # Seeded starting offset avoids always giving systematic the same prefix.
                offset = rng.randrange(len(grid))
                started = perf_counter()
                n = hits = 0
                unique = set()
                first = None
                while (
                    n < config["count"]
                    if regime == "count"
                    else perf_counter() - started < config["seconds"]
                ):
                    index = (
                        (offset + n) % len(grid)
                        if method == "systematic"
                        else rng.randrange(len(grid))
                    )
                    hit = any(detections(execute(grid[index], runner)).values())
                    n += 1
                    hits += hit
                    unique.add(index)
                    if hit and first is None:
                        first = n
                rows.append(
                    {
                        "mutation": description["name"],
                        "regime": regime,
                        "method": method,
                        "seed": seed,
                        "executions": n,
                        "hits": hits,
                        "first_detection": first,
                        "unique": len(unique),
                        "seconds": perf_counter() - started,
                    }
                )
    return rows


def scaling(protocol):
    config = protocol["scaling"]
    rows = []
    for horizon, steps in product(config["horizons"], config["steps"]):
        samples = []
        for _ in range(config["repeats"]):
            gc.collect()
            tracemalloc.start()
            result = explore(
                horizon=horizon,
                steps=steps,
                events=4 * steps + 2,
                queue_bound=horizon,
                max_states=config["max_states"],
                seconds=config["seconds"],
            )
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            samples.append({"result": result, "python_peak_bytes": peak})
        rows.append(
            {
                "horizon": horizon,
                "steps": steps,
                "samples": samples,
                "median_seconds": statistics.median(
                    s["result"]["seconds"] for s in samples
                ),
                "median_python_peak_bytes": statistics.median(
                    s["python_peak_bytes"] for s in samples
                ),
            }
        )
    return rows


def extension_accounting(result, *, expected_valid, expected_projected, expected_searches):
    observable = [
        mutation
        for mutation in result["mutations"]
        if mutation["classification"] == "observable"
    ]
    equivalent = [
        mutation
        for mutation in result["mutations"]
        if mutation["classification"] == "delivery_equivalent"
    ]
    scaling_verdicts = [
        sample["result"]["verdict"]
        for row in result["scaling"]
        for sample in row["samples"]
    ]
    counts = {
        "classified_mutations": len(observable) + len(equivalent),
        "observable_mutations": len(observable),
        "observable_detected": sum(
            mutation["detections"]["either"] > 0 for mutation in observable
        ),
        "equivalent_controls": len(equivalent),
        "equivalent_observed": sum(
            mutation["detections"]["either"] > 0 for mutation in equivalent
        ),
        "fault_controls": len(result["equivalence_fault_controls"]),
        "fault_controls_identical": sum(
            control["identical"]
            for control in result["equivalence_fault_controls"]
        ),
        "search_trials": len(result["searches"]),
        "scaling_pass": scaling_verdicts.count("pass"),
        "scaling_unknown": scaling_verdicts.count("unknown"),
        "scaling_fail": scaling_verdicts.count("fail"),
    }
    complete = (
        result["valid_cases"] == expected_valid
        and result["valid_failures"] == 0
        and result["projected_cases"] == expected_projected
        and counts["classified_mutations"] == len(result["mutations"])
        and counts["observable_mutations"] > 0
        and counts["observable_detected"] == counts["observable_mutations"]
        and counts["equivalent_controls"] > 0
        and counts["equivalent_observed"] == 0
        and counts["fault_controls"] == 3
        and counts["fault_controls_identical"] == counts["fault_controls"]
        and counts["search_trials"] == expected_searches
        and bool(scaling_verdicts)
        and counts["scaling_fail"] == 0
        and len(scaling_verdicts)
        == counts["scaling_pass"] + counts["scaling_unknown"]
    )
    return counts, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "papers/action-delivery/extension-protocol.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="diagnostics only, recorded in output",
    )
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    grids = domains(protocol)
    valid = [execute(s) for grid in grids.values() for s in grid]
    correspondence = [
        project_trace(execute(s), s["horizon"], s["steps"]) for s in grids["async"]
    ]
    print(
        f"Valid executions: {len(valid)}; projected traces: {len(correspondence)}",
        flush=True,
    )
    mutations, controls, runners = mutation_study(protocol, grids)
    print("Mutation enumeration complete", flush=True)
    result = {
        "schema": 2,
        "scope": "protocol_fixed_extension",
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "provenance": provenance(ROOT),
        "skipped_search": args.skip_search,
        "valid_cases": len(valid),
        "valid_failures": sum(
            not c["stopped"]
            or any(c[k]["verdict"] != "pass" for k in ("monitor", "boundary"))
            for c in valid
        ),
        "projected_cases": len(correspondence),
        "projected_transitions": sum(correspondence),
        "mutations": mutations,
        "equivalence_fault_controls": controls,
        "searches": [] if args.skip_search else searches(protocol, grids, runners),
    }
    print("Search comparison complete; measuring scaling", flush=True)
    result["scaling"] = scaling(protocol)
    observable_count = sum(
        mutation["classification"] == "observable"
        for mutation in protocol["mutations"]
    )
    expected_searches = (
        0
        if args.skip_search
        else observable_count * 2 * 2 * len(protocol["search"]["seeds"])
    )
    accounting, complete = extension_accounting(
        result,
        expected_valid=sum(len(grid) for grid in grids.values()),
        expected_projected=len(grids["async"]),
        expected_searches=expected_searches,
    )
    result["status"] = "diagnostic" if args.skip_search else "completed"
    result["expectations_met"] = None if args.skip_search else complete
    result["outcome_counts"] = accounting
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    result["generation_command"] = [sys.executable, *sys.argv]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "expectations_met": result["expectations_met"],
                "outcome_counts": result["outcome_counts"],
            }
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
