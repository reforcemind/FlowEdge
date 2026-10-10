"""Independent finite-trace contract monitor.

This module imports no runner or candidate-model code. A pass is conditional on
complete observation at the documented Python API boundary, not physical motion.
"""

from dataclasses import asdict, dataclass


@dataclass
class Report:
    verdict: str
    reason: str
    event: int | None = None
    expected: object = None
    observed: object = None
    scope: str = "finite_trace"
    contract: str = "delivery-v1"

    def record(self):
        return asdict(self)


class Violation(Exception):
    def __init__(self, reason, expected=None, observed=None):
        self.reason, self.expected, self.observed = reason, expected, observed


def origin(kind, request, observation, slot):
    return {"kind": kind, "request": request, "observation": observation, "slot": slot}


def require(condition, reason, expected=None, observed=None):
    if not condition:
        raise Violation(reason, expected, observed)


def check_trace(events):
    """Return pass/fail/unknown; a missing origin or integrity gap is unknown.

    A complete, uninterrupted violating prefix can fail even if its suffix is
    absent. Missing required observations in a completed step fail only under
    the declared observation-coverage assumption.
    """
    if not isinstance(events, (list, tuple)) or not events:
        return Report("unknown", "empty_trace")
    if any(not isinstance(event, dict) for event in events):
        return Report("unknown", "invalid_event_record")
    run = events[0].get("run")
    for i, event in enumerate(events):
        if event.get("schema") != 1 or not run or event.get("run") != run:
            return Report("unknown", "schema_or_run_identity", i)
        if type(event.get("seq")) is not int or event["seq"] != i:
            return Report("unknown", "sequence_gap_or_reordering", i)
    if events[0].get("kind") != "run_begin":
        return Report("unknown", "missing_run_begin", 0)
    config = events[0]
    profile = config.get("profile")
    if profile not in ("sync", "async"):
        return Report("unknown", "unsupported_profile", 0)
    if profile == "async" and config.get("fixed_plan") is not True:
        return Report("unknown", "adaptive_plan_outside_profile", 0)
    observations = config.get("observations")
    needed = {"origin", "step_outcome", "send", "stop"}
    if profile == "async":
        needed.add("phase")
    if (
        not isinstance(observations, list)
        or any(not isinstance(value, str) for value in observations)
        or not needed.issubset(observations)
    ):
        return Report("unknown", "insufficient_observation_coverage", 0)
    queue, requests, complete = [], {}, set()
    disposed = set()
    held = origin("initial", -1, -1, 0)
    current = None
    selected = None
    sending = None
    sent = None
    required = None
    step_count = 0
    stopped = terminated = ended = stop_ended = fault = False
    ready_request = None
    awaiting_disposition = None
    ready_seen = False
    fresh = False
    committed = False
    bootstrap = "required" if profile == "async" else "not_applicable"
    stop_status = None
    try:
        require(
            type(config["steps"]) is int and config["steps"] > 0, "invalid_step_bound"
        )
        if profile == "sync":
            require(config["on_miss"] in ("hold", "drop", "raise"), "invalid_fallback")
        else:
            require(
                type(config["horizon"]) is int and config["horizon"] > 0,
                "invalid_horizon",
            )
        for i, e in enumerate(events[1:], 1):
            kind, tick = e["kind"], e["step"]
            require(not ended, "event_after_run_end")
            require(
                not fault
                or kind in ("stop_begin", "stop_end", "drain", "run_end"),
                "event_after_fault",
            )
            if kind == "step_begin":
                require(
                    not stopped and not terminated and current is None,
                    "illegal_step_begin",
                )
                require(
                    profile != "async" or bootstrap == "done",
                    "missing_async_bootstrap",
                )
                require(tick == step_count and tick < config["steps"], "step_order")
                current = tick
                selected = sending = sent = required = None
                fresh = committed = False
                ready_request = None
                ready_seen = False
            elif kind == "submit":
                require(not stopped and not terminated, "submit_after_termination")
                request = e["request"]
                require(request not in requests, "duplicate_request")
                require(
                    type(e["length"]) is int and e["length"] > 0, "invalid_chunk_length"
                )
                if tick == -1:
                    require(
                        profile == "async"
                        and bootstrap == "required"
                        and current is None
                        and step_count == 0
                        and not requests
                        and request == 0
                        and e["base"] == 0,
                        "illegal_bootstrap_submit",
                    )
                    bootstrap = "submitted"
                else:
                    require(e["base"] == tick, "request_base")
                require(
                    e["length"] == (1 if profile == "sync" else config["horizon"]),
                    "request_length",
                )
                require(len(set(requests) - complete) == 0, "multiple_pending_requests")
                requests[request] = (e["base"], e["length"])
            elif kind == "ready_check":
                require(profile == "async" and current == tick, "readiness_boundary")
                require(not ready_seen, "duplicate_readiness_check")
                ready_seen = True
                if type(e["ready"]) is not bool:
                    return Report("unknown", "missing_readiness_phase", i)
                pending = sorted(set(requests) - complete)
                expected_pending = pending[0] if pending else None
                require(
                    e["request"] == expected_pending,
                    "pending_request_mismatch",
                    expected_pending,
                    e["request"],
                )
                require(
                    not e["ready"] or e["request"] is not None,
                    "ready_without_request",
                )
                ready_request = e["request"] if e["ready"] else None
                if e["request"] is not None:
                    require(
                        e["request"] in requests and e["request"] not in complete,
                        "unknown_pending_request",
                    )
            elif kind == "complete":
                request = e["request"]
                require(
                    request in requests and request not in complete,
                    "unexpected_completion",
                )
                complete.add(request)
                if profile == "async":
                    if tick == -1:
                        require(
                            bootstrap == "submitted"
                            and current is None
                            and step_count == 0
                            and request == 0,
                            "illegal_bootstrap_completion",
                        )
                        bootstrap = "completed"
                    else:
                        require(
                            ready_request == request,
                            "completion_without_ready_check",
                        )
                    ready_request = None
                    require(awaiting_disposition is None, "missing_chunk_disposition")
                    awaiting_disposition = request
                else:
                    require(
                        current == tick and requests[request][0] == tick,
                        "sync_completion_step",
                    )
                    period = config["period_ms"]
                    late = period is not None and e["elapsed_ns"] > period * 1e6
                    require(e["late"] == late, "deadline_classification")
                    fresh = not late
                    required = config["on_miss"] if late else "send"
                    selected = origin("policy", request, tick, 0) if fresh else held
            elif kind in ("accept", "discard"):
                require(profile == "async" and not stopped, "illegal_queue_update")
                request = e["request"]
                require(request in complete, "queue_update_before_completion")
                require(request not in disposed, "duplicate_chunk_disposition")
                require(request == awaiting_disposition, "unexpected_chunk_disposition")
                awaiting_disposition = None
                require(tick == -1 or tick == current, "queue_update_outside_step")
                disposed.add(request)
                base, length = requests[request]
                offset = 0 if tick == -1 else tick - base
                if tick == -1:
                    require(
                        bootstrap == "completed"
                        and current is None
                        and step_count == 0
                        and request == 0
                        and kind == "accept"
                        and e.get("offset") == 0
                        and e.get("length") == length,
                        "illegal_bootstrap_disposition",
                    )
                if kind == "discard":
                    require(offset >= length, "discarded_live_chunk")
                else:
                    require(0 <= offset < length, "accepted_expired_chunk")
                    require(
                        e["offset"] == offset and e["length"] == length - offset,
                        "chunk_alignment",
                        offset,
                        e["offset"],
                    )
                    # Contract is stated as eligibility of absolute slots, not a slice helper.
                    queue = [
                        origin("policy", request, base, slot)
                        for slot in range(length)
                        if base + slot >= base + offset
                    ]
                if tick == -1:
                    bootstrap = "done"
            elif kind in ("dequeue", "fallback"):
                require(ready_request is None, "missing_ready_completion")
                require(awaiting_disposition is None, "missing_chunk_disposition")
                require(ready_seen, "missing_readiness_check")
                require(
                    profile == "async" and current == tick and selected is None,
                    "duplicate_or_outside_step_selection",
                )
                if kind == "dequeue":
                    require(bool(queue), "dequeue_from_empty_queue")
                    selected = queue.pop(0)
                else:
                    require(not queue, "fallback_with_nonempty_queue")
                    selected = origin("fallback", -1, tick, 0)
                require(
                    e["origin"] == selected, "selected_origin", selected, e["origin"]
                )
                required = "send"
            elif kind == "send_begin":
                require(not stopped and not terminated, "send_after_termination")
                require(
                    current == tick and sending is None and sent is None,
                    "duplicate_or_outside_step_send",
                )
                require(
                    required in ("send", "hold") and selected is not None,
                    "unexpected_send",
                )
                require(e["origin"] == selected, "sent_origin", selected, e["origin"])
                sending = e["origin"]
            elif kind == "send_end":
                require(
                    current == tick and sending is not None, "send_end_without_begin"
                )
                require(e["origin"] == sending, "send_origin_changed")
                require(e["status"] in ("ok", "error"), "invalid_send_status")
                sent = e["status"]
                sending = None
            elif kind == "held_commit":
                require(
                    profile == "sync"
                    and current == tick
                    and fresh
                    and sent == "ok"
                    and not committed,
                    "held_commit_without_fresh_success",
                )
                require(e["origin"] == selected, "held_origin", selected, e["origin"])
                held = e["origin"]
                committed = True
            elif kind == "skip":
                require(
                    current == tick and required == "drop" and sent is None,
                    "unexpected_skip",
                )
                sent = "skip"
            elif kind == "step_end":
                require(current == tick and sending is None, "step_end_without_begin")
                expected = "skip" if required == "drop" else "send"
                require(required not in (None, "raise"), "missing_step_decision")
                require(
                    e["outcome"] == expected
                    and sent == ("skip" if expected == "skip" else "ok"),
                    "missing_or_wrong_step_outcome",
                    expected,
                    e["outcome"],
                )
                require(not fresh or committed, "missing_held_commit")
                step_count += 1
                current = None
            elif kind == "terminate":
                require(
                    current == tick and required == "raise", "unexpected_termination"
                )
                terminated = True
            elif kind == "fault":
                require(not fault and e.get("step") == step_count, "invalid_fault")
                if not (terminated or sent == "error"):
                    return Report("unknown", "unexplained_fault", i)
                fault = True
            elif kind == "stop_begin":
                require(
                    not stopped and sending is None, "duplicate_stop_or_unfinished_send"
                )
                stopped = True
            elif kind == "stop_end":
                require(stopped and not stop_ended, "stop_end_without_begin")
                require(e["status"] in ("ok", "error"), "invalid_stop_status")
                stop_ended = True
                stop_status = e["status"]
            elif kind == "drain":
                require(stopped and e["request"] in requests, "illegal_drain")
            elif kind == "run_end":
                require(stopped and stop_ended, "missing_stop_attempt")
                require(e["completed"] == step_count, "completed_step_count")
                require(e["status"] in ("ok", "error"), "invalid_run_status")
                if e["status"] == "ok":
                    require(
                        current is None and not fault and stop_status == "ok",
                        "unfinished_successful_run",
                    )
                    if profile == "sync":
                        require(step_count == config["steps"], "short_sync_run")
                elif not fault:
                    return Report("unknown", "error_without_fault_observation", i)
                ended = True
            else:
                return Report("unknown", "unsupported_event:" + kind, i)
    except Violation as error:
        return Report(
            "fail",
            error.reason,
            i if "i" in locals() else 0,
            error.expected,
            error.observed,
        )
    except (KeyError, TypeError, ValueError) as error:
        return Report(
            "unknown",
            "missing_or_invalid_observation:" + str(error),
            i if "i" in locals() else 0,
        )
    if not ended:
        return Report("unknown", "truncated_trace", len(events))
    return Report(
        "pass", "observed_contract_satisfied_assuming_complete_instrumentation"
    )
