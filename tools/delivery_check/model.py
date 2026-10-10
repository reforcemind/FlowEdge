"""Finite phase model plus an independently represented eligibility contract.

No Python threads, payloads, or wall-clock deadlines are modeled. Completion is
an explicit action before or after a readiness check. Bounds cut analysis, not
the runtime queue. The state key includes the contract state and search depth.
"""

from collections import deque
from dataclasses import asdict, dataclass, replace
from time import perf_counter


@dataclass(frozen=True)
class State:
    tick: int
    phase: str
    queue: tuple  # (request, base, slot)
    pending: tuple | None  # (request, base)
    ready: bool
    next_request: int


@dataclass(frozen=True)
class Contract:
    request: int = 0
    base: int = 0
    first_eligible: int = 0
    consumed: int = 0

    def selection(self, horizon, tick):
        slot = self.first_eligible + self.consumed
        return (self.request, self.base, slot) if slot < horizon else (-1, tick, 0)


def successors(
    state, contract, horizon, steps, mutation=None, refill_threshold=0.75
):
    """Yield action, successor pair, and an optional contract violation."""
    if state.phase == "stopped":
        if mutation == "send_after_stop":
            yield "send_after_stop", state, contract, "send_after_termination"
        return
    yield "stop", replace(state, phase="stopped"), contract, None
    if state.pending is not None and not state.ready:
        yield "complete", replace(state, ready=True), contract, None
    if state.phase == "before_check":
        new = replace(state, phase="deliver")
        expected = contract
        if state.pending is not None and state.ready:
            request, base = state.pending
            offset = state.tick - base
            if offset < horizon:
                actual_offset = (
                    max(0, offset - 1) if mutation == "offset_minus_one" else offset
                )
                new = replace(
                    new,
                    queue=tuple(
                        (request, base, j) for j in range(actual_offset, horizon)
                    ),
                )
                # Eligibility is derived from absolute slot times, not candidate offset.
                eligible = [j for j in range(horizon) if base + j >= state.tick]
                expected = Contract(request, base, min(eligible), 0)
            new = replace(new, pending=None, ready=False)
        yield "check", new, expected, None
    elif state.phase == "deliver":
        observed = state.queue[0] if state.queue else (-1, state.tick, 0)
        expected = contract.selection(horizon, state.tick)
        issue = (
            None
            if observed == expected
            else {
                "property": "origin_alignment",
                "expected": expected,
                "observed": observed,
            }
        )
        new = replace(state, queue=state.queue[1:], phase="after_send")
        advanced = replace(contract, consumed=contract.consumed + (observed[0] != -1))
        yield "send", new, advanced, issue
    elif state.phase == "after_send":
        new = state
        if state.pending is None and len(state.queue) <= int(
            horizon * refill_threshold
        ):
            new = replace(
                new,
                pending=(state.next_request, state.tick),
                next_request=state.next_request + 1,
            )
        new = replace(
            new,
            tick=state.tick + 1,
            phase="before_check" if state.tick + 1 < steps else "stopped",
        )
        yield "advance", new, contract, None


def explore(
    *,
    events=20,
    horizon=3,
    queue_bound=3,
    steps=4,
    max_states=100000,
    seconds=None,
    mutation=None,
):
    if min(events, horizon, queue_bound, steps, max_states) <= 0:
        raise ValueError("all bounds must be positive")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive")
    if mutation not in (None, "offset_minus_one", "send_after_stop"):
        raise ValueError("unknown model mutation")
    started = perf_counter()
    initial = State(
        0, "before_check", tuple((0, 0, j) for j in range(horizon)), None, False, 1
    )
    todo = deque([(initial, Contract(), ())])
    seen = {(initial, Contract(), 0)}
    transitions = depth_frontier = outside_queue = 0
    verdict, reason, witness = "pass", "bounded_product_exhausted", None
    if horizon > queue_bound:
        todo.clear()
        outside_queue = 1
        verdict, reason = "unknown", "initial_queue_exceeds_analysis_bound"
    while todo:
        if seconds is not None and perf_counter() - started >= seconds:
            verdict, reason = "unknown", "time_budget_exhausted"
            break
        state, contract, history = todo.popleft()
        if len(history) == events:
            depth_frontier += 1
            continue
        for action, new, expected, issue in successors(
            state, contract, horizon, steps, mutation
        ):
            transitions += 1
            path = (*history, action)
            if issue:
                verdict, reason = "fail", "contract_violation"
                witness = {"actions": path, "violation": issue, "state": asdict(state)}
                todo.clear()
                break
            if len(new.queue) > queue_bound:
                outside_queue += 1
                continue
            key = (new, expected, len(path))
            if key not in seen:
                if len(seen) >= max_states:
                    verdict, reason = "unknown", "state_budget_exhausted"
                    todo.clear()
                    break
                seen.add(key)
                todo.append((new, expected, path))
        if verdict != "pass":
            break
    if outside_queue and verdict == "pass":
        verdict, reason = "unknown", "queue_analysis_boundary"
    return {
        "scope": "bounded_model",
        "model": "async-phases-v1",
        "contract": "eligibility-v1",
        "verdict": verdict,
        "reason": reason,
        "bounds": {
            "events": events,
            "horizon": horizon,
            "queue": queue_bound,
            "steps": steps,
            "max_states": max_states,
            "seconds": seconds,
        },
        "states": len(seen),
        "transitions": transitions,
        "event_frontier": depth_frontier,
        "outside_queue": outside_queue,
        "seconds": perf_counter() - started,
        "witness": witness,
        "mutation": mutation,
    }


def replay(actions, *, horizon=3, steps=4, mutation=None):
    state = State(
        0, "before_check", tuple((0, 0, j) for j in range(horizon)), None, False, 1
    )
    contract = Contract()
    issue = None
    for action in actions:
        choices = [
            item
            for item in successors(state, contract, horizon, steps, mutation)
            if item[0] == action
        ]
        if len(choices) != 1:
            raise ValueError("infeasible model witness")
        _, state, contract, issue = choices[0]
        if issue:
            break
    return {"verdict": "fail" if issue else "pass", "violation": issue}
