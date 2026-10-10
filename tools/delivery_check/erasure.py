"""Finite executable sanity check for the paper's payload-erasure lemma."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

Pair = tuple[int, str]


@dataclass(frozen=True)
class Concrete:
    queue: tuple[Pair, ...] = ()
    selected: Pair | None = None
    held: Pair | None = None
    outcome: str = "idle"
    stopped: bool = False


@dataclass(frozen=True)
class Abstract:
    queue: tuple[str, ...] = ()
    selected: str | None = None
    held: str | None = None
    outcome: str = "idle"
    stopped: bool = False


def erase(state: Concrete) -> Abstract:
    def label(item):
        return None if item is None else item[1]

    return Abstract(
        tuple(item[1] for item in state.queue),
        label(state.selected),
        label(state.held),
        state.outcome,
        state.stopped,
    )


def erase_event(event: tuple) -> tuple:
    kind, *args = event
    if kind in ("append", "replace"):
        return (kind, tuple(item[1] for item in args[0]))
    if kind == "create":
        return (kind, args[1])
    return event


def concrete_step(state: Concrete, event: tuple):
    kind, *args = event
    if kind == "create":
        return replace(state, queue=state.queue + ((args[0], args[1]),)), None
    if kind == "copy":
        return replace(state, queue=tuple(state.queue)), None
    if kind == "slice":
        return replace(state, queue=state.queue[args[0] :]), None
    if kind == "append":
        return replace(state, queue=state.queue + tuple(args[0])), None
    if kind == "replace":
        return replace(state, queue=tuple(args[0]), selected=None), None
    if kind == "pop":
        if not state.queue or state.stopped:
            return None
        return replace(state, queue=state.queue[1:], selected=state.queue[0]), (
            "select",
            state.queue[0][1],
        )
    if kind == "send":
        if state.selected is None or state.stopped:
            return None
        success = args[0]
        return (
            replace(
                state,
                held=state.selected if success else state.held,
                outcome="send" if success else "fault",
            ),
            ("send", state.selected[1], success),
        )
    if kind == "hold":
        if state.held is None or state.stopped:
            return None
        return replace(state, outcome="send"), ("hold", state.held[1])
    if kind == "skip":
        if state.stopped:
            return None
        return replace(state, outcome="skip"), ("skip",)
    if kind == "stop":
        return replace(state, stopped=True, outcome="stop"), ("stop",)
    raise ValueError(kind)


def abstract_step(state: Abstract, event: tuple):
    kind, *args = event
    if kind == "create":
        return replace(state, queue=state.queue + (args[0],)), None
    if kind == "copy":
        return replace(state, queue=tuple(state.queue)), None
    if kind == "slice":
        return replace(state, queue=state.queue[args[0] :]), None
    if kind == "append":
        return replace(state, queue=state.queue + tuple(args[0])), None
    if kind == "replace":
        return replace(state, queue=tuple(args[0]), selected=None), None
    if kind == "pop":
        if not state.queue or state.stopped:
            return None
        return replace(state, queue=state.queue[1:], selected=state.queue[0]), (
            "select",
            state.queue[0],
        )
    if kind == "send":
        if state.selected is None or state.stopped:
            return None
        success = args[0]
        return (
            replace(
                state,
                held=state.selected if success else state.held,
                outcome="send" if success else "fault",
            ),
            ("send", state.selected, success),
        )
    if kind == "hold":
        if state.held is None or state.stopped:
            return None
        return replace(state, outcome="send"), ("hold", state.held)
    if kind == "skip":
        if state.stopped:
            return None
        return replace(state, outcome="skip"), ("skip",)
    if kind == "stop":
        return replace(state, stopped=True, outcome="stop"), ("stop",)
    raise ValueError(kind)


def pair_domain() -> tuple[Pair, ...]:
    return tuple(product((0, 1), ("a", "b")))


def operation_domain() -> tuple[tuple, ...]:
    pairs = pair_domain()
    sequences = tuple((pair,) for pair in pairs) + tuple(product(pairs, repeat=2))
    return (
        *(("create", payload, origin) for payload, origin in pairs),
        ("copy",),
        ("slice", 0),
        ("slice", 1),
        *(("append", sequence) for sequence in sequences),
        *(("replace", sequence) for sequence in sequences),
        ("pop",),
        ("send", True),
        ("send", False),
        ("hold",),
        ("skip",),
        ("stop",),
    )


def negative_controls() -> list[dict]:
    controls = []

    left = Concrete(selected=(0, "a"))
    right = Concrete(selected=(1, "a"))
    controls.append(
        {
            "name": "payload_dependent_branch",
            "detected": erase(left) == erase(right)
            and ("skip" if left.selected[0] == 0 else "send")
            != ("skip" if right.selected[0] == 0 else "send"),
        }
    )

    state = Concrete(queue=((0, "a"), (1, "b")))
    concrete = concrete_step(state, ("slice", 1))
    wrong_abstract = abstract_step(erase(state), ("slice", 0))
    controls.append(
        {
            "name": "slice_offset_zero",
            "detected": erase(concrete[0]) != wrong_abstract[0],
        }
    )

    event = ("append", ((0, "a"), (1, "b")))
    concrete = concrete_step(Concrete(), event)
    wrong_abstract = abstract_step(Abstract(), ("append", ("a",)))
    controls.append(
        {
            "name": "append_drops_tail",
            "detected": erase(concrete[0]) != wrong_abstract[0],
        }
    )

    state = Concrete(selected=(0, "a"), held=(1, "b"))
    concrete = concrete_step(state, ("send", False))
    abstract = abstract_step(erase(state), ("send", False))
    wrong_concrete = replace(concrete[0], held=state.selected)
    controls.append(
        {
            "name": "failed_send_commits_held",
            "detected": erase(wrong_concrete) != abstract[0],
        }
    )

    concrete = Concrete()
    abstract = abstract_step(erase(concrete), ("stop",))
    controls.append(
        {
            "name": "stop_is_ignored",
            "detected": erase(concrete) != abstract[0],
        }
    )
    return controls


def evaluate() -> dict:
    pairs = pair_domain()
    queues = ((),) + tuple((item,) for item in pairs) + tuple(product(pairs, repeat=2))
    optionals = (None,) + pairs
    states = [
        Concrete(queue, selected, held, outcome, stopped)
        for queue, selected, held, outcome, stopped in product(
            queues, optionals, optionals, ("idle", "send", "skip"), (False, True)
        )
    ]
    checked = enabled = 0
    witness = None
    for state, event in product(states, operation_domain()):
        concrete = concrete_step(state, event)
        abstract = abstract_step(erase(state), erase_event(event))
        checked += 1
        if (concrete is None) != (abstract is None):
            witness = {"state": repr(state), "event": repr(event), "reason": "guard"}
            break
        if concrete is None:
            continue
        enabled += 1
        if erase(concrete[0]) != abstract[0] or concrete[1] != abstract[1]:
            witness = {
                "state": repr(state),
                "event": repr(event),
                "concrete": repr(concrete),
                "abstract": repr(abstract),
            }
            break

    controls = negative_controls()
    return {
        "scope": "finite_erasure_sanity_check",
        "payloads": [0, 1],
        "origins": ["a", "b"],
        "states": len(states),
        "operations": len(operation_domain()),
        "state_operation_pairs_checked": checked,
        "enabled_transitions": enabled,
        "negative_controls": controls,
        "negative_controls_detected": sum(c["detected"] for c in controls),
        "witness": witness,
        "limitations": [
            "finite domains do not prove the general proposition",
            "the checker tests the declared operation semantics, not Python source equivalence",
            "payload-dependent branches are excluded and demonstrated by a negative control",
        ],
        "verdict": "pass"
        if witness is None and all(c["detected"] for c in controls)
        else "fail",
    }
