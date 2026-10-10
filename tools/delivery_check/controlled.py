"""Deterministic executions of the production runners with distinct tensor slots."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from flowedge_lerobot.async_rollout import run_async_rollout
from flowedge_lerobot.rollout import DeadlineMissed, run_rollout

from .monitor import Report, check_trace


def observed_execution_signature(case):
    """Canonical event/send signature for detecting duplicate controlled cases."""
    return json.dumps(
        {"events": case["events"], "attempts": case["attempts"]},
        sort_keys=True,
        separators=(",", ":"),
    )


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += round(seconds * 1e9)


class Robot:
    def __init__(self, fail_send=None, end_after=None):
        self.actions = []
        self.attempts = []
        self.stopped = False
        self.fail_send = fail_send
        self.end_after = end_after

    def reset(self):
        pass

    def observe(self):
        return np.array([len(self.actions)], dtype=np.float32)

    @property
    def done(self):
        return self.end_after is not None and len(self.actions) >= self.end_after

    def send_action(self, action):
        if self.stopped:
            raise AssertionError("send after stop")
        self.attempts.append(np.array(action, copy=True).tolist())
        if len(self.actions) == self.fail_send:
            raise OSError("controlled send failure")
        self.actions.append(np.array(action, copy=True).tolist())

    def stop(self):
        self.stopped = True


@dataclass(frozen=True)
class Completion:
    delay: int
    phase: str = "before_check"

    def __post_init__(self):
        if self.delay < 0 or self.phase not in ("before_check", "after_check"):
            raise ValueError("invalid completion schedule")


class ControlledFuture:
    def __init__(self, clock, schedule, function, args):
        self.clock, self.function, self.args = clock, function, args
        self.target = clock() + schedule.delay * 1_000_000
        self.phase = schedule.phase
        self.ready = False
        self.executed = False
        self.value = self.error = None
        self.was_cancelled = False

    def done(self):
        if (
            self.clock() > self.target
            or self.clock() == self.target
            and self.phase == "before_check"
        ):
            self.ready = True
        elif self.clock() == self.target:
            # Completion just after this check becomes visible at the next check.
            self.phase = "before_check"
            return False
        return self.ready

    def cancelled(self):
        return self.was_cancelled

    def result(self):
        if not self.executed:
            self.executed = True
            try:
                self.value = self.function(*self.args)
            except BaseException as error:  # noqa: BLE001 - mirror Future exception storage.
                self.error = error
        if self.error is not None:
            raise self.error
        return self.value


class ControlledExecutor:
    """Models one already-running request; shutdown drains it without delivering."""

    def __init__(self, clock, schedules):
        self.clock, self.schedules = clock, schedules
        self.futures = []
        self.shutdown_errors = []

    def submit(self, function, *args):
        index = len(self.futures)
        schedule = self.schedules[min(index, len(self.schedules) - 1)]
        future = ControlledFuture(self.clock, schedule, function, args)
        self.futures.append(future)
        return future

    def shutdown(self, **kwargs):
        for future in self.futures:
            if not future.executed:
                try:
                    future.result()
                except BaseException as error:  # noqa: BLE001 - mirror executor shutdown.
                    self.shutdown_errors.append(error)


def payload(origin):
    if origin["kind"] == "initial":
        return [0.0]
    if origin["kind"] == "fallback":
        return [-1.0]
    return [float(1000 * (origin["request"] + 1) + origin["slot"] + 1)]


def check_boundary(events, attempts):
    """Compare observed tensors with labels without asking the monitor for a label."""
    sends = [event for event in events if event["kind"] == "send_begin"]
    if len(sends) != len(attempts):
        return Report(
            "fail",
            "independent_send_count",
            expected=len(sends),
            observed=len(attempts),
        )
    for event, actual in zip(sends, attempts):
        if "origin" not in event:
            return Report("unknown", "missing_origin", event["seq"])
        expected = payload(event["origin"])
        if actual != expected:
            return Report(
                "fail", "tensor_origin_disagreement", event["seq"], expected, actual
            )
    return Report("pass", "distinct_tensor_boundary_agrees")


def execute_async(
    schedules=None,
    *,
    steps=6,
    horizon=4,
    fail_send=None,
    end_after=None,
    runner=run_async_rollout,
    invalid_request=None,
    trace=True,
):
    schedules = (Completion(1),) if schedules is None else schedules
    if not schedules or not 0 < horizon < 1000:
        raise ValueError("nonempty schedules and horizon below 1000 required")
    clock, robot, events = Clock(), Robot(fail_send, end_after), []
    executor = ControlledExecutor(clock, schedules)

    def predict(history, plan, request):
        if request == invalid_request:
            return np.full((horizon, 1), np.nan)
        return (1000 * (request + 1) + np.arange(horizon, dtype=np.float32) + 1)[
            :, None
        ]

    error = None
    try:
        runner(
            robot,
            predict,
            fallback=lambda _: np.array([-1.0]),
            steps=steps,
            period_ms=1,
            action_steps=horizon,
            clock=clock,
            wait=clock.wait,
            executor_factory=lambda **kwargs: executor,
            event_sink=events.append if trace else None,
            run_id="controlled-async",
        )
    except (OSError, ValueError) as caught:
        error = type(caught).__name__
    return {
        "events": events,
        "actions": robot.actions,
        "attempts": robot.attempts,
        "error": error,
        "stopped": robot.stopped,
        "monitor": check_trace(events).record(),
        "boundary": check_boundary(events, robot.attempts).record(),
    }


def execute_sync(
    durations=(0, 2_000_000),
    *,
    mode="hold",
    fail_send=None,
    runner=run_rollout,
    trace=True,
):
    clock, robot, events = Clock(), Robot(fail_send), []

    class Policy:
        metadata = SimpleNamespace(horizon=1)
        action_dim = 1

        def reset(self):
            self.request = 0

        def select_action_into(self, condition, noise, action, **kwargs):
            action.fill(1000 * (self.request + 1) + 1)
            clock.now += durations[self.request]
            self.request += 1

    error = None
    try:
        runner(
            Policy(),
            robot,
            lambda observation: observation,
            steps=len(durations),
            period_ms=1,
            on_miss=mode,
            clock=clock,
            event_sink=events.append if trace else None,
            run_id="controlled-sync",
        )
    except (OSError, DeadlineMissed, AssertionError) as caught:
        error = type(caught).__name__
    return {
        "events": events,
        "actions": robot.actions,
        "attempts": robot.attempts,
        "error": error,
        "stopped": robot.stopped,
        "monitor": check_trace(events).record(),
        "boundary": check_boundary(events, robot.attempts).record(),
    }
