"""Small, backend-neutral rollout loop for LeRobot robot adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import perf_counter_ns
from typing import Protocol

import numpy as np

from .delivery_trace import DeliveryTrace, Origin
from .diffusion import FlowEdgeDiffusionPolicy

ON_MISS = ("hold", "drop", "raise")


class DeadlineMissed(RuntimeError):
    """Raised when ``on_miss='raise'`` and a control period is exceeded."""


class RolloutRobot(Protocol):
    """Minimal robot surface needed by the deployment loop."""

    def reset(self) -> None: ...

    def observe(self) -> np.ndarray: ...

    def send_action(self, action: np.ndarray) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True)
class RolloutResult:
    """Bounded result returned after a rollout completes or fails."""

    steps: int
    stopped: bool
    elapsed_ms: float = 0.0
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    missed_deadlines: int = 0
    on_miss: str = "hold"


MAX_ROLLOUT_STEPS = 100_000


def run_rollout(
    policy: FlowEdgeDiffusionPolicy,
    robot: RolloutRobot,
    encode_condition: Callable[[np.ndarray], np.ndarray],
    *,
    steps: int,
    seed: int = 0,
    diffusion_steps: int = 10,
    scheduler: str = "ddim",
    period_ms: float | None = None,
    on_miss: str = "hold",
    sync=None,
    clock=None,
    event_sink=None,
    run_id=None,
) -> RolloutResult:
    """Run a bounded single-action loop suitable for SO-100/SO-101 integration shims.

    ``encode_condition`` is injected so LeRobot owns camera/state feature encoding and
    normalization. ``sync`` runs after sampling so CUDA miss counts include device work.
    ``stop`` is attempted after entering the protected robot lifecycle. Validation,
    policy reset, and buffer allocation precede that boundary. ``clock`` and
    ``event_sink`` support controlled routing tests; neither proves physical timing.

    Joint limits and emergency-stop stay in the robot adapter. On a missed
    ``period_ms`` the plugin emits the last sent action (``hold``), skips
    ``send_action`` (``drop``), or raises ``DeadlineMissed`` (``raise``).
    """

    if steps <= 0 or steps > MAX_ROLLOUT_STEPS:
        raise ValueError(f"steps must be from 1 through {MAX_ROLLOUT_STEPS}")
    if diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be positive")
    if period_ms is not None and (not isfinite(period_ms) or period_ms <= 0):
        raise ValueError("period_ms must be positive")
    if on_miss not in ON_MISS:
        raise ValueError("on_miss must be hold, drop, or raise")

    policy.reset()
    rng = np.random.default_rng(seed)
    noise = np.empty((policy.metadata.horizon, policy.action_dim), dtype=np.float32)
    action = np.empty(policy.action_dim, dtype=np.float32)
    held = np.zeros(policy.action_dim, dtype=np.float32)
    durations_ns = np.empty(steps, dtype=np.int64)
    clock = perf_counter_ns if clock is None else clock
    rollout_start = clock()
    trace = DeliveryTrace(event_sink, run_id)
    tracing = event_sink is not None
    held_origin = Origin("initial", -1, -1, 0) if tracing else None
    completed = 0
    missed = 0
    status = "ok"
    try:
        trace.emit(
            "run_begin",
            profile="sync",
            steps=steps,
            on_miss=on_miss,
            period_ms=period_ms,
            observation_boundary="select_action_into",
            observations=["origin", "step_outcome", "send", "stop"],
        )
        robot.reset()
        deadline_ns = None if period_ms is None else period_ms * 1e6
        for tick in range(steps):
            trace.emit("step_begin", tick)
            step_start = clock()
            condition = encode_condition(robot.observe())
            rng.standard_normal(noise.shape, dtype=np.float32, out=noise)
            trace.emit("submit", tick, request=tick, base=tick, length=1)
            policy.select_action_into(
                condition,
                noise,
                action,
                steps=diffusion_steps,
                scheduler=scheduler,
                seed=seed + completed,
            )
            if sync is not None:
                sync()
            elapsed_ns = clock() - step_start
            late = deadline_ns is not None and elapsed_ns > deadline_ns
            trace.emit("complete", tick, request=tick, elapsed_ns=elapsed_ns, late=late)
            origin = Origin("policy", tick, tick, 0) if tracing else None
            if late:
                missed += 1
                if on_miss == "raise":
                    trace.emit("terminate", tick, reason="deadline")
                    raise DeadlineMissed(
                        f"control period {period_ms} ms exceeded ({elapsed_ns / 1e6:.3f} ms)"
                    )
                if on_miss == "hold":
                    if tracing:
                        trace.send(robot, held, tick, held_origin)
                    else:
                        robot.send_action(held)
                else:
                    trace.emit("skip", tick, reason="deadline")
            else:
                if tracing:
                    trace.send(robot, action, tick, origin)
                else:
                    robot.send_action(action)
                np.copyto(held, action)
                held_origin = origin
                if tracing:
                    trace.origin("held_commit", tick, held_origin)
            trace.emit(
                "step_end",
                tick,
                outcome="skip" if late and on_miss == "drop" else "send",
            )
            durations_ns[completed] = elapsed_ns
            completed += 1
    except BaseException:
        status = "error"
        trace.emit("fault", completed)
        raise
    finally:
        try:
            trace.stop(robot)
        except BaseException:
            status = "error"
            raise
        finally:
            trace.emit("run_end", status=status, completed=completed)
    elapsed_ns = clock() - rollout_start
    if completed == 0:
        return RolloutResult(
            steps=0, stopped=True, elapsed_ms=elapsed_ns / 1e6, on_miss=on_miss
        )
    samples = durations_ns[:completed]
    p50, p99 = np.percentile(samples, (50, 99))
    return RolloutResult(
        steps=completed,
        stopped=True,
        elapsed_ms=elapsed_ns / 1e6,
        p50_ms=float(p50 / 1e6),
        p99_ms=float(p99 / 1e6),
        max_ms=float(samples.max() / 1e6),
        missed_deadlines=missed,
        on_miss=on_miss,
    )
