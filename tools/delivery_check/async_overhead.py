"""Same-host synthetic asynchronous observation cost, not inference latency."""

import argparse
import gc
import hashlib
import inspect
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/lerobot/src"))

from flowedge_lerobot.async_rollout import run_async_rollout

from .controlled import Clock, Completion, ControlledExecutor

BASE = "97f572064e3e785eca1663bfab709941d441e906"


def measure(steps=2000, repeats=7):
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASE}:integrations/lerobot/src/flowedge_lerobot/async_rollout.py",
        ],
        cwd=ROOT,
        text=True,
    )
    namespace = dict(run_async_rollout.__globals__)
    exec(compile(source, "<pinned-original-async>", "exec"), namespace)  # noqa: S102 - trusted repository revision.
    baseline = namespace["run_async_rollout"]

    class Robot:
        def reset(self):
            pass

        def observe(self):
            return np.zeros(1, dtype=np.float32)

        def send_action(self, action):
            pass

        def stop(self):
            pass

    variants = ("original", "disabled", "discard_sink", "list_sink")
    rows = []
    for repeat in range(repeats + 1):
        for variant in variants[repeat % 4 :] + variants[: repeat % 4]:
            clock, events = Clock(), []
            executor = ControlledExecutor(clock, (Completion(1),))

            def factory(executor=executor, **kwargs):
                return executor

            namespace["ThreadPoolExecutor"] = factory
            function = baseline if variant == "original" else run_async_rollout
            kwargs = {} if variant == "original" else {"executor_factory": factory}
            if variant in ("discard_sink", "list_sink"):
                kwargs["event_sink"] = (
                    (lambda event: None) if variant == "discard_sink" else events.append
                )
            gc.collect()
            started = perf_counter_ns()
            function(
                Robot(),
                lambda *args: np.ones((4, 1), dtype=np.float32),
                fallback=lambda _: np.zeros(1, dtype=np.float32),
                steps=steps,
                period_ms=1,
                action_steps=4,
                clock=clock,
                wait=clock.wait,
                **kwargs,
            )
            duration = perf_counter_ns() - started
            if repeat:
                rows.append(
                    {
                        "variant": variant,
                        "repeat": repeat,
                        "ns_per_step": duration / steps,
                        "recorded_events": len(events),
                    }
                )
    return {
        "scope": "synthetic_async_observation_cost",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "baseline_revision": BASE,
        "steps": steps,
        "repeats": repeats,
        "runner_sha256": hashlib.sha256(
            inspect.getsource(run_async_rollout).encode()
        ).hexdigest(),
        "discarded_warmup_rounds": 1,
        "gc_enabled": gc.isenabled(),
        "workload": "one-dimensional chunk of four; controlled executor; no physical waits or neural inference",
        "rows": rows,
        "median_ns_per_step": {
            v: statistics.median(r["ns_per_step"] for r in rows if r["variant"] == v)
            for v in variants
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.repeats) <= 0:
        parser.error("positive steps and repeats required")
    result = measure(args.steps, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["median_ns_per_step"]))


if __name__ == "__main__":
    main()
