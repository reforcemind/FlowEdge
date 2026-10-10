"""Same-process observation-cost measurement; no real-time guarantee."""

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
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations/lerobot/src"))

from flowedge_lerobot.rollout import run_rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats <= 0:
        parser.error("positive steps and repeats required")
    original = subprocess.check_output(
        [
            "git",
            "show",
            "97f572064e3e785eca1663bfab709941d441e906:integrations/lerobot/src/flowedge_lerobot/rollout.py",
        ],
        cwd=ROOT,
        text=True,
    )
    namespace = dict(run_rollout.__globals__)
    exec(compile(original, "<pinned-original-rollout>", "exec"), namespace)  # noqa: S102 - trusted pinned repository source.
    baseline = namespace["run_rollout"]

    class Policy:
        metadata = SimpleNamespace(horizon=1)
        action_dim = 1

        def reset(self):
            pass

        def select_action_into(self, condition, noise, action, **kwargs):
            action.fill(1.0)

    class Robot:
        def reset(self):
            pass

        def observe(self):
            return np.zeros(1, dtype=np.float32)

        def send_action(self, action):
            pass

        def stop(self):
            pass

    rows = []
    variants = ("original", "disabled", "discard_sink", "list_sink")
    for repeat in range(args.repeats + 1):
        for variant in variants[repeat % 4 :] + variants[: repeat % 4]:
            events = []
            function = baseline if variant == "original" else run_rollout
            kwargs = {} if variant == "original" else {"event_sink": None}
            if variant in ("discard_sink", "list_sink"):
                kwargs["event_sink"] = (
                    (lambda event: None) if variant == "discard_sink" else events.append
                )
            gc.collect()
            started = perf_counter_ns()
            function(Policy(), Robot(), lambda value: value, steps=args.steps, **kwargs)
            elapsed = perf_counter_ns() - started
            if repeat:
                rows.append(
                    {
                        "variant": variant,
                        "repeat": repeat,
                        "ns_per_step": elapsed / args.steps,
                        "recorded_events": len(events),
                    }
                )
    result = {
        "scope": "observation_overhead",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "steps": args.steps,
        "repeats": args.repeats,
        "discarded_warmup_rounds": 1,
        "baseline_revision": "97f572064e3e785eca1663bfab709941d441e906",
        "runner_sha256": hashlib.sha256(
            inspect.getsource(run_rollout).encode()
        ).hexdigest(),
        "workload": "synthetic one-dimensional synchronous policy; no budget; GC enabled",
        "rows": rows,
        "median_ns_per_step": {
            name: statistics.median(
                r["ns_per_step"] for r in rows if r["variant"] == name
            )
            for name in variants
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["median_ns_per_step"]))


if __name__ == "__main__":
    main()
