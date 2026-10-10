"""Paired origin-only/tensor queue microbenchmark under an identical event tape.

This isolates erasure's materialization/copy cost. It is not an inference or
end-to-end checker benchmark, and does not explore a larger state space.
"""

import argparse
import gc
import json
import platform
import statistics
from collections import deque
from itertools import product
from pathlib import Path
from time import perf_counter_ns

import numpy as np


def tape(horizon, requests=100):
    events = []
    for request in range(requests):
        delay = request % (horizon + 1)
        events.append(("replace", request, delay))
        events.extend(("send", request, i) for i in range(horizon))
    return events


def route(events, horizon, dim=None):
    queue, observed = deque(), []
    for operation, request, offset in events:
        if operation == "replace":
            if offset < horizon:
                queue.clear()
                if dim is None:
                    queue.extend((request, j) for j in range(offset, horizon))
                else:
                    # Materialize the full returned chunk before copying its suffix.
                    chunk = np.repeat(
                        (1000 * (request + 1) + np.arange(horizon, dtype=np.float32))[
                            :, None
                        ],
                        dim,
                        axis=1,
                    )
                    queue.extend(
                        ((request, j), chunk[j].copy()) for j in range(offset, horizon)
                    )
        elif queue:
            value = queue.popleft()
            observed.append(value if dim is None else value[0])
            if dim is not None:
                # Copying send consumer, the ownership assumption used by the study.
                value[1].copy()
        else:
            observed.append((-1, 0))
    return observed


def evaluate(repeats=7):
    rows = []
    for horizon, dim in product((4, 16, 64), (1, 8, 64, 512)):
        events = tape(horizon)
        expected = route(events, horizon)
        if route(events, horizon, dim) != expected:
            raise AssertionError("payload erasure changed observed origin sequence")
        samples = []
        for repeat in range(repeats + 1):
            order = ("origins", "tensors") if repeat % 2 else ("tensors", "origins")
            for method in order:
                gc.collect()
                started = perf_counter_ns()
                route(events, horizon, None if method == "origins" else dim)
                duration = perf_counter_ns() - started
                if repeat:
                    samples.append(
                        {
                            "repeat": repeat,
                            "method": method,
                            "ns_per_event": duration / len(events),
                        }
                    )
        rows.append(
            {
                "horizon": horizon,
                "action_dim": dim,
                "events": len(events),
                "origins_agree": True,
                "rows": samples,
                "median_ns_per_event": {
                    m: statistics.median(
                        s["ns_per_event"] for s in samples if s["method"] == m
                    )
                    for m in ("origins", "tensors")
                },
            }
        )
    return {
        "scope": "paired_queue_materialization_microbenchmark",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeats": repeats,
        "discarded_warmup_rounds": 1,
        "gc_enabled": gc.isenabled(),
        "rows": rows,
        "excludes": [
            "neural inference",
            "contract monitoring",
            "state exploration",
            "native device transfers",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paired_domains": len(result["rows"]), "origins_agree": True}))


if __name__ == "__main__":
    main()
