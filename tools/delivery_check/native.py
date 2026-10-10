"""Small real-inference routing integration; no robot or task-success claim."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.build_dir.resolve()))
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "integrations/lerobot/src")
    )
    from flowedge_lerobot.async_rollout import run_async_rollout
    from flowedge_lerobot.diffusion import FlowEdgeDiffusionPolicy
    from flowedge_lerobot.rollout import run_rollout

    from .controlled import Clock, Completion, ControlledExecutor
    from .monitor import check_trace

    policy = FlowEdgeDiffusionPolicy.from_checkpoint(str(args.checkpoint), threads=0)
    condition = np.zeros(policy.contract.condition_dim, dtype=np.float32)
    chunks = {}

    class Robot:
        def __init__(self):
            self.sent = []

        def reset(self):
            pass

        def observe(self):
            return condition.copy()

        def send_action(self, action):
            self.sent.append(action.copy())

        def stop(self):
            pass

    class ObservedPolicy:
        metadata = policy.metadata
        action_dim = policy.action_dim

        def reset(self):
            self.request = 0
            policy.reset()

        def select_action_into(self, condition, noise, action, **kwargs):
            policy.select_action_into(condition, noise, action, **kwargs)
            chunks[self.request] = action.copy()[None, :]
            self.request += 1

    events, robot = [], Robot()
    run_rollout(
        ObservedPolicy(),
        robot,
        lambda obs: obs,
        steps=2,
        diffusion_steps=1,
        event_sink=events.append,
        run_id="native-sync",
    )

    def boundary(events, sent):
        labels = [e["origin"] for e in events if e["kind"] == "send_begin"]
        if len(labels) != len(sent):
            return False
        return all(
            np.array_equal(value, chunks[label["request"]][label["slot"]])
            for label, value in zip(labels, sent)
        )

    rows = [
        {
            "profile": "sync",
            "trace": events,
            "monitor": check_trace(events).record(),
            "boundary": boundary(events, robot.sent),
            "actions": [a.tolist() for a in robot.sent],
        }
    ]
    chunks.clear()
    events, robot, clock = [], Robot(), Clock()
    executor = ControlledExecutor(clock, (Completion(1),))

    def predict(history, plan, request):
        rng = np.random.default_rng(request)
        noise = rng.standard_normal(
            (policy.metadata.horizon, policy.action_dim), dtype=np.float32
        )
        chunk = policy.predict_action_chunk(history[-1], noise, steps=1, seed=request)
        chunks[request] = chunk.copy()
        return chunk

    run_async_rollout(
        robot,
        predict,
        fallback=lambda _: np.zeros(policy.action_dim),
        steps=3,
        period_ms=1,
        action_steps=policy.action_steps,
        clock=clock,
        wait=clock.wait,
        executor_factory=lambda **kwargs: executor,
        event_sink=events.append,
        run_id="native-async",
    )
    rows.append(
        {
            "profile": "async",
            "trace": events,
            "monitor": check_trace(events).record(),
            "boundary": boundary(events, robot.sent),
            "actions": [a.tolist() for a in robot.sent],
        }
    )
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    result = {
        "scope": "real_inference_integration",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        "numerical_correctness": "covered separately by parity gates",
        "diffusion_steps": 1,
        "observation": "zero encoded condition; no environment rollout",
        "rows": rows,
    }
    result["verdict"] = (
        "pass"
        if all(row["monitor"]["verdict"] == "pass" and row["boundary"] for row in rows)
        else "fail"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": result["scope"], "verdict": result["verdict"]}))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
