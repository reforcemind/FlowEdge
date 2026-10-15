"""Closed-loop PushT evaluation; task success is distinct from policy replay parity."""

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .diffusion import FlowEdgeDiffusionPolicy
from .observation import DiffusionObservationEncoder, validate_source_pair


class PushTRobot:
    def __init__(self, *, seed, max_steps):
        import gym_pusht  # noqa: F401
        import gymnasium as gym

        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            max_episode_steps=max_steps,
        )
        self.seed = seed

    def reset(self):
        self.observation, self.info = self.env.reset(seed=self.seed)
        self.done = self.success = False
        self.max_reward = 0.0
        self.steps = 0

    def observe(self):
        return {
            "observation.state": torch.as_tensor(
                self.observation["agent_pos"], dtype=torch.float32
            ),
            "observation.image": torch.from_numpy(self.observation["pixels"].copy())
            .permute(2, 0, 1)
            .float()
            / 255,
        }

    def send_action(self, action):
        self.observation, reward, terminated, truncated, self.info = self.env.step(
            np.asarray(action)
        )
        self.steps += 1
        self.done = terminated or truncated
        self.success = bool(self.info.get("is_success", terminated))
        self.max_reward = max(self.max_reward, float(reward))

    def stop(self):
        self.env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--backend", choices=("flowedge", "lerobot"), default="flowedge"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--steps", type=int, default=10, help="complete DDIM integration steps"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--precision",
        choices=("exact-f32", "approx-fp16-deep"),
        default="exact-f32",
        help="explicit native diffusion precision; approximate mode requires CUDA",
    )
    parser.add_argument(
        "--period-ms", type=float, help="enable periodic asynchronous delivery"
    )
    parser.add_argument(
        "--plans", help="calibrated DeadlineFlow JSON portfolio; requires --period-ms"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.episodes, args.max_steps, args.steps, args.threads) <= 0:
        parser.error("counts must be positive")
    if args.plans and (args.period_ms is None or args.backend != "flowedge"):
        parser.error("--plans requires asynchronous FlowEdge execution")
    torch.set_num_threads(args.threads)
    validate_source_pair(args.checkpoint, args.source)
    encoder = DiffusionObservationEncoder(args.source)
    native = FlowEdgeDiffusionPolicy.from_checkpoint(
        args.checkpoint,
        threads=args.threads,
        device=args.device,
        precision=args.precision,
    )
    policy = native
    if args.backend == "lerobot":
        from .reference import TorchDiffusionReference

        policy = TorchDiffusionReference(args.source, native.metadata.condition_dim)
    reports = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        robot = PushTRobot(seed=episode_seed, max_steps=args.max_steps)
        encoder.reset()
        started = perf_counter()
        timing = None
        if args.period_ms is None:
            rng = np.random.default_rng(episode_seed)
            queue = deque()
            try:
                robot.reset()
                while not robot.done:
                    encoder.observe(robot.observe())
                    if not queue:
                        noise = rng.standard_normal(
                            (native.metadata.horizon, native.action_dim),
                            dtype=np.float32,
                        )
                        queue.extend(
                            policy.predict_action_chunk(
                                encoder.condition(), noise, steps=args.steps
                            )
                        )
                    robot.send_action(queue.popleft())
            finally:
                robot.stop()
        else:
            from .async_rollout import run_async_rollout
            from .deadline import CpuPlanSelector

            selector = CpuPlanSelector.from_json(args.plans) if args.plans else None
            plan = (
                next(iter(selector.plans.values()))
                if selector
                else {"id": 0, "steps": args.steps, "solver": "ddim"}
            )

            def predict(history, selected, request_seed):
                encoder.reset()
                for observation in history:
                    encoder.observe(observation)
                noise = np.random.default_rng(request_seed).standard_normal(
                    (native.metadata.horizon, native.action_dim), dtype=np.float32
                )
                return policy.predict_action_chunk(
                    encoder.condition(), noise, steps=selected["steps"]
                )

            timing = asdict(
                run_async_rollout(
                    robot,
                    predict,
                    fallback=lambda obs: obs["observation.state"].numpy(),
                    steps=args.max_steps,
                    period_ms=args.period_ms,
                    action_steps=native.action_steps,
                    observation_steps=native.metadata.observation_steps,
                    seed=episode_seed,
                    fixed_plan=plan,
                    selector=selector,
                )
            )
        reports.append(
            {
                "seed": episode_seed,
                "success": robot.success,
                "steps": robot.steps,
                "max_reward": robot.max_reward,
                "elapsed_s": perf_counter() - started,
                "timing": timing,
            }
        )
    from .benchmark import sha256

    result = {
        "schema_version": 1,
        "environment": "gym_pusht/PushT-v0",
        "backend": args.backend,
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_sha256": sha256(Path(args.source) / "model.safetensors"),
        "mode": "asynchronous" if args.period_ms else "synchronous_simulation",
        "inference_steps": args.steps if not args.plans else None,
        "plans_sha256": sha256(args.plans) if args.plans else None,
        "period_ms": args.period_ms,
        "threads": args.threads,
        "device": args.device,
        "precision": native.precision,
        "episodes": reports,
        "success_rate": sum(r["success"] for r in reports) / len(reports),
        "limitations": [
            "Simulation only; no robot-hardware validation.",
            "Small episode counts do not establish a reliable task-success estimate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"success_rate": result["success_rate"], "output": str(args.output)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
