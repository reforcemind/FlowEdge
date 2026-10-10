"""Entry point: python -m flowedge_dev verify delivery --help."""

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations/lerobot/src"))

from tools.delivery_check.controlled import Completion, execute_async, execute_sync
from tools.delivery_check.experiments import evaluate, mutated_runner
from tools.delivery_check.model import explore, replay
from tools.delivery_check.monitor import check_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("smoke", "trace", "explore", "evaluate", "replay"),
        default="smoke",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--events", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--queue", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-states", type=int, default=100000)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--mutation", choices=("offset_minus_one", "send_after_stop"))
    parser.add_argument("--upstream", action="store_true")
    parser.add_argument("--fetch-upstream", action="store_true")
    parser.add_argument("--cache", type=Path, default=ROOT / "build-delivery/upstream")
    args = parser.parse_args()
    if args.command in ("trace", "replay") and args.input is None:
        parser.error("--input is required for trace/replay")
    if args.command == "smoke":
        cases = [execute_sync(), execute_async()]
        model = explore()
        result = {
            "scope": "controlled_tests",
            "cases": cases,
            "model": model,
            "verdict": "pass"
            if all(
                c[k]["verdict"] == "pass"
                for c in cases
                for k in ("monitor", "boundary")
            )
            and model["verdict"] == "pass"
            else "fail",
        }
    elif args.command == "explore":
        result = explore(
            events=args.events,
            horizon=args.horizon,
            queue_bound=args.queue,
            steps=args.steps,
            max_states=args.max_states,
            seconds=args.seconds,
            mutation=args.mutation,
        )
    elif args.command == "trace":
        text = args.input.read_text(encoding="utf-8")
        events = (
            json.loads(text)
            if text.lstrip().startswith("[")
            else [json.loads(line) for line in text.splitlines() if line]
        )
        result = check_trace(events).record()
    elif args.command == "evaluate":
        result = evaluate(
            ROOT,
            upstream=args.upstream or args.fetch_upstream,
            fetch=args.fetch_upstream,
            cache=args.cache,
        )
        result["generated_utc"] = datetime.now(timezone.utc).isoformat()
        result["generation_command"] = [sys.executable, *sys.argv]
    else:
        witness = json.loads(args.input.read_text(encoding="utf-8"))
        if witness.get("scope") == "bounded_model":
            result = replay(
                witness["witness"]["actions"],
                horizon=witness["bounds"]["horizon"],
                steps=witness["bounds"]["steps"],
                mutation=witness.get("mutation"),
            )
        else:
            schedule = witness["schedule"]
            runner, _ = mutated_runner(witness["name"])
            if schedule["profile"] == "async":
                result = execute_async(
                    tuple(Completion(**c) for c in schedule["completions"]),
                    runner=runner,
                    steps=schedule["steps"],
                    horizon=schedule["horizon"],
                )
            else:
                result = execute_sync(
                    schedule["durations"], runner=runner, mode=schedule["mode"]
                )
            result["verdict"] = (
                "fail"
                if any(result[k]["verdict"] == "fail" for k in ("monitor", "boundary"))
                else "pass"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.command == "evaluate":
            for mutation in result["mutations"]:
                (args.output.parent / (mutation["name"] + ".json")).write_text(
                    json.dumps(mutation, indent=2) + "\n", encoding="utf-8"
                )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "scope",
                        "status",
                        "expectations_met",
                        "verdict",
                        "verdict_counts",
                        "summary",
                        "seconds",
                    )
                    if k in result
                }
            )
        )
    else:
        print(json.dumps(result, indent=2))
    if args.command == "evaluate":
        return 0 if result["expectations_met"] else 1
    return {"pass": 0, "fail": 1, "unknown": 2}.get(result.get("verdict"), 2)


if __name__ == "__main__":
    raise SystemExit(main())
