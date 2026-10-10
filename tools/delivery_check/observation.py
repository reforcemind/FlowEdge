"""Summarize bounded mutation sensitivity across three online checks.

The unit is a mutation, not an execution: a check rejects a mutation when at
least one declared case produces its failing verdict. Non-rejection can mean a
missing contract obligation even when the recorded evidence is sufficient. The
result does not estimate natural-defect recall or prove evidence completeness.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("monitor", "tensor", "lifecycle")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development_rows(result: dict) -> list[dict]:
    rows = []
    for mutation in result["mutations"]:
        witness = mutation.get("witness") or {}
        counts = {
            "monitor": int((witness.get("monitor") or {}).get("verdict") == "fail"),
            "tensor": int((witness.get("boundary") or {}).get("verdict") == "fail"),
            "lifecycle": int(witness.get("stopped") is False),
        }
        rows.append(_row("development", mutation["name"], counts))
    return rows


def _study_rows(round_name: str, result: dict) -> list[dict]:
    return [
        _row(round_name, mutation["name"], mutation["detections"])
        for mutation in result["mutations"]
        if mutation.get("classification") == "observable"
    ]


def _row(round_name: str, name: str, counts: dict) -> dict:
    normalized = {channel: int(counts.get(channel, 0)) for channel in CHANNELS}
    return {
        "id": f"{round_name}:{name}",
        "round": round_name,
        "name": name,
        "channel_exposing_cases": normalized,
        "detected_by": [
            channel for channel in CHANNELS if normalized[channel] > 0
        ],
        "offline_projection_rejections": int(counts.get("projection", 0)),
    }


def _combinations(rows: list[dict]) -> list[dict]:
    detectable = [row for row in rows if row["detected_by"]]
    combinations = []
    for size in range(1, len(CHANNELS) + 1):
        for selected in itertools.combinations(CHANNELS, size):
            selected_set = set(selected)
            detected = [
                row["id"]
                for row in rows
                if selected_set.intersection(row["detected_by"])
            ]
            missed = [row["id"] for row in rows if row["id"] not in detected]
            combinations.append(
                {
                    "channels": list(selected),
                    "detected_mutations": len(detected),
                    "missed": missed,
                    "covers_all_detectable": all(
                        selected_set.intersection(row["detected_by"])
                        for row in detectable
                    ),
                    "covers_all_observable": not missed,
                }
            )
    return combinations


def analyze(development: Path, extension: Path, heldout: Path) -> dict:
    paths = {
        "development": development,
        "extension": extension,
        "heldout": heldout,
    }
    inputs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    rows = [
        *_development_rows(inputs["development"]),
        *_study_rows("extension", inputs["extension"]),
        *_study_rows("heldout", inputs["heldout"]),
    ]
    combinations = _combinations(rows)
    covering = [row for row in combinations if row["covers_all_detectable"]]
    minimum_size = min((len(row["channels"]) for row in covering), default=None)
    minimal_sets = [
        row["channels"]
        for row in covering
        if len(row["channels"]) == minimum_size
    ]
    survivors = [row["id"] for row in rows if not row["detected_by"]]
    projected_survivors = [
        row["id"]
        for row in rows
        if not row["detected_by"] and row["offline_projection_rejections"] > 0
    ]
    per_channel = {
        channel: sum(channel in row["detected_by"] for row in rows)
        for channel in CHANNELS
    }
    summary = {
        "observable_mutations": len(rows),
        "detected_mutations": len(rows) - len(survivors),
        "survivors": len(survivors),
        "survivor_ids": survivors,
        "online_check_survivor_ids": survivors,
        "offline_projection_rejected_survivors": projected_survivors,
        "per_channel_detected_mutations": per_channel,
        "all_observable_covered": not survivors,
        "all_observable_rejected_by_online_checks": not survivors,
        "minimal_channel_sets_for_detectable_mutations": minimal_sets,
    }
    result = {
        "schema": 1,
        "scope": "bounded_mutation_observation_adequacy",
        "unit": (
            "mutation-level rejection by the online monitor, tensor, or lifecycle "
            "check in at least one declared execution; not evidence-channel "
            "sufficiency or per-execution completeness"
        ),
        "limitations": [
            "author-designed finite mutation families",
            "no estimate of natural-defect prevalence or recall",
            "a channel set is adequate only for the listed detectable mutations",
            "non-rejection may reflect a missing contract obligation rather than "
            "missing event evidence",
            "the offline phase-model projection is reported separately from the "
            "three online checks",
        ],
        "inputs": {
            name: {
                "path": path.as_posix(),
                "sha256": _digest(path),
                "scope": inputs[name].get("scope"),
            }
            for name, path in paths.items()
        },
        "mutations": rows,
        "channel_combinations": combinations,
        "summary": summary,
    }
    result["expectations_met"] = (
        len({row["id"] for row in rows}) == len(rows)
        and all(rows)
        and summary["detected_mutations"] + summary["survivors"]
        == summary["observable_mutations"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        type=Path,
        default=ROOT / "papers/action-delivery/results/development.json",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        default=ROOT / "papers/action-delivery/results/extension-linux.json",
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=ROOT / "papers/action-delivery/results/heldout.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.development, args.extension, args.heldout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"expectations_met": result["expectations_met"], **result["summary"]}))
    return 0 if result["expectations_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
