"""Run the matched Python/TLA+ action-delivery baseline.

This is a bounded comparison, not a benchmark of the two implementation
languages.  The Python explorer and the TLA+ specification use the same
phase model and bounds.  TLC is optional because the clean reproduction
archive intentionally excludes Java and the TLC jar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

from .model import explore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "papers/action-delivery/results/general-tool-baseline.json"
DEFAULT_JAR = ROOT / "build-delivery/tla2tools.jar"
DEFAULT_STORED_TLC = ROOT / "papers/action-delivery/results/tlc.json"
BOUNDARY = {
    "events": 20,
    "horizon": 3,
    "queue": 3,
    "steps": 4,
}


OBLIGATIONS = [
    {
        "id": "origin_alignment",
        "description": "Every observed send has the next eligible request, base, and slot.",
        "python": "contract_violation in model.explore",
        "tla": "Agreement invariant",
        "scope": "bounded valid product",
    },
    {
        "id": "termination_guard",
        "description": "A stopped execution has no ordinary send successor.",
        "python": "stopped phase has no normal successors",
        "tla": "Next excludes Send when phase = stopped",
        "scope": "transition guard, not an independent liveness proof",
    },
    {
        "id": "payload_erasure",
        "description": "Payload values are not part of this matched model.",
        "python": "not modeled",
        "tla": "not modeled",
        "scope": "covered separately by the erasure study",
    },
    {
        "id": "physical_safety",
        "description": "Actuator behavior and robot safety are outside both encodings.",
        "python": "not modeled",
        "tla": "not modeled",
        "scope": "outside this baseline",
    },
]

FEATURE_MATRIX = [
    {
        "obligation": "bounded asynchronous origin alignment",
        "specialized_phase_model": "encoded",
        "direct_tla": "encoded",
        "broader_artifact": "monitor plus replay",
        "matched_baseline": True,
    },
    {
        "obligation": "no ordinary send after model stop",
        "specialized_phase_model": "guarded by construction; mutant injects violation",
        "direct_tla": "guarded by Next; mutant config injects violation",
        "broader_artifact": "monitor plus source audit",
        "matched_baseline": True,
    },
    {
        "obligation": "synchronous deadline and miss policy",
        "specialized_phase_model": "not encoded",
        "direct_tla": "not encoded",
        "broader_artifact": "finite monitor cases",
        "matched_baseline": False,
    },
    {
        "obligation": "send failure and held-state commit",
        "specialized_phase_model": "not encoded",
        "direct_tla": "not encoded",
        "broader_artifact": "monitor plus tensor observer",
        "matched_baseline": False,
    },
    {
        "obligation": "physical stop effect",
        "specialized_phase_model": "not encoded",
        "direct_tla": "not encoded",
        "broader_artifact": "lifecycle observer",
        "matched_baseline": False,
    },
    {
        "obligation": "three-valued incomplete-trace semantics",
        "specialized_phase_model": "not encoded",
        "direct_tla": "not encoded",
        "broader_artifact": "finite trace monitor",
        "matched_baseline": False,
    },
    {
        "obligation": "pinned LeRobot queue API",
        "specialized_phase_model": "not encoded",
        "direct_tla": "not encoded",
        "broader_artifact": "component adapter tests",
        "matched_baseline": False,
    },
    {
        "obligation": "payload-erasure conditions",
        "specialized_phase_model": "not encoded",
        "direct_tla": "not encoded",
        "broader_artifact": "separate finite semantic check",
        "matched_baseline": False,
    },
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_metrics(paths: list[Path]) -> dict:
    """Count physical nonblank lines; keep the metric language-neutral."""

    rows = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "physical_lines": len(lines),
                "nonblank_lines": sum(bool(line.strip()) for line in lines),
                "sha256": _digest(path),
            }
        )
    return {
        "metric": "physical nonblank source lines",
        "note": (
            "Descriptive effort proxy only; it is not a cross-language productivity "
            "or quality measure. Configuration files are counted with the TLA+ model."
        ),
        "files": rows,
        "physical_lines": sum(row["physical_lines"] for row in rows),
        "nonblank_lines": sum(row["nonblank_lines"] for row in rows),
    }


def _python_row(mutation: str | None) -> tuple[dict, float]:
    started = perf_counter()
    result = explore(
        events=BOUNDARY["events"],
        horizon=BOUNDARY["horizon"],
        queue_bound=BOUNDARY["queue"],
        steps=BOUNDARY["steps"],
        mutation=mutation,
    )
    elapsed = perf_counter() - started
    return (
        {
            "verdict": result["verdict"],
            "reason": result["reason"],
            "states": result["states"],
            "transitions": result["transitions"],
            "seconds": elapsed,
            "mutation": mutation or "none",
        },
        elapsed,
    )


def _python_run() -> dict:
    started = perf_counter()
    valid, _ = _python_row(None)
    mutant, _ = _python_row("offset_minus_one")
    return {
        "tool": "FlowEdge bounded Python explorer",
        "source_files": ["tools/delivery_check/model.py"],
        "valid": valid,
        "mutant": mutant,
        "pair_seconds": perf_counter() - started,
    }


def _read_tlc_rows(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for row in result.get("rows", []):
        rows[row["config"]] = {
            "verdict": row["verdict"],
            "generated": row.get("generated"),
            "distinct": row.get("distinct"),
            "seconds": row.get("seconds"),
        }
    return {
        "valid": rows.get("Delivery"),
        "mutant": rows.get("Mutant"),
        "source_sha256": {
            key.replace("\\", "/"): value
            for key, value in result.get("source_sha256", {}).items()
        },
        "jar_sha256": result.get("jar_sha256"),
    }


def _tlc_run(jar: Path, repeat: int, stored: Path) -> dict:
    if not jar.is_file() or shutil.which("java") is None:
        if stored.is_file():
            parsed = _read_tlc_rows(stored)
            expected_sources = {
                path.relative_to(ROOT).as_posix(): _digest(path)
                for path in (
                    ROOT / "papers/action-delivery/tla/Delivery.tla",
                    ROOT / "papers/action-delivery/tla/Delivery.cfg",
                    ROOT / "papers/action-delivery/tla/Mutant.cfg",
                )
            }
            if parsed["source_sha256"] == expected_sources:
                valid_seconds = parsed["valid"].get("seconds") if parsed["valid"] else None
                mutant_seconds = parsed["mutant"].get("seconds") if parsed["mutant"] else None
                return {
                    "availability": "stored",
                    "runtime_source": "historical stored TLC result",
                    "reason": (
                        f"TLC unavailable (jar={jar.is_file()}, java={shutil.which('java') is not None})"
                    ),
                    "tool": "TLC v1.8.0",
                    "valid": parsed["valid"],
                    "mutant": parsed["mutant"],
                    "jar_sha256": parsed["jar_sha256"],
                    "pair_seconds": valid_seconds + mutant_seconds
                    if valid_seconds is not None and mutant_seconds is not None
                    else None,
                    "stored_result": stored.relative_to(ROOT).as_posix(),
                }
            return {
                "availability": "unavailable",
                "reason": "stored TLC source hashes do not match current specification",
                "tool": "TLC v1.8.0",
            }
        return {
            "availability": "unavailable",
            "reason": (
                f"TLC unavailable (jar={jar.is_file()}, java={shutil.which('java') is not None})"
            ),
            "tool": "TLC v1.8.0",
        }

    runs = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="flowedge-baseline-", dir=ROOT / "tmp") as temp:
        for index in range(repeat):
            output = Path(temp) / f"tlc-{index}.json"
            command = [
                sys.executable,
                "-m",
                "tools.delivery_check.tlc",
                "--jar",
                str(jar),
                "--output",
                str(output),
            ]
            started = perf_counter()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=180,
            )
            elapsed = perf_counter() - started
            if completed.returncode != 0 or not output.is_file():
                failures.append(
                    {
                        "repeat": index,
                        "returncode": completed.returncode,
                        "seconds": elapsed,
                        "output": completed.stdout[-4000:],
                    }
                )
                continue
            parsed = _read_tlc_rows(output)
            parsed["pair_seconds"] = elapsed
            parsed["repeat"] = index
            runs.append(parsed)
    if failures:
        return {
            "availability": "error",
            "tool": "TLC v1.8.0",
            "failures": failures,
            "runs": runs,
        }

    valid = runs[0]["valid"]
    mutant = runs[0]["mutant"]
    return {
        "availability": "available",
        "tool": "TLC v1.8.0",
        "valid": valid,
        "mutant": mutant,
        "pair_seconds": {
            "median": median(row["pair_seconds"] for row in runs),
            "min": min(row["pair_seconds"] for row in runs),
            "max": max(row["pair_seconds"] for row in runs),
            "repeats": repeat,
        },
        "runs": runs,
    }


def compare(python: dict, tla: dict) -> dict:
    valid_tla = tla.get("valid")
    mutant_tla = tla.get("mutant")
    valid_match = bool(
        valid_tla
        and python["valid"]["verdict"] == "pass"
        and valid_tla["verdict"] == "pass"
        and python["valid"]["states"] == valid_tla["distinct"]
    )
    mutant_detected = (
        python["mutant"]["verdict"] == "fail"
        and mutant_tla is not None
        and mutant_tla["verdict"] == "fail"
    )
    return {
        "valid_distinct_state_counts_equal": valid_match,
        "python_valid_states": python["valid"]["states"],
        "tla_valid_distinct_states": valid_tla.get("distinct") if valid_tla else None,
        "tla_valid_generated_states": valid_tla.get("generated") if valid_tla else None,
        "same_valid_verdict": bool(valid_tla and valid_tla["verdict"] == "pass"),
        "both_detect_offset_mutation": mutant_detected,
        "runtime_comparison": (
            "descriptive only: Python is in-process; TLC includes JVM startup, parsing, and both configs"
        ),
    }


def run(*, output: Path, jar: Path, repeat: int, skip_tlc: bool, stored: Path) -> dict:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    python_rows = [_python_run() for _ in range(repeat)]
    python = {
        "tool": python_rows[0]["tool"],
        "source_files": python_rows[0]["source_files"],
        "valid": python_rows[0]["valid"],
        "mutant": python_rows[0]["mutant"],
        "pair_seconds": {
            "median": median(row["pair_seconds"] for row in python_rows),
            "min": min(row["pair_seconds"] for row in python_rows),
            "max": max(row["pair_seconds"] for row in python_rows),
            "repeats": repeat,
        },
        "runs": python_rows,
    }
    tla = (
        {"availability": "skipped", "tool": "TLC v1.8.0"}
        if skip_tlc
        else _tlc_run(jar, repeat, stored)
    )
    tla_sources = [
        ROOT / "papers/action-delivery/tla/Delivery.tla",
        ROOT / "papers/action-delivery/tla/Delivery.cfg",
        ROOT / "papers/action-delivery/tla/Mutant.cfg",
    ]
    specialized_metrics = _source_metrics([ROOT / "tools/delivery_check/model.py"])
    general_metrics = _source_metrics(tla_sources)
    result = {
        "schema": 1,
        "scope": "matched_general_tool_baseline",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "boundary": BOUNDARY,
        },
        "obligations": OBLIGATIONS,
        "feature_matrix": FEATURE_MATRIX,
        "specialized": {
            **python,
            "source_metrics": specialized_metrics,
        },
        "general_tool": {
            **tla,
            "source_files": [path.relative_to(ROOT).as_posix() for path in tla_sources],
            "source_metrics": general_metrics,
            "source_sha256": {path.relative_to(ROOT).as_posix(): _digest(path) for path in tla_sources},
        },
        "effort_evidence": {
            "recorded_person_hours": None,
            "independent_authors": False,
            "proxy": "physical nonblank source lines",
            "specialized_nonblank_lines": specialized_metrics["nonblank_lines"],
            "direct_tla_nonblank_lines": general_metrics["nonblank_lines"],
            "witness_handling": {
                "specialized": "model action sequence replayed by model.replay",
                "direct_tla": (
                    "TLC invariant counterexample retained in the stored raw output; "
                    "no shared source-level replay adapter"
                ),
            },
            "conclusion": (
                "No specification-effort advantage is demonstrated. The direct TLA+ "
                "encoding is the smaller source artifact under this descriptive proxy, "
                "and both authorship and person-time are uncontrolled."
            ),
        },
    }
    result["comparison"] = compare(python, tla)
    result["expectations_met"] = (
        result["comparison"]["valid_distinct_state_counts_equal"]
        and result["comparison"]["both_detect_offset_mutation"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--stored-tlc", type=Path, default=DEFAULT_STORED_TLC)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--skip-tlc",
        action="store_true",
        help="record TLC as skipped instead of invoking the optional general tool",
    )
    args = parser.parse_args()
    result = run(
        output=args.output,
        jar=args.jar,
        repeat=args.repeat,
        skip_tlc=args.skip_tlc,
        stored=args.stored_tlc,
    )
    print(
        json.dumps(
            {
                "expectations_met": result["expectations_met"],
                "python_states": result["comparison"]["python_valid_states"],
                "tla_states": result["comparison"]["tla_valid_distinct_states"],
                "tla_availability": result["general_tool"].get("availability"),
            }
        )
    )
    if result["general_tool"].get("availability") == "error":
        return 1
    return 0 if result["expectations_met"] or args.skip_tlc else 1


if __name__ == "__main__":
    raise SystemExit(main())
