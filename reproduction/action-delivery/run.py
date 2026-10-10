"""One-command reproduction of the deterministic action-delivery study core."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / f"{ROOT.name}-reproduction-output"
DEFAULT_CACHE = ROOT.parent / f"{ROOT.name}-upstream-cache"


def resolved_path(path: Path) -> Path:
    """Return an absolute path without requiring the final target to exist."""
    return path.expanduser().resolve()


def is_within(path: Path, directory: Path) -> bool:
    path = resolved_path(path)
    directory = resolved_path(directory)
    return path == directory or path.is_relative_to(directory)


def run(command: list[str], *, env: dict[str, str]) -> dict:
    started = perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": perf_counter() - started,
    }


def delivery_test_methods() -> int:
    count = 0
    for path in (ROOT / "integrations/lerobot/tests").glob("test_delivery*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
    return count


def extension_summary(result: dict) -> dict:
    observable = [
        mutation
        for mutation in result["mutations"]
        if mutation["classification"] == "observable"
    ]
    equivalent = [
        mutation
        for mutation in result["mutations"]
        if mutation["classification"] == "delivery_equivalent"
    ]
    scaling_verdicts = [
        sample["result"]["verdict"]
        for row in result["scaling"]
        for sample in row["samples"]
    ]
    return {
        "extension_status_completed": result.get("status") == "completed",
        "extension_expectations_met": result.get("expectations_met") is True,
        "extension_valid_cases": result["valid_cases"],
        "extension_valid_failures": result["valid_failures"],
        "extension_projected_cases": result["projected_cases"],
        "extension_observable_mutations": len(observable),
        "extension_observable_detected": sum(
            mutation["detections"]["either"] > 0 for mutation in observable
        ),
        "extension_equivalent_controls": len(equivalent),
        "extension_equivalent_observed": sum(
            mutation["detections"]["either"] > 0 for mutation in equivalent
        ),
        "extension_fault_controls_identical": sum(
            control["identical"]
            for control in result["equivalence_fault_controls"]
        ),
        "extension_search_trials": len(result["searches"]),
        "extension_scaling_pass": scaling_verdicts.count("pass"),
        "extension_scaling_unknown": scaling_verdicts.count("unknown"),
        "extension_scaling_fail": scaling_verdicts.count("fail"),
    }


def corruption_summary(result: dict) -> dict:
    return {
        "corruption_status_completed": result.get("status") == "completed",
        "corruption_expectations_met": result.get("expectations_met") is True,
        "corruption_base_cases": result["base_cases"]["total"],
        "corruption_classes": result["summary"]["classes"],
        "corruption_generated": result["generated_corruptions"],
        "corruption_mismatches": result["summary"]["mismatches"],
        "corruption_unexpected_passes": result["summary"]["unexpected_passes"],
    }


def heldout_summary(result: dict) -> dict:
    counts = result["outcome_counts"]
    return {
        "heldout_status_completed": result.get("status") == "completed",
        "heldout_expectations_met": result.get("expectations_met") is True,
        "heldout_cases": counts["cases"],
        "heldout_observable_mutations": counts["observable_mutations"],
        "heldout_observable_detected": counts["observable_detected"],
        "heldout_survivors": (
            counts["observable_mutations"] - counts["observable_detected"]
        ),
        "heldout_execution_errors": counts["execution_errors"],
        "heldout_projection_rejected_mutations": counts.get(
            "projection_rejected_mutations", 0
        ),
        "heldout_baseline_projection_rejections": counts.get(
            "baseline_projection_rejections", 0
        ),
        "heldout_async_distinct_executions": result.get(
            "domain_characterization", {}
        )
        .get("async", {})
        .get("distinct_observed_executions"),
        "heldout_async_live_accept_cases": result.get(
            "domain_characterization", {}
        )
        .get("async", {})
        .get("live_accept_cases"),
    }


def baseline_summary(result: dict) -> dict:
    comparison = result["comparison"]
    return {
        "baseline_expectations_met": result.get("expectations_met") is True,
        "baseline_python_states": comparison["python_valid_states"],
        "baseline_tla_states": comparison["tla_valid_distinct_states"],
        "baseline_both_detect": comparison["both_detect_offset_mutation"],
    }


def observation_summary(result: dict) -> dict:
    summary = result["summary"]
    return {
        "observation_expectations_met": result.get("expectations_met") is True,
        "observation_mutations": summary["observable_mutations"],
        "observation_detected": summary["detected_mutations"],
        "observation_survivors": summary["survivors"],
        "observation_projection_rejected_survivors": len(
            summary.get("offline_projection_rejected_survivors", [])
        ),
        "observation_all_covered": summary["all_observable_covered"],
        "observation_minimal_channels": summary[
            "minimal_channel_sets_for_detectable_mutations"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--upstream", action="store_true")
    group.add_argument("--fetch-upstream", action="store_true")
    parser.add_argument(
        "--extension",
        action="store_true",
        help="also rerun and semantically check the protocol-fixed extension",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.cache = resolved_path(args.cache)
    args.output = resolved_path(args.output)
    manifest = ROOT / "artifact-manifest.json"
    if manifest.is_file():
        unsafe = [
            (name, path)
            for name, path in (("--output", args.output), ("--cache", args.cache))
            if is_within(path, ROOT)
        ]
        if unsafe:
            details = ", ".join(f"{name}={path}" for name, path in unsafe)
            parser.error(
                "strict archive outputs and caches must be outside the extracted "
                f"archive: {details}"
            )

    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    integration = str(ROOT / "integrations/lerobot/src")
    env["PYTHONPATH"] = (
        integration
        + os.pathsep
        + str(ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    records = []
    if manifest.is_file():
        verify_command = [
            sys.executable,
            "-m",
            "tools.delivery_check.artifact",
            "verify",
            "--root",
            str(ROOT),
        ]
        verify_record = run(verify_command, env=env)
        records.append(verify_record)
        if verify_record["returncode"] != 0:
            return 1

    args.output.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            sys.executable,
            "-m",
            "tools.verification.verify_delivery",
            "smoke",
            "--output",
            str(args.output / "smoke.json"),
        ],
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "integrations/lerobot/tests",
            "-p",
            "test_delivery*.py",
            "-v",
        ],
    ]
    evaluate = [
        sys.executable,
        "-m",
        "tools.verification.verify_delivery",
        "evaluate",
        "--output",
        str(args.output / "development.json"),
        "--cache",
        str(args.cache),
    ]
    if args.fetch_upstream:
        evaluate.append("--fetch-upstream")
    elif args.upstream:
        evaluate.append("--upstream")
    commands.append(evaluate)
    commands.extend(
        [
            [
                sys.executable,
                "-m",
                "tools.delivery_check.heldout",
                "evaluate",
                "--protocol",
                str(ROOT / "papers/action-delivery/heldout-protocol.json"),
                "--output",
                str(args.output / "heldout.json"),
            ],
            [
                sys.executable,
                "-m",
                "tools.delivery_check.baseline",
                "--output",
                str(args.output / "general-tool-baseline.json"),
            ],
        ]
    )
    if args.extension:
        commands.append(
            [
                sys.executable,
                "-m",
                "tools.delivery_check.extension",
                "--output",
                str(args.output / "extension.json"),
            ]
        )
        commands.append(
            [
                sys.executable,
                "-m",
                "tools.delivery_check.corruptions",
                "--output",
                str(args.output / "monitor-corruptions.json"),
            ]
        )
    extension_input = (
        args.output / "extension.json"
        if args.extension
        else ROOT / "papers/action-delivery/results/extension-linux.json"
    )
    commands.append(
        [
            sys.executable,
            "-m",
            "tools.delivery_check.observation",
            "--development",
            str(args.output / "development.json"),
            "--extension",
            str(extension_input),
            "--heldout",
            str(args.output / "heldout.json"),
            "--output",
            str(args.output / "observation-adequacy.json"),
        ]
    )

    for command in commands:
        record = run(command, env=env)
        records.append(record)
        if record["returncode"] != 0:
            report = {"schema": 1, "verdict": "fail", "commands": records}
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            return 1

    result = json.loads((args.output / "development.json").read_text(encoding="utf-8"))
    smoke = json.loads((args.output / "smoke.json").read_text(encoding="utf-8"))
    observed = {
        key: result["summary"][key]
        for key in (
            "sync_cases",
            "async_cases",
            "sync_distinct_observed_executions",
            "async_distinct_observed_executions",
            "async_live_accept_cases",
            "routing_failures",
            "mutants_detected",
            "source_audit_failures",
            "erasure_state_operation_pairs",
            "monitor_negative_controls_detected",
        )
    }
    observed.update(
        {
            "delivery_test_methods": delivery_test_methods(),
            "smoke_cases": len(smoke["cases"]),
            "smoke_model_states": smoke["model"]["states"],
            "smoke_model_transitions": smoke["model"]["transitions"],
            "erasure_negative_controls": result["erasure"][
                "negative_controls_detected"
            ],
            "bounded_pass": result["verdict_counts"]["pass"],
            "bounded_fail": result["verdict_counts"]["fail"],
            "bounded_unknown_expected": result["verdict_counts"]["unknown_expected"],
            "bounded_unknown_unexpected": result["verdict_counts"][
                "unknown_unexpected"
            ],
        }
    )
    heldout = json.loads((args.output / "heldout.json").read_text(encoding="utf-8"))
    observed.update(heldout_summary(heldout))
    baseline = json.loads(
        (args.output / "general-tool-baseline.json").read_text(encoding="utf-8")
    )
    observed.update(baseline_summary(baseline))
    observation = json.loads(
        (args.output / "observation-adequacy.json").read_text(encoding="utf-8")
    )
    observed.update(observation_summary(observation))
    if args.extension:
        extension = json.loads(
            (args.output / "extension.json").read_text(encoding="utf-8")
        )
        observed.update(extension_summary(extension))
        corruptions = json.loads(
            (args.output / "monitor-corruptions.json").read_text(encoding="utf-8")
        )
        observed.update(corruption_summary(corruptions))

    wanted = {key: expected[key] for key in observed}
    failures = [] if observed == wanted else [{"expected": wanted, "observed": observed}]
    if args.upstream or args.fetch_upstream:
        upstream = {
            "upstream_parameter_cases": len(result["lerobot"]["cases"]),
            "upstream_histories": result["lerobot"]["histories"]["histories"],
        }
        expected_upstream = {key: expected[key] for key in upstream}
        if upstream != expected_upstream or result["lerobot"]["verdict"] != "pass":
            failures.append({"expected": expected_upstream, "observed": upstream})
    report = {
        "schema": 1,
        "scope": "deterministic_cpu_reproduction",
        "python": sys.version,
        "expected": wanted,
        "observed": observed,
        "upstream_requested": args.upstream or args.fetch_upstream,
        "extension_requested": args.extension,
        "heldout_included": True,
        "baseline_included": True,
        "baseline_tla_availability": baseline["general_tool"].get("availability"),
        "observation_adequacy_included": True,
        "commands": records,
        "failures": failures,
        "verdict": "pass" if not failures else "fail",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": report["verdict"], "observed": observed}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
