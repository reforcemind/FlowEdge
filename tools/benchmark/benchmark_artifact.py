#!/usr/bin/env python3
"""Validate and render a portable LeRobot/FlowEdge benchmark artifact.

The artifact intentionally stores measurements rather than collecting them. This
keeps the runner-specific setup (LeRobot, robot processor, CPU affinity, and
RSS tooling) out of FlowEdge while making a published comparison auditable and
repeatable on the same host.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
QUANTILES = ("p50", "p95", "p99")
BACKENDS = ("flowedge", "lerobot")
STATUSES = ("measured", "not_measured", "unavailable", "blocked")


class ArtifactError(ValueError):
    """An artifact does not satisfy the published benchmark contract."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be an object")
    return value


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ArtifactError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or result < 0.0:
        raise ArtifactError(f"{name} must be finite and non-negative")
    return result


def _quantiles(value: Any, name: str) -> dict[str, float]:
    values = _mapping(value, name)
    result = {key: _number(values.get(key), f"{name}.{key}") for key in QUANTILES}
    if not result["p50"] <= result["p95"] <= result["p99"]:
        raise ArtifactError(f"{name} must satisfy p50 <= p95 <= p99")
    return result


def _measurement(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {"status": "not_measured", "reason": "no measurement supplied"}
    measurement = _mapping(value, name)
    status = _text(measurement.get("status"), f"{name}.status")
    if status not in STATUSES:
        raise ArtifactError(f"{name}.status must be one of {', '.join(STATUSES)}")
    if status != "measured":
        _text(measurement.get("reason"), f"{name}.reason")
        return {"status": status, "reason": measurement["reason"]}
    for field in ("startup_ms", "rss_mb", "throughput_hz"):
        if field == "rss_mb" and measurement.get(field) is None:
            _text(measurement.get("rss_reason"), f"{name}.rss_reason")
            continue
        _number(
            measurement.get(field), f"{name}.{field}", positive=field == "throughput_hz"
        )
    for field in ("encoder_ms", "policy_ms", "end_to_end_ms"):
        _quantiles(measurement.get(field), f"{name}.{field}")
    allocations = _mapping(measurement.get("allocations"), f"{name}.allocations")
    for field in ("setup", "hot_path"):
        value = allocations.get(field)
        if value is None:
            _text(allocations.get("reason"), f"{name}.allocations.reason")
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ArtifactError(
                f"{name}.allocations.{field} must be a non-negative integer"
            )
    return measurement


def validate(document: Any) -> dict[str, Any]:
    root = _mapping(document, "artifact")
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"schema_version must be {SCHEMA_VERSION}")
    _text(root.get("captured_at"), "captured_at")
    for section in ("model", "processor", "contract", "hardware", "commands"):
        _mapping(root.get(section), section)
    model = root["model"]
    for field in ("id", "revision", "sha256"):
        _text(model.get(field), f"model.{field}")
    processor = root["processor"]
    for field in ("id", "revision"):
        _text(processor.get(field), f"processor.{field}")
    _mapping(processor.get("stats"), "processor.stats")
    contract = root["contract"]
    for field in ("observation_schema_hash", "action_units", "normalization"):
        _text(contract.get(field), f"contract.{field}")
    for field in (
        "observation_steps",
        "action_steps",
        "action_dim",
        "batch_size",
        "inference_steps",
    ):
        value = contract.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ArtifactError(f"contract.{field} must be a positive integer")
    hardware = root["hardware"]
    for field in ("host", "os", "cpu", "compiler", "build_type"):
        _text(hardware.get(field), f"hardware.{field}")
    if not isinstance(hardware.get("threads"), int) or hardware["threads"] <= 0:
        raise ArtifactError("hardware.threads must be a positive integer")
    accelerator = hardware.get("accelerator")
    if accelerator is not None:
        _text(accelerator, "hardware.accelerator")
        if accelerator not in {"cpu", "cuda"}:
            raise ArtifactError("hardware.accelerator must be cpu or cuda")
    if hardware.get("cuda_device") is not None:
        _text(hardware.get("cuda_device"), "hardware.cuda_device")
        if accelerator != "cuda":
            raise ArtifactError("hardware.cuda_device requires accelerator=cuda")
    commands = root["commands"]
    for field in BACKENDS:
        _text(commands.get(field), f"commands.{field}")
    comparison = root.get("comparison")
    if comparison is not None:
        comparison = _mapping(comparison, "comparison")
        for field in ("candidate", "reference", "reference_framework"):
            _text(comparison.get(field), f"comparison.{field}")
    measurements = _mapping(root.get("measurements"), "measurements")
    for backend in BACKENDS:
        _measurement(measurements.get(backend), f"measurements.{backend}")
    limitations = root.get("limitations", [])
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) and item.strip() for item in limitations
    ):
        raise ArtifactError("limitations must be a list of non-empty strings")
    return root


def _display(value: Any, suffix: str = "") -> str:
    if value is None:
        return "not measured"
    return f"{value:.3f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"


def _metric(
    measurement: dict[str, Any],
    field: str,
    quantile: str | None = None,
    suffix: str = "",
) -> str:
    if measurement.get("status") != "measured":
        return "not measured"
    value = measurement[field] if quantile is None else measurement[field][quantile]
    return _display(value, suffix)


def _triplet(measurement: dict[str, Any], field: str) -> str:
    return " / ".join(
        _metric(measurement, field, quantile, " ms") for quantile in QUANTILES
    )


def _comparison(measurements: dict[str, Any]) -> list[str]:
    flowedge = measurements.get("flowedge") or {"status": "not_measured"}
    lerobot = measurements.get("lerobot") or {"status": "not_measured"}
    if flowedge.get("status") != "measured" or lerobot.get("status") != "measured":
        return []
    rows = [
        "",
        "## Relative comparison",
        "",
        "| Metric | FlowEdge / LeRobot |",
        "|---|---:|",
    ]
    for label, field in (
        ("Policy p50", "policy_ms"),
        ("End-to-end p50", "end_to_end_ms"),
    ):
        flow = flowedge[field]["p50"]
        ref = lerobot[field]["p50"]
        rows.append(
            f"| {label} | {flow / ref:.3f}x |" if ref else f"| {label} | unavailable |"
        )
    rows.append(
        f"| Throughput | {flowedge['throughput_hz'] / lerobot['throughput_hz']:.3f}x |"
    )
    return rows


def _cuda_host(hardware: dict[str, Any]) -> str:
    device = hardware.get("cuda_device")
    if not device:
        return ""
    return f"; CUDA `{device}`"


def render(document: dict[str, Any]) -> str:
    measurements = document["measurements"]
    model, processor = document["model"], document["processor"]
    contract, hardware, commands = (
        document["contract"],
        document["hardware"],
        document["commands"],
    )
    comparison = document.get("comparison", {})
    rows = [
        "# LeRobot / FlowEdge edge benchmark",
        "",
        "> These are measurements of one checkpoint on one declared host, not universal deployment guarantees.",
        "",
        "## Run identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Captured | `{document['captured_at']}` |",
        f"| Model | `{model['id']}` @ `{model['revision']}` |",
        f"| Model SHA-256 | `{model['sha256']}` |",
        f"| Processor | `{processor['id']}` @ `{processor['revision']}` |",
        f"| Processor stats | `{json.dumps(processor['stats'], sort_keys=True, separators=(',', ':'))}` |",
        f"| Observation contract | `{contract['observation_schema_hash']}`; {contract['observation_steps']} steps |",
        f"| Action contract | {contract['action_dim']} dims x {contract['action_steps']} steps; {contract['action_units']} |",
        f"| Run shape | batch {contract['batch_size']}; {contract['inference_steps']} inference steps |",
        f"| Host | `{hardware['host']}`; {hardware['os']}; {hardware['cpu']}; {hardware['threads']} threads{_cuda_host(hardware)} |",
        f"| Build | `{hardware['compiler']}`; {hardware['build_type']} |",
        f"| Comparison | {comparison.get('candidate', 'FlowEdge')} vs {comparison.get('reference', 'LeRobot reference')} |",
        f"| Reference framework | `{comparison.get('reference_framework', 'unspecified')}` |",
    ]
    if comparison.get("scope"):
        rows.append(f"| Scope | {comparison['scope']} |")
    if "reference_compile" in comparison:
        rows.append(f"| Reference compile mode | `{comparison['reference_compile']}` |")
        rows.append(
            f"| Compiled scope | {comparison.get('compiled_scope', 'unspecified')} |"
        )
    parity = document.get("parity") or {}
    if isinstance(parity, dict) and "max_abs_error" in parity:
        rows.append(
            f"| Max abs error | {parity['max_abs_error']:.6g} "
            f"(atol={parity.get('atol')}, rtol={parity.get('rtol')}) |"
        )
    rows += [
        "",
        "## Measurements",
        "",
        "Encoder and policy are reported separately. End-to-end includes both and is the number to use for control-loop budgeting. The reference row is the same LeRobot policy executed through PyTorch.",
        "",
        "| Backend | Startup | Encoder p50 / p95 / p99 | Policy p50 / p95 / p99 | E2E p50 / p95 / p99 | Throughput | RSS | Setup / hot allocations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in BACKENDS:
        measurement = measurements.get(backend) or {"status": "not_measured"}
        if measurement.get("status") != "measured":
            reason = measurement.get("reason", "not measured")
            rows.append(f"| {backend} | not measured | - | - | - | - | - | - |")
            rows.append(f"| status | `{reason}` | | | | | | |")
            continue

        allocations = measurement["allocations"]
        rows.append(
            f"| {backend} | {_display(measurement['startup_ms'], ' ms')} | "
            f"{_triplet(measurement, 'encoder_ms')} | {_triplet(measurement, 'policy_ms')} | "
            f"{_triplet(measurement, 'end_to_end_ms')} | "
            f"{_display(measurement['throughput_hz'], ' Hz')} | {_display(measurement['rss_mb'], ' MiB')} | "
            f"{_display(allocations['setup'])} / {_display(allocations['hot_path'])} |"
        )
    rows += _comparison(measurements)
    rows += [
        "",
        "## Commands",
        "",
        "```text",
        f"FlowEdge: {commands['flowedge']}",
        f"LeRobot:  {commands['lerobot']}",
        "```",
        "",
    ]
    rows += [
        "## Limits",
        "",
        "- Repeat on the target CPU, OS, compiler, thread count, power mode, and checkpoint before making a deployment decision.",
        "- RSS includes process/runtime state and is not a per-inference allocation measurement.",
        "- The zero-allocation contract applies to supported native paths; unknown full-policy counts remain unmeasured.",
    ]
    for limitation in document.get("limitations", []):
        rows.append(f"- {limitation}")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="JSON artifact to validate")
    parser.add_argument("--report", type=Path, help="write Markdown report")
    args = parser.parse_args()
    try:
        document = validate(json.loads(args.artifact.read_text(encoding="utf-8")))
        report = render(document)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding="utf-8")
        print(report, end="")
    except (OSError, json.JSONDecodeError, ArtifactError) as error:
        print(f"benchmark artifact failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
