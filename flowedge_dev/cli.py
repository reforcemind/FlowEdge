"""One entry for maintainer benches, parity, and deployment pipelines.

    python -m flowedge_dev bench policy -- <run_policy_report args>
    python -m flowedge_dev verify ulp models/mamba_flow.safetensors
    python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from . import ROOT

# Everyday commands first. Scripts stay in tools/ and convert/; this module is the front door.
_COMMANDS: dict[str, dict[str, Path]] = {
    "bench": {
        "policy": ROOT / "tools/benchmark/run_policy_report.py",
        "mix": ROOT / "tools/benchmark/export_kernel_mix.py",
        "budgets": ROOT / "tools/benchmark/report_budgets.py",
        "hybrid": ROOT / "tools/benchmark/run_smolvla_hybrid_report.py",
        "artifact": ROOT / "tools/benchmark/benchmark_artifact.py",
    },
    "verify": {
        "ulp": ROOT / "tools/verification/verify_ulp.py",
        "diffusion": ROOT / "tools/verification/verify_diffusion.py",
        "head": ROOT / "tools/verification/verify_external_head.py",
        "diffusion-public": ROOT / "tools/verification/verify_diffusion_public.py",
        "smolvla": ROOT / "tools/verification/verify_smolvla_hybrid.py",
        "transformer": ROOT / "tools/verification/verify_transformer_reference.py",
        "delivery": ROOT / "tools/verification/verify_delivery.py",
    },
    "pipeline": {
        "convert": ROOT / "convert/convert.py",
        "inspect": ROOT / "convert/policy_inspect.py",
        "rollout": ROOT / "tools/verification/run_edge_dp_rollout.py",
        "capture": ROOT / "tools/verification/capture_lerobot_frame.py",
    },
}

_HELP = """FlowEdge maintainer CLI (checkout only; not shipped in the runtime wheel).

groups:
  bench      policy | mix | budgets | hybrid | artifact
  verify     ulp | diffusion | head | diffusion-public | smolvla | transformer | delivery
  pipeline   convert | inspect | rollout | capture

Everyday:
  python -m flowedge_dev bench policy <checkpoint> --source ... --revision ... --output ...
  python -m flowedge_dev verify ulp models/mamba_flow.safetensors
  python -m flowedge_dev pipeline rollout models/diffusion_pusht.flowedge.safetensors
  python -m flowedge_dev pipeline convert <src> <dst> --arch mamba|diffusion|transformer
"""


def _exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _run_script(script: Path, argv: list[str]) -> int:
    if not script.is_file():
        print(f"error: missing {script.relative_to(ROOT)}", file=sys.stderr)
        return 2
    sys.argv = [str(script), *argv]
    sys.path.insert(0, str(ROOT / "integrations" / "lerobot" / "src"))
    sys.path.insert(0, str(ROOT))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return _exit_code(exc)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flowedge_dev",
        description=_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("group", choices=sorted(_COMMANDS))
    parser.add_argument("command")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="forwarded to the tool")
    ns = parser.parse_args(argv)
    table = _COMMANDS[ns.group]
    if ns.command not in table:
        known = ", ".join(sorted(table))
        print(f"error: {ns.group} commands: {known}", file=sys.stderr)
        return 2
    forwarded = ns.args[1:] if ns.args[:1] == ["--"] else ns.args
    return _run_script(table[ns.command], forwarded)
