"""Installed entry point for the supported LeRobot deployment workflow."""

from __future__ import annotations

import argparse
import importlib
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="flowedge-edge",
        description="Package, preflight, or compare a supported LeRobot policy.",
    )
    parser.add_argument("command", choices=("package", "preflight", "bench"))
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    remaining = argv[1:]
    modules = {
        "package": "flowedge_lerobot.deployment",
        "preflight": "flowedge_lerobot.preflight",
        "bench": "flowedge_lerobot.benchmark",
    }
    return importlib.import_module(modules[args.command]).main(remaining) or 0


if __name__ == "__main__":
    sys.exit(main())
