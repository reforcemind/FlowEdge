"""Pinned, optional TLC baseline. Its state encoding is not a Python state key."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

URL = "https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
SHA1 = "85b172970b0a1d283b9e45679b8ecf5b4de72ab2"  # Published release checksum.
SHA256 = "9d36716ffb5e49d1ba8fae4651eba59f3189887e12eb90e204a42d2e6e993fef"
MAX_JAR_BYTES = 100 * 1024 * 1024


def verify_jar(data):
    if len(data) > MAX_JAR_BYTES:
        raise ValueError("TLC jar exceeds download size limit")
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("TLC jar SHA-256 mismatch")
    if hashlib.sha1(data).hexdigest() != SHA1:
        raise ValueError("TLC release SHA-1 mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--java", default="java", help="Java 11 or later executable")
    parser.add_argument(
        "--jar", type=Path, default=Path("build-delivery/tla2tools.jar")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("papers/action-delivery/results/tlc.json")
    )
    args = parser.parse_args()
    if not args.jar.exists() and args.fetch:
        with urlopen(URL, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_JAR_BYTES:
                raise ValueError("TLC jar exceeds download size limit")
            data = response.read(MAX_JAR_BYTES + 1)
        verify_jar(data)
        args.jar.parent.mkdir(parents=True, exist_ok=True)
        args.jar.write_bytes(data)
    data = args.jar.read_bytes()
    verify_jar(data)
    rows = []
    sources = [
        Path("papers/action-delivery/tla", name)
        for name in ("Delivery.tla", "Delivery.cfg", "Mutant.cfg")
    ]
    spec_dir = Path("build-delivery/tlc/spec").resolve()
    spec_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copyfile(source, spec_dir / source.name)
    for config in ("Delivery", "Mutant"):
        command = [
            args.java,
            "-Xms32m",
            "-Xmx256m",
            "-XX:+UseSerialGC",
            "-cp",
            str(args.jar.resolve()),
            "tlc2.TLC",
            "-workers",
            "1",
            "-seed",
            "1",
            "-metadir",
            str(Path("build-delivery/tlc", config).resolve()),
            "-config",
            config + ".cfg",
            "Delivery.tla",
        ]
        started = perf_counter()
        run = subprocess.run(
            command,
            cwd=spec_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
        counts = re.search(
            r"([\d,]+) states generated, ([\d,]+) distinct states found", run.stdout
        )
        outcome = "unknown"
        if (
            run.returncode == 0
            and "Model checking completed. No error has been found." in run.stdout
        ):
            outcome = "pass"
        elif "Invariant Agreement is violated" in run.stdout:
            outcome = "fail"
        rows.append(
            {
                "config": config,
                "verdict": outcome,
                "exit_code": run.returncode,
                "seconds": perf_counter() - started,
                "command": command,
                "cwd": str(spec_dir),
                "raw": run.stdout,
                "generated": int(counts[1].replace(",", "")) if counts else None,
                "distinct": int(counts[2].replace(",", "")) if counts else None,
            }
        )
    result = {
        "scope": "bounded_model",
        "tool": "TLC v1.8.0",
        "jar_sha256": hashlib.sha256(data).hexdigest(),
        "bounds": {"horizon": 3, "steps": 4, "events": 20},
        "rows": rows,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "config": r["config"],
                    "verdict": r["verdict"],
                    "distinct": r["distinct"],
                }
                for r in rows
            ]
        )
    )
    return 0 if [r["verdict"] for r in rows] == ["pass", "fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
