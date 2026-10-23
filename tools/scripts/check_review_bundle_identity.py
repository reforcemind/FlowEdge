#!/usr/bin/env python3
"""Maintainer-only author-identity scan for an anonymized review zip.

This script is intentionally NOT part of the delivered review bundle. It knows
author-specific tokens so it can fail packaging before upload without embedding
those tokens in the artifact that reviewers receive.

Usage:
  python tools/scripts/check_review_bundle_identity.py build-delivery/action-delivery-review-anonymized.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

# Author-specific; keep out of INCLUDE / the review zip.
_AUTHOR_PATTERNS = (
    re.compile(rb"igorj", re.IGNORECASE),
    re.compile(rb"jankowski", re.IGNORECASE),
    re.compile(rb"/home/igor\b"),
    re.compile(rb"Users[/\\]+igorj", re.IGNORECASE),
)


def scan_root(root: Path) -> list[tuple[str, str]]:
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        for pattern in _AUTHOR_PATTERNS:
            for match in pattern.finditer(data):
                hits.append(
                    (relative, match.group(0).decode("utf-8", errors="replace"))
                )
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Anonymized review zip to scan")
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"missing archive: {args.archive}")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with zipfile.ZipFile(args.archive) as bundle:
            bundle.extractall(root)
        hits = scan_root(root)
    if hits:
        print("author identity leaks:", file=sys.stderr)
        for relative, token in hits:
            print(f"  {relative}: {token!r}", file=sys.stderr)
        return 1
    print(f"pass: no author-specific tokens in {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
