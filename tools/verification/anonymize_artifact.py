"""Remove workstation-specific paths and public repo identifiers from artifacts.

Artifact construction imports the byte transformation before manifest hashes
are computed. The standalone command is for an unmanifested derived tree; it
refuses to invalidate an existing artifact manifest.

Author-specific identity greps belong in a maintainer pre-release script outside
the delivered review bundle, not in this module.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_WINDOWS_USER = re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\"'\s]+")
_WSL_USER = re.compile(r"/mnt/[A-Za-z]/Users/[^/\"']+")
_HOME_USER = re.compile(r"/home/[^/\"']+")
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
_IDENTITY_PATTERNS = (_WINDOWS_USER, _WSL_USER, _HOME_USER)

# Local-checkout revision strings that can identify a public repository.
# Third-party pins (e.g. LeRobot) are left intact. Concatenate from parts so
# anonymize_text_bytes does not rewrite this module when it is itself shipped
# inside an anonymized review bundle.
_PUBLIC_REPO_SHAS = (
    "97f57206" + "4e3e785eca1663bfab709941d441e906",
    "fda8b642" + "68f466d68a269e42875430f47025c598",
    "40734f12" + "ec9b142586cec9f2402a3675f4ff241f",
)
_REPO_REVISION_KEYS = frozenset(
    {
        "flowedge_base_revision",
        "artifact_revision",
        "baseline_revision",
        "base_revision",
    }
)


def anonymize_string(value: str) -> str:
    value = _WINDOWS_USER.sub("<windows-user>", value)
    value = _WSL_USER.sub("<wsl-user>", value)
    value = _HOME_USER.sub("<home-user>", value)
    for sha in _PUBLIC_REPO_SHAS:
        value = value.replace(sha, "<source-snapshot>")
    return value


def transform(value):
    if isinstance(value, str):
        return anonymize_string(value)
    if isinstance(value, list):
        return [transform(item) for item in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in _REPO_REVISION_KEYS:
                out[key] = None
            else:
                out[key] = transform(item)
        return out
    return value


def anonymize_json_bytes(data: bytes) -> bytes:
    text = data.decode("utf-8")
    value = json.loads(text)
    sanitized = transform(value)
    if sanitized == value:
        return data
    return (json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def anonymize_text_bytes(data: bytes) -> bytes:
    """Redact public local-repo SHAs from non-JSON text payloads."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    sanitized = text
    for sha in _PUBLIC_REPO_SHAS:
        sanitized = sanitized.replace(sha, "<source-snapshot>")
    if sanitized == text:
        return data
    return sanitized.encode("utf-8")


def identity_path_matches(text: str) -> list[str]:
    return sorted(
        {
            match.group(0)
            for pattern in _IDENTITY_PATTERNS
            for match in pattern.finditer(text)
        }
    )


def identity_path_matches_json_bytes(data: bytes) -> list[str]:
    matches = []

    def visit(value):
        if isinstance(value, str):
            matches.extend(identity_path_matches(value))
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(json.loads(data.decode("utf-8")))
    return sorted(set(matches))


def identity_token_matches(data: bytes) -> list[str]:
    """Generic residual tokens (emails). Author-specific names are not scanned here."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return sorted({match.group(0) for match in _EMAIL.finditer(text)})


def public_repo_sha_matches(data: bytes) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return sorted({sha for sha in _PUBLIC_REPO_SHAS if sha in text})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Derived artifact root to sanitize")
    args = parser.parse_args()
    if (args.root / "artifact-manifest.json").is_file():
        parser.error(
            "refusing to invalidate artifact-manifest.json; rebuild the archive "
            "with `python -m tools.delivery_check.artifact build --anonymized`"
        )
    changed = 0
    for path in sorted(args.root.rglob("*.json")):
        if path.name == "artifact-manifest.json":
            continue
        original = path.read_bytes()
        sanitized = anonymize_json_bytes(original)
        if sanitized != original:
            path.write_bytes(sanitized)
            changed += 1
    print(json.dumps({"root": str(args.root), "json_files_changed": changed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
