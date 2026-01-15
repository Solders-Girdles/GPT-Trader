#!/usr/bin/env python3
"""Ensure deprecation shims are registered in docs/DEPRECATIONS.md."""

from __future__ import annotations

import re
from pathlib import Path

MARKERS = ("DeprecationWarning", ".. deprecated::")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    docs_path = repo_root / "docs" / "DEPRECATIONS.md"
    src_root = repo_root / "src"

    if not docs_path.exists():
        print("::error::Missing docs/DEPRECATIONS.md")
        return 1

    doc_text = docs_path.read_text(encoding="utf-8")
    allowed_paths = set(re.findall(r"src/[A-Za-z0-9_./-]+\.py", doc_text))
    if not allowed_paths:
        print("::error::No src paths found in docs/DEPRECATIONS.md")
        return 1

    missing: list[str] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in MARKERS):
            rel_path = path.relative_to(repo_root).as_posix()
            if rel_path not in allowed_paths:
                missing.append(rel_path)

    if missing:
        print("::error::Deprecation shims missing from docs/DEPRECATIONS.md:")
        for path in sorted(missing):
            print(f" - {path}")
        return 1

    print("Deprecation registry check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
