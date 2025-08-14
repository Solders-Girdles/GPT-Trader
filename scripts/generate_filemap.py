#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "ARCHITECTURE_FILEMAP.md"

# Directories to include and print as sections (relative to repo root)
SECTIONS: list[tuple[str, Path]] = [
    ("src/bot/", ROOT / "src" / "bot"),
    ("tests/", ROOT / "tests"),
    ("scripts/", ROOT / "scripts"),
    ("examples/", ROOT / "examples"),
]

IGNORE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".DS_Store",
    ".git",
    "node_modules",
}

ALLOW_FILE_EXT = {".py", ".md", ".sh"}


def iter_files(base: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(base):
        # prune ignored directories
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES and not d.startswith(".")]
        for fn in filenames:
            if fn in IGNORE_DIR_NAMES:
                continue
            p = Path(dirpath) / fn
            if p.suffix and p.suffix.lower() not in ALLOW_FILE_EXT:
                continue
            yield p


def format_section(title: str, base: Path) -> str:
    rel_base = base.relative_to(ROOT)
    lines: list[str] = [f"### {title}", ""]

    # Group files by immediate subdirectory under base (one level), keeping base files at top
    groups: dict[str, list[Path]] = {}
    base_files: list[Path] = []

    for fp in sorted(iter_files(base)):
        try:
            rel = fp.relative_to(base)
        except Exception:
            continue
        parts = rel.parts
        if len(parts) == 1:
            base_files.append(rel)
        else:
            groups.setdefault(parts[0], []).append(rel)

    # Base files
    if base_files:
        lines.append("- (root)")
        for rel in base_files:
            lines.append(f"  - {rel.name}")
        lines.append("")

    # Subdirectories
    for subdir in sorted(groups.keys()):
        lines.append(f"- {subdir}/")
        for rel in groups[subdir]:
            # indent files under their subdir (omit the first segment)
            sub_rel = Path(*rel.parts[1:])
            lines.append(f"  - {sub_rel.as_posix()}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate() -> str:
    header = (
        "## Architecture File Map\n\n"
        "This document is auto-generated. Run `python scripts/generate_filemap.py` to refresh it.\n\n"
    )

    content = [header]
    for title, base in SECTIONS:
        if not base.exists():
            continue
        content.append(format_section(title, base))

    content.append(
        "---\n\nNote: Only key file types are listed (py/md/sh). Hidden and cache dirs are skipped.\n"
    )
    return "\n".join(content)


def main() -> None:
    out = generate()
    DOC_PATH.write_text(out, encoding="utf-8")
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
