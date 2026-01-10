#!/usr/bin/env python3
"""
Audit markdown link targets for missing files.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "experiments",
    "logs",
    "runtime_data",
    "var",
}
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit markdown link targets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan (default: repo root).",
    )
    return parser.parse_args()


def iter_markdown_files(root: Path) -> list[Path]:
    markdown_files: list[Path] = []
    for path in root.rglob("*.md"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        markdown_files.append(path)
    return markdown_files


def iter_links(content: str) -> list[str]:
    links: list[str] = []
    for target in LINK_PATTERN.findall(content):
        target = target.strip()
        if not target:
            continue
        if target.startswith(EXTERNAL_PREFIXES):
            continue
        if target.startswith("#"):
            continue
        target = target.split()[0].strip("'\"")
        target = unquote(target)
        target = target.split("#", maxsplit=1)[0]
        if target:
            links.append(target)
    return links


def resolve_target(source: Path, target: str, root: Path) -> Path:
    if target.startswith("/"):
        return (root / target.lstrip("/")).resolve()
    return (source.parent / target).resolve()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    markdown_files = iter_markdown_files(root)
    missing: list[tuple[Path, str]] = []

    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        for target in iter_links(content):
            resolved = resolve_target(path, target, root)
            if not resolved.exists():
                missing.append((path, target))

    if missing:
        print("Missing markdown link targets:")
        for source, target in missing:
            rel_source = source.relative_to(root)
            print(f"- {rel_source}: {target}")
        print(f"\nTotal missing: {len(missing)}")
        return 1

    print(f"✓ Markdown link audit passed ({len(markdown_files)} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
