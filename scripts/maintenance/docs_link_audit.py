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
REPO_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./-])(?P<path>(?:src|scripts|config|tests)/[^\s`\"')]+)"
)
EXCLUDED_DIRS = {
    ".git",
    ".uv-cache",
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


def iter_repo_path_references(content: str) -> list[str]:
    references: list[str] = []
    for match in REPO_PATH_PATTERN.finditer(content):
        candidate = match.group("path")
        candidate = candidate.split("::", maxsplit=1)[0]
        if not candidate:
            continue
        if any(token in candidate for token in ("...", "*", "?", "<", ">", "{", "}")):
            continue
        candidate = candidate.rstrip(".,;:")
        if not candidate:
            continue
        references.append(candidate)
    return references


def should_check_repo_paths(source: Path, *, root: Path) -> bool:
    docs_root = root / "docs"
    try:
        source.relative_to(docs_root)
    except ValueError:
        return False

    excluded_files = {
        docs_root / "CHANGELOG.md",
        docs_root / "DEPRECATIONS.md",
    }
    if source in excluded_files:
        return False

    excluded_dirs = {
        docs_root / "adr",
    }
    for excluded_dir in excluded_dirs:
        try:
            source.relative_to(excluded_dir)
        except ValueError:
            continue
        return False

    return True


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    markdown_files = iter_markdown_files(root)
    missing_links: list[tuple[Path, str]] = []
    missing_paths: list[tuple[Path, str]] = []

    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        for target in iter_links(content):
            resolved = resolve_target(path, target, root)
            if not resolved.exists():
                missing_links.append((path, target))

        if should_check_repo_paths(path, root=root):
            for candidate in iter_repo_path_references(content):
                resolved = (root / candidate).resolve()
                if not resolved.exists():
                    missing_paths.append((path, candidate))

    if missing_links or missing_paths:
        print("How to reproduce: python scripts/maintenance/docs_link_audit.py")
        print()
        if missing_links:
            grouped_links: dict[Path, list[str]] = {}
            for source, target in missing_links:
                grouped_links.setdefault(source, []).append(target)
            print("Missing markdown link targets:")
            for source, targets in grouped_links.items():
                rel_source = source.relative_to(root)
                print(f"- {rel_source}")
                for target in targets:
                    print(f"  - {target}")
            print(f"\nTotal missing links: {len(missing_links)}")

        if missing_paths:
            if missing_links:
                print()
            grouped_paths: dict[Path, list[str]] = {}
            for source, candidate in missing_paths:
                grouped_paths.setdefault(source, []).append(candidate)
            print("Missing repo path references (src/scripts/config/tests):")
            for source, candidates in grouped_paths.items():
                rel_source = source.relative_to(root)
                print(f"- {rel_source}")
                for candidate in candidates:
                    print(f"  - {candidate}")
            print(f"\nTotal missing paths: {len(missing_paths)}")
        return 1

    print(f"✓ Markdown link audit passed ({len(markdown_files)} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
