#!/usr/bin/env python3
"""
Ensure all non-archived docs are reachable from docs/README.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
DOCS_ENTRYPOINT = DOCS_ROOT / "README.md"
ARCHIVE_DIR = DOCS_ROOT / "archive"

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check docs reachability from docs/README.md")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan (default: repo root).",
    )
    return parser.parse_args()


def is_archived(path: Path, archive_dir: Path) -> bool:
    try:
        path.relative_to(archive_dir)
    except ValueError:
        return False
    return True


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


def collect_docs(docs_root: Path, archive_dir: Path) -> list[Path]:
    return [
        path.resolve() for path in docs_root.rglob("*.md") if not is_archived(path, archive_dir)
    ]


def walk_reachable(
    start: Path,
    docs_root: Path,
    archive_dir: Path,
    repo_root: Path,
) -> set[Path]:
    reachable: set[Path] = set()
    queue = [start.resolve()]

    while queue:
        current = queue.pop()
        if current in reachable:
            continue
        reachable.add(current)

        try:
            content = current.read_text(encoding="utf-8")
        except OSError:
            continue

        for target in iter_links(content):
            resolved = resolve_target(current, target, repo_root)
            if resolved.is_dir():
                candidate = resolved / "README.md"
                if candidate.exists():
                    resolved = candidate.resolve()

            if resolved.suffix.lower() != ".md":
                continue
            if not resolved.exists():
                continue

            try:
                resolved.relative_to(docs_root)
            except ValueError:
                continue
            if is_archived(resolved, archive_dir):
                continue

            if resolved not in reachable:
                queue.append(resolved)

    return reachable


def main() -> int:
    args = parse_args()
    repo_root = args.root.resolve()
    docs_root = repo_root / "docs"
    archive_dir = docs_root / "archive"
    entrypoint = docs_root / "README.md"

    if not entrypoint.exists():
        print(f"Missing docs entrypoint: {entrypoint}", file=sys.stderr)
        return 1

    docs = collect_docs(docs_root, archive_dir)
    reachable = walk_reachable(entrypoint, docs_root, archive_dir, repo_root)
    orphaned = sorted(set(docs) - reachable)

    if orphaned:
        print("Docs not reachable from docs/README.md:")
        for path in orphaned:
            print(f"- {path.relative_to(repo_root)}")
        print(f"\nTotal orphaned: {len(orphaned)}")
        return 1

    print(f"✓ Docs reachability check passed ({len(docs)} files reachable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
