#!/usr/bin/env python3
"""
Bundle legacy modules (archived experimental slices + gpt_trader PoC) into a tarball.

Run this from a commit that still has the legacy directories in-tree. If the
targets are missing, the script will skip bundling and report which paths were
absent.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "var" / "legacy"
LEGACY_TARGETS = [
    Path("archived/experimental"),
    Path("src/gpt_trader"),
]


@dataclass
class BundlePlan:
    targets: list[Path]
    output: Path
    dry_run: bool


def parse_args() -> BundlePlan:
    parser = argparse.ArgumentParser(
        description="Create a tar.gz archive containing legacy modules for safekeeping."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the tar.gz file to create (default: var/legacy/legacy_bundle_<timestamp>.tar.gz).",
    )
    parser.add_argument(
        "--include",
        type=Path,
        action="append",
        help="Additional paths to include in the bundle (relative to repo root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without creating the tarball.",
    )

    args = parser.parse_args()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = DEFAULT_OUTPUT_DIR / f"legacy_bundle_{stamp}.tar.gz"

    include_targets = list(LEGACY_TARGETS)
    if args.include:
        for extra in args.include:
            include_targets.append(extra if extra.is_absolute() else Path(extra))

    normalized = []
    for path in include_targets:
        normalized.append(path if path.is_absolute() else REPO_ROOT / path)

    return BundlePlan(targets=normalized, output=output, dry_run=bool(args.dry_run))


def resolve_targets(targets: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    resolved: list[Path] = []
    missing: list[Path] = []

    for target in targets:
        if target.exists():
            resolved.append(target.resolve())
        else:
            missing.append(target)

    return resolved, missing


def create_manifest(targets: Iterable[Path], bundle_path: Path) -> bytes:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle_path),
        "targets": [str(path.relative_to(REPO_ROOT)) for path in targets],
    }
    return json.dumps(manifest, indent=2).encode("utf-8")


def bundle_legacy(plan: BundlePlan) -> None:
    targets, missing = resolve_targets(plan.targets)

    if missing:
        print("Skipping missing targets:")
        for target in missing:
            try:
                display = target.relative_to(REPO_ROOT)
            except ValueError:
                display = target
            print(f"  - {display}")

    if not targets:
        print("No legacy targets found; bundle not created.")
        return

    print("Legacy bundle plan:")
    for target in targets:
        print(f"  - {target.relative_to(REPO_ROOT)}")
    print(f"Archive: {plan.output}")

    if plan.dry_run:
        print("Dry run: archive not created.")
        return

    plan.output.parent.mkdir(parents=True, exist_ok=True)

    manifest_bytes = create_manifest(targets, plan.output)
    manifest_info = tarfile.TarInfo(name="legacy_manifest.json")
    manifest_info.size = len(manifest_bytes)
    manifest_info.mtime = int(time.time())

    with tarfile.open(plan.output, "w:gz") as tar:
        for target in targets:
            relative = target.relative_to(REPO_ROOT)
            tar.add(target, arcname=str(relative))
        tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

    size_mb = plan.output.stat().st_size / (1024 * 1024)
    print(f"Legacy bundle created: {plan.output} ({size_mb:.2f} MB)")


def main() -> None:
    plan = parse_args()
    try:
        bundle_legacy(plan)
    except FileNotFoundError as exc:
        print(str(exc))


if __name__ == "__main__":
    main()
