#!/usr/bin/env python3
"""Validate legacy test triage manifest and marker usage.

This repository uses `tests/_triage/legacy_tests.yaml` as the source of truth for
tests slated for deletion or modernization.

Rules:
- Any test file marked with `pytest.mark.legacy_delete` or `pytest.mark.legacy_modernize`
  must have an entry in the manifest.
- Any manifest entry must correspond to an existing test file and include the
  matching marker in that file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

MANIFEST_PATH = Path("tests/_triage/legacy_tests.yaml")
TEST_ROOT = Path("tests")

VALID_ACTIONS = {"delete", "modernize"}
VALID_STATUSES = {"todo", "in_progress", "done"}
MARKER_BY_ACTION = {
    "delete": "legacy_delete",
    "modernize": "legacy_modernize",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}

    payload = yaml.safe_load(_read_text(MANIFEST_PATH)) or {}
    if not isinstance(payload, dict):
        raise ValueError("legacy_tests.yaml must contain a mapping at the top-level")

    version = payload.get("version")
    if version != 1:
        raise ValueError(f"legacy_tests.yaml version must be 1 (got {version!r})")

    tests = payload.get("tests", {})
    if not isinstance(tests, dict):
        raise ValueError("legacy_tests.yaml `tests` must be a mapping")

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in tests.items():
        if not isinstance(key, str):
            raise ValueError("legacy_tests.yaml test paths must be strings")
        if not isinstance(value, dict):
            raise ValueError(f"legacy_tests.yaml entry for {key} must be a mapping")
        normalized[key.replace("\\", "/")] = value

    return normalized


def find_legacy_marked_test_files() -> dict[str, str]:
    marked: dict[str, str] = {}
    for test_file in sorted(TEST_ROOT.rglob("test_*.py")):
        text = _read_text(test_file)
        flags = []
        for marker in MARKER_BY_ACTION.values():
            if f".{marker}" in text:
                flags.append(marker)

        if not flags:
            continue
        if len(flags) > 1:
            raise ValueError(
                f"{test_file.as_posix()} includes multiple legacy markers: {', '.join(flags)}"
            )

        marked[test_file.as_posix()] = flags[0]

    return marked


def validate_manifest_entries(manifest: dict[str, dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for path_str, meta in sorted(manifest.items()):
        path = Path(path_str)
        if path_str.startswith("./"):
            problems.append(f"{path_str}: remove leading './' (use repo-relative paths)")
        if not path_str.startswith("tests/"):
            problems.append(f"{path_str}: must live under tests/")
        if path.suffix != ".py":
            problems.append(f"{path_str}: must be a .py file")
        if not path.exists():
            problems.append(f"{path_str}: file does not exist")

        action = meta.get("action")
        if action not in VALID_ACTIONS:
            problems.append(f"{path_str}: action must be one of {sorted(VALID_ACTIONS)}")
            continue

        reason = meta.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"{path_str}: reason is required")

        status = meta.get("status")
        if status is not None and status not in VALID_STATUSES:
            problems.append(f"{path_str}: status must be one of {sorted(VALID_STATUSES)}")

    return problems


def validate_marker_alignment(
    manifest: dict[str, dict[str, Any]],
    marked_files: dict[str, str],
) -> list[str]:
    problems: list[str] = []

    for file_path, marker in sorted(marked_files.items()):
        meta = manifest.get(file_path)
        if not meta:
            problems.append(
                f"{file_path}: has {marker} marker but is missing from legacy_tests.yaml"
            )
            continue
        action = meta.get("action")
        expected = MARKER_BY_ACTION.get(action)
        if expected and marker != expected:
            problems.append(
                f"{file_path}: marker {marker} does not match manifest action {action!r} (expected {expected})"
            )

    for file_path, meta in sorted(manifest.items()):
        action = meta.get("action")
        expected = MARKER_BY_ACTION.get(action)
        if not expected:
            continue
        marker = marked_files.get(file_path)
        if marker != expected:
            problems.append(
                f"{file_path}: manifest action {action!r} requires marker {expected} (found {marker or 'none'})"
            )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate legacy test triage manifest.")
    parser.parse_args(argv)

    try:
        manifest = load_manifest()
        marked_files = find_legacy_marked_test_files()
        problems = []
        problems.extend(validate_manifest_entries(manifest))
        problems.extend(validate_marker_alignment(manifest, marked_files))
    except Exception as exc:
        print(f"Legacy test triage check failed: {exc}", file=sys.stderr)
        return 2

    if problems:
        print("Legacy test triage issues detected:\n", file=sys.stderr)
        for item in problems:
            print(f"- {item}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
