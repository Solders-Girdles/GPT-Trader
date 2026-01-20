#!/usr/bin/env python3
"""Lightweight guardrails for test suite hygiene."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

THRESHOLD = 240
UNIT_ALLOWED_PREFIXES = (
    "tests/unit/gpt_trader/",
    "tests/unit/scripts/",
    "tests/unit/support/",
)
INTEGRATION_TEST_PREFIX = "tests/integration/"
CONTRACT_TEST_PREFIX = "tests/contract/"
REAL_API_TEST_PREFIX = "tests/real_api/"
ALLOWLIST: set[str] = set()

SLEEP_ALLOWLIST: set[str] = set()


def scan(paths: Sequence[str]) -> int:
    root = pathlib.Path.cwd()
    test_files: list[pathlib.Path] = []
    if paths:
        for entry in paths:
            p = pathlib.Path(entry)
            if p.is_dir():
                test_files.extend(p.rglob("test_*.py"))
            elif p.name.startswith("test_") and p.suffix == ".py":
                test_files.append(p)
    else:
        test_files = list(pathlib.Path("tests").rglob("test_*.py"))

    problems: list[str] = []

    normalized = [path.resolve() for path in test_files]

    for path in normalized:
        rel = path.relative_to(root)
        rel_str = rel.as_posix()
        text = path.read_text(encoding="utf-8")
        if not text:
            line_count = 0
        else:
            line_count = text.count("\n")
            if not text.endswith("\n"):
                line_count += 1

        if rel_str.startswith("tests/unit/") and not rel_str.startswith(UNIT_ALLOWED_PREFIXES):
            problems.append(
                f"{rel} is a unit test outside the supported layout. Place unit tests under `tests/unit/gpt_trader/` (or `tests/unit/scripts/`, `tests/unit/support/`)."
            )

        if rel_str.startswith(INTEGRATION_TEST_PREFIX) and ".mark.integration" not in text:
            problems.append(
                f"{rel} is under `tests/integration/` but is missing the `pytest.mark.integration` marker."
            )

        if rel_str.startswith(CONTRACT_TEST_PREFIX) and ".mark.contract" not in text:
            problems.append(
                f"{rel} is under `tests/contract/` but is missing the `pytest.mark.contract` marker."
            )

        if rel_str.startswith(REAL_API_TEST_PREFIX) and ".mark.real_api" not in text:
            problems.append(
                f"{rel} is under `tests/real_api/` but is missing the `pytest.mark.real_api` marker."
            )

        if rel_str.startswith("tests/unit/") and (
            ".mark.integration" in text or ".mark.contract" in text or ".mark.real_api" in text
        ):
            problems.append(
                f"{rel} is under `tests/unit/` but is marked integration/contract/real_api. Reclassify it under `tests/integration/`, `tests/contract/`, or `tests/real_api/`."
            )

        if rel_str.startswith(INTEGRATION_TEST_PREFIX) and ".mark.real_api" in text:
            problems.append(
                f"{rel} is under `tests/integration/` but is marked real_api. Move it to `tests/real_api/`."
            )

        if line_count > THRESHOLD and rel_str not in ALLOWLIST:
            problems.append(
                f"{rel} exceeds {THRESHOLD} lines ({line_count}). Split into smaller modules or add to the allowlist with justification."
            )

        if "time.sleep(" in text and "fake_clock" not in text and rel_str not in SLEEP_ALLOWLIST:
            problems.append(
                f"{rel} calls time.sleep without using fake_clock fixture. Use fake_clock or justify with an explicit helper."
            )

    if problems:
        print("Test hygiene issues detected:\n", file=sys.stderr)
        for item in problems:
            print(f"- {item}", file=sys.stderr)
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check test hygiene constraints.")
    parser.add_argument("paths", nargs="*", help="Optional subset of files or directories to scan.")
    args = parser.parse_args(argv)
    return scan(args.paths)


if __name__ == "__main__":
    raise SystemExit(main())
