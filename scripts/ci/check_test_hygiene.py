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
ALLOWLIST = {
    # Coverage expansion tests (C1 task)
    # Phase 2 pain points remediation: critical service coverage
    # Optimization and CLI test suites
    # TUI Phase 3.5 comprehensive test suites
    # TUI reliability/fault-injection tests
    # Architecture migration and observability tests
    # Pre-existing TUI and feature test suites
}

SLEEP_ALLOWLIST = {
    "tests/unit/gpt_trader/utilities/performance/test_timing.py",  # timing utility coverage requires real sleep for precision tests
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_response_cache.py",  # TTL-based cache expiration requires real time elapsed
}


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
