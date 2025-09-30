#!/usr/bin/env python3
"""Lightweight guardrails for test suite hygiene."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

THRESHOLD = 240
ALLOWLIST = {
    "tests/unit/bot_v2/features/brokerages/coinbase/test_product_catalog.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_specs_quantization.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_performance.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_order_payloads.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_integration.py",
    "tests/unit/bot_v2/features/live_trade/test_pnl_comprehensive.py",
    "tests/unit/bot_v2/features/live_trade/test_risk_core.py",
    "tests/unit/bot_v2/features/live_trade/test_risk_runtime.py",  # 343 lines: comprehensive runtime guard tests
    "tests/unit/bot_v2/orchestration/test_perps_bot.py",
    "tests/unit/bot_v2/orchestration/test_live_execution.py",
    "tests/unit/bot_v2/test_broker_behavioral_contract.py",  # 358 lines: broker contract validation suite
    "tests/unit/bot_v2/security/test_secrets_manager.py",  # 482 lines: comprehensive security test suite (encryption, vault, threading)
    "tests/unit/bot_v2/security/test_security_validator.py",  # 575 lines: comprehensive security validation (injection, rate limiting, trading limits)
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
        text = path.read_text(encoding="utf-8")
        line_count = text.count("\n") + 1

        if line_count > THRESHOLD and str(rel) not in ALLOWLIST:
            problems.append(
                f"{rel} exceeds {THRESHOLD} lines ({line_count}). Split into smaller modules or add to the allowlist with justification."
            )

        if "time.sleep(" in text and "fake_clock" not in text:
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
