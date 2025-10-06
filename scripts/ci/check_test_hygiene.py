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
    "tests/unit/bot_v2/features/live_trade/test_risk_runtime_comprehensive.py",  # 1178 lines: comprehensive risk runtime test suite (5 classes, 46 tests covering margin, leverage, position limits, account snapshots)
    "tests/unit/bot_v2/orchestration/test_perps_bot.py",
    "tests/unit/bot_v2/orchestration/test_live_execution.py",
    "tests/unit/bot_v2/orchestration/test_lifecycle_service.py",  # 436 lines: comprehensive lifecycle service test suite (40 tests covering initialization, background tasks, error handling, cleanup)
    "tests/unit/bot_v2/test_broker_behavioral_contract.py",  # 358 lines: broker contract validation suite
    "tests/unit/bot_v2/security/test_secrets_manager.py",  # 482 lines: comprehensive security test suite (encryption, vault, threading)
    "tests/unit/bot_v2/security/test_security_validator.py",  # 575 lines: comprehensive security validation (injection, rate limiting, trading limits)
    "tests/unit/bot_v2/persistence/test_config_store.py",  # 421 lines: comprehensive persistence tests for bot config CRUD, thread safety, error recovery
    "tests/unit/bot_v2/persistence/test_event_store.py",  # 494 lines: comprehensive JSONL event store tests (trades, positions, metrics, errors, concurrency)
    "tests/unit/bot_v2/orchestration/test_broker_factory.py",  # 377 lines: comprehensive broker configuration tests (credential priority, auth detection, URL overrides)
    "tests/unit/bot_v2/state/backup/test_backup_operations.py",  # 1046 lines: comprehensive backup operations test suite (creation, restoration, compression, encryption, S3, cleanup, concurrency, error handling, metadata, storage tiers, async scheduling)
    "tests/unit/bot_v2/state/backup/test_backup_recovery_integration.py",  # 424 lines: integration test suite for backup and recovery workflows (batch operations, state coherence, error scenarios)
    "tests/unit/bot_v2/features/paper_trade/test_paper_trade_strategies.py",  # 1052 lines: comprehensive paper trading strategy test suite (9 classes, 72 tests covering momentum, mean reversion, volatility, breakout, MA crossover, scalping strategies)
    # Week 3 integration tests (operational audit Q4 2025) - comprehensive scenario coverage
    "tests/integration/brokerages/test_coinbase_streaming_failover.py",  # 361 lines: 6 WebSocket failover scenarios (reconnect, auth errors, heartbeat timeout, message corruption, graceful disconnect, concurrent failures)
    "tests/integration/streaming/test_websocket_rest_fallback.py",  # 378 lines: 7 WebSocket/REST fallback scenarios (WebSocket unavailable, partial failure, REST degradation, recovery validation, latency comparison, concurrent fallback, dual-mode operation)
    "tests/integration/orchestration/test_broker_outage_handling.py",  # 420 lines: 8 broker outage handling scenarios (full outage detection, graceful degradation, position sync, recovery workflow, partial API failures, concurrent outages, data staleness, error propagation)
}

LINE_ALLOWLIST_PATH = pathlib.Path("tests/.hygiene_line_allowlist")
SLEEP_ALLOWLIST_PATH = pathlib.Path("tests/.hygiene_sleep_allowlist")


def _load_allowlist(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()

    entries: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


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

    line_allowlist = set(ALLOWLIST)
    line_allowlist.update(_load_allowlist(LINE_ALLOWLIST_PATH))
    sleep_allowlist = _load_allowlist(SLEEP_ALLOWLIST_PATH)

    normalized = [path.resolve() for path in test_files]

    for path in normalized:
        rel = path.relative_to(root)
        rel_str = str(rel)
        text = path.read_text(encoding="utf-8")
        line_count = text.count("\n") + 1

        if line_count > THRESHOLD and rel_str not in line_allowlist:
            problems.append(
                f"{rel} exceeds {THRESHOLD} lines ({line_count}). Split into smaller modules or add to the allowlist with justification."
            )

        if "time.sleep(" in text and "fake_clock" not in text and rel_str not in sleep_allowlist:
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
