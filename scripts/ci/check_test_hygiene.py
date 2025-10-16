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
    "tests/unit/bot_v2/features/brokerages/coinbase/client/test_base.py",  # comprehensive client contract coverage
    "tests/unit/bot_v2/features/brokerages/coinbase/rest/test_base.py",  # REST validation matrix
    "tests/unit/bot_v2/data_providers/test_coinbase_provider.py",  # integration heavy provider tests
    "tests/unit/bot_v2/features/live_trade/test_pnl_comprehensive.py",
    "tests/unit/bot_v2/features/live_trade/test_risk_core.py",
    "tests/unit/bot_v2/features/live_trade/test_risk_runtime.py",  # 343 lines: comprehensive runtime guard tests
    "tests/unit/bot_v2/orchestration/test_perps_bot.py",
    "tests/unit/bot_v2/orchestration/test_live_execution.py",
    "tests/unit/bot_v2/orchestration/test_execution_coordinator.py",  # complex orchestration harness
    "tests/unit/bot_v2/orchestration/test_runtime_coordinator.py",  # integration-heavy runtime checks
    "tests/unit/bot_v2/orchestration/test_strategy_orchestrator.py",  # strategy routing contract suite
    "tests/unit/bot_v2/orchestration/test_system_monitor.py",  # monitoring integration matrix
    "tests/unit/bot_v2/test_broker_behavioral_contract.py",  # 358 lines: broker contract validation suite
    "tests/unit/bot_v2/utilities/test_async_utils_advanced.py",  # legacy async helpers coverage
    "tests/unit/bot_v2/persistence/test_event_store.py",  # event store contract and normalization coverage
    "tests/unit/bot_v2/features/brokerages/coinbase/test_helpers.py",  # scenario builder helpers consolidated
    "tests/unit/bot_v2/utilities/test_performance_monitoring_advanced.py",  # needs refactor but tracked separately
    "tests/unit/bot_v2/orchestration/test_partial_fills.py",  # legacy regression suite
    "tests/unit/bot_v2/monitoring/test_health_checks.py",  # extensive health matrix
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_permissions.py",  # permissions coverage table
    "tests/unit/bot_v2/persistence/test_config_store.py",  # config persistence contract coverage
    "tests/unit/bot_v2/utilities/test_datetime_helpers.py",  # datetime edge-case coverage
    "tests/unit/bot_v2/utilities/test_performance_monitoring_core.py",  # performance collector matrix
    "tests/unit/bot_v2/utilities/test_trading_operations_core.py",  # trading ops regression suite
    "tests/unit/bot_v2/orchestration/test_execution_runtime_guards.py",  # guardrail regression suite
    "tests/unit/bot_v2/utilities/test_import_utils.py",  # compatibility shims validation
    "tests/unit/bot_v2/config/test_schemas.py",  # schema contract coverage
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_websocket.py",  # websocket integration matrix
    "tests/unit/bot_v2/features/live_trade/test_coinbase_pnl.py",  # pnl engine behavioural coverage
    "tests/unit/bot_v2/utilities/test_console_logging_functions.py",  # logging formatting matrix
    "tests/unit/bot_v2/types/test_trading.py",  # trading dataclass behaviours
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_market_data.py",  # market data coverage
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_system.py",  # system integration harness
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_accounts.py",  # account snapshot + treasury surface coverage
    "tests/unit/bot_v2/errors/test_base.py",  # error taxonomy coverage
    "tests/unit/bot_v2/monitoring/test_configuration_guardian.py",  # guardian behaviour matrix
    "tests/unit/bot_v2/utilities/test_trading_operations_integration.py",  # trading ops integration harness
    "tests/unit/bot_v2/utilities/test_async_utils_core.py",  # async helper regression suite
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_trading.py",  # trading API coverage
    "tests/unit/bot_v2/logging/test_setup.py",  # logging stack regression suite
    "tests/unit/bot_v2/validation/test_calculation_validator.py",  # calculation validator coverage
    "tests/unit/bot_v2/features/brokerages/coinbase/test_market_data_service.py",  # market data service contract
    "tests/unit/bot_v2/features/data/test_data_module.py",  # 295 lines: comprehensive data module contract with storage/cache stubs
    "tests/unit/bot_v2/cli/test_commands_orders.py",  # 254 lines: comprehensive CLI orders command coverage
    "tests/unit/bot_v2/utilities/test_console_logging_core.py",  # 336 lines: console logging contract and fallback behavior matrix
    "tests/unit/bot_v2/utilities/test_logging_patterns.py",  # 494 lines: extensive structured logging patterns and decorator coverage
}

SLEEP_ALLOWLIST = {
    "tests/unit/bot_v2/utilities/test_performance_monitoring_advanced.py",
    "tests/unit/bot_v2/utilities/test_performance_monitoring_core.py",
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

        if "time.sleep(" in text and "fake_clock" not in text and str(rel) not in SLEEP_ALLOWLIST:
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
