#!/usr/bin/env python3
"""Lightweight guardrails for test suite hygiene."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

THRESHOLD = 240
ALLOWLIST = {
    "tests/integration/test_coinbase_auth_smoke.py",  # full credential negotiation + auth fallback smoke flow
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_product_catalog.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_specs_quantization.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_performance.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_order_payloads.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_integration.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_base.py",  # comprehensive client contract coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_base.py",  # REST validation matrix
    "tests/unit/gpt_trader/data_providers/test_coinbase_provider.py",  # integration heavy provider tests
    "tests/unit/gpt_trader/features/live_trade/test_pnl_comprehensive.py",
    "tests/unit/gpt_trader/features/live_trade/test_risk_core.py",
    "tests/unit/gpt_trader/features/live_trade/test_risk_runtime.py",  # 343 lines: comprehensive runtime guard tests
    "tests/unit/gpt_trader/orchestration/test_perps_bot.py",
    "tests/unit/gpt_trader/orchestration/test_live_execution.py",
    "tests/unit/gpt_trader/orchestration/test_execution_coordinator.py",  # complex orchestration harness
    "tests/unit/gpt_trader/orchestration/test_runtime_coordinator.py",  # integration-heavy runtime checks
    "tests/unit/gpt_trader/orchestration/test_strategy_orchestrator.py",  # strategy routing contract suite
    "tests/integration/test_composition_root.py",  # composition root integration coverage
    "tests/unit/app/test_container.py",  # exhaustive container unit scenarios
    "tests/integration/test_reduce_only_state_manager_integration.py",  # comprehensive reduce-only state manager integration coverage
    "tests/unit/gpt_trader/orchestration/test_state_manager.py",  # exhaustive reduce-only state manager unit scenarios
    "tests/integration/gpt_trader/features/live_trade/test_circuit_breaker_integration.py",  # extensive circuit breaker regression suite
    "tests/unit/gpt_trader/orchestration/configuration/test_core.py",  # configuration validation matrix
    "tests/unit/gpt_trader/logging/test_orchestration_helpers.py",  # structured logging helper coverage
    "tests/unit/gpt_trader/orchestration/test_system_monitor.py",  # monitoring integration matrix
    "tests/unit/gpt_trader/test_broker_behavioral_contract.py",  # 358 lines: broker contract validation suite
    "tests/unit/gpt_trader/utilities/test_async_utils_advanced.py",  # legacy async helpers coverage
    "tests/unit/gpt_trader/persistence/test_event_store.py",  # event store contract and normalization coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_helpers.py",  # scenario builder helpers consolidated
    "tests/unit/gpt_trader/utilities/test_performance_monitoring_advanced.py",  # needs refactor but tracked separately
    "tests/unit/gpt_trader/orchestration/test_partial_fills.py",  # legacy regression suite
    "tests/unit/gpt_trader/monitoring/test_health_checks.py",  # extensive health matrix
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_permissions.py",  # permissions coverage table
    "tests/unit/gpt_trader/persistence/test_config_store.py",  # config persistence contract coverage
    "tests/unit/gpt_trader/utilities/test_datetime_helpers.py",  # datetime edge-case coverage
    "tests/unit/gpt_trader/utilities/test_performance_monitoring_core.py",  # performance collector matrix
    "tests/unit/gpt_trader/utilities/test_trading_operations_core.py",  # trading ops regression suite
    "tests/unit/gpt_trader/orchestration/test_execution_runtime_guards.py",  # guardrail regression suite
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_websocket.py",  # websocket integration matrix
    "tests/unit/gpt_trader/features/live_trade/test_coinbase_pnl.py",  # pnl engine behavioural coverage
    "tests/unit/gpt_trader/utilities/test_console_logging_functions.py",  # logging formatting matrix
    "tests/unit/gpt_trader/types/test_trading.py",  # trading dataclass behaviours
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_market_data.py",  # market data coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_system.py",  # system integration harness
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_accounts.py",  # account snapshot + treasury surface coverage
    "tests/unit/gpt_trader/errors/test_base.py",  # error taxonomy coverage
    "tests/unit/gpt_trader/monitoring/test_configuration_guardian.py",  # guardian behaviour matrix
    "tests/unit/gpt_trader/utilities/test_trading_operations_integration.py",  # trading ops integration harness
    "tests/unit/gpt_trader/utilities/test_async_utils_core.py",  # async helper regression suite
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_trading.py",  # trading API coverage
    "tests/unit/gpt_trader/logging/test_setup.py",  # logging stack regression suite
    "tests/unit/gpt_trader/validation/test_calculation_validator.py",  # calculation validator coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_market_data_service.py",  # market data service contract
    "tests/unit/gpt_trader/features/data/test_data_module.py",  # 295 lines: comprehensive data module contract with storage/cache stubs
    "tests/unit/gpt_trader/security/security_validator/test_order_validation.py",  # 335 lines: full order validation matrix across pricing/limits/error paths
    "tests/unit/gpt_trader/security/security_validator/test_rate_limiting.py",  # 305 lines: rate limiter escalation, blocking, and reset regression suite
    "tests/unit/gpt_trader/security/security_validator/test_suspicious_activity.py",  # 294 lines: suspicious activity detection heuristics and alert coverage
    "tests/unit/gpt_trader/security/security_validator/test_symbol_validation.py",  # 243 lines: comprehensive symbol/blocklist validation cases
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_contract_suite.py",  # REST contract suite covering retries, fallbacks, and error surfaces
    "tests/unit/gpt_trader/features/live_trade/strategies/shared/test_shared_helpers.py",  # shared strategy helper permutations across venues
    "tests/unit/gpt_trader/features/position_sizing/test_position_sizing.py",  # covers regime, strategy, and kelly sizing permutations in one harness
    "tests/unit/gpt_trader/cli/test_commands_orders.py",  # 254 lines: comprehensive CLI orders command coverage
    "tests/unit/gpt_trader/utilities/test_console_logging_core.py",  # 336 lines: console logging contract and fallback behavior matrix
    "tests/unit/gpt_trader/utilities/test_logging_patterns.py",  # 494 lines: extensive structured logging patterns and decorator coverage
    "tests/unit/gpt_trader/features/analyze/test_analyze_strategies.py",  # 280 lines: comprehensive strategy signal matrix (MA/momentum/reversion/volatility/breakout)
    "tests/property/test_coinbase_invariants.py",  # property-based coverage for invariants across stochastic market scenarios
    "tests/unit/gpt_trader/monitoring/test_guard_manager_e2e.py",  # end-to-end guard orchestration scenarios spanning async workflows
    "tests/unit/gpt_trader/monitoring/test_system_logger.py",  # structured logging sink contract covering emit and fallback paths
    "tests/unit/gpt_trader/orchestration/test_orchestration_async.py",  # async orchestration behaviours with concurrency guard rails
    "tests/unit/gpt_trader/persistence/test_json_file_store_contract.py",  # persistence contract with locking, rotation, and corruption scenarios
}

SLEEP_ALLOWLIST = {
    "tests/unit/gpt_trader/utilities/test_performance_monitoring_advanced.py",
    "tests/unit/gpt_trader/utilities/test_performance_monitoring_core.py",
    "tests/unit/gpt_trader/persistence/test_json_file_store_contract.py",  # uses real sleep to validate file system based locking semantics
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
