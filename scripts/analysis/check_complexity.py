#!/usr/bin/env python3
"""
Check file complexity (line count) in the codebase.
Enforces a maximum line count for test files to prevent monolithic test suites.
"""

import argparse
import os
import sys
from pathlib import Path

# Files to ignore (legacy technical debt)
IGNORE_LIST = {
    "tests/unit/gpt_trader/features/live_trade/risk/test_pre_trade_checks_coverage.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_websocket_enhanced.py",
    "tests/unit/gpt_trader/features/live_trade/risk/test_live_risk_manager_coverage.py",
    "tests/unit/gpt_trader/orchestration/test_live_execution.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_market_data_service_coverage.py",
    "tests/integration/gpt_trader/features/live_trade/test_reconciliation_integration.py",
    "tests/unit/gpt_trader/orchestration/coordinators/test_execution.py",
    "tests/unit/gpt_trader/features/live_trade/test_advanced_execution.py",
    "tests/unit/gpt_trader/orchestration/coordinators/test_runtime.py",
    "tests/unit/gpt_trader/orchestration/coordinators/test_execution_enhanced.py",
    "tests/unit/gpt_trader/orchestration/test_runtime_coordinator.py",
    "tests/unit/gpt_trader/monitoring/test_health_checks.py",
    "tests/integration/gpt_trader/features/live_trade/test_market_condition_integration.py",
    "tests/integration/gpt_trader/features/live_trade/conftest.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_base.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_contract_suite.py",
    "tests/unit/gpt_trader/features/live_trade/risk/test_runtime_monitoring_coverage.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_websocket_handler_coverage.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_specs_quantization.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_helpers.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_base.py",
    "tests/unit/gpt_trader/features/live_trade/test_coinbase_pnl.py",
    "tests/unit/gpt_trader/features/live_trade/risk/conftest.py",
    "tests/unit/gpt_trader/orchestration/perps_bot/test_config_and_streaming.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_market_data.py",
    "tests/unit/gpt_trader/orchestration/test_telemetry_coordinator.py",
    "tests/unit/gpt_trader/features/live_trade/test_risk_core.py",
    "tests/unit/gpt_trader/orchestration/test_orchestration_async.py",
    "tests/unit/gpt_trader/data_providers/test_coinbase_provider.py",
    "tests/unit/gpt_trader/monitoring/test_guard_manager_e2e.py",
    "tests/unit/gpt_trader/orchestration/perps_bot/test_lifecycle.py",
    "tests/unit/gpt_trader/orchestration/test_strategy_coordinator.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_transports_coverage.py",
    "tests/unit/gpt_trader/orchestration/test_system_monitor_enhanced.py",
    "tests/unit/gpt_trader/orchestration/perps_bot/test_coordinator_integration.py",
    "tests/unit/gpt_trader/orchestration/test_metrics_publisher_enhanced.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_market_data_service.py",
    "tests/integration/gpt_trader/features/live_trade/test_circuit_breaker_integration.py",
    "tests/unit/gpt_trader/utilities/test_logging_patterns.py",
    "tests/unit/gpt_trader/orchestration/test_system_monitor.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_websocket_coverage.py",
    "tests/unit/gpt_trader/monitoring/test_system_logger.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/conftest.py",
    "tests/integration/test_coinbase_auth_smoke.py",
    "tests/shared/mock_brokers.py",
    "tests/unit/gpt_trader/orchestration/coordinators/telemetry/test_telemetry_lifecycle.py",
    "tests/unit/gpt_trader/features/analyze/test_analyze_strategies.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_websocket.py",
    "tests/unit/gpt_trader/orchestration/coordinators/test_error_handling.py",
    "tests/unit/gpt_trader/orchestration/coordinators/test_metric_emission.py",
    "tests/unit/gpt_trader/orchestration/execution/test_validation_pre_trade.py",
    "tests/support/deterministic_broker.py",
    "tests/unit/gpt_trader/utilities/test_async_utils_core.py",  # naming: allow
    "tests/unit/gpt_trader/utilities/test_async_utils_advanced.py",  # naming: allow
    "tests/unit/gpt_trader/orchestration/test_state_manager.py",
    "tests/unit/gpt_trader/orchestration/execution/state_collection/test_error_handling.py",
    "tests/unit/gpt_trader/persistence/test_event_store.py",
    "tests/unit/app/test_container.py",
    "tests/property/test_coinbase_invariants.py",
    "tests/unit/gpt_trader/orchestration/strategy_orchestrator/test_orch_main_edge_cases.py",
    "tests/fixtures/behavioral/helpers.py",
    "tests/fixtures/behavioral/market_data.py",
    "tests/unit/gpt_trader/features/position_sizing/test_position_sizing.py",
    "tests/unit/gpt_trader/orchestration/configuration/test_core.py",
    "tests/unit/gpt_trader/orchestration/execution/state_collection/test_state_diff.py",
    "tests/unit/gpt_trader/utilities/test_performance_monitoring_advanced.py",
    "tests/unit/gpt_trader/orchestration/test_runtime_settings_utils.py",  # naming: allow
    "tests/unit/gpt_trader/persistence/test_json_file_store_contract.py",
    "tests/unit/gpt_trader/orchestration/perps_bot/conftest.py",
    "tests/unit/gpt_trader/types/test_trading.py",
    "tests/integration/gpt_trader/features/live_trade/test_broker_error_propagation_simple.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_accounts.py",
    "tests/unit/gpt_trader/test_broker_behavioral_contract.py",
    "tests/unit/gpt_trader/utilities/test_trading_operations_integration.py",
    "tests/unit/gpt_trader/logging/test_setup.py",
    "tests/unit/orchestration/test_state_manager_core.py",
    "tests/unit/gpt_trader/orchestration/strategy_orchestrator/test_orch_main_execution.py",
    "tests/unit/gpt_trader/orchestration/execution_coordinator/test_decision_execution.py",
    "tests/unit/gpt_trader/features/live_trade/test_risk_runtime.py",
    "tests/unit/gpt_trader/utilities/test_trading_operations_core.py",
    "tests/unit/gpt_trader/monitoring/test_guard_manager_e2e.py",
    "tests/unit/gpt_trader/orchestration/perps_bot/test_lifecycle.py",
    "tests/unit/gpt_trader/orchestration/test_strategy_coordinator.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_transports_coverage.py",
    "tests/unit/gpt_trader/security/security_validator/test_rate_limiting.py",
    "tests/unit/gpt_trader/utilities/test_console_logging_functions.py",
    "tests/unit/gpt_trader/features/live_trade/engines/test_execution_enhanced.py",
    "tests/unit/gpt_trader/features/live_trade/engines/test_execution.py",
    "tests/unit/gpt_trader/features/live_trade/engines/test_runtime.py",
    "tests/unit/gpt_trader/features/live_trade/engines/telemetry/test_telemetry_lifecycle.py",
    "tests/unit/gpt_trader/features/live_trade/engines/test_error_handling.py",
    "tests/unit/gpt_trader/features/live_trade/engines/test_metric_emission.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_websocket_messaging.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_websocket_connection.py",
    "tests/unit/gpt_trader/features/live_trade/engines/telemetry/test_telemetry_streaming.py",
    "tests/unit/gpt_trader/features/live_trade/engines/test_simple_telemetry.py",
    "tests/unit/gpt_trader/features/live_trade/engines/telemetry/test_telemetry_async.py",
    "tests/unit/gpt_trader/orchestration/execution/test_order_submission.py",
    "tests/unit/gpt_trader/orchestration/execution/test_validation.py",
    "tests/unit/gpt_trader/orchestration/execution/test_guards.py",
    "tests/unit/gpt_trader/orchestration/execution/test_state_collection.py",
    "tests/unit/gpt_trader/security/security_validator/test_order_validation.py",
    "tests/unit/gpt_trader/config/test_runtime_settings_utils.py",  # naming: allow
    "tests/unit/gpt_trader/security/test_ip_allowlist_enforcer.py",
    "tests/unit/gpt_trader/utilities/test_console_logging_core.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_permissions.py",
    # C1 coverage expansion tests
    "tests/unit/gpt_trader/backtesting/simulation/test_funding_tracker.py",
    "tests/unit/gpt_trader/orchestration/test_live_execution_engine.py",
    "tests/unit/gpt_trader/backtesting/simulation/test_simulated_broker.py",
    "tests/unit/gpt_trader/backtesting/simulation/test_fill_model.py",
    "tests/unit/gpt_trader/orchestration/execution/test_order_event_recorder.py",
    "tests/unit/gpt_trader/features/live_trade/test_risk_manager.py",
    "tests/unit/gpt_trader/backtesting/validation/test_decision_logger.py",
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_orders.py",
    "tests/unit/gpt_trader/features/live_trade/engines/test_telemetry_streaming.py",
    "tests/unit/gpt_trader/backtesting/metrics/test_report.py",
    "tests/unit/gpt_trader/orchestration/test_derivatives_discovery.py",
    "tests/property/test_margin_invariants.py",
    "tests/unit/gpt_trader/orchestration/test_account_telemetry.py",
    "tests/property/test_pnl_invariants.py",
    "tests/unit/gpt_trader/preflight/test_checks_connectivity.py",
    "tests/unit/gpt_trader/validation/test_rules.py",
    "tests/property/test_liquidation_invariants.py",
    "tests/unit/gpt_trader/backtesting/engine/test_bar_runner.py",
    "tests/unit/gpt_trader/backtesting/metrics/test_risk.py",
    "tests/property/test_fee_invariants.py",
    "tests/unit/gpt_trader/orchestration/test_symbols.py",
    "tests/unit/gpt_trader/orchestration/strategy_orchestrator/test_orchestrator.py",
    "tests/unit/gpt_trader/backtesting/metrics/test_statistics.py",
    "tests/unit/gpt_trader/backtesting/validation/test_validator.py",
    "tests/unit/gpt_trader/monitoring/daily_report/test_analytics.py",
    "tests/unit/gpt_trader/features/live_trade/engines/test_telemetry_health.py",
    "tests/unit/gpt_trader/orchestration/trading_bot/test_bot.py",
    "tests/unit/gpt_trader/features/live_trade/strategies/test_perps_baseline.py",
    "tests/unit/gpt_trader/backtesting/simulation/test_fee_calculator.py",
    "tests/unit/gpt_trader/monitoring/daily_report/test_models.py",
    "tests/unit/gpt_trader/monitoring/test_alert_types.py",
    "tests/unit/gpt_trader/orchestration/execution/test_broker_executor.py",
}


def count_lines(file_path: Path) -> int:
    """Count lines in a file, ignoring empty lines and comments might be too complex for now,
    so we just count total lines as a proxy for complexity."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def check_complexity(
    search_path: str, max_lines: int, ignore_list: set[str]
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Scan files and check against max_lines.
    Returns (violations, ignored_violations).
    """
    violations = []
    ignored_violations = []

    root_path = Path(search_path)

    if not root_path.exists():
        print(f"Path not found: {search_path}")
        return [], []

    # Walk through the directory
    for root, _, files in os.walk(root_path):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = Path(root) / file
            # Get path relative to execution directory (repo root usually)
            try:
                rel_path = file_path.relative_to(os.getcwd())
            except ValueError:
                # If not relative to cwd, just use the path as is
                rel_path = file_path

            str_path = str(rel_path)

            line_count = count_lines(file_path)

            if line_count > max_lines:
                if str_path in ignore_list:
                    ignored_violations.append((str_path, line_count))
                else:
                    violations.append((str_path, line_count))

    return violations, ignored_violations


def main():
    parser = argparse.ArgumentParser(description="Check file complexity (line count).")
    parser.add_argument(
        "--path",
        type=str,
        default="tests",
        help="Directory to scan (default: tests)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=300,
        help="Maximum allowed lines per file (default: 300)",
    )

    args = parser.parse_args()

    print(f"Scanning '{args.path}' for files exceeding {args.max_lines} lines...")

    violations, ignored = check_complexity(args.path, args.max_lines, IGNORE_LIST)

    if ignored:
        print(f"\n⚠️  Ignored {len(ignored)} files (legacy debt):")
        for path, count in sorted(ignored, key=lambda x: x[1], reverse=True):
            print(f"  {path}: {count} lines")

    if violations:
        print(f"\n❌ Found {len(violations)} files exceeding limit:")
        for path, count in sorted(violations, key=lambda x: x[1], reverse=True):
            print(f"  {path}: {count} lines")
        print("\nPlease refactor these files to be smaller and more focused.")
        sys.exit(1)

    print("\n✅ All checked files are within complexity limits.")
    sys.exit(0)


if __name__ == "__main__":
    main()
