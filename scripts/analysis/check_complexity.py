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
    "tests/unit/bot_v2/features/live_trade/risk/test_pre_trade_checks_coverage.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_websocket_enhanced.py",

    "tests/unit/bot_v2/features/live_trade/risk/test_live_risk_manager_coverage.py",
    "tests/unit/bot_v2/orchestration/test_live_execution.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_market_data_service_coverage.py",
    "tests/integration/bot_v2/features/live_trade/test_reconciliation_integration.py",
    "tests/unit/bot_v2/orchestration/coordinators/test_execution.py",
    "tests/unit/bot_v2/features/live_trade/test_advanced_execution.py",
    "tests/unit/bot_v2/orchestration/coordinators/test_runtime.py",
    "tests/unit/bot_v2/orchestration/coordinators/test_execution_enhanced.py",
    "tests/unit/bot_v2/orchestration/test_runtime_coordinator.py",
    "tests/unit/bot_v2/monitoring/test_health_checks.py",
    "tests/integration/bot_v2/features/live_trade/test_market_condition_integration.py",
    "tests/integration/bot_v2/features/live_trade/conftest.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/client/test_base.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/rest/test_contract_suite.py",
    "tests/unit/bot_v2/features/live_trade/risk/test_runtime_monitoring_coverage.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_websocket_handler_coverage.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_specs_quantization.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_helpers.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/rest/test_base.py",
    "tests/unit/bot_v2/features/live_trade/test_coinbase_pnl.py",
    "tests/unit/bot_v2/features/live_trade/risk/conftest.py",
    "tests/unit/bot_v2/orchestration/perps_bot/test_config_and_streaming.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_market_data.py",
    "tests/unit/bot_v2/orchestration/test_telemetry_coordinator.py",
    "tests/unit/bot_v2/features/live_trade/test_risk_core.py",
    "tests/unit/bot_v2/orchestration/test_orchestration_async.py",
    "tests/unit/bot_v2/data_providers/test_coinbase_provider.py",
    "tests/unit/bot_v2/monitoring/test_guard_manager_e2e.py",
    "tests/unit/bot_v2/orchestration/perps_bot/test_lifecycle.py",
    "tests/unit/bot_v2/orchestration/test_strategy_coordinator.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_transports_coverage.py",
    "tests/unit/bot_v2/orchestration/test_system_monitor_enhanced.py",
    "tests/unit/bot_v2/orchestration/perps_bot/test_coordinator_integration.py",
    "tests/unit/bot_v2/orchestration/test_metrics_publisher_enhanced.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_market_data_service.py",
    "tests/integration/bot_v2/features/live_trade/test_circuit_breaker_integration.py",
    "tests/unit/bot_v2/utilities/test_logging_patterns.py",
    "tests/unit/bot_v2/orchestration/test_system_monitor.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_websocket_coverage.py",
    "tests/unit/bot_v2/monitoring/test_system_logger.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/conftest.py",
    "tests/integration/test_coinbase_auth_smoke.py",
    "tests/shared/mock_brokers.py",
    "tests/unit/bot_v2/orchestration/coordinators/telemetry/test_telemetry_lifecycle.py",
    "tests/unit/bot_v2/features/analyze/test_analyze_strategies.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_websocket.py",
    "tests/unit/bot_v2/orchestration/coordinators/test_error_handling.py",
    "tests/unit/bot_v2/orchestration/coordinators/test_metric_emission.py",
    "tests/unit/bot_v2/orchestration/execution/test_validation_pre_trade.py",
    "tests/support/deterministic_broker.py",
    "tests/unit/bot_v2/utilities/test_async_utils_core.py",
    "tests/unit/bot_v2/utilities/test_async_utils_advanced.py",
    "tests/unit/bot_v2/orchestration/test_state_manager.py",
    "tests/unit/bot_v2/orchestration/execution/state_collection/test_error_handling.py",
    "tests/unit/bot_v2/persistence/test_event_store.py",
    "tests/unit/app/test_container.py",
    "tests/property/test_coinbase_invariants.py",
    "tests/unit/bot_v2/orchestration/strategy_orchestrator/test_orch_main_edge_cases.py",
    "tests/fixtures/behavioral/helpers.py",
    "tests/fixtures/behavioral/market_data.py",
    "tests/unit/bot_v2/features/position_sizing/test_position_sizing.py",
    "tests/unit/bot_v2/orchestration/configuration/test_core.py",
    "tests/unit/bot_v2/orchestration/execution/state_collection/test_state_diff.py",
    "tests/unit/bot_v2/utilities/test_performance_monitoring_advanced.py",
    "tests/unit/bot_v2/orchestration/test_runtime_settings_utils.py",
    "tests/unit/bot_v2/persistence/test_json_file_store_contract.py",
    "tests/unit/bot_v2/orchestration/perps_bot/conftest.py",
    "tests/unit/bot_v2/types/test_trading.py",
    "tests/integration/bot_v2/features/live_trade/test_broker_error_propagation_simple.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_accounts.py",
    "tests/unit/bot_v2/test_broker_behavioral_contract.py",
    "tests/unit/bot_v2/utilities/test_trading_operations_integration.py",
    "tests/unit/bot_v2/logging/test_setup.py",
    "tests/unit/orchestration/test_state_manager_core.py",
    "tests/unit/bot_v2/orchestration/strategy_orchestrator/test_orch_main_execution.py",
    "tests/unit/bot_v2/orchestration/execution_coordinator/test_decision_execution.py",
    "tests/unit/bot_v2/features/live_trade/test_risk_runtime.py",
    "tests/unit/bot_v2/utilities/test_trading_operations_core.py",
    "tests/unit/bot_v2/monitoring/test_two_person_rule.py",
    "tests/fixtures/factories.py",
    "tests/unit/bot_v2/persistence/test_config_store.py",
    "tests/unit/bot_v2/utilities/test_console_logging_core.py",
    "tests/unit/bot_v2/features/optimize/test_decision_logger.py",
    "tests/unit/bot_v2/security/security_validator/test_order_validation.py",
    "tests/unit/bot_v2/orchestration/test_perps_bot.py",
    "tests/unit/bot_v2/orchestration/coordinators/telemetry/test_telemetry_streaming.py",
    "tests/integration/test_reduce_only_state_manager_integration.py",
    "tests/unit/bot_v2/orchestration/test_execution_runtime_guards.py",
    "tests/unit/bot_v2/orchestration/order_reconciler/test_reconcile_flow.py",
    "tests/fixtures/product_factory.py",
    "tests/unit/bot_v2/features/brokerages/coinbase/test_coinbase_permissions.py",
    "tests/unit/bot_v2/utilities/test_import_utils.py",
    "tests/unit/bot_v2/orchestration/coordinators/test_simple_telemetry.py",
    "tests/integration/test_composition_root.py",
    "tests/unit/bot_v2/orchestration/execution/state_collection/test_collect_snapshots.py",
    "tests/unit/bot_v2/utilities/test_performance_monitoring_core.py",
    "tests/unit/bot_v2/monitoring/test_configuration_guardian.py",
    "tests/unit/bot_v2/security/security_validator/test_rate_limiting.py",
    "tests/unit/bot_v2/utilities/test_console_logging_functions.py",
}


def count_lines(file_path: Path) -> int:
    """Count lines in a file, ignoring empty lines and comments might be too complex for now,
    so we just count total lines as a proxy for complexity."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
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
