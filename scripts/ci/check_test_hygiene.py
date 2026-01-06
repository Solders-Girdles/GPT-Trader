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
    "tests/unit/gpt_trader/utilities/test_async_utils_advanced.py",  # legacy async helpers coverage  # naming: allow
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
    "tests/unit/gpt_trader/utilities/test_async_utils_core.py",  # async helper regression suite  # naming: allow
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
    "tests/unit/gpt_trader/orchestration/execution/test_state_collection.py",  # state collection aggregation matrix
    "tests/unit/gpt_trader/security/test_ip_allowlist_enforcer.py",  # IP allowlist enforcement scenarios
    "tests/unit/gpt_trader/orchestration/execution/test_order_submission.py",  # order submission flow coverage
    "tests/unit/gpt_trader/config/test_runtime_settings_utils.py",  # runtime settings validation matrix  # naming: allow
    "tests/unit/gpt_trader/preflight/test_context.py",  # preflight context scenarios
    "tests/unit/gpt_trader/security/test_request_validator.py",  # request validation coverage
    "tests/unit/gpt_trader/orchestration/execution/test_validation.py",  # execution validation matrix
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_transports_coverage.py",  # transport layer coverage
    "tests/unit/gpt_trader/security/test_input_sanitizer.py",  # input sanitization edge cases
    "tests/contract/test_coinbase_api_contract.py",  # API contract compliance suite
    "tests/unit/gpt_trader/preflight/test_core.py",  # preflight core validation
    "tests/unit/gpt_trader/orchestration/execution/test_guards.py",  # execution guard scenarios
    "tests/unit/gpt_trader/orchestration/test_symbols.py",  # symbol management coverage
    "tests/property/test_fee_invariants.py",  # property-based fee calculation invariants with comprehensive edge cases
    "tests/property/test_liquidation_invariants.py",  # property-based liquidation safety invariants across margin scenarios
    "tests/property/test_margin_invariants.py",  # property-based margin requirement invariants with leverage permutations
    "tests/property/test_pnl_invariants.py",  # property-based PnL calculation invariants covering all position/entry combinations
    "tests/unit/gpt_trader/preflight/test_checks_connectivity.py",  # comprehensive connectivity check scenarios with retry/timeout coverage
    "tests/unit/gpt_trader/orchestration/execution/test_broker_executor.py",  # broker communication contract and async handling coverage
    "tests/unit/gpt_trader/orchestration/execution/test_order_event_recorder.py",  # order event recording and telemetry coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_orders.py",  # comprehensive REST orders mixin coverage
    "tests/unit/gpt_trader/features/live_trade/test_risk_manager.py",  # LiveRiskManager risk validation, volatility breakers, daily PnL tracking
    "tests/unit/gpt_trader/orchestration/test_live_execution_engine.py",  # LiveExecutionEngine orchestration, order flow, risk validation
    "tests/unit/gpt_trader/orchestration/test_derivatives_discovery.py",  # derivatives eligibility discovery and safety gating
    "tests/unit/gpt_trader/orchestration/test_account_telemetry.py",  # account telemetry collection and publishing
    "tests/unit/gpt_trader/features/live_trade/engines/test_telemetry_streaming.py",  # WebSocket streaming telemetry with async/threading patterns
    "tests/unit/gpt_trader/features/live_trade/engines/test_telemetry_health.py",  # telemetry health check and mark extraction coverage
    "tests/unit/gpt_trader/features/live_trade/engines/test_strategy_engine.py",  # strategy engine dynamic sizing, position tracking, risk format validation
    "tests/unit/gpt_trader/backtesting/simulation/test_funding_tracker.py",  # comprehensive funding rate tracking with perpetuals scenarios
    "tests/unit/gpt_trader/backtesting/simulation/test_fee_calculator.py",  # fee tier matrix coverage across volume brackets
    "tests/unit/gpt_trader/backtesting/simulation/test_simulated_broker.py",  # broker protocol contract with order lifecycle scenarios
    "tests/unit/gpt_trader/backtesting/simulation/test_fill_model.py",  # order fill simulation covering slippage/liquidity/partial fills
    "tests/unit/gpt_trader/validation/test_composite_validators.py",  # composite validator chain permutations
    "tests/unit/gpt_trader/validation/test_config_validators.py",  # configuration validation matrix
    "tests/unit/gpt_trader/validation/test_data_validators.py",  # data validation edge cases and error handling
    "tests/unit/gpt_trader/orchestration/strategy_orchestrator/test_orchestrator.py",  # strategy orchestration routing and signal aggregation
    "tests/unit/gpt_trader/orchestration/trading_bot/test_bot.py",  # trading bot lifecycle and state management scenarios
    # Coverage expansion tests (C1 task)
    "tests/unit/gpt_trader/features/live_trade/test_indicators.py",  # indicator calculation coverage
    "tests/unit/gpt_trader/utilities/performance/test_timing.py",  # timing utilities coverage
    "tests/unit/gpt_trader/backtesting/metrics/test_risk.py",  # risk metrics calculation matrix
    "tests/unit/gpt_trader/errors/test_error_patterns.py",  # error pattern decorator coverage
    "tests/unit/gpt_trader/backtesting/validation/test_decision_logger.py",  # decision logging scenarios
    "tests/unit/gpt_trader/monitoring/test_alert_types.py",  # alert type coverage
    "tests/unit/gpt_trader/app/test_health_server.py",  # health server endpoint coverage
    "tests/unit/gpt_trader/validation/test_rules.py",  # validation rule matrix
    "tests/unit/gpt_trader/monitoring/daily_report/test_analytics.py",  # daily report analytics
    "tests/unit/gpt_trader/backtesting/metrics/test_statistics.py",  # trade statistics coverage
    "tests/unit/gpt_trader/backtesting/engine/test_clock.py",  # simulation clock scenarios
    "tests/unit/gpt_trader/backtesting/engine/test_bar_runner.py",  # bar runner orchestration
    "tests/unit/gpt_trader/monitoring/daily_report/test_models.py",  # daily report model coverage
    "tests/unit/gpt_trader/backtesting/metrics/test_report.py",  # backtest report generation
    "tests/unit/gpt_trader/backtesting/validation/test_validator.py",  # validation scenarios
    "tests/unit/gpt_trader/features/live_trade/strategies/test_perps_baseline.py",  # baseline strategy coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_websocket_mixin.py",  # WebSocket streaming mixin with threading patterns
    "tests/integration/test_optimize_feature.py",  # optimization feature integration coverage
    "tests/unit/gpt_trader/features/optimize/test_walk_forward.py",  # walk-forward analysis comprehensive scenarios
    "tests/unit/gpt_trader/monitoring/notifications/test_notification_service.py",  # notification backend coverage
    "tests/unit/gpt_trader/monitoring/test_heartbeat.py",  # heartbeat monitoring scenarios
    "tests/unit/gpt_trader/monitoring/test_status_reporter.py",  # status reporting coverage
    "tests/integration/test_funding_pnl_integration.py",  # funding and PnL calculation scenarios
    "tests/unit/gpt_trader/features/live_trade/strategies/mean_reversion/test_strategy.py",  # comprehensive Z-Score, volatility targeting, and exit logic coverage
    "tests/unit/gpt_trader/persistence/test_orders_store.py",  # comprehensive order persistence contract with lifecycle, recovery, and integrity validation
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_ws_events.py",  # WebSocket event dispatcher testing with typed handlers and dispatcher routing
    "tests/unit/gpt_trader/orchestration/test_bootstrap.py",  # comprehensive bootstrap module coverage with profile loading and container initialization
    "tests/unit/gpt_trader/features/live_trade/strategies/test_stateful_indicators.py",  # Welford algorithm accuracy, numerical stability, and reset behavior
    "tests/unit/gpt_trader/orchestration/configuration/test_profiles.py",  # YAML-first profile loading with validation and fallback testing
    # Phase 2 pain points remediation: critical service coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_order_service.py",  # OrderService protocol coverage: place/cancel/list/get orders
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_portfolio_service.py",  # PortfolioService coverage: balances, positions, INTX/CFM ops
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_product_service.py",  # ProductService coverage: products, quotes, candles, tickers
    "tests/unit/gpt_trader/orchestration/test_hybrid_paper_broker.py",  # HybridPaperBroker paper trading simulation: orders, positions, market data
    # Optimization and CLI test suites
    "tests/unit/gpt_trader/features/optimize/objectives/test_constraints.py",  # constraint objective validation matrix
    "tests/unit/gpt_trader/features/optimize/objectives/test_single.py",  # single objective validation matrix
    "tests/unit/gpt_trader/cli/commands/optimize/test_commands.py",  # CLI command integration tests
    "tests/unit/gpt_trader/cli/commands/optimize/test_config_loader.py",  # config loader edge cases
    "tests/unit/gpt_trader/cli/test_response.py",  # CLI response formatting coverage
    # Intelligence and Strategy Dev tests (large modules)
    "tests/unit/gpt_trader/features/intelligence/ensemble/test_voting.py",
    "tests/unit/gpt_trader/features/intelligence/ensemble/test_orchestrator.py",
    "tests/unit/gpt_trader/features/intelligence/ensemble/test_adaptive.py",
    "tests/unit/gpt_trader/features/intelligence/backtesting/test_batch_regime.py",
    "tests/unit/gpt_trader/features/intelligence/sizing/test_position_sizer.py",
    "tests/unit/gpt_trader/features/intelligence/regime/test_detector.py",
    "tests/unit/gpt_trader/features/strategy_dev/monitor/test_metrics.py",
    "tests/unit/gpt_trader/features/strategy_dev/monitor/test_alerts.py",
    "tests/unit/gpt_trader/features/strategy_dev/lab/test_experiment.py",
    "tests/unit/gpt_trader/tui/test_log_manager.py",  # comprehensive TUI log handler coverage including threading, markup handling, and error scenarios
    # TUI Phase 3.5 comprehensive test suites
    "tests/unit/gpt_trader/tui/services/test_action_dispatcher.py",  # action dispatcher routing and handler coverage
    "tests/unit/gpt_trader/tui/test_widget_interactions.py",  # widget interaction and integration scenarios
    "tests/unit/gpt_trader/tui/state_management/test_validators.py",  # state validator permutation matrix
    "tests/unit/gpt_trader/tui/mixins/test_event_handlers.py",  # mixin event handler coverage across components
    "tests/unit/gpt_trader/tui/services/test_credential_validator.py",  # credential validation flow comprehensive coverage
    "tests/unit/gpt_trader/tui/test_screen_flows.py",  # screen navigation and lifecycle scenarios
    "tests/unit/gpt_trader/tui/state_management/test_delta_updater.py",  # delta state update comprehensive scenarios
    "tests/unit/gpt_trader/tui/test_events.py",  # custom event type coverage and dispatch patterns
    "tests/unit/gpt_trader/tui/services/test_mode_service.py",  # mode service state transitions
    "tests/unit/gpt_trader/tui/widgets/test_positions.py",  # positions widget comprehensive scenarios
    "tests/unit/gpt_trader/tui/test_thresholds.py",  # unified threshold system test coverage (status levels, risk, confidence)
    "tests/unit/gpt_trader/tui/utilities/test_table_formatting.py",  # table formatting utilities with timestamp parsing, sorting, clipboard, and cell formatting
    "tests/unit/gpt_trader/tui/services/test_alert_manager.py",  # alert system comprehensive coverage with execution health alerts
    "tests/unit/gpt_trader/tui/test_staleness_helpers.py",  # staleness and execution health banner comprehensive testing
    "tests/unit/gpt_trader/tui/services/test_execution_telemetry.py",  # execution telemetry metrics collection coverage
    "tests/unit/gpt_trader/tui/services/test_onboarding_service.py",  # onboarding service state machine and persistence
    "tests/unit/gpt_trader/tui/widgets/test_strategy.py",  # strategy widget comprehensive display scenarios
    "tests/unit/gpt_trader/tui/widgets/test_risk_detail_modal.py",  # risk detail modal comprehensive scenarios
    "tests/unit/gpt_trader/features/live_trade/strategies/hybrid/test_base.py",  # hybrid strategy base comprehensive coverage
    "tests/unit/gpt_trader/features/live_trade/strategies/hybrid/test_types.py",  # hybrid strategy types validation matrix
    "tests/unit/gpt_trader/features/live_trade/signals/test_orderbook_imbalance.py",  # orderbook imbalance signal comprehensive scenarios
    "tests/unit/gpt_trader/features/live_trade/signals/test_spread.py",  # spread signal comprehensive scenarios
    "tests/unit/gpt_trader/features/live_trade/execution/test_router.py",  # execution router comprehensive flow coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_api_resilience.py",  # API resilience patterns comprehensive coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_circuit_breaker.py",  # circuit breaker state machine coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_metrics.py",  # API metrics collection comprehensive scenarios
    "tests/tui/test_state_logic.py",  # TUI state logic comprehensive scenarios
    "tests/unit/gpt_trader/tui/services/test_trading_stats_service.py",  # trading stats FIFO matching comprehensive scenarios
    "tests/unit/gpt_trader/orchestration/configuration/risk/test_cfm_risk_config.py",  # CFM risk config validation matrix
    "tests/unit/gpt_trader/tui/test_snapshots.py",  # TUI snapshot and layout guardrail comprehensive scenarios
    "tests/unit/gpt_trader/tui/widgets/test_system_monitor.py",  # system monitor widget comprehensive state scenarios
    # TUI reliability/fault-injection tests
    "tests/integration/test_tui_degraded_paths.py",  # TUI degraded state handling with partial/missing data scenarios
    "tests/unit/gpt_trader/tui/widgets/test_account.py",  # account widget partial state and signature caching tests
    "tests/unit/gpt_trader/tui/widgets/test_position_card.py",  # position card widget resilience with missing data
    "tests/integration/test_validation_escalation.py",  # validation escalation integration flow coverage
    "tests/integration/test_container_lifecycle.py",  # container lifecycle and service registration coverage
}

SLEEP_ALLOWLIST = {
    "tests/unit/gpt_trader/utilities/test_performance_monitoring_advanced.py",
    "tests/unit/gpt_trader/utilities/test_performance_monitoring_core.py",
    "tests/unit/gpt_trader/persistence/test_json_file_store_contract.py",  # uses real sleep to validate file system based locking semantics
    "tests/unit/gpt_trader/features/live_trade/engines/test_telemetry_streaming.py",  # uses time.sleep for run_in_executor cancellation test
    "tests/unit/gpt_trader/utilities/performance/test_timing.py",  # timing utility coverage requires real sleep for precision tests
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_websocket_mixin.py",  # uses time.sleep for WebSocket thread synchronization
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
