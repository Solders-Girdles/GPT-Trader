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
ALLOWLIST = {
    "tests/unit/gpt_trader/logging/test_runtime_helpers.py",  # runtime logging helper coverage
    "tests/unit/gpt_trader/monitoring/test_health_signals.py",  # health signal model contract coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_permissions.py",  # permissions coverage table
    "tests/unit/gpt_trader/utilities/test_datetime_helpers.py",  # datetime edge-case coverage
    "tests/unit/gpt_trader/utilities/test_console_logging_functions.py",  # logging formatting matrix
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_accounts.py",  # account snapshot + treasury surface coverage
    "tests/unit/gpt_trader/errors/test_base.py",  # error taxonomy coverage
    "tests/unit/gpt_trader/validation/test_calculation_validator.py",  # calculation validator coverage
    "tests/unit/gpt_trader/features/data/test_data_module.py",  # 295 lines: comprehensive data module contract with storage/cache stubs
    "tests/unit/gpt_trader/security/security_validator/test_rate_limiting.py",  # 305 lines: rate limiter escalation, blocking, and reset regression suite
    "tests/unit/gpt_trader/security/security_validator/test_suspicious_activity.py",  # 294 lines: suspicious activity detection heuristics and alert coverage
    "tests/unit/gpt_trader/security/security_validator/test_symbol_validation.py",  # 243 lines: comprehensive symbol/blocklist validation cases
    "tests/unit/gpt_trader/cli/test_commands_orders.py",  # 254 lines: comprehensive CLI orders command coverage
    "tests/unit/gpt_trader/utilities/test_console_logging_core.py",  # 336 lines: console logging contract and fallback behavior matrix
    "tests/unit/gpt_trader/security/test_ip_allowlist_enforcer.py",  # IP allowlist enforcement scenarios
    "tests/unit/gpt_trader/preflight/test_context.py",  # preflight context scenarios
    "tests/unit/gpt_trader/security/test_request_validator.py",  # request validation coverage
    "tests/unit/gpt_trader/security/test_input_sanitizer.py",  # input sanitization edge cases
    "tests/contract/test_coinbase_api_contract.py",  # API contract compliance suite
    "tests/unit/gpt_trader/preflight/test_core.py",  # preflight core validation
    "tests/unit/gpt_trader/backtesting/simulation/test_fee_calculator.py",  # fee tier matrix coverage across volume brackets
    "tests/unit/gpt_trader/validation/test_composite_validators.py",  # composite validator chain permutations
    "tests/unit/gpt_trader/validation/test_config_validators.py",  # configuration validation matrix
    "tests/unit/gpt_trader/validation/test_data_validators.py",  # data validation edge cases and error handling
    # Coverage expansion tests (C1 task)
    "tests/unit/gpt_trader/features/live_trade/test_indicators.py",  # indicator calculation coverage
    "tests/unit/gpt_trader/errors/test_error_patterns.py",  # error pattern decorator coverage
    "tests/unit/gpt_trader/monitoring/test_alert_types.py",  # alert type coverage
    "tests/unit/gpt_trader/app/test_health_server.py",  # health server endpoint coverage
    "tests/unit/gpt_trader/backtesting/engine/test_clock.py",  # simulation clock scenarios
    "tests/unit/gpt_trader/features/optimize/test_walk_forward.py",  # walk-forward analysis comprehensive scenarios
    "tests/unit/gpt_trader/monitoring/notifications/test_notification_service.py",  # notification backend coverage
    "tests/unit/gpt_trader/monitoring/test_heartbeat.py",  # heartbeat monitoring scenarios
    "tests/unit/gpt_trader/persistence/test_orders_store.py",  # comprehensive order persistence contract with lifecycle, recovery, and integrity validation
    "tests/unit/gpt_trader/app/test_bootstrap.py",  # comprehensive bootstrap module coverage with profile loading and container initialization
    # Phase 2 pain points remediation: critical service coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/rest/test_portfolio_service.py",  # PortfolioService coverage: balances, positions, INTX/CFM ops
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
    # TUI Phase 3.5 comprehensive test suites
    "tests/unit/gpt_trader/tui/state_management/test_validators.py",  # state validator permutation matrix
    "tests/unit/gpt_trader/tui/services/test_mode_service.py",  # mode service state transitions
    "tests/unit/gpt_trader/tui/services/test_onboarding_service.py",  # onboarding service state machine and persistence
    "tests/unit/gpt_trader/tui/widgets/test_strategy.py",  # strategy widget comprehensive display scenarios
    "tests/unit/gpt_trader/tui/widgets/test_risk_detail_modal.py",  # risk detail modal comprehensive scenarios
    "tests/unit/gpt_trader/features/live_trade/strategies/hybrid/test_types.py",  # hybrid strategy types validation matrix
    "tests/unit/gpt_trader/features/live_trade/signals/test_orderbook_imbalance.py",  # orderbook imbalance signal comprehensive scenarios
    "tests/unit/gpt_trader/features/live_trade/execution/test_router.py",  # execution router comprehensive flow coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_api_resilience.py",  # API resilience patterns comprehensive coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_circuit_breaker.py",  # circuit breaker state machine coverage
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_metrics.py",  # API metrics collection comprehensive scenarios
    "tests/unit/gpt_trader/features/live_trade/risk/test_cfm_risk_config.py",  # CFM risk config validation matrix
    # TUI reliability/fault-injection tests
    "tests/unit/gpt_trader/tui/widgets/test_account.py",  # account widget partial state and signature caching tests
    "tests/unit/gpt_trader/tui/widgets/test_position_card.py",  # position card widget resilience with missing data
    "tests/integration/test_validation_escalation.py",  # validation escalation integration flow coverage
    "tests/integration/test_container_lifecycle.py",  # container lifecycle and service registration coverage
    "tests/unit/gpt_trader/tui/screens/test_strategy_detail_screen.py",  # signal detail formatting, backtest display, delta formatting, entry/exit badges
    "tests/unit/gpt_trader/config/test_bot_config_env.py",  # BotConfig env/profile wiring: RISK_* prefixes, daily loss limit, reduce-only enforcement, mark staleness
    "tests/unit/gpt_trader/monitoring/test_metrics_collector.py",  # metrics collector coverage: counters, gauges, histograms with labels
    # Architecture migration and observability tests
    "tests/unit/gpt_trader/tui/test_app.py",  # TUI app lifecycle, mode switching, and refresh scheduling
    "tests/integration/test_crash_recovery.py",  # crash recovery scenarios and state restoration
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_ws_reconnection.py",  # WebSocket reconnection reliability scenarios
    "tests/unit/gpt_trader/features/live_trade/test_state_recovery.py",  # state recovery and persistence contract
    # Pre-existing TUI and feature test suites
    "tests/unit/gpt_trader/tui/widgets/test_performance_dashboard.py",  # performance dashboard widget scenarios
    "tests/unit/gpt_trader/tui/services/test_performance_service.py",  # performance service comprehensive coverage
    "tests/unit/gpt_trader/tui/test_trade_matcher.py",  # trade matcher FIFO comprehensive scenarios
    "tests/unit/gpt_trader/tui/widgets/test_cfm_balance.py",  # CFM balance widget scenarios
    "tests/unit/gpt_trader/features/live_trade/strategies/test_ensemble_profile.py",  # ensemble profile comprehensive coverage
    "tests/unit/gpt_trader/features/live_trade/signals/test_vwap.py",  # VWAP signal comprehensive scenarios
    "tests/unit/gpt_trader/app/config/test_profile_loader.py",  # profile loader comprehensive coverage
    "tests/unit/gpt_trader/app/containers/test_risk_validation.py",  # risk validation container scenarios
    "tests/unit/gpt_trader/backtesting/validation/test_decision_logger_logging_and_retrieval.py",  # decision logger modularized coverage (242 lines)
    "tests/unit/gpt_trader/features/brokerages/coinbase/test_transports_real_transport.py",  # transport layer coverage across real transport scenarios (272 lines)
}

SLEEP_ALLOWLIST = {
    "tests/unit/gpt_trader/utilities/performance/test_timing.py",  # timing utility coverage requires real sleep for precision tests
    "tests/unit/gpt_trader/features/brokerages/coinbase/client/test_response_cache.py",  # TTL-based cache expiration requires real time elapsed
    "tests/unit/gpt_trader/tui/services/test_performance_service.py",  # uses time.sleep for rate limiting window tests
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
        line_count = text.count("\n") + 1

        if rel_str.startswith("tests/unit/") and not rel_str.startswith(UNIT_ALLOWED_PREFIXES):
            problems.append(
                f"{rel} is a unit test outside the supported layout. Place unit tests under `tests/unit/gpt_trader/` (or `tests/unit/scripts/`, `tests/unit/support/`)."
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
