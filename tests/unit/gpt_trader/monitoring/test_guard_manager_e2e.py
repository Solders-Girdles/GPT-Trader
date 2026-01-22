"""Tests for guard manager defaults, guard states, and runtime behavior."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import GuardConfig, GuardStatus
from gpt_trader.monitoring.guards.builtins import (
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)
from gpt_trader.monitoring.guards.manager import (
    RuntimeGuardManager,
    create_default_runtime_guard_manager,
)


class TestDefaultGuardManagerCreation:
    def test_create_default_runtime_guard_manager(self):
        config = {
            "circuit_breakers": {
                "daily_loss_limit": 500.0,
                "stale_mark_seconds": 10.0,
                "error_threshold": 5.0,
                "position_timeout_seconds": 900.0,
            },
            "risk_management": {
                "max_drawdown_pct": 3.0,
            },
        }

        manager = create_default_runtime_guard_manager(config)

        expected_guards = {
            "daily_loss",
            "stale_mark",
            "error_rate",
            "position_stuck",
            "max_drawdown",
        }
        assert set(manager.guards.keys()) == expected_guards

        daily_loss_guard = manager.guards["daily_loss"]
        assert daily_loss_guard.config.threshold == 500.0
        assert daily_loss_guard.config.auto_shutdown is True

        stale_mark_guard = manager.guards["stale_mark"]
        assert stale_mark_guard.config.threshold == 10.0

        error_rate_guard = manager.guards["error_rate"]
        assert error_rate_guard.config.threshold == 5.0
        assert error_rate_guard.config.auto_shutdown is True

        position_stuck_guard = manager.guards["position_stuck"]
        assert position_stuck_guard.config.threshold == 900.0

        drawdown_guard = manager.guards["max_drawdown"]
        assert drawdown_guard.config.threshold == 3.0
        assert drawdown_guard.config.auto_shutdown is True

    def test_create_default_with_missing_config(self):
        manager = create_default_runtime_guard_manager({})

        daily_loss_guard = manager.guards["daily_loss"]
        assert daily_loss_guard.config.threshold == 500.0

        stale_mark_guard = manager.guards["stale_mark"]
        assert stale_mark_guard.config.threshold == 15.0


class TestGuardStateTransitions:
    def test_guard_status_transitions_healthy_to_warning(self):
        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        assert guard.status == GuardStatus.HEALTHY

        result = guard.check({"pnl": -40.0})
        assert result is None
        assert guard.status == GuardStatus.HEALTHY

        result = guard.check({"pnl": -60.0})
        assert result is None
        assert guard.status == GuardStatus.WARNING

    def test_guard_status_transitions_warning_to_breached(self):
        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        guard.check({"pnl": -60.0})
        assert guard.status == GuardStatus.WARNING

        result = guard.check({"pnl": -50.0})
        assert result is not None
        assert guard.status == GuardStatus.BREACHED
        assert guard.breach_count == 1

    def test_guard_status_transitions_breached_to_warning(self, frozen_time):
        from datetime import timedelta

        guard = DailyLossGuard(GuardConfig(name="test", threshold=100.0))

        guard.check({"pnl": -110.0})
        assert guard.status == GuardStatus.BREACHED

        frozen_time.tick(delta=timedelta(days=1))
        result = guard.check({"pnl": -10.0})
        assert result is None
        assert guard.status == GuardStatus.HEALTHY

        guard.check({"pnl": -110.0})
        assert guard.status == GuardStatus.BREACHED

    def test_position_stuck_guard_state_tracking(self, frozen_time):
        from datetime import timedelta

        guard = PositionStuckGuard(GuardConfig(name="position_stuck", threshold=60.0))

        result = guard.check({"positions": {}})
        assert result is None

        positions = {"BTC-USD": {"quantity": 1.0}}
        result = guard.check({"positions": positions})
        assert result is None

        frozen_time.tick(delta=timedelta(minutes=2))
        result = guard.check({"positions": positions})

        assert result is not None
        assert "Stuck positions detected" in result.message

        result = guard.check({"positions": {"BTC-USD": {"quantity": 0.0}}})
        assert result is None

    def test_drawdown_guard_peak_tracking(self):
        guard = DrawdownGuard(GuardConfig(name="drawdown", threshold=10.0))

        guard.check({"equity": Decimal("1000")})
        assert guard.peak_equity == Decimal("1000")

        guard.check({"equity": Decimal("1100")})
        assert guard.peak_equity == Decimal("1100")

        guard.check({"equity": Decimal("1050")})
        assert guard.current_drawdown == pytest.approx(Decimal("4.545454545454545"))

        result = guard.check({"equity": Decimal("980")})
        assert result is not None
        assert "Maximum drawdown breached" in result.message


class TestRuntimeGuardManager:
    def test_guard_registration_and_status_tracking(self):
        manager = RuntimeGuardManager()
        manager.add_guard(
            DailyLossGuard(
                GuardConfig(name="daily_loss", threshold=1000.0, severity=AlertSeverity.CRITICAL)
            )
        )
        manager.add_guard(
            StaleMarkGuard(
                GuardConfig(name="stale_mark", threshold=30.0, severity=AlertSeverity.ERROR)
            )
        )

        status = manager.get_status()
        assert set(status) == {"daily_loss", "stale_mark"}
        assert status["daily_loss"]["status"] == GuardStatus.HEALTHY.value
        assert status["stale_mark"]["status"] == GuardStatus.HEALTHY.value

    def test_alert_fan_out_to_multiple_handlers(self):
        manager = RuntimeGuardManager()
        calls: list[str] = []

        def handler(alert) -> None:
            calls.append(alert.guard_name)

        manager.add_alert_handler(handler)
        manager.add_alert_handler(handler)
        manager.add_guard(
            ErrorRateGuard(
                GuardConfig(name="error_rate", threshold=0.5, severity=AlertSeverity.ERROR)
            )
        )

        alerts = manager.check_all({"error": True})

        assert len(alerts) == 1
        assert calls == ["error_rate", "error_rate"]

    def test_guard_cooldown_prevents_alert_spam(self):
        manager = RuntimeGuardManager()
        manager.add_guard(
            ErrorRateGuard(
                GuardConfig(
                    name="error_rate",
                    threshold=0.5,
                    severity=AlertSeverity.ERROR,
                    cooldown_seconds=300,
                )
            )
        )

        assert len(manager.check_all({"error": True})) == 1
        assert manager.check_all({"error": True}) == []

    def test_multiple_guards_trigger_multiple_alerts(self):
        manager = RuntimeGuardManager()
        manager.add_guard(
            DailyLossGuard(
                GuardConfig(name="daily_loss", threshold=100.0, severity=AlertSeverity.CRITICAL)
            )
        )
        manager.add_guard(
            ErrorRateGuard(
                GuardConfig(name="error_rate", threshold=0.5, severity=AlertSeverity.ERROR)
            )
        )

        alerts = manager.check_all({"pnl": -200.0, "error": True})

        guard_names = {alert.guard_name for alert in alerts}
        assert guard_names == {"daily_loss", "error_rate"}
