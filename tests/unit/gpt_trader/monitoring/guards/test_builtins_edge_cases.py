"""Edge case tests for built-in guard implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import GuardConfig
from gpt_trader.monitoring.guards.builtins import (
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)


def _config(name: str, *, threshold: float, window: int = 60) -> GuardConfig:
    return GuardConfig(
        name=name,
        threshold=threshold,
        window_seconds=window,
        severity=AlertSeverity.ERROR,
        cooldown_seconds=0,
    )


class TestDailyLossGuardEdgeCases:
    """Edge case tests for DailyLossGuard."""

    def test_invalid_pnl_type_returns_no_alert(self) -> None:
        guard = DailyLossGuard(_config("daily_loss", threshold=100))
        result = guard.check({"pnl": "not_a_number"})
        assert result is None

    def test_missing_pnl_uses_zero(self) -> None:
        guard = DailyLossGuard(_config("daily_loss", threshold=100))
        result = guard.check({})
        assert result is None
        assert guard.daily_pnl == Decimal("0")

    def test_zero_threshold_returns_no_alert(self) -> None:
        guard = DailyLossGuard(_config("daily_loss", threshold=0))
        result = guard.check({"pnl": -1000})
        assert result is None


class TestStaleMarkGuardEdgeCases:
    """Edge case tests for StaleMarkGuard."""

    def test_non_string_symbol_returns_no_alert(self) -> None:
        guard = StaleMarkGuard(_config("stale_mark", threshold=30))
        result = guard.check({"symbol": 123, "mark_timestamp": datetime.now(UTC)})
        assert result is None

    def test_missing_mark_timestamp_returns_no_alert(self) -> None:
        guard = StaleMarkGuard(_config("stale_mark", threshold=30))
        result = guard.check({"symbol": "BTC-PERP"})
        assert result is None

    def test_invalid_iso_string_returns_no_alert(self) -> None:
        guard = StaleMarkGuard(_config("stale_mark", threshold=30))
        result = guard.check({"symbol": "BTC-PERP", "mark_timestamp": "not-a-date"})
        assert result is None

    def test_non_datetime_type_returns_no_alert(self) -> None:
        guard = StaleMarkGuard(_config("stale_mark", threshold=30))
        result = guard.check({"symbol": "BTC-PERP", "mark_timestamp": ["list"]})
        assert result is None

    def test_float_timestamp_converted(self) -> None:
        guard = StaleMarkGuard(_config("stale_mark", threshold=30))
        recent_time = datetime.now(UTC).timestamp()
        result = guard.check({"symbol": "BTC-PERP", "mark_timestamp": recent_time})
        assert result is None
        assert "BTC-PERP" in guard.last_marks


class TestPositionStuckGuardEdgeCases:
    """Edge case tests for PositionStuckGuard."""

    def test_non_mapping_positions_returns_no_alert(self) -> None:
        guard = PositionStuckGuard(_config("stuck", threshold=60))
        result = guard.check({"positions": "not_a_dict"})
        assert result is None

    def test_non_string_symbol_skipped(self) -> None:
        guard = PositionStuckGuard(_config("stuck", threshold=60))
        result = guard.check({"positions": {123: {"quantity": 1}}})
        assert result is None
        assert len(guard.position_times) == 0

    def test_non_mapping_position_skipped(self) -> None:
        guard = PositionStuckGuard(_config("stuck", threshold=60))
        result = guard.check({"positions": {"BTC-PERP": "not_a_dict"}})
        assert result is None
        assert len(guard.position_times) == 0

    def test_invalid_size_value_skipped(self) -> None:
        guard = PositionStuckGuard(_config("stuck", threshold=60))
        result = guard.check({"positions": {"BTC-PERP": {"quantity": "invalid"}}})
        assert result is None
        assert len(guard.position_times) == 0

    def test_size_field_fallback(self) -> None:
        guard = PositionStuckGuard(_config("stuck", threshold=60))
        result = guard.check({"positions": {"BTC-PERP": {"size": 1}}})
        assert result is None
        assert "BTC-PERP" in guard.position_times

    def test_contracts_field_fallback(self) -> None:
        guard = PositionStuckGuard(_config("stuck", threshold=60))
        result = guard.check({"positions": {"BTC-PERP": {"contracts": 1}}})
        assert result is None
        assert "BTC-PERP" in guard.position_times


class TestDrawdownGuardEdgeCases:
    """Edge case tests for DrawdownGuard."""

    def test_invalid_equity_type_returns_no_alert(self) -> None:
        guard = DrawdownGuard(_config("drawdown", threshold=10))
        result = guard.check({"equity": "not_a_number"})
        assert result is None

    def test_missing_equity_uses_zero(self) -> None:
        guard = DrawdownGuard(_config("drawdown", threshold=10))
        result = guard.check({})
        assert result is None
        assert guard.peak_equity == Decimal("0")

    def test_updates_peak_equity(self) -> None:
        guard = DrawdownGuard(_config("drawdown", threshold=10))
        guard.check({"equity": 1000})
        assert guard.peak_equity == Decimal("1000")
        guard.check({"equity": 1200})
        assert guard.peak_equity == Decimal("1200")
        guard.check({"equity": 1100})
        assert guard.peak_equity == Decimal("1200")

    def test_calculates_drawdown_percent(self) -> None:
        guard = DrawdownGuard(_config("drawdown", threshold=20))
        guard.check({"equity": 1000})
        guard.check({"equity": 850})
        assert guard.current_drawdown == Decimal("15")

    def test_triggers_on_drawdown_breach(self) -> None:
        guard = DrawdownGuard(_config("drawdown", threshold=10))
        guard.check({"equity": 1000})
        alert = guard.check({"equity": 800})
        assert alert is not None
        assert "Maximum drawdown breached" in alert.message


class TestErrorRateGuardEdgeCases:
    """Edge case tests for ErrorRateGuard."""

    def test_non_error_event_does_not_add_to_times(self) -> None:
        guard = ErrorRateGuard(_config("error_rate", threshold=5, window=60))
        guard.check({"error": False})
        assert len(guard.error_times) == 0

    def test_missing_error_key_does_not_add_to_times(self) -> None:
        guard = ErrorRateGuard(_config("error_rate", threshold=5, window=60))
        guard.check({})
        assert len(guard.error_times) == 0
