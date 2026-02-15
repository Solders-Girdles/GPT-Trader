"""Tests for LiveRiskManager daily tracking, metrics, and reset."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.risk.manager as risk_manager_module
from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", MagicMock())


class TestTrackDailyPnl:
    """Tests for track_daily_pnl method."""

    def test_first_call_sets_start_equity(self) -> None:
        """Should set start of day equity on first call."""
        manager = LiveRiskManager()

        result = manager.track_daily_pnl(Decimal("10000"), {})

        assert result is False
        assert manager._start_of_day_equity == Decimal("10000")

    def test_no_config_returns_false(self) -> None:
        """Should return False when no config."""
        manager = LiveRiskManager()
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False

    def test_no_daily_loss_limit_returns_false(self) -> None:
        """Should return False when no daily_loss_limit_pct in config."""
        config = MockConfig(daily_loss_limit_pct=None)
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False

    def test_loss_within_limit(self) -> None:
        """Should return False when loss within limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.10"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9500"), {})  # 5% loss

        assert result is False
        assert manager._daily_pnl_triggered is False

    def test_loss_at_limit_triggers(self) -> None:
        """Should trigger when loss meets the limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9500"), {})  # 5% loss

        assert result is True
        assert manager._daily_pnl_triggered is True
        assert manager._reduce_only_mode is True
        assert manager._reduce_only_reason == "daily_loss_limit_breached"

    def test_loss_exceeds_limit_triggers(self) -> None:
        """Should trigger when loss exceeds limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9000"), {})  # 10% loss

        assert result is True
        assert manager._daily_pnl_triggered is True
        assert manager._reduce_only_mode is True
        assert manager._reduce_only_reason == "daily_loss_limit_breached"

    def test_profit_does_not_trigger(self) -> None:
        """Should not trigger on profit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("11000"), {})  # 10% profit

        assert result is False

    def test_zero_start_equity_skips(self) -> None:
        """Should skip when start of day equity is zero."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("0")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False


class TestResetDailyTracking:
    """Tests for reset_daily_tracking method."""

    def test_resets_all_daily_state(self) -> None:
        """Should reset all daily tracking state."""
        manager = LiveRiskManager()
        manager._start_of_day_equity = Decimal("10000")
        manager._daily_pnl_triggered = True
        manager._reduce_only_mode = True
        manager._reduce_only_reason = "daily_loss_limit"

        manager.reset_daily_tracking()

        assert manager._start_of_day_equity is None
        assert manager._daily_pnl_triggered is False
        assert manager._reduce_only_mode is False
        assert manager._reduce_only_reason == ""

    def test_reset_is_idempotent(self) -> None:
        """Should handle multiple resets safely."""
        manager = LiveRiskManager()

        manager.reset_daily_tracking()
        manager.reset_daily_tracking()

        assert manager._start_of_day_equity is None


class TestAppendRiskMetrics:
    """Tests for append_risk_metrics method."""

    def test_appends_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should append metrics with timestamp."""
        monkeypatch.setattr(risk_manager_module.time, "time", lambda: 12345.0)
        manager = LiveRiskManager()

        manager.append_risk_metrics(Decimal("10000"), {"BTC-USD": {"pnl": Decimal("100")}})

        assert len(manager._risk_metrics) == 1
        assert manager._risk_metrics[0]["timestamp"] == 12345.0
        assert manager._risk_metrics[0]["equity"] == "10000"
        assert manager._risk_metrics[0]["positions"] == {"BTC-USD": {"pnl": "100"}}
        assert manager._risk_metrics[0]["reduce_only_mode"] is False

    def test_captures_reduce_only_mode(self) -> None:
        """Should capture reduce_only_mode state."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True

        manager.append_risk_metrics(Decimal("10000"), {})

        assert manager._risk_metrics[0]["reduce_only_mode"] is True

    def test_limits_to_100_metrics(self) -> None:
        """Should keep only last 100 metrics."""
        manager = LiveRiskManager()

        for i in range(150):
            manager.append_risk_metrics(Decimal(str(i)), {})

        assert len(manager._risk_metrics) == 100
        assert manager._risk_metrics[0]["equity"] == "50"
        assert manager._risk_metrics[-1]["equity"] == "149"

    def test_converts_nested_decimals_to_strings(self) -> None:
        """Should convert nested Decimal values to strings."""
        manager = LiveRiskManager()
        positions = {
            "BTC-USD": {
                "pnl": Decimal("123.456"),
                "size": Decimal("-0.5"),
            },
            "ETH-USD": {
                "pnl": Decimal("0"),
            },
        }

        manager.append_risk_metrics(Decimal("9999.99"), positions)

        result_positions = manager._risk_metrics[0]["positions"]
        assert result_positions["BTC-USD"]["pnl"] == "123.456"
        assert result_positions["BTC-USD"]["size"] == "-0.5"
        assert result_positions["ETH-USD"]["pnl"] == "0"


class TestDailyWorkflow:
    """Workflow tests for daily tracking and metrics."""

    def test_daily_workflow(self) -> None:
        """Should handle typical daily workflow."""
        config = MockConfig(
            max_leverage=Decimal("10"),
            daily_loss_limit_pct=Decimal("0.05"),
            min_liquidation_buffer_pct=Decimal("0.10"),
        )
        manager = LiveRiskManager(config=config)

        manager.reset_daily_tracking()
        assert manager._start_of_day_equity is None

        manager.track_daily_pnl(Decimal("10000"), {})
        assert manager._start_of_day_equity == Decimal("10000")

        manager.append_risk_metrics(Decimal("10000"), {})
        assert len(manager._risk_metrics) == 1

        assert manager.track_daily_pnl(Decimal("9800"), {}) is False
        assert manager.track_daily_pnl(Decimal("9000"), {}) is True
        assert manager.is_reduce_only_mode() is True

        manager.reset_daily_tracking()
        assert manager.is_reduce_only_mode() is False
