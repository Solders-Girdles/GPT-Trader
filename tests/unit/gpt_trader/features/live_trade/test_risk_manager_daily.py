"""Tests for LiveRiskManager daily tracking: PnL, staleness, and reset."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

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


class TestCheckMarkStaleness:
    """Tests for check_mark_staleness method."""

    def test_no_update_is_stale(self) -> None:
        """Should return True when no update recorded."""
        manager = LiveRiskManager()

        assert manager.check_mark_staleness("BTC-USD") is True

    def test_recent_update_not_stale(self) -> None:
        """Should return False when update is recent."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time()

        assert manager.check_mark_staleness("BTC-USD") is False

    def test_old_update_is_stale(self) -> None:
        """Should return True when update is old."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time() - 200

        assert manager.check_mark_staleness("BTC-USD") is True

    def test_custom_staleness_threshold(self) -> None:
        """Should use config staleness threshold."""
        config = MockConfig(mark_staleness_threshold=30.0)
        manager = LiveRiskManager(config=config)
        manager.last_mark_update["BTC-USD"] = time.time() - 50

        assert manager.check_mark_staleness("BTC-USD") is True  # 50 > 30

    def test_default_threshold_without_config(self) -> None:
        """Should use default 120 second threshold."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time() - 100

        assert manager.check_mark_staleness("BTC-USD") is False  # 100 < 120

    def test_exact_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should handle exact threshold boundary."""
        monkeypatch.setattr(time, "time", lambda: 1000.0)
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = 880.0  # Exactly 120 seconds ago

        assert manager.check_mark_staleness("BTC-USD") is False  # 120 > 120 is False
