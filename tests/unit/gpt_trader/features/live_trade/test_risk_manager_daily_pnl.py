"""Tests for LiveRiskManager.track_daily_pnl."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch):
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", lambda self: None)


class TestTrackDailyPnl:
    """Tests for track_daily_pnl method."""

    def test_first_call_sets_start_equity(self) -> None:
        """Test first call sets start of day equity."""
        manager = LiveRiskManager()

        result = manager.track_daily_pnl(Decimal("10000"), {})

        assert result is False
        assert manager._start_of_day_equity == Decimal("10000")

    def test_no_config_returns_false(self) -> None:
        """Test returns False when no config."""
        manager = LiveRiskManager()
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False

    def test_no_daily_loss_limit_returns_false(self) -> None:
        """Test returns False when no daily_loss_limit_pct in config."""
        config = MockConfig(daily_loss_limit_pct=None)
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False

    def test_loss_within_limit(self) -> None:
        """Test returns False when loss within limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.10"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        # 5% loss (9500 from 10000)
        result = manager.track_daily_pnl(Decimal("9500"), {})

        assert result is False
        assert manager._daily_pnl_triggered is False

    def test_loss_exceeds_limit_triggers(self) -> None:
        """Test triggers when loss exceeds limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        # 10% loss (9000 from 10000)
        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is True
        assert manager._daily_pnl_triggered is True
        assert manager._reduce_only_mode is True
        assert manager._reduce_only_reason == "daily_loss_limit_breached"

    def test_profit_does_not_trigger(self) -> None:
        """Test profit does not trigger loss limit."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("10000")

        # 10% profit
        result = manager.track_daily_pnl(Decimal("11000"), {})

        assert result is False

    def test_zero_start_equity_skips(self) -> None:
        """Test skips when start of day equity is zero."""
        config = MockConfig(daily_loss_limit_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        manager._start_of_day_equity = Decimal("0")

        result = manager.track_daily_pnl(Decimal("9000"), {})

        assert result is False
