"""Tests for LiveRiskManager.reset_daily_tracking."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch):
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", lambda self: None)


class TestResetDailyTracking:
    """Tests for reset_daily_tracking method."""

    def test_resets_all_daily_state(self) -> None:
        """Test resets all daily tracking state."""
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
        """Test multiple resets are safe."""
        manager = LiveRiskManager()

        manager.reset_daily_tracking()
        manager.reset_daily_tracking()

        assert manager._start_of_day_equity is None
