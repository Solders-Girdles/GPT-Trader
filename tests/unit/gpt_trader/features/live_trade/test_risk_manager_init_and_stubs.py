"""Tests for LiveRiskManager initialization and stub methods."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager


@pytest.fixture(autouse=True)
def mock_load_state():
    """Prevent LiveRiskManager from loading state during tests."""
    with patch("gpt_trader.features.live_trade.risk.manager.LiveRiskManager._load_state"):
        yield


class TestLiveRiskManagerInit:
    """Tests for LiveRiskManager initialization."""

    def test_init_no_config(self) -> None:
        """Test initialization without config."""
        manager = LiveRiskManager()

        assert manager.config is None
        assert manager.event_store is None
        assert manager._reduce_only_mode is False
        assert manager._reduce_only_reason == ""
        assert manager._daily_pnl_triggered is False
        assert manager._risk_metrics == []
        assert manager._start_of_day_equity is None

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = {"max_leverage": 10}
        manager = LiveRiskManager(config=config)

        assert manager.config == {"max_leverage": 10}

    def test_init_with_event_store(self) -> None:
        """Test initialization with event store."""
        event_store = object()
        manager = LiveRiskManager(event_store=event_store)

        assert manager.event_store is event_store

    def test_positions_default_dict(self) -> None:
        """Test positions is a defaultdict."""
        manager = LiveRiskManager()

        # Access non-existent key should return empty dict
        assert manager.positions["BTC-USD"] == {}

    def test_last_mark_update_empty(self) -> None:
        """Test last_mark_update starts empty."""
        manager = LiveRiskManager()

        assert manager.last_mark_update == {}


class TestLiveRiskManagerStubs:
    """Tests for stub methods."""

    def test_check_order_returns_true(self) -> None:
        """Test check_order always returns True."""
        manager = LiveRiskManager()

        assert manager.check_order(None) is True
        assert manager.check_order({"symbol": "BTC-USD"}) is True
        assert manager.check_order(object()) is True

    def test_update_position_is_noop(self) -> None:
        """Test update_position is a no-op."""
        manager = LiveRiskManager()

        # Should not raise
        manager.update_position(None)
        manager.update_position({"symbol": "BTC-USD"})
        assert manager.positions == {}
