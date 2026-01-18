"""Tests for LiveRiskManager.check_mark_staleness."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


@pytest.fixture(autouse=True)
def mock_load_state():
    """Prevent LiveRiskManager from loading state during tests."""
    with patch("gpt_trader.features.live_trade.risk.manager.LiveRiskManager._load_state"):
        yield


class TestCheckMarkStaleness:
    """Tests for check_mark_staleness method."""

    def test_no_update_is_stale(self) -> None:
        """Test returns True when no update recorded."""
        manager = LiveRiskManager()

        assert manager.check_mark_staleness("BTC-USD") is True

    def test_recent_update_not_stale(self) -> None:
        """Test returns False when update is recent."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time()

        assert manager.check_mark_staleness("BTC-USD") is False

    def test_old_update_is_stale(self) -> None:
        """Test returns True when update is old."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time() - 200  # 200 seconds ago

        assert manager.check_mark_staleness("BTC-USD") is True

    def test_custom_staleness_threshold(self) -> None:
        """Test uses config staleness threshold."""
        config = MockConfig(mark_staleness_threshold=30.0)
        manager = LiveRiskManager(config=config)
        manager.last_mark_update["BTC-USD"] = time.time() - 50  # 50 seconds ago

        # 50 > 30, so stale
        assert manager.check_mark_staleness("BTC-USD") is True

    def test_default_threshold_without_config(self) -> None:
        """Test default 120 second threshold without config."""
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = time.time() - 100  # 100 seconds ago

        # 100 < 120, so not stale
        assert manager.check_mark_staleness("BTC-USD") is False

    @patch("time.time")
    def test_exact_boundary(self, mock_time: Any) -> None:
        """Test behavior at exact threshold boundary."""
        mock_time.return_value = 1000.0
        manager = LiveRiskManager()
        manager.last_mark_update["BTC-USD"] = 880.0  # Exactly 120 seconds ago

        # 1000 - 880 = 120, and 120 > 120 is False
        assert manager.check_mark_staleness("BTC-USD") is False
