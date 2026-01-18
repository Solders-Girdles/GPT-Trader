"""Tests for LiveRiskManager reduce-only mode helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager


@pytest.fixture(autouse=True)
def mock_load_state():
    """Prevent LiveRiskManager from loading state during tests."""
    with patch("gpt_trader.features.live_trade.risk.manager.LiveRiskManager._load_state"):
        yield


class TestReduceOnlyMode:
    """Tests for reduce-only mode methods."""

    def test_default_not_reduce_only(self) -> None:
        """Test default state is not reduce-only."""
        manager = LiveRiskManager()

        assert manager.is_reduce_only_mode() is False

    def test_set_reduce_only_true(self) -> None:
        """Test setting reduce-only to True."""
        manager = LiveRiskManager()

        manager.set_reduce_only_mode(True)

        assert manager.is_reduce_only_mode() is True

    def test_set_reduce_only_false(self) -> None:
        """Test setting reduce-only to False."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True

        manager.set_reduce_only_mode(False)

        assert manager.is_reduce_only_mode() is False

    def test_set_reduce_only_with_reason(self) -> None:
        """Test setting reduce-only with reason."""
        manager = LiveRiskManager()

        manager.set_reduce_only_mode(True, reason="liquidation_warning")

        assert manager._reduce_only_mode is True
        assert manager._reduce_only_reason == "liquidation_warning"

    def test_set_reduce_only_clears_reason_when_false(self) -> None:
        """Test setting reduce-only to False clears reason."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True
        manager._reduce_only_reason = "some_reason"

        manager.set_reduce_only_mode(False, reason="")

        assert manager._reduce_only_mode is False
        assert manager._reduce_only_reason == ""
