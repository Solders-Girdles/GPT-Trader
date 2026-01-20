"""Tests for LiveRiskManager.append_risk_metrics."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.risk.manager as risk_manager_module
from gpt_trader.features.live_trade.risk.manager import LiveRiskManager


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", MagicMock())


class TestAppendRiskMetrics:
    """Tests for append_risk_metrics method."""

    def test_appends_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test appends metrics with timestamp."""
        monkeypatch.setattr(risk_manager_module.time, "time", lambda: 12345.0)
        manager = LiveRiskManager()

        manager.append_risk_metrics(Decimal("10000"), {"BTC-USD": {"pnl": Decimal("100")}})

        assert len(manager._risk_metrics) == 1
        assert manager._risk_metrics[0]["timestamp"] == 12345.0
        assert manager._risk_metrics[0]["equity"] == "10000"
        assert manager._risk_metrics[0]["positions"] == {"BTC-USD": {"pnl": "100"}}
        assert manager._risk_metrics[0]["reduce_only_mode"] is False

    def test_captures_reduce_only_mode(self) -> None:
        """Test captures reduce_only_mode state."""
        manager = LiveRiskManager()
        manager._reduce_only_mode = True

        manager.append_risk_metrics(Decimal("10000"), {})

        assert manager._risk_metrics[0]["reduce_only_mode"] is True

    def test_limits_to_100_metrics(self) -> None:
        """Test keeps only last 100 metrics."""
        manager = LiveRiskManager()

        for i in range(150):
            manager.append_risk_metrics(Decimal(str(i)), {})

        assert len(manager._risk_metrics) == 100
        # First metric should be #50 (0-49 removed)
        assert manager._risk_metrics[0]["equity"] == "50"
        assert manager._risk_metrics[-1]["equity"] == "149"

    def test_converts_nested_decimals_to_strings(self) -> None:
        """Test converts nested Decimal values to strings."""
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
