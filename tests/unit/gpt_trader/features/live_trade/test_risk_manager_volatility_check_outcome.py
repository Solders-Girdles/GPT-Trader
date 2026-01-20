"""Tests for VolatilityCheckOutcome."""

from __future__ import annotations

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager, VolatilityCheckOutcome


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch):
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", lambda self: None)


class TestVolatilityCheckOutcome:
    """Tests for VolatilityCheckOutcome dataclass."""

    def test_default_values(self) -> None:
        """Test default values for VolatilityCheckOutcome."""
        outcome = VolatilityCheckOutcome()

        assert outcome.triggered is False
        assert outcome.symbol == ""
        assert outcome.reason == ""

    def test_custom_values(self) -> None:
        """Test creating VolatilityCheckOutcome with custom values."""
        outcome = VolatilityCheckOutcome(
            triggered=True,
            symbol="BTC-USD",
            reason="High volatility detected",
        )

        assert outcome.triggered is True
        assert outcome.symbol == "BTC-USD"
        assert outcome.reason == "High volatility detected"

    def test_to_payload_not_triggered(self) -> None:
        """Test to_payload for non-triggered outcome."""
        outcome = VolatilityCheckOutcome()
        payload = outcome.to_payload()

        assert payload == {
            "triggered": False,
            "symbol": "",
            "reason": "",
        }

    def test_to_payload_triggered(self) -> None:
        """Test to_payload for triggered outcome."""
        outcome = VolatilityCheckOutcome(
            triggered=True,
            symbol="ETH-USD",
            reason="Volatility exceeded 5%",
        )
        payload = outcome.to_payload()

        assert payload == {
            "triggered": True,
            "symbol": "ETH-USD",
            "reason": "Volatility exceeded 5%",
        }
