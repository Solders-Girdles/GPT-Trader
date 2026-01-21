"""Tests for LiveRiskManager volatility circuit breaker and outcome."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.risk.manager import (
    LiveRiskManager,
    VolatilityCheckOutcome,
)
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch):
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", lambda self: None)


class TestVolatilityCheckOutcome:
    """Tests for VolatilityCheckOutcome dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        outcome = VolatilityCheckOutcome()

        assert outcome.triggered is False
        assert outcome.symbol == ""
        assert outcome.reason == ""

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        outcome = VolatilityCheckOutcome(
            triggered=True,
            symbol="BTC-USD",
            reason="High volatility detected",
        )

        assert outcome.triggered is True
        assert outcome.symbol == "BTC-USD"
        assert outcome.reason == "High volatility detected"

    def test_to_payload_not_triggered(self) -> None:
        """Should serialize non-triggered outcome."""
        outcome = VolatilityCheckOutcome()
        payload = outcome.to_payload()

        assert payload == {
            "triggered": False,
            "symbol": "",
            "reason": "",
        }

    def test_to_payload_triggered(self) -> None:
        """Should serialize triggered outcome."""
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


class TestCheckVolatilityCircuitBreaker:
    """Tests for check_volatility_circuit_breaker method."""

    def test_empty_closes_not_triggered(self) -> None:
        """Should not trigger for empty closes."""
        manager = LiveRiskManager()

        result = manager.check_volatility_circuit_breaker("BTC-USD", [])

        assert result.triggered is False
        assert result.symbol == "BTC-USD"

    def test_too_few_closes_not_triggered(self) -> None:
        """Should not trigger for fewer than 5 closes."""
        manager = LiveRiskManager()
        closes = [Decimal("100"), Decimal("101"), Decimal("102"), Decimal("103")]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_no_config_not_triggered(self) -> None:
        """Should not trigger without config."""
        manager = LiveRiskManager()
        closes = [Decimal(str(i)) for i in range(100, 110)]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_no_volatility_threshold_not_triggered(self) -> None:
        """Should not trigger without volatility_threshold_pct."""
        config = MockConfig(volatility_threshold_pct=None)
        manager = LiveRiskManager(config=config)
        closes = [Decimal(str(i)) for i in range(100, 110)]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_low_volatility_not_triggered(self) -> None:
        """Should not trigger when volatility below threshold."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.10"))
        manager = LiveRiskManager(config=config)
        closes = [Decimal("100"), Decimal("101"), Decimal("99"), Decimal("100"), Decimal("100")]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_high_volatility_triggers(self) -> None:
        """Should trigger when volatility exceeds threshold."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        closes = [Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100"), Decimal("150")]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is True
        assert result.symbol == "BTC-USD"
        assert "exceeds threshold" in result.reason
        assert manager._reduce_only_mode is True
        assert "volatility_breaker_BTC-USD" in manager._reduce_only_reason

    def test_zero_average_not_triggered(self) -> None:
        """Should not trigger when average is zero."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        closes = [Decimal("0")] * 5

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False
