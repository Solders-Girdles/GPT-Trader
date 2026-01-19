"""Tests for LiveRiskManager.check_volatility_circuit_breaker."""

from __future__ import annotations

from decimal import Decimal
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


class TestCheckVolatilityCircuitBreaker:
    """Tests for check_volatility_circuit_breaker method."""

    def test_empty_closes_not_triggered(self) -> None:
        """Test returns not triggered for empty closes."""
        manager = LiveRiskManager()

        result = manager.check_volatility_circuit_breaker("BTC-USD", [])

        assert result.triggered is False
        assert result.symbol == "BTC-USD"

    def test_too_few_closes_not_triggered(self) -> None:
        """Test returns not triggered for fewer than 5 closes."""
        manager = LiveRiskManager()
        closes = [Decimal("100"), Decimal("101"), Decimal("102"), Decimal("103")]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_no_config_not_triggered(self) -> None:
        """Test returns not triggered without config."""
        manager = LiveRiskManager()
        closes = [Decimal(str(i)) for i in range(100, 110)]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_no_volatility_threshold_not_triggered(self) -> None:
        """Test returns not triggered without volatility_threshold_pct."""
        config = MockConfig(volatility_threshold_pct=None)
        manager = LiveRiskManager(config=config)
        closes = [Decimal(str(i)) for i in range(100, 110)]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_low_volatility_not_triggered(self) -> None:
        """Test returns not triggered when volatility is below threshold."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.10"))
        manager = LiveRiskManager(config=config)
        # Small variance - max deviation ~2.5% from mean
        closes = [Decimal("100"), Decimal("101"), Decimal("99"), Decimal("100"), Decimal("100")]

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False

    def test_high_volatility_triggers(self) -> None:
        """Test triggers when volatility exceeds threshold."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        # High variance - includes value far from mean
        closes = [Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100"), Decimal("150")]
        # Mean = 110, max deviation = 40, volatility = 40/110 ≈ 36%

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is True
        assert result.symbol == "BTC-USD"
        assert "exceeds threshold" in result.reason
        assert manager._reduce_only_mode is True
        assert "volatility_breaker_BTC-USD" in manager._reduce_only_reason

    def test_zero_average_not_triggered(self) -> None:
        """Test returns not triggered when average is zero."""
        config = MockConfig(volatility_threshold_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        closes = [Decimal("0")] * 5

        result = manager.check_volatility_circuit_breaker("BTC-USD", closes)

        assert result.triggered is False
