"""Tests for RiskGateValidator - trading safety gate validation.

This module tests the RiskGateValidator's ability to:
- Check volatility circuit breakers with kill switch detection
- Validate market data staleness
- Handle exceptions gracefully
- Properly gate trading based on risk checks
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.risk import RiskGateValidator
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    return Mock()


@pytest.fixture
def validator(mock_risk_manager):
    """Create RiskGateValidator instance."""
    return RiskGateValidator(mock_risk_manager)


@pytest.fixture
def sample_marks():
    """Create sample mark prices."""
    return [Decimal(f"{50000 + i * 100}") for i in range(30)]


class TestValidateGates:
    """Test main validate_gates method."""

    def test_all_gates_pass(self, validator, mock_risk_manager, sample_marks):
        """All gates pass when no risks detected."""
        # Setup: no circuit breaker trigger, no staleness
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)
        mock_risk_manager.check_mark_staleness.return_value = False

        result = validator.validate_gates("BTC-USD", sample_marks, lookback_window=20)

        assert result is True
        mock_risk_manager.check_volatility_circuit_breaker.assert_called_once()
        mock_risk_manager.check_mark_staleness.assert_called_once_with("BTC-USD")

    def test_volatility_circuit_breaker_kill_switch_blocks(
        self, validator, mock_risk_manager, sample_marks
    ):
        """Volatility CB with KILL_SWITCH action blocks trading."""
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(
            triggered=True, action=CircuitBreakerAction.KILL_SWITCH
        )

        result = validator.validate_gates("BTC-USD", sample_marks, lookback_window=20)

        assert result is False
        # Should not check staleness if CB blocks
        mock_risk_manager.check_mark_staleness.assert_not_called()

    def test_volatility_circuit_breaker_non_kill_switch_passes(
        self, validator, mock_risk_manager, sample_marks
    ):
        """Volatility CB with non-KILL_SWITCH action allows trading."""
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(
            triggered=True, action=CircuitBreakerAction.REDUCE_ONLY
        )
        mock_risk_manager.check_mark_staleness.return_value = False

        result = validator.validate_gates("BTC-USD", sample_marks, lookback_window=20)

        assert result is True
        # Should still check staleness
        mock_risk_manager.check_mark_staleness.assert_called_once()

    def test_mark_staleness_blocks_trading(self, validator, mock_risk_manager, sample_marks):
        """Stale market data blocks trading."""
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)
        mock_risk_manager.check_mark_staleness.return_value = True  # Stale data

        result = validator.validate_gates("BTC-USD", sample_marks, lookback_window=20)

        assert result is False

    def test_volatility_check_exception_does_not_block(
        self, validator, mock_risk_manager, sample_marks, caplog
    ):
        """Exception during volatility check doesn't block trading."""
        mock_risk_manager.check_volatility_circuit_breaker.side_effect = Exception("API error")
        mock_risk_manager.check_mark_staleness.return_value = False

        with caplog.at_level("DEBUG"):
            result = validator.validate_gates("BTC-USD", sample_marks, lookback_window=20)

        assert result is True
        assert "Volatility circuit breaker check failed" in caplog.text
        assert "BTC-USD" in caplog.text

    def test_staleness_check_exception_does_not_block(
        self, validator, mock_risk_manager, sample_marks, caplog
    ):
        """Exception during staleness check doesn't block trading."""
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)
        mock_risk_manager.check_mark_staleness.side_effect = Exception("Cache error")

        with caplog.at_level("DEBUG"):
            result = validator.validate_gates("BTC-USD", sample_marks, lookback_window=20)

        assert result is True
        assert "Mark staleness check failed" in caplog.text

    def test_both_checks_exception_handling(
        self, validator, mock_risk_manager, sample_marks, caplog
    ):
        """Both checks can fail without blocking trading."""
        mock_risk_manager.check_volatility_circuit_breaker.side_effect = Exception("CB error")
        mock_risk_manager.check_mark_staleness.side_effect = Exception("Stale error")

        with caplog.at_level("DEBUG"):
            result = validator.validate_gates("BTC-USD", sample_marks, lookback_window=20)

        assert result is True
        assert "Volatility circuit breaker check failed" in caplog.text
        assert "Mark staleness check failed" in caplog.text


class TestCheckVolatilityCircuitBreaker:
    """Test volatility circuit breaker check."""

    def test_not_triggered(self, validator, mock_risk_manager, sample_marks):
        """Circuit breaker not triggered."""
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)

        result = validator._check_volatility_circuit_breaker(
            "BTC-USD", sample_marks, lookback_window=20
        )

        assert result is True

    def test_triggered_with_kill_switch(self, validator, mock_risk_manager, sample_marks):
        """Circuit breaker triggered with KILL_SWITCH action."""
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(
            triggered=True, action=CircuitBreakerAction.KILL_SWITCH
        )

        result = validator._check_volatility_circuit_breaker(
            "BTC-USD", sample_marks, lookback_window=20
        )

        assert result is False

    def test_triggered_without_kill_switch(self, validator, mock_risk_manager, sample_marks):
        """Circuit breaker triggered but not KILL_SWITCH action."""
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(
            triggered=True, action=CircuitBreakerAction.REDUCE_ONLY
        )

        result = validator._check_volatility_circuit_breaker(
            "BTC-USD", sample_marks, lookback_window=20
        )

        assert result is True

    def test_uses_correct_lookback_window(self, validator, mock_risk_manager, sample_marks):
        """Uses specified lookback window for volatility check."""
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)

        validator._check_volatility_circuit_breaker("BTC-USD", sample_marks, lookback_window=15)

        # Should pass last 15 marks
        call_args = mock_risk_manager.check_volatility_circuit_breaker.call_args
        passed_marks = call_args[0][1]
        assert len(passed_marks) == 15
        assert passed_marks == list(sample_marks[-15:])

    def test_exception_handling(self, validator, mock_risk_manager, sample_marks, caplog):
        """Handles exception during volatility check."""
        mock_risk_manager.check_volatility_circuit_breaker.side_effect = ValueError("Invalid data")

        with caplog.at_level("DEBUG"):
            result = validator._check_volatility_circuit_breaker(
                "ETH-USD", sample_marks, lookback_window=20
            )

        assert result is True
        assert "Volatility circuit breaker check failed for ETH-USD" in caplog.text


class TestCheckMarkStaleness:
    """Test mark staleness check."""

    def test_data_fresh(self, validator, mock_risk_manager):
        """Fresh data passes check."""
        mock_risk_manager.check_mark_staleness.return_value = False

        result = validator._check_mark_staleness("BTC-USD")

        assert result is True
        mock_risk_manager.check_mark_staleness.assert_called_once_with("BTC-USD")

    def test_data_stale(self, validator, mock_risk_manager):
        """Stale data fails check."""
        mock_risk_manager.check_mark_staleness.return_value = True

        result = validator._check_mark_staleness("BTC-USD")

        assert result is False

    def test_exception_handling(self, validator, mock_risk_manager, caplog):
        """Handles exception during staleness check."""
        mock_risk_manager.check_mark_staleness.side_effect = KeyError("No cache entry")

        with caplog.at_level("DEBUG"):
            result = validator._check_mark_staleness("ETH-USD")

        assert result is True
        assert "Mark staleness check failed for ETH-USD" in caplog.text


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_validates_with_minimum_marks(self, validator, mock_risk_manager):
        """Validates with minimum number of marks."""
        marks = [Decimal("50000"), Decimal("50100")]
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)
        mock_risk_manager.check_mark_staleness.return_value = False

        result = validator.validate_gates("BTC-USD", marks, lookback_window=5)

        assert result is True
        # Should pass whatever marks available (even if < lookback_window)
        call_args = mock_risk_manager.check_volatility_circuit_breaker.call_args
        passed_marks = call_args[0][1]
        assert len(passed_marks) == 2

    def test_lookback_window_larger_than_marks(self, validator, mock_risk_manager):
        """Handles lookback window larger than available marks."""
        marks = [Decimal("50000"), Decimal("50100"), Decimal("50200")]
        mock_risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)
        mock_risk_manager.check_mark_staleness.return_value = False

        result = validator.validate_gates("BTC-USD", marks, lookback_window=100)

        assert result is True
        # Should use all available marks
        call_args = mock_risk_manager.check_volatility_circuit_breaker.call_args
        passed_marks = call_args[0][1]
        assert passed_marks == list(marks)
