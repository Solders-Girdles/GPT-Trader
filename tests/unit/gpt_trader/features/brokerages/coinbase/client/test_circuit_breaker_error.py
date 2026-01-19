"""Tests for CircuitOpenError exception."""

from gpt_trader.features.brokerages.coinbase.client.circuit_breaker import CircuitOpenError


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_error_message(self) -> None:
        """Test error message format."""
        error = CircuitOpenError("orders", 30.0)

        assert error.category == "orders"
        assert error.time_until_retry == 30.0
        assert "orders" in str(error)
        assert "30.0" in str(error)
