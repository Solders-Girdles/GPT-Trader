"""Tests for derivatives_discovery - discovery_data contents."""

from __future__ import annotations

from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.derivatives_discovery import (
    discover_derivatives_eligibility,
)


class TestDiscoveryData:
    """Tests for discovery_data contents."""

    def test_discovery_data_contains_us_details(self) -> None:
        """Test that discovery_data includes US discovery details."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = {"balance": "1000"}

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert "us_derivatives" in result.discovery_data
        assert result.discovery_data["us_derivatives"]["enabled"] is True
        assert result.discovery_data["us_derivatives"]["accessible"] is True
        assert result.discovery_data["us_derivatives"]["error"] is None

    def test_discovery_data_contains_intx_details(self) -> None:
        """Test that discovery_data includes INTX discovery details."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"uuid": "perp-123", "type": "perpetuals", "name": "INTX"}
        ]
        broker.get_intx_portfolio.return_value = {"balance": "5000"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert "intx_derivatives" in result.discovery_data
        assert result.discovery_data["intx_derivatives"]["enabled"] is True
        assert result.discovery_data["intx_derivatives"]["accessible"] is True
        assert result.discovery_data["intx_derivatives"]["error"] is None

    def test_discovery_data_contains_error_details(self) -> None:
        """Test that discovery_data includes error details."""
        broker = Mock()
        broker.get_cfm_balance_summary.side_effect = RuntimeError("API error")

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert result.discovery_data["us_derivatives"]["error"] is not None
        assert "API error" in result.discovery_data["us_derivatives"]["error"]
