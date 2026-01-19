"""Tests for derivatives_discovery - INTX market behavior."""

from __future__ import annotations

from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.derivatives_discovery import (
    discover_derivatives_eligibility,
)


class TestDiscoverDerivativesINTX:
    """Tests for INTX (International) derivatives discovery."""

    def test_intx_market_accessible(self) -> None:
        """Test successful INTX derivatives discovery."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"uuid": "portfolio-123", "type": "perpetuals", "name": "My INTX"}
        ]
        broker.get_intx_portfolio.return_value = {"balance": "5000", "margin": "1000"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_derivatives_enabled is True
        assert result.intx_portfolio_accessible is True
        assert result.error_message is None
        assert result.reduce_only_required is False

    def test_intx_market_broker_missing_list_portfolios(self) -> None:
        """Test INTX discovery when broker lacks list_portfolios."""
        broker = Mock(spec=[])

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_derivatives_enabled is False
        assert result.intx_portfolio_accessible is False
        assert "Broker does not support portfolio listing" in (result.error_message or "")
        assert result.reduce_only_required is True

    def test_intx_market_no_portfolios(self) -> None:
        """Test INTX discovery when no portfolios exist."""
        broker = Mock()
        broker.list_portfolios.return_value = []

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_derivatives_enabled is True
        assert result.intx_portfolio_accessible is False
        assert "No portfolios accessible" in (result.error_message or "")
        assert result.reduce_only_required is True

    def test_intx_market_no_perpetuals_portfolio(self) -> None:
        """Test INTX discovery when no perpetuals portfolio found."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"uuid": "spot-123", "type": "spot", "name": "Spot Trading"}
        ]

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_derivatives_enabled is True
        assert result.intx_portfolio_accessible is False
        assert "No INTX portfolio found" in (result.error_message or "")

    def test_intx_market_portfolio_found_by_name(self) -> None:
        """Test INTX discovery finds portfolio by name."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"uuid": "perp-123", "type": "trading", "name": "My INTX Perpetuals"}
        ]
        broker.get_intx_portfolio.return_value = {"balance": "1000"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is True

    def test_intx_market_portfolio_found_by_intx_type(self) -> None:
        """Test INTX discovery finds portfolio by INTX type."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"uuid": "intx-123", "type": "INTX", "name": "Trading"}
        ]
        broker.get_intx_portfolio.return_value = {"balance": "1000"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is True

    def test_intx_market_portfolio_missing_uuid(self) -> None:
        """Test INTX discovery when portfolio has no UUID."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"type": "perpetuals", "name": "INTX"}  # Missing uuid
        ]

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is False
        assert "INTX portfolio missing UUID" in (result.error_message or "")

    def test_intx_market_broker_missing_get_intx_portfolio(self) -> None:
        """Test INTX discovery when broker lacks get_intx_portfolio."""
        broker = Mock(spec=["list_portfolios"])
        broker.list_portfolios.return_value = [
            {"uuid": "perp-123", "type": "perpetuals", "name": "INTX"}
        ]

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is False
        assert "INTX portfolio endpoints not available" in (result.error_message or "")

    def test_intx_market_portfolio_details_empty(self) -> None:
        """Test INTX discovery when portfolio details are empty."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"uuid": "perp-123", "type": "perpetuals", "name": "INTX"}
        ]
        broker.get_intx_portfolio.return_value = None

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is False
        assert "INTX portfolio details unavailable" in (result.error_message or "")

    def test_intx_market_api_error(self) -> None:
        """Test INTX discovery when API call fails."""
        broker = Mock()
        broker.list_portfolios.side_effect = RuntimeError("API error")

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_derivatives_enabled is True
        assert result.intx_portfolio_accessible is False
        assert "INTX discovery error" in (result.error_message or "")

    def test_intx_market_attribute_error(self) -> None:
        """Test INTX discovery when AttributeError occurs."""
        broker = Mock()
        broker.list_portfolios.side_effect = AttributeError("Missing attr")

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_derivatives_enabled is False
        assert result.intx_portfolio_accessible is False
        assert "INTX methods not available" in (result.error_message or "")

    def test_intx_market_non_dict_portfolio_entry(self) -> None:
        """Test INTX discovery skips non-dict portfolio entries."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            "invalid",  # Not a dict
            {"uuid": "perp-123", "type": "perpetuals", "name": "INTX"},
        ]
        broker.get_intx_portfolio.return_value = {"balance": "1000"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is True

    def test_portfolios_list_is_not_list(self) -> None:
        """Test handling when portfolios response is not a list."""
        broker = Mock()
        broker.list_portfolios.return_value = {"error": "unexpected format"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is False

    def test_portfolio_with_perpetuals_in_name_lowercase(self) -> None:
        """Test case-insensitive matching for perpetuals in name."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"uuid": "perp-123", "type": "custom", "name": "PERPETUALS Trading"}
        ]
        broker.get_intx_portfolio.return_value = {"balance": "1000"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is True

    def test_portfolio_with_intx_in_name(self) -> None:
        """Test finding portfolio with 'intx' in name."""
        broker = Mock()
        broker.list_portfolios.return_value = [
            {"uuid": "intx-123", "type": "trading", "name": "My INTX Account"}
        ]
        broker.get_intx_portfolio.return_value = {"balance": "1000"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is True
