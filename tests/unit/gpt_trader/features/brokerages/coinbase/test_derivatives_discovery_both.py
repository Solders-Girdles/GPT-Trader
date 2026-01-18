"""Tests for derivatives_discovery - BOTH markets behavior."""

from __future__ import annotations

from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.derivatives_discovery import (
    discover_derivatives_eligibility,
)


class TestDiscoverDerivativesBoth:
    """Tests for BOTH markets discovery."""

    def test_both_markets_accessible(self) -> None:
        """Test when both markets are accessible."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = {"balance": "1000"}
        broker.list_portfolios.return_value = [
            {"uuid": "perp-123", "type": "perpetuals", "name": "INTX"}
        ]
        broker.get_intx_portfolio.return_value = {"balance": "5000"}

        result = discover_derivatives_eligibility(broker, requested_market="BOTH")

        assert result.us_derivatives_enabled is True
        assert result.intx_derivatives_enabled is True
        assert result.cfm_portfolio_accessible is True
        assert result.intx_portfolio_accessible is True
        assert result.reduce_only_required is False
        assert result.error_message is None

    def test_both_markets_us_only_accessible(self) -> None:
        """Test when only US market is accessible."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = {"balance": "1000"}
        broker.list_portfolios.return_value = []  # No INTX portfolio

        result = discover_derivatives_eligibility(broker, requested_market="BOTH")

        assert result.cfm_portfolio_accessible is True
        assert result.intx_portfolio_accessible is False
        assert result.reduce_only_required is False

    def test_both_markets_intx_only_accessible(self) -> None:
        """Test when only INTX market is accessible."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = None  # CFM unavailable
        broker.list_portfolios.return_value = [
            {"uuid": "perp-123", "type": "perpetuals", "name": "INTX"}
        ]
        broker.get_intx_portfolio.return_value = {"balance": "5000"}

        result = discover_derivatives_eligibility(broker, requested_market="BOTH")

        assert result.cfm_portfolio_accessible is False
        assert result.intx_portfolio_accessible is True
        assert result.reduce_only_required is False

    def test_both_markets_neither_accessible(self) -> None:
        """Test when neither market is accessible."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = None
        broker.list_portfolios.return_value = []

        result = discover_derivatives_eligibility(broker, requested_market="BOTH")

        assert result.cfm_portfolio_accessible is False
        assert result.intx_portfolio_accessible is False
        assert result.reduce_only_required is True

    def test_both_markets_errors_combined(self) -> None:
        """Test that errors from both discoveries are combined."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = None
        broker.list_portfolios.return_value = []

        result = discover_derivatives_eligibility(broker, requested_market="BOTH")

        assert result.error_message is not None
        assert "CFM balance summary unavailable" in result.error_message
        assert "No portfolios accessible" in result.error_message
