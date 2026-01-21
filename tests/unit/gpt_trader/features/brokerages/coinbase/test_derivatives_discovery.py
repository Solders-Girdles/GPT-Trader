"""Tests for derivatives_discovery - eligibility, data contents, NONE/BOTH markets."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.derivatives_discovery import (
    DerivativesEligibility,
    discover_derivatives_eligibility,
)


class TestDerivativesEligibility:
    """Tests for DerivativesEligibility dataclass."""

    def test_eligibility_creation(self) -> None:
        """Test creating a DerivativesEligibility instance."""
        eligibility = DerivativesEligibility(
            us_derivatives_enabled=True,
            intx_derivatives_enabled=False,
            cfm_portfolio_accessible=True,
            intx_portfolio_accessible=False,
            error_message=None,
            reduce_only_required=False,
            discovery_data={"requested_market": "US"},
        )

        assert eligibility.us_derivatives_enabled is True
        assert eligibility.intx_derivatives_enabled is False
        assert eligibility.cfm_portfolio_accessible is True
        assert eligibility.intx_portfolio_accessible is False
        assert eligibility.error_message is None
        assert eligibility.reduce_only_required is False
        assert eligibility.discovery_data == {"requested_market": "US"}

    def test_eligibility_is_frozen(self) -> None:
        """Test that DerivativesEligibility is immutable."""
        eligibility = DerivativesEligibility(
            us_derivatives_enabled=True,
            intx_derivatives_enabled=False,
            cfm_portfolio_accessible=True,
            intx_portfolio_accessible=False,
            error_message=None,
            reduce_only_required=False,
            discovery_data={},
        )

        with pytest.raises(AttributeError):
            eligibility.us_derivatives_enabled = False  # type: ignore[misc]


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


class TestDiscoverDerivativesNone:
    """Tests for discovery when NONE market is requested."""

    def test_none_market_skips_discovery(self) -> None:
        """Test that NONE market skips all discovery."""
        broker = Mock()

        result = discover_derivatives_eligibility(broker, requested_market="NONE")

        assert result.us_derivatives_enabled is False
        assert result.intx_derivatives_enabled is False
        assert result.cfm_portfolio_accessible is False
        assert result.intx_portfolio_accessible is False
        assert result.error_message is None
        assert result.reduce_only_required is False
        assert result.discovery_data["requested_market"] == "NONE"

        broker.get_cfm_balance_summary.assert_not_called()
        broker.list_portfolios.assert_not_called()


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
