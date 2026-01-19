"""Tests for derivatives_discovery - eligibility dataclass and NONE market behavior."""

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
