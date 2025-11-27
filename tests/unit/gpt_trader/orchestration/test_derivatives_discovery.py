"""Tests for derivatives_discovery - safety-critical startup gating."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from gpt_trader.orchestration.derivatives_discovery import (
    DerivativesEligibility,
    discover_derivatives_eligibility,
)

# ============================================================
# Test: DerivativesEligibility dataclass
# ============================================================


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


# ============================================================
# Test: discover_derivatives_eligibility - NONE market
# ============================================================


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

        # Broker should not be called
        broker.get_cfm_balance_summary.assert_not_called()
        broker.list_portfolios.assert_not_called()


# ============================================================
# Test: discover_derivatives_eligibility - US market
# ============================================================


class TestDiscoverDerivativesUS:
    """Tests for US derivatives (CFM) discovery."""

    def test_us_market_accessible(self) -> None:
        """Test successful US derivatives discovery."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = {
            "balance": "1000",
            "available": "900",
        }

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert result.us_derivatives_enabled is True
        assert result.cfm_portfolio_accessible is True
        assert result.error_message is None
        assert result.reduce_only_required is False

    def test_us_market_broker_missing_method(self) -> None:
        """Test US discovery when broker lacks CFM method."""
        broker = Mock(spec=[])  # No methods

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert result.us_derivatives_enabled is False
        assert result.cfm_portfolio_accessible is False
        assert "Broker does not support CFM endpoints" in (result.error_message or "")
        assert result.reduce_only_required is True

    def test_us_market_empty_response(self) -> None:
        """Test US discovery when CFM returns empty response."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = None

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert result.us_derivatives_enabled is True
        assert result.cfm_portfolio_accessible is False
        assert "CFM balance summary unavailable" in (result.error_message or "")
        assert result.reduce_only_required is True

    def test_us_market_invalid_response_type(self) -> None:
        """Test US discovery when CFM returns non-dict."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = "invalid"

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert result.us_derivatives_enabled is True
        assert result.cfm_portfolio_accessible is False

    def test_us_market_api_error(self) -> None:
        """Test US discovery when API call fails."""
        broker = Mock()
        broker.get_cfm_balance_summary.side_effect = RuntimeError("API error")

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert result.us_derivatives_enabled is True
        assert result.cfm_portfolio_accessible is False
        assert "CFM discovery error" in (result.error_message or "")
        assert result.reduce_only_required is True

    def test_us_market_attribute_error(self) -> None:
        """Test US discovery when AttributeError occurs."""
        broker = Mock()
        broker.get_cfm_balance_summary.side_effect = AttributeError("Missing attr")

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert result.us_derivatives_enabled is False
        assert result.cfm_portfolio_accessible is False
        assert "CFM methods not available" in (result.error_message or "")


# ============================================================
# Test: discover_derivatives_eligibility - INTX market
# ============================================================


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


# ============================================================
# Test: discover_derivatives_eligibility - BOTH markets
# ============================================================


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
        # At least one accessible, so no reduce-only required
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
        # At least one accessible, so no reduce-only required
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
        # Both error messages should be present
        assert "CFM balance summary unavailable" in result.error_message
        assert "No portfolios accessible" in result.error_message


# ============================================================
# Test: fail_on_inaccessible flag
# ============================================================


class TestFailOnInaccessible:
    """Tests for fail_on_inaccessible behavior."""

    def test_fail_on_inaccessible_true_triggers_reduce_only(self) -> None:
        """Test that fail_on_inaccessible=True triggers reduce-only."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = None

        result = discover_derivatives_eligibility(
            broker, requested_market="US", fail_on_inaccessible=True
        )

        assert result.reduce_only_required is True

    def test_fail_on_inaccessible_false_no_reduce_only(self) -> None:
        """Test that fail_on_inaccessible=False doesn't trigger reduce-only."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = None

        result = discover_derivatives_eligibility(
            broker, requested_market="US", fail_on_inaccessible=False
        )

        assert result.reduce_only_required is False

    def test_fail_on_inaccessible_when_accessible(self) -> None:
        """Test that fail_on_inaccessible doesn't affect accessible markets."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = {"balance": "1000"}

        result = discover_derivatives_eligibility(
            broker, requested_market="US", fail_on_inaccessible=True
        )

        assert result.cfm_portfolio_accessible is True
        assert result.reduce_only_required is False


# ============================================================
# Test: discovery_data contents
# ============================================================


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


# ============================================================
# Test: Edge cases
# ============================================================


class TestDerivativesDiscoveryEdgeCases:
    """Tests for edge cases in derivatives discovery."""

    def test_portfolios_list_is_not_list(self) -> None:
        """Test handling when portfolios response is not a list."""
        broker = Mock()
        broker.list_portfolios.return_value = {"error": "unexpected format"}

        result = discover_derivatives_eligibility(broker, requested_market="INTX")

        assert result.intx_portfolio_accessible is False

    def test_empty_dict_balance_summary(self) -> None:
        """Test handling when CFM returns empty dict."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = {}

        result = discover_derivatives_eligibility(broker, requested_market="US")

        # Empty dict is falsy, so it's treated as unavailable
        assert result.cfm_portfolio_accessible is False
        assert "CFM balance summary unavailable" in (result.error_message or "")

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
