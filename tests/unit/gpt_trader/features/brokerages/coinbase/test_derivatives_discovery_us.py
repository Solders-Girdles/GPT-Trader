"""Tests for derivatives_discovery - US market behavior."""

from __future__ import annotations

from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.derivatives_discovery import (
    discover_derivatives_eligibility,
)


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

    def test_empty_dict_balance_summary(self) -> None:
        """Test handling when CFM returns empty dict."""
        broker = Mock()
        broker.get_cfm_balance_summary.return_value = {}

        result = discover_derivatives_eligibility(broker, requested_market="US")

        assert result.cfm_portfolio_accessible is False
        assert "CFM balance summary unavailable" in (result.error_message or "")


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
