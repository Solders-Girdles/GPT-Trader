"""Tests for PortfolioService positions and derivatives."""

from decimal import Decimal
from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService

from .portfolio_service_test_base import PortfolioServiceTestBase


class TestPortfolioServicePositions(PortfolioServiceTestBase):
    def test_list_positions_returns_positions(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        """Test listing positions when derivatives are supported."""
        from gpt_trader.core import Position

        mock_endpoints.supports_derivatives.return_value = True
        mock_client.list_positions.return_value = [
            Position(
                symbol="BTC-PERP",
                quantity=Decimal("0.5"),
                entry_price=Decimal("50000.00"),
                unrealized_pnl=Decimal("100.00"),
                mark_price=Decimal("51000.00"),
                realized_pnl=Decimal("0.00"),
                side="LONG",
            )
        ]

        result = portfolio_service.list_positions()

        assert len(result) == 1
        assert result[0].symbol == "BTC-PERP"

    def test_list_positions_returns_empty_when_derivatives_not_supported(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        """Test list_positions returns empty when derivatives not supported."""
        mock_endpoints.supports_derivatives.return_value = False

        result = portfolio_service.list_positions()

        assert result == []
        mock_client.list_positions.assert_not_called()

    def test_list_positions_handles_exception(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        """Test list_positions handles API exception."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.list_positions.side_effect = Exception("API error")

        result = portfolio_service.list_positions()

        assert result == []

    def test_get_position_returns_position(
        self,
        portfolio_service: PortfolioService,
        mock_client: Mock,
        mock_endpoints: Mock,
    ) -> None:
        """Test getting a single position."""
        mock_endpoints.supports_derivatives.return_value = True
        mock_client.get_cfm_position.return_value = {
            "product_id": "BTC-PERP",
            "side": "LONG",
            "contracts": "0.5",
            "entry_price": "50000.00",
            "unrealized_pnl": "100.00",
            "realized_pnl": "100.00",
        }

        result = portfolio_service.get_position("BTC-PERP")

        assert result is not None
        assert result.symbol == "BTC-PERP"

    def test_get_position_returns_none_when_not_supported(
        self,
        portfolio_service: PortfolioService,
        mock_endpoints: Mock,
    ) -> None:
        """Test get_position returns None when derivatives not supported."""
        mock_endpoints.supports_derivatives.return_value = False

        result = portfolio_service.get_position("BTC-PERP")

        assert result is None
