"""Tests for `StateCollector.require_product` and end-to-end state flow."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Balance, MarketType, Product
from gpt_trader.features.live_trade.execution.state_collection import StateCollector
from gpt_trader.features.live_trade.risk import ValidationError


class TestRequireProduct:
    """Tests for require_product method."""

    def test_returns_provided_product(
        self, collector: StateCollector, mock_product: Product
    ) -> None:
        """Test that provided product is returned directly."""
        result = collector.require_product("BTC-PERP", mock_product)

        assert result is mock_product

    def test_fetches_from_broker_when_none(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        """Test that product is fetched from broker when None."""
        mock_broker.get_product.return_value = mock_product

        result = collector.require_product("BTC-PERP", None)

        assert result is mock_product
        mock_broker.get_product.assert_called_once_with("BTC-PERP")

    def test_raises_validation_error_when_not_found(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test that ValidationError is raised when product not found."""
        mock_broker.get_product.return_value = None

        with pytest.raises(ValidationError, match="Product not found"):
            collector.require_product("UNKNOWN-PERP", None)

    def test_provides_synthetic_product_in_integration_mode(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test that synthetic product is provided in integration mode."""
        mock_broker.get_product.return_value = None

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        result = collector.require_product("BTC-PERP", None)

        assert result.symbol == "BTC-PERP"
        assert result.base_asset == "BTC"
        assert result.quote_asset == "PERP"
        assert result.market_type == MarketType.PERPETUAL

    def test_synthetic_product_parses_symbol_without_dash(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test synthetic product parsing when symbol has no dash."""
        mock_broker.get_product.return_value = None

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        result = collector.require_product("BTCUSD", None)

        assert result.symbol == "BTCUSD"
        assert result.base_asset == "BTCUSD"
        assert result.quote_asset == "USD"


class TestStateCollectionFlow:
    """State collection workflow tests."""

    def test_full_state_collection_flow(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test complete state collection flow."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "USD,USDC")
        mock_broker.list_balances.return_value = [
            Balance(asset="USD", total=Decimal("10000"), available=Decimal("8000")),
            Balance(asset="USDC", total=Decimal("5000"), available=Decimal("5000")),
        ]
        mock_broker.list_positions.return_value = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("0.5"),
                side="long",
                entry_price=Decimal("50000"),
                mark_price=Decimal("51000"),
            ),
        ]

        collector = StateCollector(mock_broker, mock_config)

        # Collect account state
        balances, equity, collateral, total, positions = collector.collect_account_state()

        # Verify balances
        assert len(balances) == 2
        assert equity == Decimal("13000")  # 8000 + 5000
        assert len(collateral) == 2
        assert total == Decimal("15000")  # 10000 + 5000

        # Build positions dict
        pos_dict = collector.build_positions_dict(positions)

        assert "BTC-PERP" in pos_dict
        assert pos_dict["BTC-PERP"]["quantity"] == Decimal("0.5")
