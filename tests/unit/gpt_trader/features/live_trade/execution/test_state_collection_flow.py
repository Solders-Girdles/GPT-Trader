"""Tests for StateCollector flow and pricing helpers."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Balance, MarketType, Product
from gpt_trader.features.live_trade.execution.state_collection import StateCollector
from gpt_trader.features.live_trade.risk import ValidationError


def make_product(**overrides: object) -> SimpleNamespace:
    data = {
        "symbol": "BTC-PERP",
        "price_increment": Decimal("0.01"),
        "quote_increment": Decimal("0.01"),
        "bid_price": None,
        "ask_price": None,
        "price": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def configure_broker(
    mock_broker: MagicMock,
    *,
    mark_price: Decimal | None = None,
    mark_exception: Exception | None = None,
    quote_last: Decimal | None = None,
) -> None:
    mock_broker.get_mark_price = MagicMock()
    if mark_exception is not None:
        mock_broker.get_mark_price.side_effect = mark_exception
    else:
        mock_broker.get_mark_price.return_value = mark_price

    if quote_last is None:
        mock_broker.get_quote = MagicMock(return_value=None)
    else:
        mock_broker.get_quote = MagicMock(return_value=SimpleNamespace(last=quote_last))


class TestRequireProduct:
    def test_returns_provided_product(
        self, collector: StateCollector, mock_product: Product
    ) -> None:
        result = collector.require_product("BTC-PERP", mock_product)
        assert result is mock_product

    def test_fetches_from_broker_when_none(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        mock_broker.get_product.return_value = mock_product
        result = collector.require_product("BTC-PERP", None)
        assert result is mock_product
        mock_broker.get_product.assert_called_once_with("BTC-PERP")

    def test_raises_validation_error_when_not_found(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        mock_broker.get_product.return_value = None
        with pytest.raises(ValidationError, match="Product not found"):
            collector.require_product("UNKNOWN-PERP", None)

    def test_provides_synthetic_product_in_integration_mode(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
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
        mock_broker.get_product.return_value = None
        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        result = collector.require_product("BTCUSD", None)
        assert result.symbol == "BTCUSD"
        assert result.base_asset == "BTCUSD"
        assert result.quote_asset == "USD"


@pytest.mark.parametrize(
    "price,mark_price,mark_exception,bid_price,ask_price,quote_last,product_price,quote_increment,expected",
    [
        (
            Decimal("50000"),
            Decimal("51000"),
            None,
            None,
            None,
            None,
            None,
            Decimal("0.01"),
            Decimal("50000"),
        ),
        (None, Decimal("51000"), None, None, None, None, None, Decimal("0.01"), Decimal("51000")),
        (
            None,
            None,
            None,
            Decimal("49000"),
            Decimal("51000"),
            None,
            None,
            Decimal("0.01"),
            Decimal("50000"),
        ),
        (None, None, None, None, None, Decimal("52000"), None, Decimal("0.01"), Decimal("52000")),
        (None, None, None, None, None, None, Decimal("53000"), Decimal("0.01"), Decimal("53000")),
        (None, None, None, None, None, None, None, Decimal("0.01"), Decimal("1")),
        (
            Decimal("0"),
            Decimal("51000"),
            None,
            None,
            None,
            None,
            None,
            Decimal("0.01"),
            Decimal("51000"),
        ),
        (
            None,
            None,
            RuntimeError("API error"),
            None,
            None,
            Decimal("52000"),
            None,
            Decimal("0.01"),
            Decimal("52000"),
        ),
    ],
)
def test_resolve_effective_price(
    price,
    mark_price,
    mark_exception,
    bid_price,
    ask_price,
    quote_last,
    product_price,
    quote_increment,
    expected,
    collector: StateCollector,
    mock_broker: MagicMock,
) -> None:
    configure_broker(
        mock_broker, mark_price=mark_price, mark_exception=mark_exception, quote_last=quote_last
    )
    product = make_product(
        bid_price=bid_price,
        ask_price=ask_price,
        price=product_price,
        quote_increment=quote_increment,
    )
    result = collector.resolve_effective_price(
        symbol="BTC-PERP", side="buy", price=price, product=product
    )
    assert result == expected


class TestStateCollectionFlow:
    def test_full_state_collection_flow(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
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
        balances, equity, collateral, total, positions = collector.collect_account_state()
        assert len(balances) == 2
        assert equity == Decimal("13000")
        assert len(collateral) == 2
        assert total == Decimal("15000")
        pos_dict = collector.build_positions_dict(positions)
        assert "BTC-PERP" in pos_dict
        assert pos_dict["BTC-PERP"]["quantity"] == Decimal("0.5")
