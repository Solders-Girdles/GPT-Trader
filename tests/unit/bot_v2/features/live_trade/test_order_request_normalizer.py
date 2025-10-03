"""Unit tests for OrderRequestNormalizer."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, Mock

import pytest

from bot_v2.errors import ExecutionError
from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    Order,
    OrderSide,
    OrderType,
    Product,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.advanced_execution_models.models import (
    NormalizedOrderRequest,
    OrderConfig,
)
from bot_v2.features.live_trade.order_request_normalizer import OrderRequestNormalizer


@pytest.fixture
def mock_broker():
    """Create mock broker."""
    broker = MagicMock()
    broker.get_product.return_value = Product(
        symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.SPOT,
        step_size=Decimal("0.0001"),
        min_size=Decimal("0.0001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
    )
    broker.get_quote.return_value = Quote(
        symbol="BTC-USD",
        bid=Decimal("50000"),
        ask=Decimal("50010"),
        last=Decimal("50005"),
        ts=datetime.now(),
    )
    return broker


@pytest.fixture
def config():
    """Create order config."""
    return OrderConfig(reject_on_cross=True)


@pytest.fixture
def normalizer(mock_broker, config):
    """Create normalizer instance."""
    pending_orders: dict[str, Order] = {}
    client_order_map: dict[str, str] = {}
    return OrderRequestNormalizer(
        broker=mock_broker,
        pending_orders=pending_orders,
        client_order_map=client_order_map,
        config=config,
    )


class TestOrderRequestNormalizer:
    """Test OrderRequestNormalizer functionality."""

    def test_normalize_basic_request(self, normalizer):
        """Test normalizing a basic order request."""
        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert result is not None
        assert result.symbol == "BTC-USD"
        assert result.side == OrderSide.BUY
        assert result.quantity == Decimal("0.1")
        assert result.order_type == OrderType.MARKET
        assert result.client_id is not None
        assert result.product is not None
        assert result.quote is None  # Not fetched for market orders

    def test_normalize_generates_client_id(self, normalizer):
        """Test that client_id is generated if not provided."""
        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert result is not None
        assert result.client_id.startswith("BTC-USD_buy_")
        assert len(result.client_id.split("_")) == 4  # symbol_side_timestamp_random

    def test_normalize_uses_provided_client_id(self, normalizer):
        """Test that provided client_id is used."""
        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
            client_id="my-custom-id",
        )

        assert result is not None
        assert result.client_id == "my-custom-id"

    def test_normalize_converts_quantity_to_decimal(self, normalizer):
        """Test that integer quantity is converted to Decimal."""
        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=1,  # int
            order_type=OrderType.MARKET,
        )

        assert result is not None
        assert result.quantity == Decimal("1")
        assert isinstance(result.quantity, Decimal)

    def test_normalize_fetches_quote_for_post_only_limit(self, normalizer, mock_broker):
        """Test that quote is fetched for post-only LIMIT orders."""
        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000"),
            post_only=True,
        )

        assert result is not None
        assert result.quote is not None
        assert result.quote.bid == Decimal("50000")
        mock_broker.get_quote.assert_called_once_with("BTC-USD")

    def test_normalize_skips_quote_for_non_post_only(self, normalizer, mock_broker):
        """Test that quote is not fetched for non-post-only orders."""
        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000"),
            post_only=False,
        )

        assert result is not None
        assert result.quote is None
        mock_broker.get_quote.assert_not_called()

    def test_normalize_skips_quote_for_market_orders(self, normalizer, mock_broker):
        """Test that quote is not fetched for market orders even if post_only."""
        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
            post_only=True,  # Should be ignored for market orders
        )

        assert result is not None
        assert result.quote is None
        mock_broker.get_quote.assert_not_called()

    def test_normalize_raises_on_missing_quote_for_post_only(self, normalizer, mock_broker):
        """Test that ExecutionError is raised if quote unavailable for post-only."""
        mock_broker.get_quote.return_value = None

        with pytest.raises(ExecutionError, match="Could not get quote"):
            normalizer.normalize(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.LIMIT,
                limit_price=Decimal("50000"),
                post_only=True,
            )

    def test_normalize_raises_on_quote_fetch_error(self, normalizer, mock_broker):
        """Test that ExecutionError is raised if quote fetch fails."""
        mock_broker.get_quote.side_effect = Exception("API error")

        with pytest.raises(ExecutionError, match="Could not get quote"):
            normalizer.normalize(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                order_type=OrderType.LIMIT,
                limit_price=Decimal("50000"),
                post_only=True,
            )

    def test_normalize_handles_missing_product(self, normalizer, mock_broker):
        """Test that normalization continues if product unavailable."""
        mock_broker.get_product.return_value = None

        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert result is not None
        assert result.product is None  # Should be None but not fail

    def test_normalize_handles_product_fetch_error(self, normalizer, mock_broker):
        """Test that normalization continues if product fetch fails."""
        mock_broker.get_product.side_effect = Exception("API error")

        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert result is not None
        assert result.product is None  # Should be None but not fail

    def test_normalize_returns_none_for_duplicate(self, normalizer):
        """Test that None is returned for duplicate client_id."""
        # Add existing order to maps
        normalizer.client_order_map["existing-id"] = "order-123"
        normalizer.pending_orders["order-123"] = Mock(spec=Order)

        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
            client_id="existing-id",
        )

        assert result is None

    def test_get_existing_order_returns_order(self, normalizer):
        """Test getting existing order by client_id."""
        # Setup existing order
        existing_order = Mock(spec=Order)
        existing_order.id = "order-123"
        normalizer.client_order_map["existing-id"] = "order-123"
        normalizer.pending_orders["order-123"] = existing_order

        result = normalizer.get_existing_order("existing-id")

        assert result is existing_order

    def test_get_existing_order_returns_none_for_missing(self, normalizer):
        """Test that None is returned for non-existent client_id."""
        result = normalizer.get_existing_order("non-existent")
        assert result is None

    def test_normalize_preserves_all_parameters(self, normalizer):
        """Test that all parameters are preserved in normalized request."""
        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.5"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("55000"),
            stop_price=Decimal("54000"),
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
            post_only=False,
            client_id="custom-123",
            leverage=5,
        )

        assert result is not None
        assert result.symbol == "BTC-USD"
        assert result.side == OrderSide.SELL
        assert result.quantity == Decimal("0.5")
        assert result.order_type == OrderType.LIMIT
        assert result.limit_price == Decimal("55000")
        assert result.stop_price == Decimal("54000")
        assert result.time_in_force == TimeInForce.IOC
        assert result.reduce_only is True
        assert result.post_only is False
        assert result.client_id == "custom-123"
        assert result.leverage == 5

    def test_normalize_with_reject_on_cross_disabled(self, mock_broker):
        """Test that quote is not fetched when reject_on_cross is disabled."""
        config = OrderConfig(reject_on_cross=False)
        normalizer = OrderRequestNormalizer(
            broker=mock_broker,
            pending_orders={},
            client_order_map={},
            config=config,
        )

        result = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000"),
            post_only=True,
        )

        assert result is not None
        assert result.quote is None  # Not fetched when reject_on_cross=False
        mock_broker.get_quote.assert_not_called()

    def test_client_id_uniqueness(self, normalizer):
        """Test that generated client_ids are unique."""
        result1 = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        result2 = normalizer.normalize(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert result1 is not None
        assert result2 is not None
        assert result1.client_id != result2.client_id
