"""Tests for DeterministicBroker - test infrastructure broker."""

from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.orchestration.deterministic_broker import DeterministicBroker


@pytest.fixture
def broker():
    """Create DeterministicBroker instance."""
    return DeterministicBroker(equity=Decimal("100000"))


class TestBrokerInitialization:
    """Tests for broker initialization."""

    def test_initializes_with_default_equity(self):
        """Initializes with default equity."""
        broker = DeterministicBroker()
        assert broker.equity == Decimal("100000")

    def test_initializes_with_custom_equity(self):
        """Initializes with custom equity."""
        broker = DeterministicBroker(equity=Decimal("50000"))
        assert broker.equity == Decimal("50000")

    def test_initializes_not_connected(self, broker):
        """Starts disconnected."""
        assert broker._connected is False

    def test_initializes_products(self, broker):
        """Has predefined products."""
        assert "BTC-PERP" in broker._products
        assert "ETH-PERP" in broker._products
        assert "XRP-PERP" in broker._products

    def test_initializes_marks(self, broker):
        """Has predefined marks."""
        assert broker.marks["BTC-PERP"] == Decimal("50000")
        assert broker.marks["ETH-PERP"] == Decimal("3000")


class TestConnectivity:
    """Tests for connection management."""

    def test_connect_returns_true(self, broker):
        """connect() returns True and sets connected flag."""
        result = broker.connect()
        assert result is True
        assert broker._connected is True

    def test_disconnect_clears_flag(self, broker):
        """disconnect() clears connected flag."""
        broker.connect()
        broker.disconnect()
        assert broker._connected is False

    def test_validate_connection_when_connected(self, broker):
        """validate_connection() returns True when connected."""
        broker.connect()
        assert broker.validate_connection() is True

    def test_validate_connection_when_not_connected(self, broker):
        """validate_connection() returns False when not connected."""
        assert broker.validate_connection() is False

    def test_get_account_id(self, broker):
        """get_account_id() returns deterministic ID."""
        assert broker.get_account_id() == "DETERMINISTIC"


class TestAccounts:
    """Tests for account operations."""

    def test_list_balances_returns_equity(self, broker):
        """list_balances() returns single USD balance."""
        balances = broker.list_balances()
        assert len(balances) == 1
        assert balances[0].asset == "USD"
        assert balances[0].total == Decimal("100000")
        assert balances[0].available == Decimal("100000")


class TestProducts:
    """Tests for product operations."""

    def test_list_products_all(self, broker):
        """list_products() returns all products when no filter."""
        products = broker.list_products()
        assert len(products) == 3

    def test_list_products_filtered_by_market(self, broker):
        """list_products() filters by market type."""
        products = broker.list_products(market=MarketType.PERPETUAL)
        assert len(products) == 3
        assert all(p.market_type == MarketType.PERPETUAL for p in products)

    def test_get_product_existing(self, broker):
        """get_product() returns existing product."""
        product = broker.get_product("BTC-PERP")
        assert product.symbol == "BTC-PERP"
        assert product.base_asset == "BTC"

    def test_get_product_creates_default_for_unknown(self, broker):
        """get_product() creates default product for unknown symbol."""
        product = broker.get_product("UNKNOWN-PERP")
        assert product.symbol == "UNKNOWN-PERP"
        assert product.market_type == MarketType.PERPETUAL


class TestQuotes:
    """Tests for quote operations."""

    def test_get_quote_uses_mark(self, broker):
        """get_quote() uses mark from marks dict."""
        quote = broker.get_quote("BTC-PERP")
        assert quote.last == Decimal("50000")
        assert quote.bid < quote.last
        assert quote.ask > quote.last

    def test_get_quote_defaults_to_1000(self, broker):
        """get_quote() defaults to 1000 for unknown symbol."""
        quote = broker.get_quote("UNKNOWN")
        assert quote.last == Decimal("1000")


class TestCandles:
    """Tests for candle operations."""

    def test_get_candles_returns_empty(self, broker):
        """get_candles() returns empty list (not implemented)."""
        candles = broker.get_candles("BTC-PERP", "1h")
        assert candles == []


class TestOrderPlacement:
    """Tests for order placement."""

    def test_place_market_order_fills_immediately(self, broker):
        """Market orders fill immediately."""
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("1.0")
        assert order.avg_fill_price == Decimal("50000")

    def test_place_limit_order_submits(self, broker):
        """Limit orders submit but don't fill."""
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("49000"),
        )

        assert order.status == OrderStatus.SUBMITTED
        assert order.filled_quantity == Decimal("0")

    def test_place_order_requires_quantity(self, broker):
        """place_order raises ValueError when quantity is None."""
        with pytest.raises(ValueError, match="requires a quantity"):
            broker.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=None,
            )

    def test_place_order_creates_position(self, broker):
        """Market order creates position."""
        broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        positions = broker.list_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC-PERP"
        assert positions[0].quantity == Decimal("1.0")
        assert positions[0].side == "long"


class TestOrderManagement:
    """Tests for order management operations."""

    def test_get_order_existing(self, broker):
        """get_order() returns existing order."""
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            client_id="test_order",
        )

        retrieved = broker.get_order("test_order")
        assert retrieved is not None
        assert retrieved.id == "test_order"

    def test_get_order_nonexistent(self, broker):
        """get_order() returns None for nonexistent order."""
        result = broker.get_order("nonexistent")
        assert result is None

    def test_cancel_order_submitted(self, broker):
        """cancel_order() cancels submitted order."""
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            client_id="cancel_me",
        )

        success = broker.cancel_order("cancel_me")
        assert success is True

        cancelled = broker.get_order("cancel_me")
        assert cancelled.status == OrderStatus.CANCELLED

    def test_cancel_order_already_filled(self, broker):
        """cancel_order() returns False for filled order."""
        order = broker.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            client_id="already_filled",
        )

        success = broker.cancel_order("already_filled")
        assert success is False

    def test_cancel_order_nonexistent(self, broker):
        """cancel_order() returns False for nonexistent order."""
        success = broker.cancel_order("nonexistent")
        assert success is False

    def test_list_orders_all(self, broker):
        """list_orders() returns all orders."""
        broker.place_order("BTC-PERP", OrderSide.BUY, OrderType.MARKET, Decimal("1.0"))
        broker.place_order("ETH-PERP", OrderSide.SELL, OrderType.LIMIT, Decimal("10.0"))

        orders = broker.list_orders()
        assert len(orders) == 2

    def test_list_orders_filtered_by_status(self, broker):
        """list_orders() filters by status."""
        broker.place_order("BTC-PERP", OrderSide.BUY, OrderType.MARKET, Decimal("1.0"))
        broker.place_order("ETH-PERP", OrderSide.SELL, OrderType.LIMIT, Decimal("10.0"))

        filled = broker.list_orders(status=OrderStatus.FILLED)
        assert len(filled) == 1
        assert filled[0].status == OrderStatus.FILLED

    def test_list_orders_filtered_by_symbol(self, broker):
        """list_orders() filters by symbol."""
        broker.place_order("BTC-PERP", OrderSide.BUY, OrderType.MARKET, Decimal("1.0"))
        broker.place_order("ETH-PERP", OrderSide.SELL, OrderType.MARKET, Decimal("10.0"))

        btc_orders = broker.list_orders(symbol="BTC-PERP")
        assert len(btc_orders) == 1
        assert btc_orders[0].symbol == "BTC-PERP"


class TestPositions:
    """Tests for position operations."""

    def test_list_positions_empty(self, broker):
        """list_positions() returns empty list initially."""
        positions = broker.list_positions()
        assert positions == []

    def test_list_fills_returns_empty(self, broker):
        """list_fills() returns empty list (not implemented)."""
        fills = broker.list_fills()
        assert fills == []


class TestSeedPosition:
    """Tests for seed_position test helper."""

    def test_seed_position_long(self, broker):
        """seed_position() creates long position."""
        broker.seed_position("BTC-PERP", "long", Decimal("2.0"), Decimal("48000"))

        positions = broker.list_positions()
        assert len(positions) == 1
        assert positions[0].side == "long"
        assert positions[0].quantity == Decimal("2.0")
        assert positions[0].entry_price == Decimal("48000")

    def test_seed_position_short(self, broker):
        """seed_position() creates short position."""
        broker.seed_position("ETH-PERP", "short", Decimal("10.0"), Decimal("3100"))

        positions = broker.list_positions()
        assert len(positions) == 1
        assert positions[0].side == "short"
        assert positions[0].quantity == Decimal("10.0")

    def test_seed_position_requires_quantity(self, broker):
        """seed_position() raises ValueError when quantity is None."""
        with pytest.raises(ValueError, match="requires a quantity"):
            broker.seed_position("BTC-PERP", "long", None)


class TestSetMark:
    """Tests for set_mark test helper."""

    def test_set_mark_updates_marks_dict(self, broker):
        """set_mark() updates marks dictionary."""
        broker.set_mark("BTC-PERP", Decimal("51000"))
        assert broker.marks["BTC-PERP"] == Decimal("51000")

    def test_set_mark_updates_position_mark_price(self, broker):
        """set_mark() updates existing position's mark price."""
        broker.seed_position("BTC-PERP", "long", Decimal("1.0"), Decimal("50000"))
        broker.set_mark("BTC-PERP", Decimal("51000"))

        positions = broker.list_positions()
        assert positions[0].mark_price == Decimal("51000")


class TestApplyFillNewPosition:
    """Tests for _apply_fill creating new positions."""

    def test_buy_creates_long_position(self, broker):
        """Buying creates new long position."""
        broker._apply_fill("BTC-PERP", OrderSide.BUY, Decimal("1.0"), Decimal("50000"))

        positions = broker.list_positions()
        assert len(positions) == 1
        assert positions[0].side == "long"
        assert positions[0].quantity == Decimal("1.0")
        assert positions[0].entry_price == Decimal("50000")

    def test_sell_creates_short_position(self, broker):
        """Selling creates new short position."""
        broker._apply_fill("BTC-PERP", OrderSide.SELL, Decimal("1.0"), Decimal("50000"))

        positions = broker.list_positions()
        assert len(positions) == 1
        assert positions[0].side == "short"


class TestApplyFillAddToPosition:
    """Tests for _apply_fill adding to existing position."""

    def test_add_to_long_position(self, broker):
        """Adding to long position increases quantity and averages price."""
        broker.seed_position("BTC-PERP", "long", Decimal("1.0"), Decimal("50000"))
        broker._apply_fill("BTC-PERP", OrderSide.BUY, Decimal("1.0"), Decimal("52000"))

        positions = broker.list_positions()
        assert positions[0].quantity == Decimal("2.0")
        # Entry price should be average: (50000*1 + 52000*1) / 2 = 51000
        assert positions[0].entry_price == Decimal("51000")
        assert positions[0].side == "long"

    def test_add_to_short_position(self, broker):
        """Adding to short position increases quantity and averages price."""
        broker.seed_position("BTC-PERP", "short", Decimal("1.0"), Decimal("50000"))
        broker._apply_fill("BTC-PERP", OrderSide.SELL, Decimal("1.0"), Decimal("48000"))

        positions = broker.list_positions()
        assert positions[0].quantity == Decimal("2.0")
        # Entry price should be average: (50000*1 + 48000*1) / 2 = 49000
        assert positions[0].entry_price == Decimal("49000")
        assert positions[0].side == "short"


class TestApplyFillReducePosition:
    """Tests for _apply_fill reducing existing position."""

    def test_partial_reduce_long_position(self, broker):
        """Partially reducing long position decreases quantity."""
        broker.seed_position("BTC-PERP", "long", Decimal("2.0"), Decimal("50000"))
        broker._apply_fill("BTC-PERP", OrderSide.SELL, Decimal("1.0"), Decimal("51000"))

        positions = broker.list_positions()
        assert positions[0].quantity == Decimal("1.0")
        assert positions[0].entry_price == Decimal("50000")  # Entry unchanged
        assert positions[0].side == "long"

    def test_partial_reduce_short_position(self, broker):
        """Partially reducing short position decreases quantity."""
        broker.seed_position("BTC-PERP", "short", Decimal("2.0"), Decimal("50000"))
        broker._apply_fill("BTC-PERP", OrderSide.BUY, Decimal("1.0"), Decimal("49000"))

        positions = broker.list_positions()
        assert positions[0].quantity == Decimal("1.0")
        assert positions[0].side == "short"


class TestApplyFillClosePosition:
    """Tests for _apply_fill exactly closing position."""

    def test_exact_close_long_position(self, broker):
        """Exactly closing long position deletes it."""
        broker.seed_position("BTC-PERP", "long", Decimal("1.0"), Decimal("50000"))
        broker._apply_fill("BTC-PERP", OrderSide.SELL, Decimal("1.0"), Decimal("51000"))

        positions = broker.list_positions()
        assert len(positions) == 0

    def test_exact_close_short_position(self, broker):
        """Exactly closing short position deletes it."""
        broker.seed_position("BTC-PERP", "short", Decimal("1.0"), Decimal("50000"))
        broker._apply_fill("BTC-PERP", OrderSide.BUY, Decimal("1.0"), Decimal("49000"))

        positions = broker.list_positions()
        assert len(positions) == 0


class TestApplyFillFlipPosition:
    """Tests for _apply_fill flipping position (over-reducing)."""

    def test_flip_long_to_short(self, broker):
        """Over-reducing long position flips to short."""
        broker.seed_position("BTC-PERP", "long", Decimal("1.0"), Decimal("50000"))
        broker._apply_fill("BTC-PERP", OrderSide.SELL, Decimal("2.0"), Decimal("51000"))

        positions = broker.list_positions()
        assert len(positions) == 1
        assert positions[0].side == "short"
        assert positions[0].quantity == Decimal("1.0")  # Leftover
        assert positions[0].entry_price == Decimal("51000")

    def test_flip_short_to_long(self, broker):
        """Over-reducing short position flips to long."""
        broker.seed_position("BTC-PERP", "short", Decimal("1.0"), Decimal("50000"))
        broker._apply_fill("BTC-PERP", OrderSide.BUY, Decimal("2.0"), Decimal("49000"))

        positions = broker.list_positions()
        assert len(positions) == 1
        assert positions[0].side == "long"
        assert positions[0].quantity == Decimal("1.0")  # Leftover
        assert positions[0].entry_price == Decimal("49000")


class TestStreamingMethods:
    """Tests for streaming methods (not implemented)."""

    def test_stream_trades_returns_empty_iterator(self, broker):
        """stream_trades() returns empty iterator."""
        result = broker.stream_trades(["BTC-PERP"])
        assert list(result) == []

    def test_stream_orderbook_returns_empty_iterator(self, broker):
        """stream_orderbook() returns empty iterator."""
        result = broker.stream_orderbook(["BTC-PERP"])
        assert list(result) == []
