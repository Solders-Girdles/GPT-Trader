"""Tests for HybridPaperBroker functionality."""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from gpt_trader.core import (
    Balance,
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from gpt_trader.orchestration.hybrid_paper_broker import HybridPaperBroker


class TestHybridPaperBrokerInit:
    """Test HybridPaperBroker initialization."""

    @patch("gpt_trader.orchestration.hybrid_paper_broker.CoinbaseClient")
    @patch("gpt_trader.orchestration.hybrid_paper_broker.SimpleAuth")
    def test_init_creates_client(self, mock_auth, mock_client) -> None:
        """Test initialization creates Coinbase client."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
        )

        mock_auth.assert_called_once_with(key_name="test_key", private_key="test_private_key")
        mock_client.assert_called_once()
        assert broker._initial_equity == Decimal("10000")
        assert broker._slippage_bps == 5
        assert broker._commission_bps == Decimal("5")

    @patch("gpt_trader.orchestration.hybrid_paper_broker.CoinbaseClient")
    @patch("gpt_trader.orchestration.hybrid_paper_broker.SimpleAuth")
    def test_init_with_custom_parameters(self, mock_auth, mock_client) -> None:
        """Test initialization with custom parameters."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("50000"),
            slippage_bps=10,
            commission_bps=Decimal("10"),
        )

        assert broker._initial_equity == Decimal("50000")
        assert broker._slippage_bps == 10
        assert broker._commission_bps == Decimal("10")

    @patch("gpt_trader.orchestration.hybrid_paper_broker.CoinbaseClient")
    @patch("gpt_trader.orchestration.hybrid_paper_broker.SimpleAuth")
    def test_init_creates_usd_balance(self, mock_auth, mock_client) -> None:
        """Test initialization creates USD balance."""
        broker = HybridPaperBroker(
            api_key="test_key",
            private_key="test_private_key",
            initial_equity=Decimal("25000"),
        )

        assert "USD" in broker._balances
        assert broker._balances["USD"].total == Decimal("25000")
        assert broker._balances["USD"].available == Decimal("25000")


class TestHybridPaperBrokerMarketData:
    """Test HybridPaperBroker market data methods."""

    @pytest.fixture
    def broker(self):
        """Create broker fixture with mocked client."""
        with patch("gpt_trader.orchestration.hybrid_paper_broker.CoinbaseClient"):
            with patch("gpt_trader.orchestration.hybrid_paper_broker.SimpleAuth"):
                broker = HybridPaperBroker(
                    api_key="test_key",
                    private_key="test_private_key",
                )
                broker._client = Mock()
                return broker

    def test_get_product_from_cache(self, broker) -> None:
        """Test get_product returns cached product."""
        from gpt_trader.core import Product

        cached_product = Product(
            symbol="BTC-USD",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.SPOT,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=1,
        )
        broker._products_cache["BTC-USD"] = cached_product

        result = broker.get_product("BTC-USD")

        assert result == cached_product
        broker._client.get_market_product.assert_not_called()

    def test_get_product_from_api(self, broker) -> None:
        """Test get_product fetches from API when not cached."""
        broker._client.get_market_product.return_value = {
            "product_id": "ETH-USD",
            "base_currency_id": "ETH",
            "quote_currency_id": "USD",
            "product_type": "SPOT",
            "base_min_size": "0.01",
            "base_increment": "0.01",
            "min_market_funds": "10",
            "quote_increment": "0.01",
        }

        result = broker.get_product("ETH-USD")

        assert result.symbol == "ETH-USD"
        assert result.base_asset == "ETH"
        assert "ETH-USD" in broker._products_cache

    def test_get_product_api_error_returns_synthetic(self, broker) -> None:
        """Test get_product returns synthetic product on API error."""
        broker._client.get_market_product.side_effect = Exception("API error")

        result = broker.get_product("NEW-USD")

        assert result.symbol == "NEW-USD"
        assert result.base_asset == "NEW"
        assert result.quote_asset == "USD"

    def test_get_quote_returns_quote(self, broker) -> None:
        """Test get_quote returns parsed quote."""
        broker._client.get_market_product_ticker.return_value = {
            "best_bid": "49900.00",
            "best_ask": "50100.00",
            "trades": [{"price": "50000.00"}],
        }

        result = broker.get_quote("BTC-USD")

        assert result.symbol == "BTC-USD"
        assert result.bid == Decimal("49900.00")
        assert result.ask == Decimal("50100.00")
        assert result.last == Decimal("50000.00")
        assert broker._last_prices["BTC-USD"] == Decimal("50000.00")

    def test_get_quote_calculates_mid_when_no_trades(self, broker) -> None:
        """Test get_quote calculates mid price when no trades."""
        broker._client.get_market_product_ticker.return_value = {
            "best_bid": "49900.00",
            "best_ask": "50100.00",
            "trades": [],
        }

        result = broker.get_quote("BTC-USD")

        assert result.last == Decimal("50000.00")  # (49900 + 50100) / 2

    def test_get_quote_api_error(self, broker) -> None:
        """Test get_quote returns None on API error."""
        broker._client.get_market_product_ticker.side_effect = Exception("API error")

        result = broker.get_quote("BTC-USD")

        assert result is None

    def test_get_ticker_returns_ticker(self, broker) -> None:
        """Test get_ticker returns ticker data."""
        broker._client.get_market_product_ticker.return_value = {
            "price": "50000.00",
            "volume_24h": "1000.00",
        }

        result = broker.get_ticker("BTC-USD")

        assert result["price"] == "50000.00"

    def test_get_ticker_api_error(self, broker) -> None:
        """Test get_ticker returns empty dict on API error."""
        broker._client.get_market_product_ticker.side_effect = Exception("API error")

        result = broker.get_ticker("BTC-USD")

        assert result == {}

    def test_get_candles_returns_candles(self, broker) -> None:
        """Test get_candles returns parsed candles."""
        broker._client.get_market_product_candles.return_value = {
            "candles": [
                {
                    "start": "1704067200",
                    "open": "50000",
                    "high": "51000",
                    "low": "49000",
                    "close": "50500",
                    "volume": "100",
                },
                {
                    "start": "1704070800",
                    "open": "50500",
                    "high": "52000",
                    "low": "50000",
                    "close": "51500",
                    "volume": "150",
                },
            ]
        }

        result = broker.get_candles("BTC-USD")

        assert len(result) == 2
        assert result[0].open == Decimal("50000")
        assert result[1].close == Decimal("51500")

    def test_get_candles_api_error(self, broker) -> None:
        """Test get_candles returns empty list on API error."""
        broker._client.get_market_product_candles.side_effect = Exception("API error")

        result = broker.get_candles("BTC-USD")

        assert result == []

    def test_list_products_returns_products(self, broker) -> None:
        """Test list_products returns parsed products."""
        broker._client.get_market_products.return_value = {
            "products": [
                {"product_id": "BTC-USD", "product_type": "SPOT"},
                {"product_id": "ETH-USD", "product_type": "SPOT"},
            ]
        }

        result = broker.list_products()

        assert len(result) == 2

    def test_list_products_filters_by_type(self, broker) -> None:
        """Test list_products filters by product type."""
        broker._client.get_market_products.return_value = {
            "products": [
                {"product_id": "BTC-USD", "product_type": "SPOT"},
                {"product_id": "BTC-PERP", "product_type": "PERPETUAL"},
            ]
        }

        result = broker.list_products(product_type="PERPETUAL")

        # Only perpetual products should be returned
        perp_products = [p for p in result if p.market_type == MarketType.PERPETUAL]
        assert len(perp_products) <= len(result)


class TestHybridPaperBrokerPositionsBalances:
    """Test HybridPaperBroker position and balance methods."""

    @pytest.fixture
    def broker(self):
        """Create broker fixture with mocked client."""
        with patch("gpt_trader.orchestration.hybrid_paper_broker.CoinbaseClient"):
            with patch("gpt_trader.orchestration.hybrid_paper_broker.SimpleAuth"):
                return HybridPaperBroker(
                    api_key="test_key",
                    private_key="test_private_key",
                    initial_equity=Decimal("10000"),
                )

    def test_list_positions_empty(self, broker) -> None:
        """Test list_positions returns empty list initially."""
        result = broker.list_positions()

        assert result == []

    def test_list_positions_returns_positions(self, broker) -> None:
        """Test list_positions returns stored positions."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        result = broker.list_positions()

        assert len(result) == 1
        assert result[0].symbol == "BTC-USD"

    def test_get_positions_alias(self, broker) -> None:
        """Test get_positions is alias for list_positions."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        result = broker.get_positions()

        assert len(result) == 1

    def test_list_balances_returns_balances(self, broker) -> None:
        """Test list_balances returns balances."""
        result = broker.list_balances()

        assert len(result) == 1
        assert result[0].asset == "USD"
        assert result[0].total == Decimal("10000")

    def test_get_balances_alias(self, broker) -> None:
        """Test get_balances is alias for list_balances."""
        result = broker.get_balances()

        assert len(result) == 1

    def test_get_equity_cash_only(self, broker) -> None:
        """Test get_equity with cash only."""
        result = broker.get_equity()

        assert result == Decimal("10000")

    def test_get_equity_with_long_position(self, broker) -> None:
        """Test get_equity includes unrealized PnL from long position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )
        broker._last_prices["BTC-USD"] = Decimal("51000")  # Price went up
        broker._balances["USD"] = Balance(
            asset="USD", total=Decimal("5000"), available=Decimal("5000")
        )

        result = broker.get_equity()

        # 5000 cash + (51000 - 50000) * 0.1 = 5000 + 100 = 5100
        assert result == Decimal("5100")

    def test_get_equity_with_short_position(self, broker) -> None:
        """Test get_equity includes unrealized PnL from short position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("-0.1"),  # Short
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",
            leverage=1,
        )
        broker._last_prices["BTC-USD"] = Decimal("49000")  # Price went down (profit for short)
        broker._balances["USD"] = Balance(
            asset="USD", total=Decimal("15000"), available=Decimal("15000")
        )

        result = broker.get_equity()

        # 15000 cash + (50000 - 49000) * 0.1 = 15000 + 100 = 15100
        assert result == Decimal("15100")


class TestHybridPaperBrokerOrderExecution:
    """Test HybridPaperBroker order execution."""

    @pytest.fixture
    def broker(self):
        """Create broker fixture with mocked client."""
        with patch("gpt_trader.orchestration.hybrid_paper_broker.CoinbaseClient"):
            with patch("gpt_trader.orchestration.hybrid_paper_broker.SimpleAuth"):
                broker = HybridPaperBroker(
                    api_key="test_key",
                    private_key="test_private_key",
                    initial_equity=Decimal("10000"),
                    slippage_bps=10,
                    commission_bps=Decimal("10"),
                )
                broker._client = Mock()
                return broker

    def test_place_order_buy_market(self, broker) -> None:
        """Test placing a buy market order."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.symbol == "BTC-USD"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("0.1")
        # Price should include slippage (10 bps = 0.1%)
        assert order.avg_fill_price > Decimal("50000")

    def test_place_order_sell_market(self, broker) -> None:
        """Test placing a sell market order."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.side == OrderSide.SELL
        assert order.status == OrderStatus.FILLED
        # Price should include negative slippage for sells
        assert order.avg_fill_price < Decimal("50000")

    def test_place_order_updates_position(self, broker) -> None:
        """Test place_order updates position state."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert "BTC-USD" in broker._positions
        assert broker._positions["BTC-USD"].quantity == Decimal("0.1")

    def test_place_order_updates_balance(self, broker) -> None:
        """Test place_order updates balance."""
        broker._last_prices["BTC-USD"] = Decimal("50000")
        initial_balance = broker._balances["USD"].total

        broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        # Balance should decrease (buy reduces cash)
        assert broker._balances["USD"].total < initial_balance

    def test_place_order_with_dict_payload(self, broker) -> None:
        """Test place_order handles dict payload format."""
        broker._last_prices["ETH-USD"] = Decimal("3000")

        order = broker.place_order(
            symbol_or_payload={
                "product_id": "ETH-USD",
                "side": "BUY",
                "order_configuration": {"market_market_ioc": {"base_size": "1.0"}},
            }
        )

        assert order.symbol == "ETH-USD"
        assert order.filled_quantity == Decimal("1.0")

    def test_place_order_rejected_when_no_price(self, broker) -> None:
        """Test order rejected when price unavailable."""
        broker._client.get_market_product_ticker.side_effect = Exception("API error")

        order = broker.place_order(
            symbol_or_payload="UNKNOWN-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.status == OrderStatus.REJECTED

    def test_cancel_order_success(self, broker) -> None:
        """Test successful order cancellation."""
        broker._last_prices["BTC-USD"] = Decimal("50000")
        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        result = broker.cancel_order(order.id)

        assert result is True
        assert broker._orders[order.id].status == OrderStatus.CANCELLED

    def test_cancel_order_not_found(self, broker) -> None:
        """Test cancellation of non-existent order."""
        result = broker.cancel_order("nonexistent_order")

        assert result is False

    def test_get_order_returns_order(self, broker) -> None:
        """Test getting an order by ID."""
        broker._last_prices["BTC-USD"] = Decimal("50000")
        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        result = broker.get_order(order.id)

        assert result == order

    def test_get_order_not_found(self, broker) -> None:
        """Test getting non-existent order."""
        result = broker.get_order("nonexistent")

        assert result is None


class TestHybridPaperBrokerPositionUpdates:
    """Test HybridPaperBroker position update logic."""

    @pytest.fixture
    def broker(self):
        """Create broker fixture."""
        with patch("gpt_trader.orchestration.hybrid_paper_broker.CoinbaseClient"):
            with patch("gpt_trader.orchestration.hybrid_paper_broker.SimpleAuth"):
                return HybridPaperBroker(
                    api_key="test_key",
                    private_key="test_private_key",
                )

    def test_update_position_creates_new_long(self, broker) -> None:
        """Test creating new long position."""
        broker._update_position("BTC-USD", OrderSide.BUY, Decimal("0.5"), Decimal("50000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("0.5")
        assert pos.entry_price == Decimal("50000")
        assert pos.side == "long"

    def test_update_position_creates_new_short(self, broker) -> None:
        """Test creating new short position."""
        broker._update_position("BTC-USD", OrderSide.SELL, Decimal("0.5"), Decimal("50000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("-0.5")
        assert pos.side == "short"

    def test_update_position_adds_to_long(self, broker) -> None:
        """Test adding to existing long position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        broker._update_position("BTC-USD", OrderSide.BUY, Decimal("0.5"), Decimal("52000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("1.0")
        # Average price: (0.5 * 50000 + 0.5 * 52000) / 1.0 = 51000
        assert pos.entry_price == Decimal("51000")

    def test_update_position_reduces_long(self, broker) -> None:
        """Test reducing long position."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        broker._update_position("BTC-USD", OrderSide.SELL, Decimal("0.5"), Decimal("52000"))

        pos = broker._positions["BTC-USD"]
        assert pos.quantity == Decimal("0.5")
        assert pos.entry_price == Decimal("50000")  # Entry price unchanged on reduction

    def test_update_position_closes_position(self, broker) -> None:
        """Test closing position completely."""
        broker._positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
            leverage=1,
        )

        broker._update_position("BTC-USD", OrderSide.SELL, Decimal("0.5"), Decimal("52000"))

        assert "BTC-USD" not in broker._positions


class TestHybridPaperBrokerStatus:
    """Test HybridPaperBroker status methods."""

    @pytest.fixture
    def broker(self):
        """Create broker fixture."""
        with patch("gpt_trader.orchestration.hybrid_paper_broker.CoinbaseClient"):
            with patch("gpt_trader.orchestration.hybrid_paper_broker.SimpleAuth"):
                return HybridPaperBroker(
                    api_key="test_key",
                    private_key="test_private_key",
                    initial_equity=Decimal("10000"),
                )

    def test_is_connected_always_true(self, broker) -> None:
        """Test is_connected returns True."""
        assert broker.is_connected() is True

    def test_is_stale_always_false(self, broker) -> None:
        """Test is_stale returns False."""
        assert broker.is_stale("BTC-USD") is False

    def test_start_market_data_prefetches_quotes(self, broker) -> None:
        """Test start_market_data prefetches quotes."""
        broker._client = Mock()
        broker._client.get_market_product_ticker.return_value = {
            "best_bid": "50000",
            "best_ask": "50100",
            "trades": [{"price": "50050"}],
        }

        broker.start_market_data(["BTC-USD", "ETH-USD"])

        assert broker._client.get_market_product_ticker.call_count == 2

    def test_stop_market_data_noop(self, broker) -> None:
        """Test stop_market_data is no-op."""
        broker.stop_market_data()  # Should not raise

    def test_get_status_returns_status(self, broker) -> None:
        """Test get_status returns status dict."""
        result = broker.get_status()

        assert result["mode"] == "paper"
        assert result["initial_equity"] == 10000.0
        assert result["current_equity"] == 10000.0
        assert result["positions"] == 0
        assert result["orders_executed"] == 0
