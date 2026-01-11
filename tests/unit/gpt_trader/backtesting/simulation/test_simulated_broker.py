"""Comprehensive tests for SimulatedBroker."""

from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.types import FeeTier
from gpt_trader.core import MarketType, Product


class TestSimulatedBrokerInitialization:
    """Test SimulatedBroker initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        broker = SimulatedBroker()

        assert broker.equity == Decimal("100000")
        assert broker.fee_tier == FeeTier.TIER_2
        assert broker.connected is False
        assert len(broker.products) == 0
        assert len(broker.positions) == 0

    def test_custom_equity_initialization(self) -> None:
        """Test initialization with custom equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("50000"))

        assert broker.equity == Decimal("50000")
        assert broker.get_equity() == Decimal("50000")

    def test_custom_fee_tier_initialization(self) -> None:
        """Test initialization with custom fee tier."""
        broker = SimulatedBroker(fee_tier=FeeTier.TIER_5)

        assert broker.fee_tier == FeeTier.TIER_5

    def test_initial_balance_matches_equity(self) -> None:
        """Test that initial USDC balance matches equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("75000"))

        balances = broker.list_balances()
        assert len(balances) == 1
        assert balances[0].asset == "USDC"
        assert balances[0].total == Decimal("75000")
        assert balances[0].available == Decimal("75000")


class TestSimulatedBrokerProductManagement:
    """Test product registration and retrieval."""

    @pytest.fixture
    def sample_product(self) -> Product:
        """Create a sample product for testing."""
        return Product(
            symbol="BTC-PERP-USDC",
            base_asset="BTC",
            quote_asset="USDC",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.0001"),
            step_size=Decimal("0.0001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=10,
        )

    def test_register_product(self, sample_product: Product) -> None:
        """Test product registration."""
        broker = SimulatedBroker()

        broker.register_product(sample_product)

        assert "BTC-PERP-USDC" in broker.products
        assert broker.products["BTC-PERP-USDC"] == sample_product

    def test_get_product_exists(self, sample_product: Product) -> None:
        """Test retrieving a registered product."""
        broker = SimulatedBroker()
        broker.register_product(sample_product)

        retrieved = broker.get_product("BTC-PERP-USDC")

        assert retrieved is not None
        assert retrieved.symbol == "BTC-PERP-USDC"
        assert retrieved.base_asset == "BTC"

    def test_get_product_not_exists(self) -> None:
        """Test retrieving a non-existent product returns None."""
        broker = SimulatedBroker()

        retrieved = broker.get_product("NONEXISTENT-USD")

        assert retrieved is None

    def test_register_multiple_products(self) -> None:
        """Test registering multiple products."""
        broker = SimulatedBroker()

        products = [
            Product(
                symbol="BTC-PERP-USDC",
                base_asset="BTC",
                quote_asset="USDC",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.0001"),
                step_size=Decimal("0.0001"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.01"),
                leverage_max=10,
            ),
            Product(
                symbol="ETH-PERP-USDC",
                base_asset="ETH",
                quote_asset="USDC",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.01"),
                leverage_max=10,
            ),
        ]

        for product in products:
            broker.register_product(product)

        assert len(broker.products) == 2
        assert broker.get_product("BTC-PERP-USDC") is not None
        assert broker.get_product("ETH-PERP-USDC") is not None

    def test_overwrite_product(self, sample_product: Product) -> None:
        """Test that registering same symbol overwrites previous product."""
        broker = SimulatedBroker()
        broker.register_product(sample_product)

        # Create updated product with same symbol
        updated_product = Product(
            symbol="BTC-PERP-USDC",
            base_asset="BTC",
            quote_asset="USDC",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),  # Changed
            step_size=Decimal("0.001"),
            min_notional=Decimal("20"),  # Changed
            price_increment=Decimal("0.01"),
            leverage_max=10,
        )
        broker.register_product(updated_product)

        retrieved = broker.get_product("BTC-PERP-USDC")
        assert retrieved.min_size == Decimal("0.001")
        assert retrieved.min_notional == Decimal("20")


class TestSimulatedBrokerConnection:
    """Test connection lifecycle methods."""

    def test_initial_connection_state(self) -> None:
        """Test broker starts disconnected."""
        broker = SimulatedBroker()

        assert broker.connected is False
        assert broker.validate_connection() is False

    def test_connect(self) -> None:
        """Test connect() sets connected state."""
        broker = SimulatedBroker()

        result = broker.connect()

        assert result is True
        assert broker.connected is True
        assert broker.validate_connection() is True

    def test_disconnect(self) -> None:
        """Test disconnect() clears connected state."""
        broker = SimulatedBroker()
        broker.connect()

        broker.disconnect()

        assert broker.connected is False
        assert broker.validate_connection() is False

    def test_connect_disconnect_cycle(self) -> None:
        """Test multiple connect/disconnect cycles."""
        broker = SimulatedBroker()

        # First cycle
        broker.connect()
        assert broker.validate_connection() is True
        broker.disconnect()
        assert broker.validate_connection() is False

        # Second cycle
        broker.connect()
        assert broker.validate_connection() is True

    def test_get_account_id(self) -> None:
        """Test get_account_id returns fixed value."""
        broker = SimulatedBroker()

        assert broker.get_account_id() == "SIMULATED_ACCOUNT"

    def test_get_account_id_consistent(self) -> None:
        """Test get_account_id is consistent across calls."""
        broker = SimulatedBroker()

        id1 = broker.get_account_id()
        id2 = broker.get_account_id()

        assert id1 == id2


class TestSimulatedBrokerAccountInfo:
    """Test account information methods."""

    def test_get_equity(self) -> None:
        """Test get_equity returns initial equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("123456.78"))

        assert broker.get_equity() == Decimal("123456.78")

    def test_get_account_info(self) -> None:
        """Test get_account_info returns cash key."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("50000"))

        info = broker.get_account_info()

        assert "cash" in info
        assert info["cash"] == Decimal("50000")

    def test_list_balances_structure(self) -> None:
        """Test list_balances returns proper Balance objects."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))

        balances = broker.list_balances()

        assert len(balances) == 1
        balance = balances[0]
        assert hasattr(balance, "asset")
        assert hasattr(balance, "total")
        assert hasattr(balance, "available")


class TestSimulatedBrokerPositions:
    """Test position management."""

    def test_list_positions_initially_empty(self) -> None:
        """Test positions list is empty initially."""
        broker = SimulatedBroker()

        positions = broker.list_positions()

        assert positions == []
        assert len(positions) == 0

    def test_positions_dict_initially_empty(self) -> None:
        """Test positions dict is empty initially."""
        broker = SimulatedBroker()

        assert broker.positions == {}


class TestSimulatedBrokerEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_initial_equity(self) -> None:
        """Test initialization with zero equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("0"))

        assert broker.equity == Decimal("0")
        assert broker.get_equity() == Decimal("0")
        balances = broker.list_balances()
        assert balances[0].total == Decimal("0")

    def test_very_large_equity(self) -> None:
        """Test initialization with very large equity."""
        large_equity = Decimal("1000000000000")  # 1 trillion
        broker = SimulatedBroker(initial_equity_usd=large_equity)

        assert broker.equity == large_equity
        assert broker.get_equity() == large_equity

    def test_fractional_equity(self) -> None:
        """Test initialization with fractional equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("12345.6789"))

        assert broker.equity == Decimal("12345.6789")


class TestSimulatedBrokerMarketData:
    """Test market data methods."""

    def test_get_quote_no_data(self) -> None:
        """Test get_quote returns None when no data available."""
        broker = SimulatedBroker()

        quote = broker.get_quote("BTC-USD")

        assert quote is None

    def test_get_quote_with_data(self) -> None:
        """Test get_quote returns stored quote."""
        from datetime import datetime

        from gpt_trader.core import Quote

        broker = SimulatedBroker()
        test_quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )
        broker._current_quote["BTC-USD"] = test_quote

        result = broker.get_quote("BTC-USD")

        assert result is not None
        assert result.symbol == "BTC-USD"
        assert result.bid == Decimal("49990")
        assert result.ask == Decimal("50010")

    def test_get_position_no_position(self) -> None:
        """Test get_position returns None when no position exists."""
        broker = SimulatedBroker()

        position = broker.get_position("BTC-USD")

        assert position is None

    def test_get_position_with_position(self) -> None:
        """Test get_position returns stored position."""
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        test_position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("1000"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        broker.positions["BTC-USD"] = test_position

        result = broker.get_position("BTC-USD")

        assert result is not None
        assert result.symbol == "BTC-USD"
        assert result.quantity == Decimal("1.0")

    def test_get_ticker_with_quote(self) -> None:
        """Test get_ticker returns data from quote."""
        from datetime import datetime

        from gpt_trader.core import Quote

        broker = SimulatedBroker()
        test_quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )
        broker._current_quote["BTC-USD"] = test_quote

        ticker = broker.get_ticker("BTC-USD")

        assert ticker["price"] == "50000"
        assert ticker["bid"] == "49990"
        assert ticker["ask"] == "50010"

    def test_get_ticker_with_bar(self) -> None:
        """Test get_ticker returns data from bar when no quote."""
        from datetime import datetime

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        test_bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = test_bar

        ticker = broker.get_ticker("BTC-USD")

        assert ticker["price"] == "50500"
        assert ticker["volume"] == "100"

    def test_get_ticker_no_data(self) -> None:
        """Test get_ticker returns zeros when no data."""
        broker = SimulatedBroker()

        ticker = broker.get_ticker("BTC-USD")

        assert ticker["price"] == "0"
        assert ticker["bid"] == "0"
        assert ticker["ask"] == "0"
        assert ticker["volume"] == "0"

    def test_get_candles_empty(self) -> None:
        """Test get_candles returns empty list when no history."""
        broker = SimulatedBroker()

        candles = broker.get_candles("BTC-USD")

        assert candles == []

    def test_get_candles_with_history(self) -> None:
        """Test get_candles returns stored candles."""
        from datetime import datetime

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        test_candles = [
            Candle(
                ts=datetime(2024, 1, 1, i, 0, 0),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("100"),
            )
            for i in range(5)
        ]
        broker._candle_history["BTC-USD"] = test_candles

        result = broker.get_candles("BTC-USD")

        assert len(result) == 5

    def test_get_candles_with_limit(self) -> None:
        """Test get_candles respects limit parameter."""
        from datetime import datetime, timedelta

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        test_candles = [
            Candle(
                ts=base_time + timedelta(hours=i),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("100"),
            )
            for i in range(100)
        ]
        broker._candle_history["BTC-USD"] = test_candles

        result = broker.get_candles("BTC-USD", limit=10)

        assert len(result) == 10


class TestSimulatedBrokerEquityTracking:
    """Test equity tracking and drawdown calculation."""

    def test_equity_curve_initially_empty(self) -> None:
        """Test equity curve starts empty."""
        broker = SimulatedBroker()

        assert broker._equity_curve == []

    def test_get_equity_curve(self) -> None:
        """Test get_equity_curve returns stored curve."""
        from datetime import datetime

        broker = SimulatedBroker()
        broker._equity_curve = [
            (datetime(2024, 1, 1), Decimal("100000")),
            (datetime(2024, 1, 2), Decimal("101000")),
        ]

        curve = broker.get_equity_curve()

        assert len(curve) == 2
        assert curve[0][1] == Decimal("100000")

    def test_peak_equity_initialized(self) -> None:
        """Test peak equity starts at initial equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("75000"))

        assert broker._peak_equity == Decimal("75000")

    def test_max_drawdown_initially_zero(self) -> None:
        """Test max drawdown starts at zero."""
        broker = SimulatedBroker()

        assert broker._max_drawdown == Decimal("0")
        assert broker._max_drawdown_usd == Decimal("0")


class TestSimulatedBrokerStatistics:
    """Test statistics tracking."""

    def test_total_fees_initially_zero(self) -> None:
        """Test total fees starts at zero."""
        broker = SimulatedBroker()

        assert broker._total_fees_paid == Decimal("0")

    def test_trade_counters_initially_zero(self) -> None:
        """Test trade counters start at zero."""
        broker = SimulatedBroker()

        assert broker._total_trades == 0
        assert broker._winning_trades == 0
        assert broker._losing_trades == 0

    def test_slippage_initially_zero(self) -> None:
        """Test slippage tracking starts at zero."""
        broker = SimulatedBroker()

        assert broker._total_slippage_bps == Decimal("0")


class TestSimulatedBrokerMarginCalculation:
    """Test margin calculation methods."""

    def test_calculate_margin_used_no_positions(self) -> None:
        """Test margin calculation with no positions returns zero."""
        broker = SimulatedBroker()

        margin = broker._calculate_margin_used()

        assert margin == Decimal("0")

    def test_calculate_margin_used_with_position(self) -> None:
        """Test margin calculation with open position."""
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        # Position: 1 BTC at $50000 with 5x leverage
        # Notional = 1 * 50000 = 50000
        # Margin = 50000 / 5 = 10000
        test_position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        broker.positions["BTC-USD"] = test_position

        margin = broker._calculate_margin_used()

        assert margin == Decimal("10000")

    def test_calculate_margin_used_multiple_positions(self) -> None:
        """Test margin calculation with multiple positions."""
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        # BTC: 1 * 50000 / 5 = 10000
        # ETH: 10 * 3000 / 3 = 10000
        # Total = 20000
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        broker.positions["ETH-USD"] = Position(
            symbol="ETH-USD",
            quantity=Decimal("10.0"),
            side="long",
            entry_price=Decimal("3000"),
            mark_price=Decimal("3000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=3,
        )

        margin = broker._calculate_margin_used()

        assert margin == Decimal("20000")

    def test_calculate_margin_used_default_leverage(self) -> None:
        """Test margin calculation defaults to 1x leverage if not set."""
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        # 1 BTC at $50000 with no leverage (defaults to 1x)
        # Margin = 50000 / 1 = 50000
        test_position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=None,  # type: ignore[arg-type]
        )
        broker.positions["BTC-USD"] = test_position

        margin = broker._calculate_margin_used()

        assert margin == Decimal("50000")


class TestSimulatedBrokerOrderCancellation:
    """Test order cancellation."""

    def test_cancel_open_order(self) -> None:
        """Test cancelling an open order succeeds."""
        from gpt_trader.core import (
            Order,
            OrderSide,
            OrderStatus,
            OrderType,
        )

        broker = SimulatedBroker()
        # Add an open order directly
        order = Order(
            id="test-order-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            status=OrderStatus.SUBMITTED,
            price=Decimal("45000"),
        )
        broker._open_orders["test-order-123"] = order

        result = broker.cancel_order("test-order-123")

        assert result is True
        assert "test-order-123" not in broker._open_orders
        assert "test-order-123" in broker._cancelled_orders
        assert broker._cancelled_orders["test-order-123"].status == OrderStatus.CANCELLED


class TestSimulatedBrokerAccountInfoDetailed:
    """Test detailed account info methods."""

    def test_get_account_info_all_fields(self) -> None:
        """Test get_account_info returns all expected fields."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))

        info = broker.get_account_info()

        assert "cash" in info
        assert "equity" in info
        assert "unrealized_pnl" in info
        assert "realized_pnl" in info
        assert "margin_used" in info

    def test_get_account_info_with_unrealized_pnl(self) -> None:
        """Test get_account_info includes unrealized PnL from positions."""
        from gpt_trader.core import Position

        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))
        # Add position with unrealized profit
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("55000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )

        info = broker.get_account_info()

        assert info["unrealized_pnl"] == Decimal("5000")
        # Equity should include unrealized PnL
        assert info["equity"] == Decimal("105000")


class TestSimulatedBrokerMarkPrice:
    """Tests for get_mark_price method."""

    def test_get_mark_price_from_quote(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Quote

        broker = SimulatedBroker()
        quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )
        broker._current_quote["BTC-USD"] = quote

        mark = broker.get_mark_price("BTC-USD")

        assert mark == Decimal("50000")

    def test_get_mark_price_from_bar(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = bar

        mark = broker.get_mark_price("BTC-USD")

        assert mark == Decimal("50500")

    def test_get_mark_price_no_data(self) -> None:
        broker = SimulatedBroker()

        mark = broker.get_mark_price("BTC-USD")

        assert mark is None


class TestSimulatedBrokerMarketSnapshot:
    """Tests for get_market_snapshot method."""

    def test_snapshot_with_quote(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Quote

        broker = SimulatedBroker()
        quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )
        broker._current_quote["BTC-USD"] = quote

        snapshot = broker.get_market_snapshot("BTC-USD")

        assert snapshot["symbol"] == "BTC-USD"
        assert snapshot["bid"] == Decimal("49990")
        assert snapshot["ask"] == Decimal("50010")
        assert snapshot["last"] == Decimal("50000")
        assert snapshot["spread"] == Decimal("20")
        assert snapshot["mid"] == Decimal("50000")

    def test_snapshot_with_bar(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = bar

        snapshot = broker.get_market_snapshot("BTC-USD")

        assert snapshot["symbol"] == "BTC-USD"
        assert snapshot["open"] == Decimal("49000")
        assert snapshot["high"] == Decimal("51000")
        assert snapshot["low"] == Decimal("48500")
        assert snapshot["close"] == Decimal("50500")
        assert snapshot["volume"] == Decimal("100")

    def test_snapshot_empty(self) -> None:
        broker = SimulatedBroker()

        snapshot = broker.get_market_snapshot("BTC-USD")

        assert snapshot == {"symbol": "BTC-USD"}


class TestSimulatedBrokerPositionPnl:
    """Tests for get_position_pnl method."""

    def test_pnl_with_position(self) -> None:
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("55000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("1000"),
            leverage=5,
        )

        pnl = broker.get_position_pnl("BTC-USD")

        assert pnl["realized_pnl"] == Decimal("1000")
        assert pnl["unrealized_pnl"] == Decimal("5000")
        assert pnl["total_pnl"] == Decimal("6000")

    def test_pnl_no_position(self) -> None:
        broker = SimulatedBroker()

        pnl = broker.get_position_pnl("BTC-USD")

        assert pnl["realized_pnl"] == Decimal("0")
        assert pnl["unrealized_pnl"] == Decimal("0")
        assert pnl["total_pnl"] == Decimal("0")


class TestSimulatedBrokerPositionRisk:
    """Tests for get_position_risk method."""

    def test_risk_with_long_position(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Position, Quote

        broker = SimulatedBroker()
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        # Add quote for mark price
        broker._current_quote["BTC-USD"] = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )

        risk = broker.get_position_risk("BTC-USD")

        assert risk["symbol"] == "BTC-USD"
        assert risk["notional"] == Decimal("50000")
        assert risk["leverage"] == 5
        assert risk["margin_used"] == Decimal("10000")
        assert risk["entry_price"] == Decimal("50000")
        assert risk["mark_price"] == Decimal("50000")
        assert "liquidation_price" in risk

    def test_risk_no_position(self) -> None:
        broker = SimulatedBroker()

        risk = broker.get_position_risk("BTC-USD")

        assert risk == {}


class TestSimulatedBrokerPlaceOrder:
    """Tests for place_order method."""

    def test_place_order_requires_quantity(self) -> None:
        broker = SimulatedBroker()

        with pytest.raises(ValueError, match="quantity is required"):
            broker.place_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=None,  # type: ignore
            )

    def test_place_limit_order_submitted(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        # Add market data
        bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = bar

        order = broker.place_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("48000"),
        )

        assert order is not None
        assert order.status.value == "SUBMITTED"
        assert order.symbol == "BTC-USD"
        assert order.quantity == Decimal("0.1")


class TestSimulatedBrokerCancelOrder:
    """Tests for cancel_order method."""

    def test_cancel_nonexistent_order(self) -> None:
        broker = SimulatedBroker()

        result = broker.cancel_order("nonexistent-order-id")

        assert result is False

    def test_cancel_open_order(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        # Add market data
        bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = bar

        # Place a limit order
        order = broker.place_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("48000"),
        )

        # Cancel it
        result = broker.cancel_order(order.id)

        assert result is True
        assert order.id not in broker._open_orders
