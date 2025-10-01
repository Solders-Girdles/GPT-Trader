"""
Comprehensive tests for paper trading execution engine.

Tests cover:
- Order execution (buy/sell)
- Commission and slippage application
- Position management
- Trade logging
- Account status tracking
- Position limits
- Day trading rules
- Edge cases and error handling
"""

from datetime import datetime, timedelta

import pytest

from bot_v2.features.paper_trade.execution import PaperExecutor
from bot_v2.features.paper_trade.types import AccountStatus, Position, TradeLog


class TestPaperExecutorInitialization:
    """Test executor initialization."""

    def test_default_initialization(self):
        """Test executor with default parameters."""
        executor = PaperExecutor()

        assert executor.initial_capital == 100000
        assert executor.cash == 100000
        assert executor.commission == 0.001
        assert executor.slippage == 0.0005
        assert executor.max_positions == 10
        assert len(executor.positions) == 0
        assert len(executor.trade_log) == 0

    def test_custom_initialization(self):
        """Test executor with custom parameters."""
        executor = PaperExecutor(
            initial_capital=50000, commission=0.002, slippage=0.001, max_positions=5
        )

        assert executor.initial_capital == 50000
        assert executor.cash == 50000
        assert executor.commission == 0.002
        assert executor.slippage == 0.001
        assert executor.max_positions == 5


class TestBuyExecution:
    """Test buy order execution."""

    def test_execute_buy_signal(self):
        """Test successful buy execution."""
        executor = PaperExecutor(initial_capital=10000)

        trade = executor.execute_signal(
            symbol="BTC-USD",
            signal=1,
            current_price=100.0,
            timestamp=datetime.now(),
            position_size=0.95,
        )

        assert trade is not None
        assert trade.side == "buy"
        assert trade.symbol == "BTC-USD"
        assert trade.quantity > 0
        assert trade.price > 100.0  # Price + slippage
        assert trade.commission > 0

        # Check position created
        assert "BTC-USD" in executor.positions
        position = executor.positions["BTC-USD"]
        assert position.quantity == trade.quantity
        assert position.entry_price == trade.price

        # Check cash reduced
        assert executor.cash < 10000

    def test_execute_buy_applies_slippage(self):
        """Test that buy orders apply positive slippage."""
        executor = PaperExecutor(slippage=0.01)  # 1% slippage

        trade = executor.execute_signal(
            symbol="BTC-USD", signal=1, current_price=100.0, timestamp=datetime.now()
        )

        # Execution price should be higher due to slippage
        assert trade.price == 101.0  # 100 * 1.01

    def test_execute_buy_applies_commission(self):
        """Test that buy orders include commission costs."""
        executor = PaperExecutor(commission=0.01, slippage=0.0)  # 1% commission

        initial_cash = executor.cash
        trade = executor.execute_signal(
            symbol="BTC-USD", signal=1, current_price=100.0, timestamp=datetime.now()
        )

        # Calculate expected costs
        cost = trade.quantity * trade.price
        expected_commission = cost * 0.01
        expected_total = cost + expected_commission

        assert abs(trade.commission - expected_commission) < 0.01
        assert abs(initial_cash - executor.cash - expected_total) < 0.01

    def test_execute_buy_respects_position_limit(self):
        """Test that buy is rejected when position limit reached."""
        executor = PaperExecutor(max_positions=2)

        # Fill up position slots
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("ETH-USD", 1, 200.0, datetime.now())

        # Third buy should be rejected
        trade = executor.execute_signal("SOL-USD", 1, 50.0, datetime.now())

        assert trade is None
        assert len(executor.positions) == 2

    def test_execute_buy_insufficient_cash(self):
        """Test buy execution with insufficient cash."""
        executor = PaperExecutor(initial_capital=100)  # Very low capital

        trade = executor.execute_signal(
            symbol="BTC-USD",
            signal=1,
            current_price=10000.0,  # Expensive asset
            timestamp=datetime.now(),
        )

        # Should fail or get minimal shares
        assert trade is None or trade.quantity == 0

    def test_execute_buy_with_position_size(self):
        """Test buy with different position sizes."""
        executor1 = PaperExecutor(initial_capital=10000)
        executor2 = PaperExecutor(initial_capital=10000)

        trade1 = executor1.execute_signal("BTC-USD", 1, 100.0, datetime.now(), position_size=0.5)
        trade2 = executor2.execute_signal("BTC-USD", 1, 100.0, datetime.now(), position_size=1.0)

        # Larger position size should buy more shares
        assert trade2.quantity > trade1.quantity

    def test_execute_buy_ignores_existing_position(self):
        """Test that buy is ignored if position already exists."""
        executor = PaperExecutor()

        # First buy succeeds
        trade1 = executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        assert trade1 is not None

        # Second buy for same symbol is ignored
        trade2 = executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        assert trade2 is None


class TestSellExecution:
    """Test sell order execution."""

    def test_execute_sell_signal(self):
        """Test successful sell execution."""
        executor = PaperExecutor(initial_capital=10000)

        # First buy
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        initial_cash = executor.cash

        # Then sell
        trade = executor.execute_signal("BTC-USD", -1, 110.0, datetime.now())

        assert trade is not None
        assert trade.side == "sell"
        assert trade.symbol == "BTC-USD"
        assert trade.price < 110.0  # Price - slippage
        assert trade.commission > 0

        # Position should be closed
        assert "BTC-USD" not in executor.positions

        # Cash should increase
        assert executor.cash > initial_cash

    def test_execute_sell_applies_slippage(self):
        """Test that sell orders apply negative slippage."""
        executor = PaperExecutor(slippage=0.01)

        # Buy first
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())

        # Sell with slippage
        trade = executor.execute_signal("BTC-USD", -1, 110.0, datetime.now())

        # Execution price should be lower due to slippage
        assert trade.price == 108.9  # 110 * 0.99

    def test_execute_sell_applies_commission(self):
        """Test that sell orders include commission costs."""
        executor = PaperExecutor(commission=0.01, slippage=0.0)

        # Buy first
        buy_trade = executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        cash_before_sell = executor.cash

        # Sell
        sell_trade = executor.execute_signal("BTC-USD", -1, 110.0, datetime.now())

        # Calculate expected proceeds
        gross_proceeds = sell_trade.quantity * sell_trade.price
        expected_commission = gross_proceeds * 0.01
        expected_net_proceeds = gross_proceeds - expected_commission

        assert abs(sell_trade.commission - expected_commission) < 0.01
        assert abs(executor.cash - cash_before_sell - expected_net_proceeds) < 0.01

    def test_execute_sell_without_position(self):
        """Test that sell is ignored without open position."""
        executor = PaperExecutor()

        trade = executor.execute_signal("BTC-USD", -1, 100.0, datetime.now())

        assert trade is None

    def test_execute_sell_profitable_trade(self):
        """Test selling at profit."""
        executor = PaperExecutor(commission=0.0, slippage=0.0)

        initial_equity = executor.cash

        # Buy at 100, sell at 120
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("BTC-USD", -1, 120.0, datetime.now())

        # Should have profit
        assert executor.cash > initial_equity

    def test_execute_sell_losing_trade(self):
        """Test selling at loss."""
        executor = PaperExecutor(commission=0.0, slippage=0.0)

        initial_equity = executor.cash

        # Buy at 100, sell at 80
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("BTC-USD", -1, 80.0, datetime.now())

        # Should have loss
        assert executor.cash < initial_equity


class TestPositionManagement:
    """Test position tracking and updates."""

    def test_update_positions(self):
        """Test updating position values with current prices."""
        executor = PaperExecutor()

        # Create position
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        position = executor.positions["BTC-USD"]
        initial_value = position.value

        # Update with new price
        executor.update_positions({"BTC-USD": 120.0})

        # Position should reflect new price
        assert position.current_price == 120.0
        assert position.value > initial_value
        assert position.unrealized_pnl > 0

    def test_update_positions_multiple_symbols(self):
        """Test updating multiple positions."""
        executor = PaperExecutor()

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("ETH-USD", 1, 200.0, datetime.now())

        executor.update_positions({"BTC-USD": 110.0, "ETH-USD": 220.0})

        btc_pos = executor.positions["BTC-USD"]
        eth_pos = executor.positions["ETH-USD"]

        assert btc_pos.current_price == 110.0
        assert eth_pos.current_price == 220.0
        assert btc_pos.unrealized_pnl > 0
        assert eth_pos.unrealized_pnl > 0

    def test_update_positions_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        executor = PaperExecutor(commission=0.0, slippage=0.0)

        trade = executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        quantity = trade.quantity

        # Update to higher price
        executor.update_positions({"BTC-USD": 120.0})

        position = executor.positions["BTC-USD"]
        expected_pnl = (120.0 - 100.0) * quantity

        assert abs(position.unrealized_pnl - expected_pnl) < 0.01

    def test_update_positions_ignores_unknown_symbols(self):
        """Test that unknown symbols in price update are ignored."""
        executor = PaperExecutor()

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())

        # Update with unknown symbol shouldn't cause error
        executor.update_positions({"BTC-USD": 110.0, "UNKNOWN": 50.0})

        # BTC position should be updated
        assert executor.positions["BTC-USD"].current_price == 110.0


class TestAccountStatus:
    """Test account status reporting."""

    def test_get_account_status_initial(self):
        """Test account status at initialization."""
        executor = PaperExecutor(initial_capital=50000)

        status = executor.get_account_status()

        assert status.cash == 50000
        assert status.positions_value == 0
        assert status.total_equity == 50000
        assert status.buying_power == 100000  # 2x cash
        assert status.margin_used == 0
        assert status.day_trades_remaining == 3

    def test_get_account_status_with_positions(self):
        """Test account status with open positions."""
        executor = PaperExecutor(initial_capital=10000, commission=0.0, slippage=0.0)

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.update_positions({"BTC-USD": 110.0})

        status = executor.get_account_status()

        assert status.cash < 10000
        assert status.positions_value > 0
        assert status.total_equity > status.cash
        assert abs(status.total_equity - 10000) < 1000  # Approximate check

    def test_get_account_status_profitable_position(self):
        """Test account status with profitable position."""
        executor = PaperExecutor(initial_capital=10000, commission=0.0, slippage=0.0)

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.update_positions({"BTC-USD": 150.0})  # 50% gain

        status = executor.get_account_status()

        # Total equity should be higher than initial
        assert status.total_equity > 10000


class TestDayTradingRules:
    """Test day trading pattern rules."""

    def test_day_trade_counter(self):
        """Test day trade counting."""
        executor = PaperExecutor()
        timestamp = datetime.now()

        # Buy and sell same day = day trade
        executor.execute_signal("BTC-USD", 1, 100.0, timestamp)
        executor.execute_signal("BTC-USD", -1, 110.0, timestamp)

        assert executor.day_trades == 1

    def test_day_trade_counter_different_days(self):
        """Test that trades on different days don't count as day trades."""
        executor = PaperExecutor()

        today = datetime.now()
        tomorrow = today + timedelta(days=1)

        executor.execute_signal("BTC-USD", 1, 100.0, today)
        executor.execute_signal("BTC-USD", -1, 110.0, tomorrow)

        assert executor.day_trades == 0

    def test_day_trade_remaining(self):
        """Test day trades remaining calculation."""
        executor = PaperExecutor()
        timestamp = datetime.now()

        status = executor.get_account_status()
        assert status.day_trades_remaining == 3

        # Execute day trade
        executor.execute_signal("BTC-USD", 1, 100.0, timestamp)
        executor.execute_signal("BTC-USD", -1, 110.0, timestamp)

        status = executor.get_account_status()
        assert status.day_trades_remaining == 2

    def test_day_trade_reset_new_day(self):
        """Test day trade counter resets on new day."""
        executor = PaperExecutor()

        # Execute trades on day 1
        day1 = datetime(2024, 1, 1, 10, 0, 0)
        executor.execute_signal("BTC-USD", 1, 100.0, day1)
        executor.execute_signal("BTC-USD", -1, 110.0, day1)
        assert executor.day_trades == 1

        # Execute trade on day 2
        day2 = datetime(2024, 1, 2, 10, 0, 0)
        executor.execute_signal("ETH-USD", 1, 200.0, day2)

        # Counter should reset
        assert executor.day_trades == 0


class TestTradeLogging:
    """Test trade logging functionality."""

    def test_trade_log_records_buy(self):
        """Test that buy trades are logged."""
        executor = PaperExecutor()

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())

        assert len(executor.trade_log) == 1
        trade = executor.trade_log[0]
        assert isinstance(trade, TradeLog)
        assert trade.side == "buy"
        assert trade.symbol == "BTC-USD"

    def test_trade_log_records_sell(self):
        """Test that sell trades are logged."""
        executor = PaperExecutor()

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("BTC-USD", -1, 110.0, datetime.now())

        assert len(executor.trade_log) == 2
        assert executor.trade_log[0].side == "buy"
        assert executor.trade_log[1].side == "sell"

    def test_trade_log_sequential_ids(self):
        """Test that trades get sequential IDs."""
        executor = PaperExecutor()

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("BTC-USD", -1, 110.0, datetime.now())
        executor.execute_signal("ETH-USD", 1, 200.0, datetime.now())

        assert executor.trade_log[0].id == 0
        assert executor.trade_log[1].id == 1
        assert executor.trade_log[2].id == 2

    def test_trade_log_includes_metadata(self):
        """Test that trade log includes all metadata."""
        executor = PaperExecutor(commission=0.001, slippage=0.0005)

        trade = executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())

        assert trade.commission > 0
        assert trade.slippage > 0
        assert trade.timestamp is not None
        assert trade.price > 0
        assert trade.quantity > 0


class TestCloseAllPositions:
    """Test closing all positions functionality."""

    def test_close_all_positions(self):
        """Test closing all open positions."""
        executor = PaperExecutor()

        # Open multiple positions
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("ETH-USD", 1, 200.0, datetime.now())
        executor.execute_signal("SOL-USD", 1, 50.0, datetime.now())

        assert len(executor.positions) == 3

        # Close all
        prices = {"BTC-USD": 110.0, "ETH-USD": 220.0, "SOL-USD": 55.0}
        executor.close_all_positions(prices, datetime.now())

        assert len(executor.positions) == 0
        assert len(executor.trade_log) == 6  # 3 buys + 3 sells

    def test_close_all_positions_missing_prices(self):
        """Test closing positions when some prices missing."""
        executor = PaperExecutor()

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("ETH-USD", 1, 200.0, datetime.now())

        # Only provide one price
        prices = {"BTC-USD": 110.0}
        executor.close_all_positions(prices, datetime.now())

        # Only BTC should be closed
        assert "BTC-USD" not in executor.positions
        assert "ETH-USD" in executor.positions


class TestSharedTypesConversion:
    """Test conversion to shared trading types."""

    def test_get_account_snapshot(self):
        """Test conversion to AccountSnapshot."""
        executor = PaperExecutor(initial_capital=10000)

        snapshot = executor.get_account_snapshot(account_id="test-account")

        assert snapshot.account_id == "test-account"
        assert snapshot.cash == 10000
        assert snapshot.equity == 10000

    def test_get_trading_positions(self):
        """Test conversion to TradingPosition list."""
        executor = PaperExecutor()

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("ETH-USD", 1, 200.0, datetime.now())

        positions = executor.get_trading_positions()

        assert len(positions) == 2
        assert all(hasattr(p, "symbol") for p in positions)
        assert all(hasattr(p, "quantity") for p in positions)

    def test_get_trade_fills(self):
        """Test conversion to TradeFill list."""
        executor = PaperExecutor()

        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        executor.execute_signal("BTC-USD", -1, 110.0, datetime.now())

        fills = executor.get_trade_fills()

        assert len(fills) == 2
        assert all(hasattr(f, "symbol") for f in fills)
        assert all(hasattr(f, "side") for f in fills)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_execute_signal_zero(self):
        """Test that zero signal does nothing."""
        executor = PaperExecutor()

        trade = executor.execute_signal("BTC-USD", 0, 100.0, datetime.now())

        assert trade is None
        assert len(executor.positions) == 0
        assert len(executor.trade_log) == 0

    def test_execute_with_zero_price(self):
        """Test execution with zero price."""
        executor = PaperExecutor()

        trade = executor.execute_signal("BTC-USD", 1, 0.0, datetime.now())

        # Should handle gracefully (no division by zero)
        assert trade is None or trade.quantity == 0

    def test_execute_with_negative_price(self):
        """Test execution with negative price."""
        executor = PaperExecutor()

        trade = executor.execute_signal("BTC-USD", 1, -100.0, datetime.now())

        # Should handle gracefully
        assert trade is None

    def test_multiple_operations_consistency(self):
        """Test consistency across multiple operations."""
        executor = PaperExecutor(initial_capital=10000, commission=0.0, slippage=0.0)

        # Record initial state
        initial_cash = executor.cash

        # Execute multiple trades
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())
        cash_after_buy = executor.cash

        executor.execute_signal("BTC-USD", -1, 100.0, datetime.now())
        final_cash = executor.cash

        # Buying and selling at same price should result in same cash (no commission)
        assert abs(final_cash - initial_cash) < 0.01

    def test_extreme_slippage(self):
        """Test execution with extreme slippage."""
        executor = PaperExecutor(slippage=0.5)  # 50% slippage

        trade = executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())

        # Execution price should reflect extreme slippage
        assert trade.price == 150.0  # 100 * 1.5

    def test_extreme_commission(self):
        """Test execution with extreme commission."""
        executor = PaperExecutor(commission=0.1, slippage=0.0)  # 10% commission

        initial_cash = executor.cash
        executor.execute_signal("BTC-USD", 1, 100.0, datetime.now())

        # High commission should significantly impact cash
        cash_spent = initial_cash - executor.cash
        # Commission should be ~10% of position
        assert cash_spent > initial_cash * 0.05  # Significant amount
