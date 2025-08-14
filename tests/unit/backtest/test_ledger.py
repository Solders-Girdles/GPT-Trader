"""
Comprehensive unit tests for Ledger system.

Tests order execution, position tracking, P&L calculation,
trade recording, and transaction cost handling.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
from bot.exec.ledger import Fill, Ledger, Order, Position, Trade


class TestOrder:
    """Test Order dataclass."""

    def test_order_creation(self):
        """Test Order creation."""
        ts = datetime.now()
        order = Order(
            symbol="AAPL", side="BUY", qty=100, price=150.50, ts=ts, reason="Signal triggered"
        )

        assert order.symbol == "AAPL"
        assert order.side == "BUY"
        assert order.qty == 100
        assert order.price == 150.50
        assert order.ts == ts
        assert order.reason == "Signal triggered"

    def test_order_defaults(self):
        """Test Order with default values."""
        ts = datetime.now()
        order = Order(symbol="GOOGL", side="SELL", qty=50, price=2800.00, ts=ts)

        assert order.reason == ""


class TestFill:
    """Test Fill dataclass."""

    def test_fill_creation(self):
        """Test Fill creation."""
        ts_order = datetime.now()
        ts_fill = ts_order + timedelta(seconds=1)

        order = Order("AAPL", "BUY", 100, 150.00, ts_order)
        fill = Fill(order=order, qty=100, price=150.10, ts=ts_fill, cost=15.01)  # Commission

        assert fill.order == order
        assert fill.qty == 100
        assert fill.price == 150.10
        assert fill.ts == ts_fill
        assert fill.cost == 15.01

    def test_partial_fill(self):
        """Test partial fill."""
        ts = datetime.now()
        order = Order("MSFT", "BUY", 200, 300.00, ts)

        # Partial fill
        fill = Fill(
            order=order,
            qty=50,  # Only 50 of 200
            price=300.05,
            ts=ts + timedelta(seconds=1),
            cost=15.00,
        )

        assert fill.qty < order.qty
        assert fill.qty == 50


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test Position creation."""
        ts = datetime.now()
        position = Position(
            symbol="AAPL", qty=100, avg_price=150.00, entry_ts=ts, realized_pnl=500.00, costs=20.00
        )

        assert position.symbol == "AAPL"
        assert position.qty == 100
        assert position.avg_price == 150.00
        assert position.entry_ts == ts
        assert position.realized_pnl == 500.00
        assert position.costs == 20.00

    def test_position_defaults(self):
        """Test Position with default values."""
        position = Position(symbol="GOOGL")

        assert position.qty == 0
        assert position.avg_price == 0.0
        assert position.entry_ts is None
        assert position.realized_pnl == 0.0
        assert position.costs == 0.0

    def test_position_value_calculation(self):
        """Test position value calculation."""
        position = Position(symbol="MSFT", qty=100, avg_price=300.00)

        # Market value at current price
        current_price = 310.00
        market_value = position.qty * current_price
        unrealized_pnl = (current_price - position.avg_price) * position.qty

        assert market_value == 31000.00
        assert unrealized_pnl == 1000.00


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test Trade creation."""
        entry_ts = datetime(2024, 1, 1, 9, 30)
        exit_ts = datetime(2024, 1, 5, 15, 30)

        trade = Trade(
            symbol="AAPL",
            entry_ts=entry_ts,
            entry_price=150.00,
            exit_ts=exit_ts,
            exit_price=155.00,
            qty=100,
            pnl=500.00,
            rtn=0.0333,  # 3.33% return
            bars_held=4,
            reason_exit="Take profit",
        )

        assert trade.symbol == "AAPL"
        assert trade.entry_ts == entry_ts
        assert trade.entry_price == 150.00
        assert trade.exit_ts == exit_ts
        assert trade.exit_price == 155.00
        assert trade.qty == 100
        assert trade.pnl == 500.00
        assert trade.rtn == 0.0333
        assert trade.bars_held == 4
        assert trade.reason_exit == "Take profit"

    def test_trade_calculations(self):
        """Test trade P&L calculations."""
        trade = Trade(
            symbol="GOOGL",
            entry_ts=datetime.now(),
            entry_price=2800.00,
            exit_ts=datetime.now() + timedelta(days=10),
            exit_price=2900.00,
            qty=10,
            pnl=(2900.00 - 2800.00) * 10,
            rtn=(2900.00 - 2800.00) / 2800.00,
            bars_held=10,
            reason_exit="Signal",
        )

        assert trade.pnl == 1000.00
        assert abs(trade.rtn - 0.0357) < 0.0001  # ~3.57% return


class TestLedger:
    """Test Ledger class."""

    @pytest.fixture
    def ledger(self):
        """Create a fresh ledger instance."""
        return Ledger()

    def test_ledger_initialization(self, ledger):
        """Test ledger initialization."""
        assert ledger.positions == {}
        assert ledger.orders == []
        assert ledger.fills == []
        assert ledger.trades == []
        assert ledger.cash == 0.0
        assert ledger.initial_capital == 0.0

    def test_set_initial_capital(self, ledger):
        """Test setting initial capital."""
        ledger.set_initial_capital(100000.0)

        assert ledger.initial_capital == 100000.0
        assert ledger.cash == 100000.0

    def test_process_buy_order(self, ledger):
        """Test processing a buy order."""
        ledger.set_initial_capital(100000.0)

        ts = datetime.now()
        order = Order(symbol="AAPL", side="BUY", qty=100, price=150.00, ts=ts)

        # Process the order
        ledger.process_order(order)

        # Check position created
        assert "AAPL" in ledger.positions
        position = ledger.positions["AAPL"]
        assert position.qty == 100
        assert position.avg_price == 150.00

        # Check cash reduced
        assert ledger.cash == 100000.0 - (100 * 150.00)

    def test_process_sell_order(self, ledger):
        """Test processing a sell order."""
        ledger.set_initial_capital(100000.0)

        # First buy
        buy_ts = datetime.now()
        buy_order = Order("AAPL", "BUY", 100, 150.00, buy_ts)
        ledger.process_order(buy_order)

        # Then sell
        sell_ts = buy_ts + timedelta(days=5)
        sell_order = Order("AAPL", "SELL", 100, 155.00, sell_ts)
        ledger.process_order(sell_order)

        # Position should be closed
        assert ledger.positions["AAPL"].qty == 0

        # Check cash increased
        expected_cash = 100000.0 - (100 * 150.00) + (100 * 155.00)
        assert ledger.cash == expected_cash

        # Check trade recorded
        assert len(ledger.trades) == 1
        trade = ledger.trades[0]
        assert trade.symbol == "AAPL"
        assert trade.pnl == (155.00 - 150.00) * 100

    def test_partial_sell(self, ledger):
        """Test partial position sell."""
        ledger.set_initial_capital(100000.0)

        # Buy 200 shares
        buy_order = Order("MSFT", "BUY", 200, 300.00, datetime.now())
        ledger.process_order(buy_order)

        # Sell 50 shares
        sell_order = Order("MSFT", "SELL", 50, 310.00, datetime.now())
        ledger.process_order(sell_order)

        # Position should have 150 shares left
        position = ledger.positions["MSFT"]
        assert position.qty == 150
        assert position.avg_price == 300.00

    def test_averaging_positions(self, ledger):
        """Test averaging when adding to positions."""
        ledger.set_initial_capital(100000.0)

        # First buy
        order1 = Order("GOOGL", "BUY", 10, 2800.00, datetime.now())
        ledger.process_order(order1)

        # Second buy at different price
        order2 = Order("GOOGL", "BUY", 10, 2900.00, datetime.now())
        ledger.process_order(order2)

        # Check averaged price
        position = ledger.positions["GOOGL"]
        assert position.qty == 20
        expected_avg = (10 * 2800.00 + 10 * 2900.00) / 20
        assert position.avg_price == expected_avg

    def test_commission_handling(self, ledger):
        """Test commission cost handling."""
        ledger.set_initial_capital(100000.0)
        ledger.commission_rate = 0.001  # 0.1%

        # Buy with commission
        order = Order("AAPL", "BUY", 100, 150.00, datetime.now())
        ledger.process_order(order)

        # Check commission deducted from cash
        trade_value = 100 * 150.00
        commission = trade_value * 0.001
        expected_cash = 100000.0 - trade_value - commission

        assert abs(ledger.cash - expected_cash) < 0.01

        # Check costs tracked in position
        position = ledger.positions["AAPL"]
        assert position.costs > 0

    def test_insufficient_cash(self, ledger):
        """Test handling of insufficient cash for order."""
        ledger.set_initial_capital(1000.0)  # Only $1000

        # Try to buy $15000 worth
        order = Order("AAPL", "BUY", 100, 150.00, datetime.now())

        # Should handle gracefully (implementation dependent)
        # May either reject order or adjust quantity
        result = ledger.process_order(order)

        # Cash should not go negative
        assert ledger.cash >= 0

    def test_multiple_positions(self, ledger):
        """Test managing multiple positions."""
        ledger.set_initial_capital(100000.0)

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        for i, symbol in enumerate(symbols):
            order = Order(
                symbol=symbol, side="BUY", qty=10 + i * 5, price=100 + i * 50, ts=datetime.now()
            )
            ledger.process_order(order)

        # Check all positions created
        assert len(ledger.positions) == 4
        for symbol in symbols:
            assert symbol in ledger.positions
            assert ledger.positions[symbol].qty > 0

    def test_trade_history(self, ledger):
        """Test trade history recording."""
        ledger.set_initial_capital(100000.0)

        # Execute multiple round-trip trades
        trades_to_execute = [
            ("AAPL", 100, 150.00, 155.00),
            ("GOOGL", 10, 2800.00, 2850.00),
            ("MSFT", 50, 300.00, 295.00),  # Losing trade
        ]

        for symbol, qty, buy_price, sell_price in trades_to_execute:
            # Buy
            buy_order = Order(symbol, "BUY", qty, buy_price, datetime.now())
            ledger.process_order(buy_order)

            # Sell
            sell_order = Order(symbol, "SELL", qty, sell_price, datetime.now())
            ledger.process_order(sell_order)

        # Check trades recorded
        assert len(ledger.trades) == 3

        # Check P&L calculations
        assert ledger.trades[0].pnl == (155.00 - 150.00) * 100  # AAPL profit
        assert ledger.trades[1].pnl == (2850.00 - 2800.00) * 10  # GOOGL profit
        assert ledger.trades[2].pnl == (295.00 - 300.00) * 50  # MSFT loss

    def test_get_portfolio_value(self, ledger):
        """Test portfolio value calculation."""
        ledger.set_initial_capital(100000.0)

        # Create positions
        ledger.process_order(Order("AAPL", "BUY", 100, 150.00, datetime.now()))
        ledger.process_order(Order("GOOGL", "BUY", 10, 2800.00, datetime.now()))

        # Calculate portfolio value at current prices
        current_prices = {"AAPL": 155.00, "GOOGL": 2850.00}

        portfolio_value = ledger.get_portfolio_value(current_prices)

        expected_value = ledger.cash + (100 * 155.00) + (10 * 2850.00)
        assert abs(portfolio_value - expected_value) < 0.01

    def test_get_returns(self, ledger):
        """Test returns calculation."""
        ledger.set_initial_capital(100000.0)

        # Execute some trades
        ledger.process_order(Order("AAPL", "BUY", 100, 150.00, datetime.now()))

        # Calculate returns with current price
        current_prices = {"AAPL": 157.50}  # 5% gain

        returns = ledger.get_returns(current_prices)

        portfolio_value = ledger.get_portfolio_value(current_prices)
        expected_return = (portfolio_value - ledger.initial_capital) / ledger.initial_capital

        assert abs(returns - expected_return) < 0.0001

    def test_reset_ledger(self, ledger):
        """Test resetting the ledger."""
        ledger.set_initial_capital(100000.0)

        # Add some data
        ledger.process_order(Order("AAPL", "BUY", 100, 150.00, datetime.now()))
        ledger.process_order(Order("AAPL", "SELL", 100, 155.00, datetime.now()))

        # Reset
        ledger.reset()

        # Check everything cleared
        assert ledger.positions == {}
        assert ledger.orders == []
        assert ledger.fills == []
        assert ledger.trades == []
        assert ledger.cash == ledger.initial_capital

    def test_export_trades_to_dataframe(self, ledger):
        """Test exporting trades to DataFrame."""
        ledger.set_initial_capital(100000.0)

        # Execute trades
        for i in range(5):
            symbol = f"STOCK{i}"
            ledger.process_order(Order(symbol, "BUY", 100, 100.0, datetime.now()))
            ledger.process_order(Order(symbol, "SELL", 100, 105.0, datetime.now()))

        # Export to DataFrame
        df = ledger.export_trades()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "symbol" in df.columns
        assert "pnl" in df.columns
        assert "rtn" in df.columns


class TestLedgerEdgeCases:
    """Test edge cases and error conditions for Ledger."""

    @pytest.fixture
    def ledger(self):
        return Ledger()

    def test_sell_without_position(self, ledger):
        """Test selling without holding position."""
        ledger.set_initial_capital(100000.0)

        # Try to sell without buying first
        sell_order = Order("AAPL", "SELL", 100, 150.00, datetime.now())

        # Should handle gracefully (no short selling in long-only ledger)
        ledger.process_order(sell_order)

        # No position should be created
        assert "AAPL" not in ledger.positions or ledger.positions["AAPL"].qty == 0

    def test_oversell_position(self, ledger):
        """Test selling more than owned."""
        ledger.set_initial_capital(100000.0)

        # Buy 50 shares
        ledger.process_order(Order("MSFT", "BUY", 50, 300.00, datetime.now()))

        # Try to sell 100 shares
        sell_order = Order("MSFT", "SELL", 100, 310.00, datetime.now())
        ledger.process_order(sell_order)

        # Should only sell what we have
        position = ledger.positions["MSFT"]
        assert position.qty == 0  # All 50 sold, not negative

    def test_zero_quantity_order(self, ledger):
        """Test handling zero quantity orders."""
        ledger.set_initial_capital(100000.0)

        order = Order("AAPL", "BUY", 0, 150.00, datetime.now())
        ledger.process_order(order)

        # Should not create position
        assert "AAPL" not in ledger.positions or ledger.positions["AAPL"].qty == 0

    def test_negative_price_order(self, ledger):
        """Test handling negative price orders."""
        ledger.set_initial_capital(100000.0)

        # Should handle or reject appropriately
        order = Order("AAPL", "BUY", 100, -150.00, datetime.now())

        # Implementation should validate
        # Either reject or handle gracefully
        initial_cash = ledger.cash
        ledger.process_order(order)

        # Cash should not increase from negative price
        assert ledger.cash <= initial_cash

    def test_extreme_values(self, ledger):
        """Test handling extreme values."""
        ledger.set_initial_capital(1e12)  # Trillion dollars

        # Very large order
        order = Order("BRK.A", "BUY", 1000, 500000.00, datetime.now())  # $500M order
        ledger.process_order(order)

        assert "BRK.A" in ledger.positions
        assert ledger.positions["BRK.A"].qty == 1000
