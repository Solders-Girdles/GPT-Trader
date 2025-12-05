"""Unit tests for TradeMatcher class."""

from gpt_trader.tui.trade_matcher import TradeMatcher
from gpt_trader.tui.types import Trade


def create_trade(
    trade_id: str,
    symbol: str,
    side: str,
    quantity: str,
    price: str,
    fee: str = "0.00",
) -> Trade:
    """Helper to create a Trade object for testing."""
    return Trade(
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_id=f"order_{trade_id}",
        time="2024-01-15T10:00:00.000Z",
        fee=fee,
    )


class TestTradeMatcher:
    """Test suite for TradeMatcher."""

    def test_simple_long_position(self):
        """Test basic LONG position: BUY then SELL."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "50000.00", "30.00"),
            create_trade("2", "BTC-USD", "SELL", "1.0", "51000.00", "30.60"),
        ]

        pnl_map = matcher.process_trades(trades)

        # Both trades should show the same P&L (profit)
        # P&L = (51000 - 50000) * 1.0 - (30 + 30.60) = 1000 - 60.60 = 939.40
        assert pnl_map["1"] == "+939.40"
        assert pnl_map["2"] == "+939.40"

    def test_simple_short_position(self):
        """Test basic SHORT position: SELL then BUY."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "SELL", "1.0", "51000.00", "30.60"),
            create_trade("2", "BTC-USD", "BUY", "1.0", "50000.00", "30.00"),
        ]

        pnl_map = matcher.process_trades(trades)

        # Both trades should show the same P&L (profit from short)
        # P&L = (51000 - 50000) * 1.0 - (30.60 + 30) = 1000 - 60.60 = 939.40
        assert pnl_map["1"] == "+939.40"
        assert pnl_map["2"] == "+939.40"

    def test_partial_close(self):
        """Test partial position close with multiple trades."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "50000.00", "30.00"),
            create_trade("2", "BTC-USD", "SELL", "0.5", "51000.00", "15.30"),
            create_trade("3", "BTC-USD", "SELL", "0.5", "52000.00", "15.60"),
        ]

        pnl_map = matcher.process_trades(trades)

        # First trade (BUY) - accumulates P&L from both SELL matches
        # First match (0.5): (51000 - 50000) * 0.5 - (30*0.5/1.0 + 15.30) = 500 - 30.30 = 469.70
        # Second match (0.5 remaining): (52000 - 50000) * 0.5 - (30*0.5/0.5 + 15.60) = 1000 - 45.60 = 954.40
        # Total: 469.70 + 954.40 = 1424.10
        assert pnl_map["1"] == "+1424.10"

        # Second trade (SELL 0.5) - matches first 0.5 of BUY
        # P&L: (51000 - 50000) * 0.5 - (30*0.5/1.0 + 15.30) = 500 - 30.30 = 469.70
        assert pnl_map["2"] == "+469.70"

        # Third trade (SELL 0.5) - matches remaining 0.5 of BUY
        # P&L: (52000 - 50000) * 0.5 - (30*0.5/0.5 + 15.60) = 1000 - 45.60 = 954.40
        assert pnl_map["3"] == "+954.40"

    def test_multiple_symbols(self):
        """Test that trades for different symbols are matched independently."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "50000.00", "30.00"),
            create_trade("2", "ETH-USD", "BUY", "10.0", "3000.00", "18.00"),
            create_trade("3", "BTC-USD", "SELL", "1.0", "51000.00", "30.60"),
            create_trade("4", "ETH-USD", "SELL", "10.0", "3100.00", "18.60"),
        ]

        pnl_map = matcher.process_trades(trades)

        # BTC trades
        assert pnl_map["1"] == "+939.40"  # (51000-50000)*1 - (30+30.60)
        assert pnl_map["3"] == "+939.40"

        # ETH trades
        assert pnl_map["2"] == "+963.40"  # (3100-3000)*10 - (18+18.60)
        assert pnl_map["4"] == "+963.40"

    def test_unmatched_trades(self):
        """Test that unmatched trades show 'N/A'."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "50000.00", "30.00"),
            create_trade("2", "ETH-USD", "SELL", "5.0", "3000.00", "9.00"),
        ]

        pnl_map = matcher.process_trades(trades)

        # Both trades are unmatched (different symbols)
        assert pnl_map["1"] == "N/A"
        assert pnl_map["2"] == "N/A"

    def test_fifo_ordering(self):
        """Test that FIFO (first-in-first-out) matching works correctly."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "50000.00", "30.00"),
            create_trade("2", "BTC-USD", "BUY", "1.0", "51000.00", "30.60"),
            create_trade("3", "BTC-USD", "SELL", "1.5", "52000.00", "46.80"),
        ]

        pnl_map = matcher.process_trades(trades)

        # Trade 1 (BUY @ 50000) should be fully matched with first 1.0 of SELL
        # P&L: (52000 - 50000) * 1.0 - (30 + 46.80*1/1.5) = 2000 - 61.20 = 1938.80
        assert pnl_map["1"] == "+1938.80"

        # Trade 2 (BUY @ 51000) should be partially matched with remaining 0.5 of SELL
        # P&L: (52000 - 51000) * 0.5 - (30.60*0.5 + 46.80*0.5/1.5) = 500 - 30.90 = 469.10
        assert pnl_map["2"] == "+469.10"

        # Trade 3 (SELL) - matched with both BUYs, accumulates both P&Ls
        # First match (1.0 with trade 1): 1938.80
        # Second match (0.5 with trade 2): 469.10
        # Total: 1938.80 + 469.10 = 2407.90
        assert pnl_map["3"] == "+2407.90"

    def test_invalid_quantity(self):
        """Test graceful handling of invalid quantity."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "invalid", "50000.00", "30.00"),
            create_trade("2", "BTC-USD", "SELL", "1.0", "51000.00", "30.60"),
        ]

        pnl_map = matcher.process_trades(trades)

        # Trade 1 has invalid quantity, should be skipped
        assert pnl_map["1"] == "N/A"
        # Trade 2 can't match with invalid trade, stays unmatched
        assert pnl_map["2"] == "N/A"

    def test_invalid_price(self):
        """Test graceful handling of invalid price."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "invalid", "30.00"),
            create_trade("2", "BTC-USD", "SELL", "1.0", "51000.00", "30.60"),
        ]

        pnl_map = matcher.process_trades(trades)

        # Trade 1 has invalid price, should be skipped
        assert pnl_map["1"] == "N/A"
        # Trade 2 can't match with invalid trade
        assert pnl_map["2"] == "N/A"

    def test_zero_fee(self):
        """Test that zero fees are handled correctly."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "50000.00", "0.00"),
            create_trade("2", "BTC-USD", "SELL", "1.0", "51000.00", "0.00"),
        ]

        pnl_map = matcher.process_trades(trades)

        # P&L = (51000 - 50000) * 1.0 - 0 = 1000.00
        assert pnl_map["1"] == "+1000.00"
        assert pnl_map["2"] == "+1000.00"

    def test_negative_pnl(self):
        """Test that negative P&L is formatted correctly (with sign)."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "51000.00", "30.60"),
            create_trade("2", "BTC-USD", "SELL", "1.0", "50000.00", "30.00"),
        ]

        pnl_map = matcher.process_trades(trades)

        # P&L = (50000 - 51000) * 1.0 - (30.60 + 30) = -1000 - 60.60 = -1060.60
        assert pnl_map["1"] == "-1060.60"
        assert pnl_map["2"] == "-1060.60"

    def test_zero_pnl(self):
        """Test that zero P&L is formatted correctly."""
        matcher = TradeMatcher()

        # Entry and exit at same price, fees balance out to near-zero
        trades = [
            create_trade("1", "BTC-USD", "BUY", "1.0", "50000.00", "0.00"),
            create_trade("2", "BTC-USD", "SELL", "1.0", "50000.00", "0.00"),
        ]

        pnl_map = matcher.process_trades(trades)

        # P&L = (50000 - 50000) * 1.0 - 0 = 0.00
        assert pnl_map["1"] == "+0.00"
        assert pnl_map["2"] == "+0.00"

    def test_mixed_long_short_positions(self):
        """Test complex scenario with both LONG and SHORT positions."""
        matcher = TradeMatcher()

        trades = [
            # Open LONG
            create_trade("1", "BTC-USD", "BUY", "1.0", "50000.00", "30.00"),
            # Close LONG (profit)
            create_trade("2", "BTC-USD", "SELL", "1.0", "51000.00", "30.60"),
            # Open SHORT
            create_trade("3", "BTC-USD", "SELL", "0.5", "52000.00", "15.60"),
            # Close SHORT (loss)
            create_trade("4", "BTC-USD", "BUY", "0.5", "53000.00", "15.90"),
        ]

        pnl_map = matcher.process_trades(trades)

        # LONG position (trades 1 & 2): profit
        assert pnl_map["1"] == "+939.40"
        assert pnl_map["2"] == "+939.40"

        # SHORT position (trades 3 & 4): loss
        # P&L = (52000 - 53000) * 0.5 - (15.60 + 15.90) = -500 - 31.50 = -531.50
        assert pnl_map["3"] == "-531.50"
        assert pnl_map["4"] == "-531.50"

    def test_empty_trades_list(self):
        """Test that empty trades list returns empty P&L map."""
        matcher = TradeMatcher()

        pnl_map = matcher.process_trades([])

        assert pnl_map == {}

    def test_fractional_quantities(self):
        """Test with very small fractional quantities."""
        matcher = TradeMatcher()

        trades = [
            create_trade("1", "BTC-USD", "BUY", "0.001", "50000.00", "0.30"),
            create_trade("2", "BTC-USD", "SELL", "0.001", "51000.00", "0.31"),
        ]

        pnl_map = matcher.process_trades(trades)

        # P&L = (51000 - 50000) * 0.001 - (0.30 + 0.31) = 1.00 - 0.61 = 0.39
        assert pnl_map["1"] == "+0.39"
        assert pnl_map["2"] == "+0.39"
