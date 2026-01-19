"""Unit tests for TradeMatcher matching logic."""

from tests.unit.gpt_trader.tui.trade_matcher_test_helpers import create_trade

from gpt_trader.tui.trade_matcher import TradeMatcher


class TestTradeMatcherMatching:
    """Matching behavior for TradeMatcher."""

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
