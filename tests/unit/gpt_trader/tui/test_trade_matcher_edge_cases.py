"""Unit tests for TradeMatcher edge cases and formatting."""

from tests.unit.gpt_trader.tui.trade_matcher_test_helpers import create_trade

from gpt_trader.tui.trade_matcher import TradeMatcher


class TestTradeMatcherEdgeCases:
    """Edge cases and formatting behavior for TradeMatcher."""

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
