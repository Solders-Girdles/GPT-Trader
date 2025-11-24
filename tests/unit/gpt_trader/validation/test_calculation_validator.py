"""Tests for calculation validation helpers."""

import logging
from unittest.mock import patch

import pytest

from gpt_trader.validation.calculation_validator import manual_backtest_example


class TestManualBacktestExample:
    """Test the manual backtest example function."""

    def test_manual_backtest_example_returns_expected_structure(self) -> None:
        """Test that the function returns the expected dictionary structure."""
        result = manual_backtest_example()

        expected_keys = [
            "prices",
            "ma3",
            "ma5",
            "signals",
            "trades",
            "final_value",
            "return_pct",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check data types
        assert isinstance(result["prices"], list)
        assert isinstance(result["ma3"], list)
        assert isinstance(result["ma5"], list)
        assert isinstance(result["signals"], list)
        assert isinstance(result["trades"], list)
        assert isinstance(result["final_value"], (int, float))
        assert isinstance(result["return_pct"], (int, float))

    def test_manual_backtest_example_price_data(self) -> None:
        """Test that the price data is correct."""
        result = manual_backtest_example()
        prices = result["prices"]

        expected_prices = [100, 98, 96, 97, 99, 102, 104, 103, 101, 100]
        assert prices == expected_prices

    def test_manual_backtest_example_moving_averages(self) -> None:
        """Test moving average calculations."""
        result = manual_backtest_example()
        ma3 = result["ma3"]
        ma5 = result["ma5"]

        # MA3 should start from day 3 (index 2)
        assert ma3[0] is None
        assert ma3[1] is None
        assert ma3[2] == pytest.approx((100 + 98 + 96) / 3)
        assert ma3[3] == pytest.approx((98 + 96 + 97) / 3)

        # MA5 should start from day 5 (index 4)
        assert ma5[0] is None
        assert ma5[1] is None
        assert ma5[2] is None
        assert ma5[3] is None
        assert ma5[4] == pytest.approx((100 + 98 + 96 + 97 + 99) / 5)

    def test_manual_backtest_example_signal_generation(self) -> None:
        """Test that signals are generated correctly."""
        result = manual_backtest_example()
        signals = result["signals"]

        # Should have same length as prices
        assert len(signals) == len(result["prices"])

        # First few days should have no signal (insufficient data)
        assert signals[0] == "-"
        assert signals[1] == "-"

        # Check that signals are valid
        valid_signals = {"BUY", "SELL", "HOLD", "-"}
        for signal in signals:
            assert signal in valid_signals

    def test_manual_backtest_example_trading_logic(self) -> None:
        """Test that trading logic produces valid trades."""
        result = manual_backtest_example()
        trades = result["trades"]

        # Should have at least one trade
        assert len(trades) > 0

        for trade in trades:
            assert "day" in trade
            assert "action" in trade
            assert "price" in trade
            assert "cash_after" in trade

            if trade["action"] == "BUY":
                assert "shares" in trade
                assert "cost" in trade
                assert trade["shares"] > 0
                assert trade["cost"] > 0
            elif trade["action"] == "SELL":
                assert "shares" in trade
                assert "proceeds" in trade
                assert "pnl" in trade
                assert trade["shares"] > 0
                assert trade["proceeds"] > 0

    def test_manual_backtest_example_buy_sell_pairs(self) -> None:
        """Test that buys and sells are properly paired."""
        result = manual_backtest_example()
        trades = result["trades"]

        # Should alternate between BUY and SELL
        if len(trades) >= 2:
            for i in range(len(trades) - 1):
                current_action = trades[i]["action"]
                next_action = trades[i + 1]["action"]

                # Should not have consecutive buys or sells
                if current_action == "BUY":
                    assert next_action != "BUY"
                elif current_action == "SELL":
                    assert next_action != "SELL"

    def test_manual_backtest_example_financial_calculations(self) -> None:
        """Test financial calculations are reasonable."""
        result = manual_backtest_example()

        # Starting capital should be 10000
        final_value = result["final_value"]
        return_pct = result["return_pct"]

        # Final value should be positive
        assert final_value > 0

        # Return percentage should be reasonable (not extreme)
        assert -100 <= return_pct <= 1000  # Allow for significant but not insane returns

    @patch("gpt_trader.validation.calculation_validator.logger")
    def test_manual_backtest_example_logging(self, mock_logger: logging.Logger) -> None:
        """Test that the function logs expected messages."""
        manual_backtest_example()

        # Check that logger.info was called multiple times
        assert mock_logger.info.call_count > 10

        # Check for specific log messages
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]

        # Should contain the header
        assert any("MANUAL CALCULATION VALIDATION" in call for call in log_calls)
        assert any("TRADING SIMULATION" in call for call in log_calls)
        assert any("FINAL RESULTS" in call for call in log_calls)

    def test_manual_backtest_example_reproducibility(self) -> None:
        """Test that the function produces consistent results."""
        result1 = manual_backtest_example()
        result2 = manual_backtest_example()

        # Results should be identical
        assert result1 == result2

    def test_manual_backtest_example_ma_crossover_logic(self) -> None:
        """Test specific MA crossover signal logic."""
        result = manual_backtest_example()
        signals = result["signals"]
        ma3 = result["ma3"]
        ma5 = result["ma5"]

        # Find first BUY signal
        buy_days = [i for i, signal in enumerate(signals) if signal == "BUY"]

        if buy_days:
            first_buy = buy_days[0]
            # For a BUY signal, MA3 should cross above MA5
            assert ma3[first_buy] is not None
            assert ma5[first_buy] is not None
            assert ma3[first_buy] > ma5[first_buy]

            # Previous day should have MA3 <= MA5
            if first_buy > 0:
                assert ma3[first_buy - 1] is not None
                assert ma5[first_buy - 1] is not None
                assert ma3[first_buy - 1] <= ma5[first_buy - 1]

    def test_manual_backtest_example_position_management(self) -> None:
        """Test that position management is correct."""
        result = manual_backtest_example()
        trades = result["trades"]

        # Track position through trades
        position = 0
        for trade in trades:
            if trade["action"] == "BUY":
                assert position == 0  # Should only buy when not in position
                position = trade["shares"]
            elif trade["action"] == "SELL":
                assert position > 0  # Should only sell when in position
                position = 0

        # Should end with no position (all positions closed)
        assert position == 0

    def test_manual_backtest_example_cash_flow(self) -> None:
        """Test that cash flow calculations are correct."""
        result = manual_backtest_example()
        trades = result["trades"]

        starting_cash = 10000
        cash = starting_cash

        for trade in trades:
            if trade["action"] == "BUY":
                cost = trade["cost"]
                cash_after = trade["cash_after"]
                assert cash_after == cash - cost
                cash = cash_after
            elif trade["action"] == "SELL":
                proceeds = trade["proceeds"]
                cash_after = trade["cash_after"]
                assert cash_after == cash + proceeds
                cash = cash_after

        # Final cash should be in the result
        assert cash == result["final_value"]  # Assuming no open position

    def test_manual_backtest_example_buy_hold_comparison(self) -> None:
        """Test buy & hold comparison logic."""
        result = manual_backtest_example()
        prices = result["prices"]

        starting_capital = 10000
        buy_hold_shares = int(starting_capital / prices[0])
        buy_hold_value = buy_hold_shares * prices[-1]

        # Buy & hold should be calculated correctly
        assert buy_hold_shares > 0
        assert buy_hold_value > 0

        # Strategy might outperform or underperform buy & hold
        # Both should be reasonable values
        assert result["final_value"] > 0
        assert buy_hold_value > 0
