"""Tests for calculation_validator module - manual backtest validation."""

import logging
from unittest.mock import patch

import pytest

from bot_v2.validation.calculation_validator import manual_backtest_example


class TestManualBacktestExample:
    """Test suite for manual backtest calculation validation."""

    def test_returns_expected_structure(self):
        """Test that the function returns expected data structure."""
        result = manual_backtest_example()

        # Check all expected keys are present
        assert "prices" in result
        assert "ma3" in result
        assert "ma5" in result
        assert "signals" in result
        assert "trades" in result
        assert "final_value" in result
        assert "return_pct" in result

    def test_prices_are_correct(self):
        """Test that prices match expected values."""
        result = manual_backtest_example()

        expected_prices = [100, 98, 96, 97, 99, 102, 104, 103, 101, 100]
        assert result["prices"] == expected_prices

    def test_ma3_calculation(self):
        """Test MA3 moving average calculation."""
        result = manual_backtest_example()
        prices = result["prices"]
        ma3 = result["ma3"]

        # First two values should be None
        assert ma3[0] is None
        assert ma3[1] is None

        # Manual verification of MA3 calculations
        # MA3 at index 2 = (100 + 98 + 96) / 3 = 98.0
        assert ma3[2] == pytest.approx(98.0, abs=0.1)

        # MA3 at index 3 = (98 + 96 + 97) / 3 = 97.0
        assert ma3[3] == pytest.approx(97.0, abs=0.1)

        # MA3 at index 4 = (96 + 97 + 99) / 3 = 97.33
        assert ma3[4] == pytest.approx(97.33, abs=0.1)

    def test_ma5_calculation(self):
        """Test MA5 moving average calculation."""
        result = manual_backtest_example()
        ma5 = result["ma5"]

        # First four values should be None
        for i in range(4):
            assert ma5[i] is None

        # MA5 at index 4 = (100 + 98 + 96 + 97 + 99) / 5 = 98.0
        assert ma5[4] == pytest.approx(98.0, abs=0.1)

        # MA5 at index 5 = (98 + 96 + 97 + 99 + 102) / 5 = 98.4
        assert ma5[5] == pytest.approx(98.4, abs=0.1)

    def test_signal_generation(self):
        """Test that signals are generated correctly."""
        result = manual_backtest_example()
        signals = result["signals"]

        # Should have same length as prices
        assert len(signals) == len(result["prices"])

        # Early signals should be "-" (no data)
        assert signals[0] == "-"
        assert signals[1] == "-"
        assert signals[2] == "-"
        assert signals[3] == "-"

        # Signals should only be BUY, SELL, HOLD, or "-"
        valid_signals = {"BUY", "SELL", "HOLD", "-"}
        for signal in signals:
            assert signal in valid_signals

    def test_buy_signal_detection(self):
        """Test that BUY signals are detected on MA crossover."""
        result = manual_backtest_example()
        signals = result["signals"]
        ma3 = result["ma3"]
        ma5 = result["ma5"]

        # Find any BUY signals
        buy_indices = [i for i, s in enumerate(signals) if s == "BUY"]

        for idx in buy_indices:
            # At BUY signal: MA3 > MA5 and previous MA3 <= MA5
            if idx > 0 and ma3[idx] is not None and ma5[idx] is not None:
                assert ma3[idx] > ma5[idx]
                if ma3[idx - 1] is not None and ma5[idx - 1] is not None:
                    assert ma3[idx - 1] <= ma5[idx - 1]

    def test_sell_signal_detection(self):
        """Test that SELL signals are detected on MA crossover."""
        result = manual_backtest_example()
        signals = result["signals"]
        ma3 = result["ma3"]
        ma5 = result["ma5"]

        # Find any SELL signals
        sell_indices = [i for i, s in enumerate(signals) if s == "SELL"]

        for idx in sell_indices:
            # At SELL signal: MA3 < MA5 and previous MA3 >= MA5
            if idx > 0 and ma3[idx] is not None and ma5[idx] is not None:
                assert ma3[idx] < ma5[idx]
                if ma3[idx - 1] is not None and ma5[idx - 1] is not None:
                    assert ma3[idx - 1] >= ma5[idx - 1]

    def test_trade_execution(self):
        """Test that trades are executed based on signals."""
        result = manual_backtest_example()
        trades = result["trades"]

        # Should have at least one trade
        assert len(trades) > 0

        # All trades should have required fields
        for trade in trades:
            assert "day" in trade
            assert "action" in trade
            assert "price" in trade
            assert "shares" in trade

            # BUY trades need cost and cash_after
            if trade["action"] == "BUY":
                assert "cost" in trade
                assert "cash_after" in trade

            # SELL trades need proceeds, pnl, cash_after
            if trade["action"] == "SELL":
                assert "proceeds" in trade
                assert "pnl" in trade
                assert "cash_after" in trade

    def test_starting_capital(self):
        """Test that starting capital is $10,000."""
        result = manual_backtest_example()
        trades = result["trades"]

        # First trade should be a BUY
        first_trade = trades[0]
        assert first_trade["action"] == "BUY"

        # Cost should be less than or equal to $10,000
        assert first_trade["cost"] <= 10000

    def test_buy_then_sell_logic(self):
        """Test that you can only sell if you have a position."""
        result = manual_backtest_example()
        trades = result["trades"]

        # Track position state
        has_position = False

        for trade in trades:
            if trade["action"] == "BUY":
                # Can only buy if no position
                assert not has_position
                has_position = True
            elif trade["action"] == "SELL":
                # Can only sell if have position
                assert has_position
                has_position = False

    def test_final_value_calculation(self):
        """Test that final value is calculated correctly."""
        result = manual_backtest_example()
        trades = result["trades"]
        prices = result["prices"]
        final_value = result["final_value"]

        # Reconstruct final state
        capital = 10000
        position = 0
        cash = capital

        for trade in trades:
            if trade["action"] == "BUY":
                shares = trade["shares"]
                cost = trade["cost"]
                cash -= cost
                position = shares
            elif trade["action"] == "SELL":
                proceeds = trade["proceeds"]
                cash += proceeds
                position = 0

        # Final value = cash + position value
        calculated_final = cash + (position * prices[-1])
        assert final_value == pytest.approx(calculated_final, abs=0.01)

    def test_return_percentage_calculation(self):
        """Test that return percentage is calculated correctly."""
        result = manual_backtest_example()
        final_value = result["final_value"]
        return_pct = result["return_pct"]

        capital = 10000
        expected_return = ((final_value - capital) / capital) * 100
        assert return_pct == pytest.approx(expected_return, abs=0.01)

    def test_logging_output(self, caplog):
        """Test that logging output is generated."""
        with caplog.at_level(logging.INFO):
            manual_backtest_example()

        # Check that key log messages are present
        log_text = caplog.text
        assert "MANUAL CALCULATION VALIDATION" in log_text
        assert "TRADING SIMULATION" in log_text
        assert "FINAL RESULTS" in log_text
        assert "Day" in log_text  # Day column header

    def test_pnl_calculation_in_trades(self):
        """Test that P&L is calculated correctly for SELL trades."""
        result = manual_backtest_example()
        trades = result["trades"]

        sell_trades = [t for t in trades if t["action"] == "SELL"]

        for i, sell_trade in enumerate(sell_trades):
            # Find corresponding BUY trade (previous trade before this SELL)
            buy_trades_before = [t for t in trades if t["action"] == "BUY" and t["day"] < sell_trade["day"]]

            if buy_trades_before:
                last_buy = buy_trades_before[-1]
                expected_pnl = sell_trade["proceeds"] - last_buy["cost"]
                assert sell_trade["pnl"] == pytest.approx(expected_pnl, abs=0.01)

    def test_ma_values_are_floats_or_none(self):
        """Test that MA values are either float or None."""
        result = manual_backtest_example()

        for val in result["ma3"]:
            assert val is None or isinstance(val, float)

        for val in result["ma5"]:
            assert val is None or isinstance(val, (float, int))

    def test_consistent_results_on_multiple_runs(self):
        """Test that function produces consistent results."""
        result1 = manual_backtest_example()
        result2 = manual_backtest_example()

        # Results should be identical
        assert result1["prices"] == result2["prices"]
        assert result1["ma3"] == result2["ma3"]
        assert result1["ma5"] == result2["ma5"]
        assert result1["signals"] == result2["signals"]
        assert result1["final_value"] == result2["final_value"]
        assert result1["return_pct"] == result2["return_pct"]
