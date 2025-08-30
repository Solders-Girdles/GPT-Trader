#!/usr/bin/env python3
"""Test the trend_breakout strategy to verify it works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta

from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy


def test_trend_breakout():
    """Test trend_breakout strategy with real data."""
    print("Testing Trend Breakout Strategy...")

    # Initialize strategy
    params = TrendBreakoutParams(
        donchian_lookback=20,
        atr_period=14,
        atr_k=2.0,  # Shorter for testing
    )
    strategy = TrendBreakoutStrategy(params)
    print(f"✓ Strategy initialized: {strategy.name}")

    # Get some test data
    source = YFinanceSource()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months of data

    print(f"Fetching data for AAPL from {start_date.date()} to {end_date.date()}...")
    data = source.get_daily_bars("AAPL", start_date.isoformat(), end_date.isoformat())

    if data.empty:
        print("✗ Failed to fetch data")
        return False

    print(f"✓ Fetched {len(data)} days of data")

    # Generate signals
    try:
        signals = strategy.generate_signals(data)
        print("✓ Generated signals successfully")

        # Check signal structure
        expected_cols = ["signal", "donchian_upper", "donchian_lower", "atr"]
        for col in expected_cols:
            if col not in signals.columns:
                print(f"✗ Missing expected column: {col}")
                return False

        # Check for any buy signals
        buy_signals = signals[signals["signal"] == 1]
        print(f"✓ Found {len(buy_signals)} buy signals")

        # Display some statistics
        print("\nSignal Statistics:")
        print(f"  - Total days: {len(signals)}")
        print(f"  - Buy signals: {len(buy_signals)}")
        print(f"  - Signal rate: {len(buy_signals)/len(signals)*100:.1f}%")

        # Show recent signals
        recent_signals = signals.tail(10)
        print("\nLast 10 days:")
        print(recent_signals[["signal", "donchian_upper", "donchian_lower", "atr"]].round(2))

        return True

    except Exception as e:
        print(f"✗ Error generating signals: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_trend_breakout()
    if success:
        print("\n✅ Trend Breakout Strategy is WORKING!")
    else:
        print("\n❌ Trend Breakout Strategy has issues")
    sys.exit(0 if success else 1)
