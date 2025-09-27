#!/usr/bin/env python3
"""Test the momentum strategy to verify it works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta

from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.strategy.momentum import MomentumParams, MomentumStrategy


def test_momentum():
    """Test momentum strategy with real data."""
    print("Testing Momentum Strategy...")

    # Initialize strategy
    params = MomentumParams(
        roc_period=15,  # Shorter for more signals in testing
        momentum_threshold=0.02,  # 2% momentum threshold
        momentum_exit_threshold=0.005,  # 0.5% exit threshold
        volume_threshold=1.3,  # 1.3x volume threshold
        atr_period=14,
    )
    strategy = MomentumStrategy(params)
    print(f"✓ Strategy initialized: {strategy.name}")

    # Get some test data
    source = YFinanceSource()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)  # 4 months of data

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
        expected_cols = [
            "signal", "roc", "enhanced_momentum", "volume_ma", 
            "volume_ratio", "volume_confirmation", "atr"
        ]
        for col in expected_cols:
            if col not in signals.columns:
                print(f"✗ Missing expected column: {col}")
                return False

        # Check for any buy signals
        buy_signals = signals[signals["signal"] == 1]
        entry_signals = signals[signals["signal"].diff() == 1]
        print(f"✓ Found {len(buy_signals)} total buy signal periods")
        print(f"✓ Found {len(entry_signals)} distinct entry signals")

        # Display some statistics
        print("\nSignal Statistics:")
        print(f"  - Total days: {len(signals)}")
        print(f"  - Buy signal periods: {len(buy_signals)}")
        print(f"  - Entry signals: {len(entry_signals)}")
        print(f"  - Signal rate: {len(buy_signals)/len(signals)*100:.1f}%")
        
        # Volume confirmation statistics
        volume_conf_signals = signals[signals["volume_confirmation"] == 1]
        print(f"  - Volume confirmed days: {len(volume_conf_signals)}")
        print(f"  - Volume confirmation rate: {len(volume_conf_signals)/len(signals)*100:.1f}%")

        # Show recent signals
        recent_signals = signals.tail(10)
        print("\nLast 10 days:")
        print(recent_signals[["signal", "roc", "volume_ratio", "volume_confirmation"]].round(3))

        # Show entry signals with details
        if len(entry_signals) > 0:
            print(f"\nFirst few entry signals:")
            entry_details = entry_signals.head(3)[
                ["signal", "roc", "enhanced_momentum", "volume_ratio", "volume_confirmation"]
            ].round(3)
            print(entry_details)

        return True

    except Exception as e:
        print(f"✗ Error generating signals: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_momentum()
    if success:
        print("\n✅ Momentum Strategy is WORKING!")
    else:
        print("\n❌ Momentum Strategy has issues")
    sys.exit(0 if success else 1)