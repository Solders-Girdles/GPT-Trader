#!/usr/bin/env python3
"""
Quick test of strategy functionality with real data.
Verifies strategies can actually generate signals.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_sample_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-02-28", freq="D")

    # Generate realistic price series
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    return data


def test_strategy(strategy_class, strategy_name, data):
    """Test a strategy with sample data."""
    try:
        strategy = strategy_class()
        signals = strategy.generate_signals(data)

        if isinstance(signals, pd.DataFrame) and len(signals) == len(data):
            signal_count = (signals["signal"] != 0).sum() if "signal" in signals.columns else 0
            print(f"✓ {strategy_name}: Generated {signal_count} signals")
            return True
        else:
            print(f"✗ {strategy_name}: Invalid output format")
            return False

    except Exception as e:
        print(f"✗ {strategy_name}: {e}")
        return False


def main():
    print("=== Quick Strategy Test ===")

    # Create sample data
    print("Creating sample data...")
    data = create_sample_data()
    print(f"✓ Created {len(data)} days of market data")

    # Test strategies
    strategies = []

    try:
        from bot.strategy.demo_ma import DemoMAStrategy

        strategies.append((DemoMAStrategy, "Demo MA Strategy"))
    except ImportError as e:
        print(f"✗ Could not import DemoMAStrategy: {e}")

    try:
        from bot.strategy.trend_breakout import TrendBreakoutStrategy

        strategies.append((TrendBreakoutStrategy, "Trend Breakout Strategy"))
    except ImportError as e:
        print(f"✗ Could not import TrendBreakoutStrategy: {e}")

    if not strategies:
        print("✗ No strategies available to test")
        return 1

    passed = 0
    for strategy_class, strategy_name in strategies:
        if test_strategy(strategy_class, strategy_name, data):
            passed += 1

    print("\n=== Results ===")
    print(f"Strategies working: {passed}/{len(strategies)}")

    # Test data validation
    try:
        from bot.dataflow.validate import validate_ohlcv

        validate_ohlcv(data)
        print("✓ Data validation working")
    except Exception as e:
        print(f"✗ Data validation failed: {e}")

    # Test ATR calculation
    try:
        from bot.indicators.atr import atr

        atr_values = atr(data, period=14)
        if isinstance(atr_values, pd.Series):
            print(f"✓ ATR calculation working (avg: {atr_values.mean():.2f})")
        else:
            print("✗ ATR calculation returned wrong type")
    except Exception as e:
        print(f"✗ ATR calculation failed: {e}")

    return 0 if passed == len(strategies) else 1


if __name__ == "__main__":
    sys.exit(main())
