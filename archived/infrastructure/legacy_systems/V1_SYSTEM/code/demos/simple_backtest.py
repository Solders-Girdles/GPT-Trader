#!/usr/bin/env python3
"""
Simple Backtest Demo - DEMO-002
GPT-Trader Emergency Recovery

This script demonstrates a minimal working backtest using the demo_ma strategy.
Shows that the backtest engine can run (even if not fully functional).

Usage:
    poetry run python demos/simple_backtest.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from bot.backtest.engine_portfolio import BacktestEngine, BacktestConfig
    from bot.strategy.demo_ma import DemoMAStrategy
    from bot.dataflow.sources.yfinance_source import YFinanceSource
    from bot.logging import get_logger
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running with 'poetry run python demos/simple_backtest.py'")
    sys.exit(1)

# Set up logging
logger = get_logger("demo")


def create_sample_data(symbol="AAPL", days=90):
    """Create or load sample data for backtesting"""
    print(f"📊 Fetching data for {symbol}...")

    # Try to load from data directory first
    data_file = project_root / "data" / "historical" / f"{symbol}_data.csv"

    if data_file.exists():
        print(f"  ✅ Loading cached data from {data_file}")
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        return df

    # Otherwise download fresh data
    yf_source = YFinanceSource()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df = yf_source.get_daily_bars(
        symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
    )

    if df.empty:
        print(f"  ❌ No data returned, creating synthetic data")
        # Create synthetic data as fallback
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)

        df = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.randn(len(dates)) * 0.01),
                "High": prices * (1 + np.abs(np.random.randn(len(dates)) * 0.02)),
                "Low": prices * (1 - np.abs(np.random.randn(len(dates)) * 0.02)),
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

    print(f"  ✅ Data loaded: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    return df


def run_simple_backtest():
    """Run a simple backtest demonstration"""
    print("=" * 60)
    print("🚀 GPT-Trader Simple Backtest Demo")
    print("=" * 60)

    # Configuration
    SYMBOL = "AAPL"
    INITIAL_CAPITAL = 10000

    # Get data
    data = create_sample_data(SYMBOL)

    # Initialize strategy
    print("\n📈 Initializing DemoMAStrategy...")
    strategy = DemoMAStrategy(
        fast=10,  # Fast moving average window
        slow=30,  # Slow moving average window
        atr_period=14,  # ATR period for volatility
    )

    # Initialize backtest engine with config
    print("🎯 Initializing BacktestEngine...")
    config = BacktestConfig(
        start_date=data.index[0],
        end_date=data.index[-1],
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.001,  # 0.1% commission
    )

    engine = BacktestEngine(config=config)

    # Run backtest
    print("\n⚡ Running backtest...")
    try:
        results = engine.run_backtest(strategy=strategy, data=data)

        # Display results
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)

        if hasattr(results, "portfolio_value"):
            final_value = (
                results.portfolio_value[-1] if len(results.portfolio_value) > 0 else INITIAL_CAPITAL
            )
            returns = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
            print(f"Final Value: ${final_value:,.2f}")
            print(f"Total Return: {returns:.2f}%")

            if hasattr(results, "trades") and results.trades:
                print(f"Total Trades: {len(results.trades)}")
                winners = sum(1 for t in results.trades if t.get("pnl", 0) > 0)
                print(f"Win Rate: {winners/len(results.trades)*100:.1f}%")
        else:
            # Fallback display for minimal results
            print("Results object created (structure may vary)")
            print(f"Result type: {type(results)}")

            # Try to display any available attributes
            if hasattr(results, "__dict__"):
                for key, value in results.__dict__.items():
                    if not key.startswith("_"):
                        print(f"  {key}: {value}")

        print("\n✅ Backtest completed successfully!")
        print("✅ Basic backtest engine is functional")
        return True

    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        print("⚠️  This is expected - the backtest engine needs more work")
        return False


def main():
    """Main demo function"""
    success = run_simple_backtest()

    print("\n" + "=" * 60)
    print("📋 DEMO SUMMARY")
    print("=" * 60)

    if success:
        print("✅ Demo completed successfully")
        print("✅ Backtest engine can initialize and run")
        print("✅ Strategy can process data")
        print("\n🎉 BACKTEST DEMO SUCCESSFUL!")
        return 0
    else:
        print("⚠️  Demo encountered issues")
        print("📝 The backtest engine needs further development")
        print("📝 Check the error messages above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
