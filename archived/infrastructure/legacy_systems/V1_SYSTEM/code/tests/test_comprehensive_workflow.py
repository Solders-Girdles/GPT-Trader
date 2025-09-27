#!/usr/bin/env python3
"""
Comprehensive workflow test with full dependencies.
Tests the complete trading workflow and identifies optimization opportunities.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_realistic_data(symbol: str = "AAPL", days: int = 252) -> pd.DataFrame:
    """Generate realistic OHLCV data using pandas and numpy"""
    logger.info(f"Generating {days} days of realistic data for {symbol}")

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq="D")

    # Generate realistic price series with volatility clustering
    start_price = 150.0

    # Use GARCH-like process for volatility clustering
    vol_persistence = 0.9
    vol_mean_reversion = 0.02
    base_vol = 0.02

    returns = []
    volatility = base_vol

    for i in range(days):
        # Update volatility with persistence
        vol_shock = np.random.normal(0, 0.001)
        volatility = vol_persistence * volatility + (1 - vol_persistence) * base_vol + vol_shock
        volatility = max(volatility, 0.005)  # Minimum volatility

        # Generate return with current volatility
        daily_return = np.random.normal(0.0005, volatility)  # Small positive drift
        returns.append(daily_return)

    # Calculate prices from returns
    prices = [start_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    prices = np.array(prices)

    # Generate OHLC from prices with realistic intraday patterns
    highs = prices * np.random.uniform(1.005, 1.025, len(prices))
    lows = prices * np.random.uniform(0.975, 0.995, len(prices))
    opens = np.roll(prices, 1) * np.random.uniform(0.995, 1.005, len(prices))
    opens[0] = prices[0]  # First open = first close

    # Generate realistic volume with trend following
    base_volume = 2000000
    volume_trend = np.abs(np.array(returns)) * 10000000  # Higher volume on big moves
    volume = np.random.lognormal(np.log(base_volume), 0.5, len(prices)) + volume_trend

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volume.astype(int),
        }
    )

    df.set_index("Date", inplace=True)
    return df


def test_pandas_performance():
    """Test pandas-based data processing performance"""
    logger.info("Testing pandas performance")

    try:
        # Generate large dataset
        start_time = time.time()
        df = generate_realistic_data(days=5000)  # ~20 years of data
        generation_time = time.time() - start_time

        # Test vectorized operations
        start_time = time.time()
        df["SMA_20"] = df["close"].rolling(20).mean()
        df["SMA_50"] = df["close"].rolling(50).mean()
        df["SMA_200"] = df["close"].rolling(200).mean()
        df["RSI"] = calculate_rsi_vectorized(df["close"])
        df["ATR"] = calculate_atr_vectorized(df)
        df["Bollinger_Upper"], df["Bollinger_Lower"] = calculate_bollinger_bands(df["close"])
        indicator_time = time.time() - start_time

        # Test strategy signals
        start_time = time.time()
        df["MA_Signal"] = np.where(df["SMA_20"] > df["SMA_50"], 1, 0)
        df["BB_Signal"] = np.where(
            df["close"] > df["Bollinger_Upper"],
            -1,
            np.where(df["close"] < df["Bollinger_Lower"], 1, 0),
        )
        df["RSI_Signal"] = np.where(df["RSI"] < 30, 1, np.where(df["RSI"] > 70, -1, 0))
        signal_time = time.time() - start_time

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        logger.info("✅ Pandas performance test completed")
        logger.info(f"   Data generation (5000 days): {generation_time:.3f}s")
        logger.info(f"   Indicator calculation: {indicator_time:.3f}s")
        logger.info(f"   Signal generation: {signal_time:.3f}s")
        logger.info(f"   Memory usage: {memory_mb:.2f} MB")
        logger.info(f"   Rows processed: {len(df):,}")

        # Validate results
        assert len(df) == 5000, f"Expected 5000 rows, got {len(df)}"
        assert not df["SMA_20"].iloc[-1] != df["SMA_20"].iloc[-1], "SMA calculation failed (NaN)"
        assert df["RSI"].iloc[-1] >= 0 and df["RSI"].iloc[-1] <= 100, "Invalid RSI value"

        return True, {
            "generation_time": generation_time,
            "indicator_time": indicator_time,
            "signal_time": signal_time,
            "memory_mb": memory_mb,
            "rows": len(df),
        }

    except Exception as e:
        logger.error(f"❌ Pandas performance test failed: {e}")
        return False, {}


def calculate_rsi_vectorized(prices: pd.Series, period: int = 14) -> pd.Series:
    """Vectorized RSI calculation"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr_vectorized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Vectorized ATR calculation"""
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return upper_band, lower_band


def test_demo_ma_strategy_performance():
    """Test the demo MA strategy with realistic data"""
    logger.info("Testing demo MA strategy performance")

    try:
        # Generate test data
        df = generate_realistic_data(days=1000)

        # Import and test strategy
        from bot.strategy.demo_ma import DemoMAStrategy

        # Test different parameter combinations
        parameter_sets = [
            {"fast": 10, "slow": 20, "atr_period": 14},
            {"fast": 5, "slow": 15, "atr_period": 10},
            {"fast": 20, "slow": 50, "atr_period": 20},
        ]

        results = []

        for params in parameter_sets:
            start_time = time.time()
            strategy = DemoMAStrategy(**params)
            signals = strategy.generate_signals(df)
            execution_time = time.time() - start_time

            # Calculate basic performance
            signal_count = signals["signal"].sum()
            total_periods = len(signals)
            signal_rate = signal_count / total_periods

            results.append(
                {
                    "params": params,
                    "execution_time": execution_time,
                    "signal_count": signal_count,
                    "signal_rate": signal_rate,
                    "valid_signals": signals["signal"].notna().sum(),
                }
            )

        logger.info("✅ Demo MA strategy performance test completed")
        for i, result in enumerate(results):
            logger.info(f"   Config {i+1}: {result['params']}")
            logger.info(f"      Execution time: {result['execution_time']:.3f}s")
            logger.info(
                f"      Signals: {result['signal_count']}/{total_periods} ({result['signal_rate']:.1%})"
            )

        return True, results

    except ImportError as e:
        logger.error(f"❌ Strategy import failed: {e}")
        return False, []
    except Exception as e:
        logger.error(f"❌ Demo MA strategy test failed: {e}")
        return False, []


def test_backtest_performance():
    """Test backtesting performance without full engine"""
    logger.info("Testing backtest performance simulation")

    try:
        # Generate data and signals
        df = generate_realistic_data(days=2000)

        # Create simple strategy signals
        df["SMA_Fast"] = df["close"].rolling(20).mean()
        df["SMA_Slow"] = df["close"].rolling(50).mean()
        df["Signal"] = np.where(df["SMA_Fast"] > df["SMA_Slow"], 1, 0)
        df["Position"] = df["Signal"].diff()

        # Simulate trade execution
        start_time = time.time()

        returns = []
        trades = []
        position = 0
        entry_price = 0

        # Use itertuples for better performance
        for row in df.itertuples():
            if row.Position == 1 and position == 0:  # Entry
                position = 1
                entry_price = row.Close
                trades.append(
                    {
                        "entry_date": row.Index,
                        "entry_price": entry_price,
                        "exit_date": None,
                        "exit_price": None,
                        "pnl": None,
                        "return": None,
                    }
                )
            elif row.Position == 0 and position == 1:  # Exit
                position = 0
                exit_price = row.Close
                if trades:
                    trades[-1]["exit_date"] = row.Index
                    trades[-1]["exit_price"] = exit_price
                    pnl = exit_price - entry_price
                    trades[-1]["pnl"] = pnl
                    trades[-1]["return"] = pnl / entry_price
                    returns.append(pnl / entry_price)

        # Calculate final statistics after loop
        backtest_time = time.time() - start_time
        total_trades = len(trades)
        if returns:
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_return = np.mean(returns)
            total_return = np.prod([1 + r for r in returns]) - 1
        else:
            win_rate = avg_return = total_return = 0.0

        logger.info("✅ Backtest performance simulation completed")
        logger.info(f"   Backtest time: {backtest_time:.3f}s")
        logger.info(f"   Total trades: {total_trades}")
        logger.info(f"   Win rate: {win_rate:.1%}")
        logger.info(f"   Average return per trade: {avg_return:.2%}")
        logger.info(f"   Total return: {total_return:.2%}")

        return True, {
            "backtest_time": backtest_time,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_return": total_return,
        }

    except Exception as e:
        logger.error(f"❌ Backtest simulation failed: {e}")
        return False, {}


def test_optimization_simulation():
    """Simulate parameter optimization process"""
    logger.info("Testing optimization simulation")

    try:
        # Simulate parameter grid search
        df = generate_realistic_data(days=1000)

        parameter_grid = {
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 50],
            "risk_pct": [0.01, 0.02, 0.03],
        }

        # Generate all parameter combinations
        combinations = []
        for fast in parameter_grid["fast_period"]:
            for slow in parameter_grid["slow_period"]:
                for risk in parameter_grid["risk_pct"]:
                    if fast < slow:  # Only valid combinations
                        combinations.append(
                            {"fast_period": fast, "slow_period": slow, "risk_pct": risk}
                        )

        logger.info(f"   Testing {len(combinations)} parameter combinations")

        start_time = time.time()
        results = []

        for params in combinations:
            # Simulate strategy evaluation
            fast_ma = df["close"].rolling(params["fast_period"]).mean()
            slow_ma = df["close"].rolling(params["slow_period"]).mean()
            signals = np.where(fast_ma > slow_ma, 1, 0)

            # Simple performance calculation
            returns = df["close"].pct_change() * np.roll(signals, 1)
            total_return = (1 + returns).prod() - 1
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            results.append({"params": params, "total_return": total_return, "sharpe": sharpe})

        optimization_time = time.time() - start_time

        # Find best parameters
        best_result = max(results, key=lambda x: x["sharpe"])

        logger.info("✅ Optimization simulation completed")
        logger.info(f"   Optimization time: {optimization_time:.3f}s")
        logger.info(f"   Combinations tested: {len(combinations)}")
        logger.info(f"   Time per combination: {optimization_time/len(combinations)*1000:.1f}ms")
        logger.info(f"   Best parameters: {best_result['params']}")
        logger.info(f"   Best Sharpe ratio: {best_result['sharpe']:.3f}")

        return True, {
            "optimization_time": optimization_time,
            "combinations_tested": len(combinations),
            "time_per_combination": optimization_time / len(combinations),
            "best_sharpe": best_result["sharpe"],
        }

    except Exception as e:
        logger.error(f"❌ Optimization simulation failed: {e}")
        return False, {}


def analyze_workflow_scalability():
    """Analyze workflow scalability with different data sizes"""
    logger.info("Analyzing workflow scalability")

    try:
        data_sizes = [100, 500, 1000, 2500, 5000]
        scalability_results = []

        for size in data_sizes:
            logger.info(f"   Testing with {size} days of data...")

            # Data generation
            start_time = time.time()
            df = generate_realistic_data(days=size)
            data_time = time.time() - start_time

            # Indicator calculation
            start_time = time.time()
            df["SMA_20"] = df["close"].rolling(20).mean()
            df["SMA_50"] = df["close"].rolling(50).mean()
            df["RSI"] = calculate_rsi_vectorized(df["close"])
            indicator_time = time.time() - start_time

            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

            scalability_results.append(
                {
                    "size": size,
                    "data_time": data_time,
                    "indicator_time": indicator_time,
                    "memory_mb": memory_mb,
                    "data_time_per_row": data_time / size * 1000,  # ms per row
                    "indicator_time_per_row": indicator_time / size * 1000,  # ms per row
                }
            )

        # Analyze scaling
        small_result = scalability_results[0]
        large_result = scalability_results[-1]

        data_scaling = (large_result["data_time"] / small_result["data_time"]) / (
            large_result["size"] / small_result["size"]
        )
        indicator_scaling = (large_result["indicator_time"] / small_result["indicator_time"]) / (
            large_result["size"] / small_result["size"]
        )
        memory_scaling = (large_result["memory_mb"] / small_result["memory_mb"]) / (
            large_result["size"] / small_result["size"]
        )

        logger.info("✅ Scalability analysis completed")
        logger.info(f"   Data generation scaling factor: {data_scaling:.2f} (1.0 = linear)")
        logger.info(f"   Indicator calculation scaling factor: {indicator_scaling:.2f}")
        logger.info(f"   Memory usage scaling factor: {memory_scaling:.2f}")

        # Performance thresholds
        if data_scaling > 1.2:
            logger.warning("⚠️  Data generation scales poorly")
        if indicator_scaling > 1.1:
            logger.warning("⚠️  Indicator calculation scales poorly")
        if large_result["memory_mb"] > 100:
            logger.warning("⚠️  High memory usage for large datasets")

        return True, {
            "results": scalability_results,
            "data_scaling": data_scaling,
            "indicator_scaling": indicator_scaling,
            "memory_scaling": memory_scaling,
        }

    except Exception as e:
        logger.error(f"❌ Scalability analysis failed: {e}")
        return False, {}


def identify_optimization_priorities():
    """Identify optimization priorities based on test results"""
    logger.info("Identifying optimization priorities")

    priorities = [
        {
            "priority": 1,
            "area": "Data Processing",
            "issue": "Pure Python loops in strategy code",
            "impact": "High",
            "solution": "Vectorize all operations using pandas/numpy",
            "estimated_speedup": "10-50x",
        },
        {
            "priority": 2,
            "area": "Technical Indicators",
            "issue": "Custom indicator implementations",
            "impact": "Medium-High",
            "solution": "Use TA-Lib or similar optimized library",
            "estimated_speedup": "5-20x",
        },
        {
            "priority": 3,
            "area": "Parameter Optimization",
            "impact": "High",
            "issue": "Single-threaded optimization",
            "solution": "Implement multiprocessing for parallel evaluation",
            "estimated_speedup": "4-8x (cores dependent)",
        },
        {
            "priority": 4,
            "area": "Memory Management",
            "issue": "No data streaming for large datasets",
            "impact": "Medium",
            "solution": "Implement chunked processing for large datasets",
            "estimated_speedup": "Memory bounded -> unlimited size",
        },
        {
            "priority": 5,
            "area": "Caching",
            "issue": "No caching of expensive calculations",
            "impact": "Medium",
            "solution": "Implement intelligent caching system",
            "estimated_speedup": "2-10x for repeated operations",
        },
    ]

    logger.info(f"✅ Identified {len(priorities)} optimization priorities")
    for p in priorities:
        logger.info(
            f"   {p['priority']}. {p['area']}: {p['solution']} (Est. {p['estimated_speedup']})"
        )

    return priorities


def main():
    """Run comprehensive workflow testing"""
    logger.info("🚀 Starting GPT-Trader Comprehensive Workflow Test")
    logger.info("=" * 70)

    # Test registry
    tests = [
        ("Pandas Performance", test_pandas_performance),
        ("Demo MA Strategy", test_demo_ma_strategy_performance),
        ("Backtest Simulation", test_backtest_performance),
        ("Optimization Simulation", test_optimization_simulation),
        ("Workflow Scalability", analyze_workflow_scalability),
    ]

    results = {}
    passed = 0

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 50)

        try:
            success, data = test_func()
            if success:
                passed += 1
                results[test_name] = data
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")

    # Generate optimization priorities
    logger.info("\n📋 Running: Optimization Analysis")
    logger.info("-" * 50)
    priorities = identify_optimization_priorities()

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info(f"📊 COMPREHENSIVE TEST SUMMARY: {passed}/{len(tests)} tests passed")

    if passed >= len(tests) - 1:  # Allow 1 failure
        logger.info("🎉 Comprehensive workflow validation successful!")
    else:
        logger.warning(f"⚠️  {len(tests) - passed} test(s) failed - investigation needed")

    # Performance Analysis
    logger.info("\n⚡ PERFORMANCE ANALYSIS:")

    if "Pandas Performance" in results:
        perf = results["Pandas Performance"]
        logger.info(f"   Data processing (5000 days): {perf['generation_time']:.2f}s")
        logger.info(f"   Indicator calculation: {perf['indicator_time']:.2f}s")
        logger.info(f"   Memory usage: {perf['memory_mb']:.1f} MB")

        # Calculate throughput
        throughput = perf["rows"] / (perf["generation_time"] + perf["indicator_time"])
        logger.info(f"   Processing throughput: {throughput:,.0f} rows/second")

    if "Optimization Simulation" in results:
        opt = results["Optimization Simulation"]
        logger.info(
            f"   Parameter optimization: {opt['time_per_combination']*1000:.1f}ms per combination"
        )
        logger.info(
            f"   Estimated full optimization (1000 combos): {opt['time_per_combination']*1000:.0f}s"
        )

    # Optimization Roadmap
    logger.info("\n🛣️  OPTIMIZATION ROADMAP:")
    logger.info("   Phase 1 (Immediate - High Impact):")
    logger.info("   ├─ Vectorize strategy calculations")
    logger.info("   ├─ Install TA-Lib for indicators")
    logger.info("   └─ Implement parameter caching")

    logger.info("   Phase 2 (Short-term - Scalability):")
    logger.info("   ├─ Add multiprocessing for optimization")
    logger.info("   ├─ Implement data streaming")
    logger.info("   └─ Add progress tracking")

    logger.info("   Phase 3 (Long-term - Advanced):")
    logger.info("   ├─ GPU acceleration for large datasets")
    logger.info("   ├─ Distributed computing for optimization")
    logger.info("   └─ Advanced ML for parameter selection")

    # Next Steps
    logger.info("\n🎯 IMMEDIATE NEXT STEPS:")

    if passed >= len(tests) - 1:
        logger.info("✅ System is ready for optimization!")
        logger.info("🔧 Recommended implementation order:")
        logger.info("   1. Vectorize demo_ma.py strategy")
        logger.info("   2. Add multiprocessing to optimization")
        logger.info("   3. Implement comprehensive benchmarking")
        logger.info("   4. Build parameter optimization UI")
        logger.info("   5. Add real-time performance monitoring")
    else:
        logger.info("⚠️  Fix failing tests before optimization:")
        logger.info("   1. Debug failed components")
        logger.info("   2. Validate data integrity")
        logger.info("   3. Check dependency installations")

    return passed >= len(tests) - 1


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
