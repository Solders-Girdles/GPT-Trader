#!/usr/bin/env python3
"""
Simple workflow test to validate our strategy evolution pipeline.
Tests the core components without heavy dependencies.
"""

import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_sample_data(symbol="AAPL", days=252):
    """Generate sample OHLCV data for testing"""
    logger.info(f"Generating {days} days of sample data for {symbol}")

    # Generate realistic price data using random walk
    np.random.seed(42)  # For reproducible results

    start_price = 150.0
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq="D")

    # Generate returns with some volatility clustering
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns ~20% annual vol
    prices = [start_price]

    for i in range(1, days):
        prices.append(prices[-1] * (1 + returns[i]))

    prices = np.array(prices)

    # Generate OHLC from prices
    high_noise = np.random.uniform(1.005, 1.02, days)
    low_noise = np.random.uniform(0.98, 0.995, days)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * np.random.uniform(0.995, 1.005, days),
            "High": prices * high_noise,
            "Low": prices * low_noise,
            "Close": prices,
            "Volume": np.random.uniform(1000000, 5000000, days).astype(int),
        }
    )

    df.set_index("Date", inplace=True)
    return df


def test_demo_ma_strategy():
    """Test the demo MA strategy without full backtest engine"""
    logger.info("Testing Demo MA Strategy")

    try:
        # Import strategy - but handle missing dependencies gracefully
        from bot.strategy.demo_ma import DemoMAStrategy

        # Generate sample data
        data = generate_sample_data()

        # Create strategy
        strategy = DemoMAStrategy(fast=10, slow=20, atr_period=14)

        # Generate signals
        signals = strategy.generate_signals(data)

        # Validate results
        assert "signal" in signals.columns, "Signal column missing"
        assert "sma_fast" in signals.columns, "Fast MA column missing"
        assert "sma_slow" in signals.columns, "Slow MA column missing"
        assert "atr" in signals.columns, "ATR column missing"

        # Check signal logic
        signal_count = signals["signal"].sum()
        logger.info(f"Generated {signal_count} buy signals out of {len(signals)} periods")

        # Validate no look-ahead bias
        first_valid_signal_idx = signals["signal"].ne(0).idxmax()
        if pd.isna(first_valid_signal_idx):
            logger.warning("No signals generated")
        else:
            first_valid_idx = signals.index.get_loc(first_valid_signal_idx)
            assert first_valid_idx >= 20, "Signals generated too early (look-ahead bias)"

        logger.info("✅ Demo MA Strategy test passed")
        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Demo MA Strategy test failed: {e}")
        return False


def test_enhanced_trend_breakout():
    """Test the enhanced trend breakout strategy"""
    logger.info("Testing Enhanced Trend Breakout Strategy")

    try:
        # This will likely fail due to missing enhanced indicators, but let's try
        from bot.strategy.enhanced_trend_breakout import (
            EnhancedTrendBreakoutParams,
            EnhancedTrendBreakoutStrategy,
        )

        # Generate sample data
        data = generate_sample_data(days=300)  # Need more data for this strategy

        # Create strategy with default parameters
        params = EnhancedTrendBreakoutParams()
        strategy = EnhancedTrendBreakoutStrategy(params)

        # Try to generate signals
        signals = strategy.generate_signals(data)

        logger.info(f"Enhanced strategy generated signals: {signals['signal'].sum()}")
        logger.info("✅ Enhanced Trend Breakout Strategy test passed")
        return True

    except ImportError as e:
        logger.warning(f"⚠️  Enhanced strategy import failed (expected): {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Enhanced Trend Breakout test failed: {e}")
        return False


def test_basic_performance_metrics():
    """Test basic performance calculation without full backtest"""
    logger.info("Testing basic performance metrics")

    try:
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year

        # Calculate basic metrics
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        # Calculate max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        logger.info("Performance Metrics:")
        logger.info(f"  Annual Return: {annual_return:.2%}")
        logger.info(f"  Volatility: {volatility:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")

        # Validate reasonable ranges
        assert -1 < annual_return < 2, f"Unrealistic annual return: {annual_return}"
        assert 0 < volatility < 1, f"Unrealistic volatility: {volatility}"
        assert -3 < sharpe < 3, f"Unrealistic Sharpe ratio: {sharpe}"
        assert -1 < max_drawdown < 0, f"Invalid max drawdown: {max_drawdown}"

        logger.info("✅ Basic performance metrics test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Performance metrics test failed: {e}")
        return False


def test_data_pipeline():
    """Test basic data pipeline functionality"""
    logger.info("Testing data pipeline")

    try:
        # Test data generation and processing
        data = generate_sample_data(symbol="TEST", days=100)

        # Validate data structure
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            assert col in data.columns, f"Missing required column: {col}"

        # Validate data integrity
        assert len(data) == 100, f"Expected 100 rows, got {len(data)}"
        assert data["high"].ge(data["low"]).all(), "High prices should be >= Low prices"
        assert data["high"].ge(data["close"]).all(), "High prices should be >= Close prices"
        assert data["low"].le(data["close"]).all(), "Low prices should be <= Close prices"
        assert data["volume"].gt(0).all(), "Volume should be positive"

        # Test basic technical indicators
        data["SMA_20"] = data["close"].rolling(20).mean()
        data["RSI"] = calculate_rsi(data["close"], period=14)

        logger.info(f"Generated data with {len(data)} rows")
        logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        logger.info("✅ Data pipeline test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Data pipeline test failed: {e}")
        return False


def calculate_rsi(prices, period=14):
    """Simple RSI calculation for testing"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def analyze_system_performance():
    """Analyze system performance characteristics"""
    logger.info("Analyzing system performance")

    try:
        import os
        import time

        import psutil

        # Memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        # CPU usage (approximate)
        cpu_percent = psutil.cpu_percent(interval=1)

        # Test computation speed
        start_time = time.time()
        data = generate_sample_data(days=1000)
        generation_time = time.time() - start_time

        start_time = time.time()
        data["SMA_50"] = data["close"].rolling(50).mean()
        data["SMA_200"] = data["close"].rolling(200).mean()
        indicator_time = time.time() - start_time

        logger.info("System Performance Analysis:")
        logger.info(f"  Memory Usage: {memory_mb:.1f} MB")
        logger.info(f"  CPU Usage: {cpu_percent:.1f}%")
        logger.info(f"  Data Generation (1000 days): {generation_time:.3f}s")
        logger.info(f"  Indicator Calculation: {indicator_time:.3f}s")

        # Performance benchmarks
        if generation_time > 1.0:
            logger.warning("⚠️  Data generation is slow, consider optimization")
        if memory_mb > 100:
            logger.warning("⚠️  High memory usage detected")
        if indicator_time > 0.1:
            logger.warning("⚠️  Slow indicator calculation")

        logger.info("✅ System performance analysis complete")
        return True

    except ImportError:
        logger.warning("⚠️  psutil not available, skipping detailed performance analysis")
        return True
    except Exception as e:
        logger.error(f"❌ System performance analysis failed: {e}")
        return False


def main():
    """Run all workflow tests"""
    logger.info("🚀 Starting GPT-Trader Workflow Tests")
    logger.info("=" * 60)

    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Basic Performance Metrics", test_basic_performance_metrics),
        ("Demo MA Strategy", test_demo_ma_strategy),
        ("Enhanced Trend Breakout", test_enhanced_trend_breakout),
        ("System Performance", analyze_system_performance),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)

        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"📊 TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Workflow is ready for optimization.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Need investigation.")

    # Identify next steps
    logger.info("\n🎯 NEXT STEPS ANALYSIS:")

    if passed >= 3:  # Core functionality working
        logger.info("✅ Core workflow components are functional")
        logger.info("🔧 Ready for strategy evolution and optimization")
        logger.info("📈 Consider implementing:")
        logger.info("   - Parameter optimization algorithms")
        logger.info("   - Multi-objective optimization")
        logger.info("   - Walk-forward analysis")
        logger.info("   - Portfolio construction improvements")
    else:
        logger.info("⚠️  Core issues need resolution before optimization:")
        logger.info("   - Fix import dependencies")
        logger.info("   - Validate data pipeline")
        logger.info("   - Test strategy implementations")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
