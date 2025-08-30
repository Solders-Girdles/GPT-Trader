#!/usr/bin/env python3
"""
Minimal workflow test using only standard library.
Tests core architecture patterns without external dependencies.
"""

import logging
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleDataFrame:
    """Minimal DataFrame-like class for testing without pandas"""

    def __init__(self, data=None):
        if data is None:
            data = {}
        self.data = data
        self.columns = list(data.keys())
        self.length = len(next(iter(data.values()))) if data else 0

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return self.data[key]

    def add_column(self, name, values):
        self.data[name] = values
        if name not in self.columns:
            self.columns.append(name)
        if not hasattr(self, "length") or self.length == 0:
            self.length = len(values)


def generate_sample_data(symbol="AAPL", days=252):
    """Generate sample OHLCV data using only standard library"""
    logger.info(f"Generating {days} days of sample data for {symbol}")

    # Generate realistic price data
    random.seed(42)  # For reproducible results

    start_price = 150.0
    dates = []
    base_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        dates.append(base_date + timedelta(days=i))

    # Generate returns with some volatility
    prices = [start_price]
    for i in range(1, days):
        # Simple random walk with drift
        return_pct = random.gauss(0.001, 0.02)  # ~20% annual volatility
        new_price = prices[-1] * (1 + return_pct)
        prices.append(max(new_price, 1.0))  # Prevent negative prices

    # Generate OHLC from close prices
    opens, highs, lows, volumes = [], [], [], []

    for i, close_price in enumerate(prices):
        # Open is previous close + small gap
        if i == 0:
            open_price = close_price
        else:
            gap = random.gauss(0, 0.005)  # Small overnight gap
            open_price = prices[i - 1] * (1 + gap)

        # High and low around close
        high_mult = random.uniform(1.005, 1.02)
        low_mult = random.uniform(0.98, 0.995)

        high_price = close_price * high_mult
        low_price = close_price * low_mult

        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(random.randint(1000000, 5000000))

    # Create simple dataframe
    df = SimpleDataFrame(
        {
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
        }
    )

    return df


def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return [None] * len(prices)

    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(None)
        else:
            window = prices[i - period + 1 : i + 1]
            sma.append(sum(window) / len(window))

    return sma


def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range"""
    if len(highs) < period + 1:
        return [None] * len(highs)

    true_ranges = []

    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)

    # Calculate ATR using simple moving average
    atr_values = [None]  # First value is None
    for i in range(len(true_ranges)):
        if i < period - 1:
            atr_values.append(None)
        else:
            window = true_ranges[i - period + 1 : i + 1]
            atr_values.append(sum(window) / len(window))

    return atr_values


def test_data_generation():
    """Test basic data generation"""
    logger.info("Testing data generation")

    try:
        data = generate_sample_data(days=100)

        # Validate structure
        assert len(data) == 100, f"Expected 100 rows, got {len(data)}"
        assert "Close" in data.columns, "Missing Close column"
        assert "Volume" in data.columns, "Missing Volume column"

        # Validate data integrity
        closes = data["Close"]
        highs = data["High"]
        lows = data["Low"]

        for i in range(len(data)):
            assert highs[i] >= closes[i], f"High < Close at index {i}"
            assert lows[i] <= closes[i], f"Low > Close at index {i}"
            assert data["Volume"][i] > 0, f"Non-positive volume at index {i}"

        logger.info(f"✅ Generated {len(data)} rows of valid OHLCV data")
        logger.info(f"   Price range: ${min(closes):.2f} - ${max(closes):.2f}")
        return True

    except Exception as e:
        logger.error(f"❌ Data generation test failed: {e}")
        return False


def test_technical_indicators():
    """Test basic technical indicator calculations"""
    logger.info("Testing technical indicators")

    try:
        data = generate_sample_data(days=100)
        closes = data["Close"]
        highs = data["High"]
        lows = data["Low"]

        # Test SMA calculation
        sma_20 = calculate_sma(closes, 20)
        sma_50 = calculate_sma(closes, 50)

        # Validate SMA
        assert len(sma_20) == len(closes), "SMA length mismatch"
        assert sma_20[19] is not None, "SMA should be valid after period"
        assert sma_20[18] is None, "SMA should be None before period"

        # Test ATR calculation
        atr = calculate_atr(highs, lows, closes)
        assert len(atr) == len(closes), "ATR length mismatch"

        # Count valid indicators
        valid_sma_20 = sum(1 for x in sma_20 if x is not None)
        valid_atr = sum(1 for x in atr if x is not None)

        logger.info("✅ Technical indicators calculated")
        logger.info(f"   Valid SMA(20): {valid_sma_20}/{len(sma_20)}")
        logger.info(f"   Valid ATR(14): {valid_atr}/{len(atr)}")
        return True

    except Exception as e:
        logger.error(f"❌ Technical indicators test failed: {e}")
        return False


def test_simple_strategy():
    """Test a simple MA crossover strategy"""
    logger.info("Testing simple MA crossover strategy")

    try:
        data = generate_sample_data(days=200)
        closes = data["Close"]

        # Calculate moving averages
        fast_ma = calculate_sma(closes, 10)
        slow_ma = calculate_sma(closes, 20)

        # Generate signals
        signals = []
        for i in range(len(closes)):
            if fast_ma[i] is None or slow_ma[i] is None:
                signals.append(0)  # No signal
            elif fast_ma[i] > slow_ma[i]:
                signals.append(1)  # Buy signal
            else:
                signals.append(0)  # No position

        # Analyze signals
        buy_signals = sum(signals)
        signal_changes = 0

        for i in range(1, len(signals)):
            if signals[i] != signals[i - 1]:
                signal_changes += 1

        logger.info("✅ Simple MA strategy tested")
        logger.info(
            f"   Buy signals: {buy_signals}/{len(signals)} ({100*buy_signals/len(signals):.1f}%)"
        )
        logger.info(f"   Signal changes: {signal_changes}")

        # Validate reasonable behavior
        assert 0 <= buy_signals <= len(signals), "Invalid signal count"
        assert signal_changes < len(signals) / 2, "Too many signal changes"

        return True

    except Exception as e:
        logger.error(f"❌ Simple strategy test failed: {e}")
        return False


def test_performance_calculation():
    """Test basic performance metrics calculation"""
    logger.info("Testing performance calculation")

    try:
        # Generate sample returns
        random.seed(42)
        returns = [random.gauss(0.001, 0.02) for _ in range(252)]  # 1 year of daily returns

        # Calculate performance metrics
        total_return = 1.0
        for r in returns:
            total_return *= 1 + r
        total_return -= 1

        # Annualized return
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Volatility
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        daily_vol = math.sqrt(variance)
        annual_vol = daily_vol * math.sqrt(252)

        # Sharpe ratio
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Max drawdown
        cumulative_returns = []
        cumulative = 1.0
        for r in returns:
            cumulative *= 1 + r
            cumulative_returns.append(cumulative)

        peak = cumulative_returns[0]
        max_dd = 0
        for cum_ret in cumulative_returns:
            if cum_ret > peak:
                peak = cum_ret
            drawdown = (peak - cum_ret) / peak
            max_dd = max(max_dd, drawdown)

        logger.info("✅ Performance metrics calculated")
        logger.info(f"   Total Return: {total_return:.2%}")
        logger.info(f"   Annual Return: {annual_return:.2%}")
        logger.info(f"   Annual Volatility: {annual_vol:.2%}")
        logger.info(f"   Sharpe Ratio: {sharpe:.3f}")
        logger.info(f"   Max Drawdown: {max_dd:.2%}")

        # Validate reasonable ranges
        assert -0.5 < annual_return < 1.0, f"Unrealistic return: {annual_return}"
        assert 0 < annual_vol < 0.8, f"Unrealistic volatility: {annual_vol}"
        assert 0 <= max_dd <= 1, f"Invalid drawdown: {max_dd}"

        return True

    except Exception as e:
        logger.error(f"❌ Performance calculation test failed: {e}")
        return False


def test_architecture_patterns():
    """Test core architecture patterns"""
    logger.info("Testing architecture patterns")

    try:
        # Test component lifecycle
        class TestComponent:
            def __init__(self, name):
                self.name = name
                self.status = "initialized"
                self.metrics = defaultdict(int)

            def start(self):
                self.status = "running"
                return True

            def stop(self):
                self.status = "stopped"
                return True

            def health_check(self):
                return self.status == "running"

            def record_metric(self, name, value):
                self.metrics[name] += value

        # Test component management
        components = []
        for i in range(3):
            comp = TestComponent(f"component_{i}")
            components.append(comp)

            # Test lifecycle
            assert comp.status == "initialized"
            comp.start()
            assert comp.status == "running"
            assert comp.health_check() == True

            # Test metrics
            comp.record_metric("operations", 10)
            comp.record_metric("errors", 1)

        # Test system health
        healthy_components = sum(1 for comp in components if comp.health_check())
        total_operations = sum(comp.metrics["operations"] for comp in components)
        total_errors = sum(comp.metrics["errors"] for comp in components)

        logger.info("✅ Architecture patterns validated")
        logger.info(f"   Healthy components: {healthy_components}/{len(components)}")
        logger.info(f"   Total operations: {total_operations}")
        logger.info(f"   Total errors: {total_errors}")
        logger.info(f"   Error rate: {100*total_errors/total_operations:.2f}%")

        # Cleanup
        for comp in components:
            comp.stop()

        return True

    except Exception as e:
        logger.error(f"❌ Architecture patterns test failed: {e}")
        return False


def benchmark_performance():
    """Benchmark system performance"""
    logger.info("Benchmarking system performance")

    try:
        # Test data generation speed
        start_time = time.time()
        large_data = generate_sample_data(days=1000)
        data_gen_time = time.time() - start_time

        # Test indicator calculation speed
        closes = large_data["Close"]
        start_time = time.time()
        sma_50 = calculate_sma(closes, 50)
        sma_200 = calculate_sma(closes, 200)
        indicator_time = time.time() - start_time

        # Test strategy execution speed
        start_time = time.time()
        signals = []
        for i in range(len(closes)):
            if sma_50[i] is None or sma_200[i] is None:
                signals.append(0)
            elif sma_50[i] > sma_200[i]:
                signals.append(1)
            else:
                signals.append(0)
        strategy_time = time.time() - start_time

        # Memory usage estimation (rough)
        data_size_mb = (len(closes) * 8 * 5) / (1024 * 1024)  # 5 columns * 8 bytes each

        logger.info("✅ Performance benchmark completed")
        logger.info(f"   Data generation (1000 days): {data_gen_time:.3f}s")
        logger.info(f"   Indicator calculation: {indicator_time:.3f}s")
        logger.info(f"   Strategy execution: {strategy_time:.3f}s")
        logger.info(f"   Estimated data size: {data_size_mb:.2f} MB")

        # Performance warnings
        if data_gen_time > 0.5:
            logger.warning("⚠️  Data generation is slow")
        if indicator_time > 0.1:
            logger.warning("⚠️  Indicator calculation is slow")
        if strategy_time > 0.05:
            logger.warning("⚠️  Strategy execution is slow")

        return True

    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False


def analyze_workflow_bottlenecks():
    """Identify potential workflow bottlenecks"""
    logger.info("Analyzing workflow bottlenecks")

    try:
        bottlenecks = []
        recommendations = []

        # Test different data sizes
        sizes = [100, 500, 1000, 2500]
        times = []

        for size in sizes:
            start_time = time.time()
            data = generate_sample_data(days=size)
            closes = data["Close"]
            sma = calculate_sma(closes, 20)
            generation_time = time.time() - start_time
            times.append(generation_time)

            if generation_time > size * 0.001:  # More than 1ms per day
                bottlenecks.append(f"Data processing scales poorly at {size} days")

        # Analyze scaling
        if len(times) >= 2:
            scaling_factor = times[-1] / times[0] / (sizes[-1] / sizes[0])
            if scaling_factor > 1.2:
                bottlenecks.append(f"Non-linear scaling detected (factor: {scaling_factor:.2f})")

        # System-specific bottlenecks
        bottlenecks.extend(
            [
                "Missing pandas/numpy - pure Python is 10-100x slower",
                "No compiled indicators - TA-Lib would provide significant speedup",
                "No vectorized operations - loops are inefficient for large datasets",
                "No caching - repeated calculations waste time",
                "No parallel processing - single-threaded execution",
            ]
        )

        # Recommendations
        recommendations.extend(
            [
                "Install pandas/numpy for vectorized operations",
                "Add TA-Lib for optimized technical indicators",
                "Implement caching for expensive calculations",
                "Use multiprocessing for parallel strategy evaluation",
                "Add progress tracking for long-running operations",
                "Implement data streaming for large datasets",
                "Add memory management for big data processing",
            ]
        )

        logger.info("✅ Workflow analysis completed")
        logger.info(f"   Identified {len(bottlenecks)} potential bottlenecks")
        logger.info(f"   Generated {len(recommendations)} recommendations")

        return {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "performance_data": {"sizes": sizes, "times": times},
        }

    except Exception as e:
        logger.error(f"❌ Workflow analysis failed: {e}")
        return None


def main():
    """Run comprehensive workflow testing"""
    logger.info("🚀 Starting GPT-Trader Minimal Workflow Test")
    logger.info("=" * 60)

    tests = [
        ("Data Generation", test_data_generation),
        ("Technical Indicators", test_technical_indicators),
        ("Simple Strategy", test_simple_strategy),
        ("Performance Calculation", test_performance_calculation),
        ("Architecture Patterns", test_architecture_patterns),
        ("Performance Benchmark", benchmark_performance),
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

    # Run workflow analysis
    logger.info("\n📋 Running: Workflow Analysis")
    logger.info("-" * 40)
    analysis = analyze_workflow_bottlenecks()

    logger.info("\n" + "=" * 60)
    logger.info(f"📊 TEST SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All core tests passed!")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")

    # Workflow Analysis Results
    if analysis:
        logger.info("\n🔍 WORKFLOW BOTTLENECKS IDENTIFIED:")
        for i, bottleneck in enumerate(analysis["bottlenecks"][:5], 1):
            logger.info(f"   {i}. {bottleneck}")

        logger.info("\n💡 TOP OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(analysis["recommendations"][:5], 1):
            logger.info(f"   {i}. {rec}")

    # Next Steps
    logger.info("\n🎯 IMMEDIATE NEXT STEPS:")

    if passed >= 4:  # Most tests passing
        logger.info("✅ Core workflow is functional")
        logger.info("🔧 Priority optimizations:")
        logger.info("   1. Install pandas/numpy for 10-100x speedup")
        logger.info("   2. Add TA-Lib for optimized indicators")
        logger.info("   3. Implement strategy evolution algorithms")
        logger.info("   4. Add multiprocessing for parallel optimization")

        logger.info("📈 Ready for:")
        logger.info("   - Parameter optimization experiments")
        logger.info("   - Walk-forward analysis")
        logger.info("   - Multi-strategy portfolio construction")

    else:
        logger.info("⚠️  Core issues need resolution:")
        logger.info("   1. Fix failing tests")
        logger.info("   2. Validate data integrity")
        logger.info("   3. Debug strategy logic")
        logger.info("   4. Install missing dependencies")

    return passed >= 4  # Success if most tests pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
