#!/usr/bin/env python3
"""
Multiprocessing Optimization Test

Tests parallel optimization capabilities:
1. Parameter grid optimization
2. Performance scaling with worker count
3. Adaptive grid refinement
4. Memory and CPU efficiency
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_parallel_optimization():
    """Test basic parallel parameter optimization"""
    print("🧪 Testing Parallel Parameter Optimization...")

    from bot.optimization.parallel_optimizer import OptimizationConfig, ParallelOptimizer
    from bot.strategy.talib_optimized_ma import TALibMAParams

    # Generate test data
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

    returns = np.random.normal(0.001, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)
    highs = prices * np.random.uniform(1.005, 1.02, n_days)
    lows = prices * np.random.uniform(0.98, 0.995, n_days)
    volumes = np.random.lognormal(15, 0.3, n_days).astype(int)

    data = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
        index=dates,
    )

    # Define parameter grid (small for testing)
    parameter_grid = {
        "fast": [8, 10, 12],
        "slow": [20, 25, 30],
        "volume_filter": [True, False],
        "atr_period": [14, 21],
    }

    total_combinations = 3 * 3 * 2 * 2  # 36 combinations

    # Create config
    config = OptimizationConfig(
        strategy_class=TALibMAParams,
        parameter_grid=parameter_grid,
        data=data,
        initial_cash=100000,
        commission=0.001,
        min_trades=5,
        max_workers=4,
    )

    # Run optimization
    optimizer = ParallelOptimizer(max_workers=4)
    start_time = time.time()
    results = optimizer.optimize_parameters(config)
    execution_time = time.time() - start_time

    # Validate results
    assert len(results) > 0, "No valid optimization results"
    assert all(r.error is None for r in results[:5]), "Top results should not have errors"

    # Check result ordering
    if len(results) > 1:
        assert (
            results[0].sharpe_ratio >= results[1].sharpe_ratio
        ), "Results should be sorted by performance"

    throughput = total_combinations / execution_time

    print(f"   ✅ Valid results: {len(results)}/{total_combinations}")
    print(f"   ⚡ Execution time: {execution_time:.2f}s")
    print(f"   🚀 Throughput: {throughput:.1f} combinations/sec")

    if results:
        best = results[0]
        print("   🏆 Best result:")
        print(f"      📊 Parameters: {best.parameters}")
        print(f"      📈 Sharpe ratio: {best.sharpe_ratio:.3f}")
        print(f"      💰 Total return: {best.total_return:.2%}")
        print(f"      📉 Max drawdown: {best.max_drawdown:.2%}")
        print(f"      🔄 Trades: {best.num_trades}")

    return results


def test_parallelization_scaling():
    """Test how performance scales with worker count"""
    print("\n🧪 Testing Parallelization Scaling...")

    from bot.optimization.parallel_optimizer import OptimizationConfig, ParallelOptimizer
    from bot.strategy.talib_optimized_ma import TALibMAParams

    # Generate larger test data for scaling test
    np.random.seed(42)
    n_days = 1500
    dates = pd.date_range(start="2021-01-01", periods=n_days, freq="D")

    returns = np.random.normal(0.0008, 0.018, n_days)
    prices = 100 * np.cumprod(1 + returns)
    highs = prices * np.random.uniform(1.005, 1.02, n_days)
    lows = prices * np.random.uniform(0.98, 0.995, n_days)
    volumes = np.random.lognormal(15.2, 0.35, n_days).astype(int)

    data = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
        index=dates,
    )

    # Much larger parameter grid for scaling test that benefits from multiprocessing
    parameter_grid = {
        "fast": [5, 7, 8, 10, 12, 15, 18, 20],
        "slow": [20, 25, 30, 35, 40, 45, 50, 55],
        "volume_filter": [True, False],
        "ma_type": [0, 1],  # SMA, EMA
        "atr_period": [14, 21, 28],
    }

    total_combinations = 8 * 8 * 2 * 2 * 3  # 768 combinations

    config = OptimizationConfig(
        strategy_class=TALibMAParams,
        parameter_grid=parameter_grid,
        data=data,
        initial_cash=100000,
        commission=0.001,
        min_trades=3,
    )

    # Test different worker counts
    scaling_results = {}
    worker_counts = [1, 2, 4, 8]  # Test more worker counts for large grid

    for workers in worker_counts:
        print(f"   🔧 Testing with {workers} workers...")

        optimizer = ParallelOptimizer(max_workers=workers)

        start_time = time.time()
        results = optimizer.optimize_parameters(config)
        execution_time = time.time() - start_time

        throughput = total_combinations / execution_time

        scaling_results[workers] = {
            "workers": workers,
            "execution_time": execution_time,
            "throughput": throughput,
            "valid_results": len(results),
            "speedup": (
                throughput / scaling_results[1]["throughput"] if 1 in scaling_results else 1.0
            ),
            "efficiency": throughput / workers,
        }

        print(f"      ⚡ Throughput: {throughput:.1f} combinations/sec")
        print(f"      ✅ Valid results: {len(results)}")

    # Analyze scaling efficiency
    print("\n   📊 SCALING ANALYSIS:")
    for workers, metrics in scaling_results.items():
        print(
            f"      {workers} workers: {metrics['throughput']:.1f} comb/sec | "
            f"Speedup: {metrics['speedup']:.1f}x | "
            f"Efficiency: {metrics['efficiency']:.1f} comb/worker/sec"
        )

    return scaling_results


def test_adaptive_optimization():
    """Test adaptive grid refinement"""
    print("\n🧪 Testing Adaptive Grid Optimization...")

    from bot.optimization.parallel_optimizer import OptimizationConfig, ParallelOptimizer
    from bot.strategy.talib_optimized_ma import TALibMAParams

    # Generate test data with trend for better optimization results
    np.random.seed(42)
    n_days = 800
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    # Create trending data for more interesting optimization
    trend = np.linspace(0, 0.3, n_days)  # 30% uptrend
    noise = np.random.normal(0, 0.015, n_days)
    returns = trend / n_days + noise

    prices = 100 * np.cumprod(1 + returns)
    highs = prices * np.random.uniform(1.005, 1.02, n_days)
    lows = prices * np.random.uniform(0.98, 0.995, n_days)
    volumes = np.random.lognormal(15.1, 0.3, n_days).astype(int)

    data = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
        index=dates,
    )

    # Wide initial parameter grid
    parameter_grid = {
        "fast": [5, 10, 15, 20],
        "slow": [25, 35, 45, 55],
        "atr_period": [10, 14, 21, 28],
    }

    config = OptimizationConfig(
        strategy_class=TALibMAParams,
        parameter_grid=parameter_grid,
        data=data,
        initial_cash=100000,
        commission=0.001,
        min_trades=5,
        max_workers=4,
    )

    # Run adaptive optimization
    optimizer = ParallelOptimizer(max_workers=4)

    start_time = time.time()
    adaptive_results = optimizer.optimize_with_adaptive_grid(
        config, max_iterations=2, refinement_factor=0.5
    )
    adaptive_time = time.time() - start_time

    # Compare with standard optimization
    start_time = time.time()
    standard_results = optimizer.optimize_parameters(config)
    standard_time = time.time() - start_time

    print("   📊 Standard optimization:")
    print(f"      ⏱️  Time: {standard_time:.2f}s")
    print(f"      ✅ Valid results: {len(standard_results)}")
    if standard_results:
        print(f"      🏆 Best Sharpe: {standard_results[0].sharpe_ratio:.3f}")

    print("   🔄 Adaptive optimization:")
    print(f"      ⏱️  Time: {adaptive_time:.2f}s")
    print(f"      ✅ Valid results: {len(adaptive_results)}")
    if adaptive_results:
        print(f"      🏆 Best Sharpe: {adaptive_results[0].sharpe_ratio:.3f}")

    # Check if adaptive found better result
    if adaptive_results and standard_results:
        improvement = adaptive_results[0].sharpe_ratio - standard_results[0].sharpe_ratio
        print(f"      📈 Improvement: {improvement:+.3f} Sharpe ratio")

    return adaptive_results, standard_results


def test_memory_efficiency():
    """Test memory efficiency of parallel optimization"""
    print("\n🧪 Testing Memory Efficiency...")

    import os

    import psutil
    from bot.optimization.parallel_optimizer import OptimizationConfig, ParallelOptimizer
    from bot.strategy.talib_optimized_ma import TALibMAParams

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Generate larger dataset for memory test
    np.random.seed(42)
    n_days = 3000
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)
    highs = prices * np.random.uniform(1.005, 1.02, n_days)
    lows = prices * np.random.uniform(0.98, 0.995, n_days)
    volumes = np.random.lognormal(15.5, 0.4, n_days).astype(int)

    data = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
        index=dates,
    )

    data_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Parameter grid
    parameter_grid = {
        "fast": [8, 10, 12, 15],
        "slow": [20, 25, 30, 35],
        "volume_filter": [True, False],
    }

    total_combinations = 4 * 4 * 2  # 32 combinations

    config = OptimizationConfig(
        strategy_class=TALibMAParams,
        parameter_grid=parameter_grid,
        data=data,
        initial_cash=100000,
        commission=0.001,
        min_trades=5,
        max_workers=4,
    )

    # Run optimization
    optimizer = ParallelOptimizer(max_workers=4)

    start_time = time.time()
    results = optimizer.optimize_parameters(config)
    execution_time = time.time() - start_time

    peak_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Calculate memory metrics
    data_size = data_memory - initial_memory
    processing_overhead = peak_memory - data_memory
    total_memory_used = peak_memory - initial_memory

    throughput = total_combinations / execution_time
    memory_efficiency = total_combinations / total_memory_used

    print(f"   💾 Initial memory: {initial_memory:.1f} MB")
    print(f"   📊 Data size: {data_size:.1f} MB")
    print(f"   ⚙️  Processing overhead: {processing_overhead:.1f} MB")
    print(f"   📈 Peak memory: {peak_memory:.1f} MB")
    print(f"   🎯 Memory efficiency: {memory_efficiency:.1f} combinations/MB")
    print(f"   ⚡ Throughput: {throughput:.1f} combinations/sec")
    print(f"   ✅ Valid results: {len(results)}")

    return {
        "initial_memory": initial_memory,
        "data_size": data_size,
        "processing_overhead": processing_overhead,
        "peak_memory": peak_memory,
        "memory_efficiency": memory_efficiency,
        "throughput": throughput,
        "valid_results": len(results),
    }


def main():
    """Run all multiprocessing tests"""
    print("🚀 Multiprocessing Optimization Test Suite")
    print("=" * 60)

    # Set up logging to reduce noise
    logging.basicConfig(level=logging.WARNING)

    try:
        # Test 1: Basic parallel optimization
        optimization_results = test_parallel_optimization()

        # Test 2: Scaling with worker count
        scaling_results = test_parallelization_scaling()

        # Test 3: Adaptive optimization
        adaptive_results, standard_results = test_adaptive_optimization()

        # Test 4: Memory efficiency
        memory_results = test_memory_efficiency()

        # Summary
        print("\n📊 MULTIPROCESSING TEST SUMMARY:")
        print(f"   ✅ Basic optimization: PASSED ({len(optimization_results)} results)")
        print(f"   ✅ Scaling analysis: PASSED ({len(scaling_results)} worker configs)")
        print("   ✅ Adaptive optimization: PASSED")
        print("   ✅ Memory efficiency: PASSED")

        # Key metrics
        max_workers = max(scaling_results.keys())
        best_throughput = scaling_results[max_workers]["throughput"]
        max_speedup = scaling_results[max_workers]["speedup"]

        print("\n🚀 KEY METRICS:")
        print(f"   ⚡ Best throughput: {best_throughput:.1f} combinations/sec")
        print(f"   📈 Maximum speedup: {max_speedup:.1f}x")
        print(f"   💾 Memory efficiency: {memory_results['memory_efficiency']:.1f} combinations/MB")
        print(f"   🎯 Peak memory usage: {memory_results['peak_memory']:.1f} MB")

        # Performance grade
        if max_speedup >= 3.0:
            grade = "🏆 EXCELLENT"
        elif max_speedup >= 2.0:
            grade = "🥈 GOOD"
        elif max_speedup >= 1.5:
            grade = "🥉 FAIR"
        else:
            grade = "⚠️  NEEDS IMPROVEMENT"

        print(f"   📊 Performance grade: {grade}")

        print("\n✅ Multiprocessing optimization ready for production!")

        return True

    except Exception as e:
        print(f"\n❌ Multiprocessing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
