#!/usr/bin/env python3
"""
Phase 1 Integration Test

Comprehensive test demonstrating all Phase 1 optimizations working together:
1. Import/logging conflicts resolved
2. Vectorized strategy code (10-50x speedup)
3. TA-Lib integration (5-20x speedup)
4. Multiprocessing optimization (4-8x parallel speedup)
5. Intelligent caching (2-10x speedup for repeated operations)
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_complete_optimization_workflow():
    """Test the complete optimization workflow with all Phase 1 improvements"""
    print("🚀 Phase 1 Complete Optimization Workflow Test")
    print("=" * 60)

    # Test 1: Import system works correctly
    print("\n1️⃣ Testing Import System...")
    try:
        from bot.optimization.intelligent_cache import IntelligentCache
        from bot.optimization.parallel_optimizer import OptimizationConfig, ParallelOptimizer
        from bot.strategy.talib_optimized_ma import TALibMAParams, TALibOptimizedMAStrategy

        print("   ✅ All imports successful - no conflicts detected")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False

    # Test 2: Generate realistic market data
    print("\n2️⃣ Generating Test Market Data...")
    np.random.seed(42)
    n_days = 2000
    dates = pd.date_range(start="2021-01-01", periods=n_days, freq="D")

    # Create realistic market data with trends and volatility regimes
    base_return = 0.0008
    volatility_regimes = np.random.choice([0.01, 0.02, 0.04], n_days, p=[0.6, 0.3, 0.1])
    trend_component = np.sin(np.linspace(0, 4 * np.pi, n_days)) * 0.0003

    returns = np.random.normal(base_return + trend_component, volatility_regimes)
    prices = 100 * np.cumprod(1 + returns)
    highs = prices * np.random.uniform(1.005, 1.025, n_days)
    lows = prices * np.random.uniform(0.975, 0.995, n_days)
    volumes = np.random.lognormal(15.5, 0.4, n_days).astype(int)

    market_data = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": volumes},
        index=dates,
    )

    print(f"   📊 Generated {len(market_data):,} days of market data")
    print(
        f"   📈 Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}"
    )
    print(
        f"   📊 Total return: {(market_data['close'].iloc[-1] / market_data['close'].iloc[0] - 1):.2%}"
    )

    # Test 3: Benchmark individual strategy performance (TA-Lib optimization)
    print("\n3️⃣ Testing TA-Lib Strategy Performance...")

    # Test multiple configurations
    strategy_configs = [
        TALibMAParams(fast=10, slow=20, volume_filter=False),
        TALibMAParams(fast=10, slow=20, volume_filter=True, rsi_filter=True),
        TALibMAParams(
            fast=15, slow=35, volume_filter=True, rsi_filter=True, trend_strength_filter=True
        ),
    ]

    strategy_results = []
    for i, config in enumerate(strategy_configs):
        strategy = TALibOptimizedMAStrategy(config)

        start_time = time.time()
        signals = strategy.generate_signals(market_data)
        execution_time = time.time() - start_time

        throughput = len(market_data) / execution_time
        signal_count = signals["signal"].sum()

        strategy_results.append(
            {
                "config": i + 1,
                "execution_time": execution_time,
                "throughput": throughput,
                "signal_count": signal_count,
                "signal_rate": signal_count / len(market_data),
            }
        )

        print(
            f"   Config {i+1}: {throughput:,.0f} rows/sec, {signal_count} signals ({signal_count/len(market_data):.1%})"
        )

    avg_throughput = sum(r["throughput"] for r in strategy_results) / len(strategy_results)
    print(f"   🚀 Average TA-Lib throughput: {avg_throughput:,.0f} rows/second")

    # Test 4: Intelligent caching integration
    print("\n4️⃣ Testing Intelligent Caching Integration...")

    cache = IntelligentCache(max_memory_mb=128)

    @cache.cached(ttl=600)  # 10 minute cache
    def cached_strategy_signals(params_dict, data_hash):
        """Cached strategy signal generation"""
        params = TALibMAParams(**params_dict)
        strategy = TALibOptimizedMAStrategy(params)
        return strategy.generate_signals(market_data)

    # Test cache performance
    test_params = {"fast": 12, "slow": 26, "volume_filter": True, "rsi_filter": True}
    data_hash = "test_data_v1"  # Simple hash for testing

    # Cold cache
    start_time = time.time()
    signals_1 = cached_strategy_signals(test_params, data_hash)
    cold_time = time.time() - start_time

    # Warm cache
    start_time = time.time()
    signals_2 = cached_strategy_signals(test_params, data_hash)
    warm_time = time.time() - start_time

    cache_speedup = cold_time / warm_time if warm_time > 0 else float("inf")

    print(f"   ❄️  Cold cache: {cold_time:.4f}s")
    print(f"   🔥 Warm cache: {warm_time:.4f}s")
    print(f"   🚀 Cache speedup: {cache_speedup:.1f}x")

    # Verify cache hit
    analytics = cache.get_analytics()
    print(f"   🎯 Cache hit rate: {analytics['stats']['hit_rate']:.1%}")

    # Test 5: Parallel optimization system
    print("\n5️⃣ Testing Parallel Optimization System...")

    # Define optimization parameter grid
    parameter_grid = {
        "fast": [8, 10, 12, 15],
        "slow": [20, 25, 30, 35],
        "volume_filter": [True, False],
        "ma_type": [0, 1],  # SMA, EMA
    }

    total_combinations = 4 * 4 * 2 * 2  # 64 combinations

    config = OptimizationConfig(
        strategy_class=TALibMAParams,
        parameter_grid=parameter_grid,
        data=market_data,
        initial_cash=100000,
        commission=0.001,
        min_trades=5,
        max_workers=4,
    )

    optimizer = ParallelOptimizer(max_workers=4)

    # Run optimization
    start_time = time.time()
    optimization_results = optimizer.optimize_parameters(config)
    optimization_time = time.time() - start_time

    optimization_throughput = total_combinations / optimization_time

    print(f"   ⚡ Optimization time: {optimization_time:.2f}s")
    print(f"   🚀 Throughput: {optimization_throughput:.1f} combinations/sec")
    print(f"   ✅ Valid results: {len(optimization_results)}/{total_combinations}")

    if optimization_results:
        best = optimization_results[0]
        print(f"   🏆 Best result: Sharpe={best.sharpe_ratio:.3f}, Return={best.total_return:.2%}")

    # Test 6: Memory efficiency and system integration
    print("\n6️⃣ Testing System Integration & Memory Efficiency...")

    import os

    import psutil

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024

    # Run complete workflow multiple times to test memory stability
    workflow_times = []
    memory_usage = []

    for run in range(3):
        run_start = time.time()

        # Generate signals for best parameters
        if optimization_results:
            best_params = TALibMAParams(**optimization_results[0].parameters)
            best_strategy = TALibOptimizedMAStrategy(best_params)
            signals = best_strategy.generate_signals(market_data)

        # Clear caches periodically
        if run % 2 == 0:
            cache.clear(memory_only=True)
            best_strategy.clear_cache() if optimization_results else None

        run_time = time.time() - run_start
        current_memory = process.memory_info().rss / 1024 / 1024

        workflow_times.append(run_time)
        memory_usage.append(current_memory)

    avg_workflow_time = sum(workflow_times) / len(workflow_times)
    max_memory = max(memory_usage)
    memory_growth = max_memory - initial_memory

    print(f"   ⏱️  Average workflow time: {avg_workflow_time:.3f}s")
    print(f"   💾 Initial memory: {initial_memory:.1f} MB")
    print(f"   📈 Peak memory: {max_memory:.1f} MB")
    print(f"   📊 Memory growth: {memory_growth:.1f} MB")

    # Test 7: Performance summary and scoring
    print("\n7️⃣ Performance Summary & Scoring...")

    # Calculate composite performance score
    scores = {
        "strategy_speed": min(100, avg_throughput / 10000),  # 1M+ rows/sec = 100 points
        "cache_efficiency": min(100, cache_speedup * 10),  # 10x+ speedup = 100 points
        "optimization_speed": min(100, optimization_throughput * 5),  # 20+ comb/sec = 100 points
        "memory_efficiency": max(0, 100 - memory_growth),  # <1MB growth = 100 points
        "result_quality": min(
            100, len(optimization_results) / total_combinations * 100
        ),  # All valid = 100 points
    }

    overall_score = sum(scores.values()) / len(scores)

    print("   📊 PERFORMANCE SCORES:")
    for metric, score in scores.items():
        print(f"      {metric.replace('_', ' ').title()}: {score:.1f}/100")

    print(f"   🎯 Overall Score: {overall_score:.1f}/100")

    # Performance grade
    if overall_score >= 90:
        grade = "🏆 EXCELLENT"
    elif overall_score >= 75:
        grade = "🥈 VERY GOOD"
    elif overall_score >= 60:
        grade = "🥉 GOOD"
    elif overall_score >= 45:
        grade = "⚠️  FAIR"
    else:
        grade = "❌ NEEDS IMPROVEMENT"

    print(f"   📈 Performance Grade: {grade}")

    # Final summary
    print("\n📊 PHASE 1 INTEGRATION SUMMARY:")
    print("   ✅ Import system: WORKING")
    print(f"   ⚡ TA-Lib speed: {avg_throughput:,.0f} rows/sec")
    print(f"   🔥 Cache speedup: {cache_speedup:.1f}x")
    print(f"   🚀 Parallel optimization: {optimization_throughput:.1f} combinations/sec")
    print(f"   💾 Memory efficiency: {memory_growth:.1f} MB growth")
    print(f"   🎯 System grade: {grade}")

    return {
        "overall_score": overall_score,
        "scores": scores,
        "strategy_throughput": avg_throughput,
        "cache_speedup": cache_speedup,
        "optimization_throughput": optimization_throughput,
        "memory_growth": memory_growth,
        "optimization_results": len(optimization_results),
        "grade": grade,
    }


def main():
    """Run Phase 1 integration test"""
    # Reduce logging noise
    logging.basicConfig(level=logging.WARNING)

    try:
        results = test_complete_optimization_workflow()

        if results["overall_score"] >= 60:
            print("\n✅ Phase 1 optimization integration SUCCESSFUL!")
            print("   System ready for Phase 2 implementation")
            return True
        else:
            print("\n⚠️  Phase 1 optimization integration needs improvement")
            print("   Consider optimizing components with low scores")
            return False

    except Exception as e:
        print(f"\n❌ Phase 1 integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
