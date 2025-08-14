"""
Performance benchmarks for Phase 1.5
Compares performance of consolidated architecture
"""

import sys
import time
import tempfile
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class PerformanceBenchmark:
    """Benchmark suite for consolidated architecture"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.process = psutil.Process()

    def start_benchmark(self, name: str):
        """Start a benchmark timer"""
        self.start_time = time.perf_counter()
        # Reset memory tracking
        self.process.memory_info()

    def end_benchmark(self, name: str) -> Dict[str, Any]:
        """End benchmark and record results"""
        elapsed = time.perf_counter() - self.start_time
        memory = self.process.memory_info().rss / 1024 / 1024  # MB

        result = {
            "name": name,
            "elapsed_seconds": elapsed,
            "memory_mb": memory,
            "timestamp": datetime.now().isoformat(),
        }

        self.results[name] = result
        return result

    def benchmark_database_operations(self):
        """Benchmark database operations"""
        from bot.core.database import DatabaseConfig, DatabaseManager

        print("Benchmarking database operations...")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatabaseConfig(database_path=Path(tmpdir) / "benchmark.db")

            # Benchmark initialization
            self.start_benchmark("db_initialization")
            db_manager = DatabaseManager(config)
            init_result = self.end_benchmark("db_initialization")
            print(f"  Database initialization: {init_result['elapsed_seconds']:.4f}s")

            # Benchmark inserts
            self.start_benchmark("db_insert_1000")
            for i in range(1000):
                db_manager.insert_record(
                    "components",
                    {
                        "component_id": f"component_{i}",
                        "component_type": "benchmark",
                        "status": "active",
                    },
                )
            insert_result = self.end_benchmark("db_insert_1000")
            print(f"  1000 inserts: {insert_result['elapsed_seconds']:.4f}s")

            # Benchmark queries
            self.start_benchmark("db_query_1000")
            for i in range(1000):
                db_manager.fetch_one(
                    "SELECT * FROM components WHERE component_id = ?", (f"component_{i}",)
                )
            query_result = self.end_benchmark("db_query_1000")
            print(f"  1000 queries: {query_result['elapsed_seconds']:.4f}s")

            # Benchmark complex query
            self.start_benchmark("db_complex_query")
            db_manager.fetch_all(
                """
                SELECT component_type, COUNT(*) as count
                FROM components
                GROUP BY component_type
                ORDER BY count DESC
            """
            )
            complex_result = self.end_benchmark("db_complex_query")
            print(f"  Complex query: {complex_result['elapsed_seconds']:.4f}s")

            db_manager.close()

    def benchmark_cli_startup(self):
        """Benchmark CLI startup time"""
        import subprocess

        print("\nBenchmarking CLI startup...")

        # Benchmark help command
        self.start_benchmark("cli_help")
        result = subprocess.run(
            [sys.executable, "-m", "bot.cli", "--help"], capture_output=True, timeout=30
        )
        help_result = self.end_benchmark("cli_help")
        print(f"  CLI help: {help_result['elapsed_seconds']:.4f}s")

        # Benchmark backtest help
        self.start_benchmark("cli_backtest_help")
        result = subprocess.run(
            [sys.executable, "-m", "bot.cli", "backtest", "--help"], capture_output=True, timeout=30
        )
        backtest_result = self.end_benchmark("cli_backtest_help")
        print(f"  Backtest help: {backtest_result['elapsed_seconds']:.4f}s")

    def benchmark_data_pipeline(self):
        """Benchmark data pipeline operations"""
        from bot.data.unified_pipeline import UnifiedDataPipeline, DataConfig

        print("\nBenchmarking data pipeline...")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DataConfig(
                cache_enabled=True, cache_dir=Path(tmpdir), validate_data=True, repair_data=True
            )

            # Initialize pipeline
            self.start_benchmark("pipeline_init")
            pipeline = UnifiedDataPipeline(config)
            init_result = self.end_benchmark("pipeline_init")
            print(f"  Pipeline initialization: {init_result['elapsed_seconds']:.4f}s")

            # Create test data
            dates = pd.date_range(end=datetime.now(), periods=10000, freq="T")
            test_data = pd.DataFrame(
                {
                    "open": np.random.randn(10000).cumsum() + 100,
                    "high": np.random.randn(10000).cumsum() + 101,
                    "low": np.random.randn(10000).cumsum() + 99,
                    "close": np.random.randn(10000).cumsum() + 100,
                    "volume": np.random.randint(1000, 10000, 10000),
                },
                index=dates,
            )

            # Benchmark validation
            self.start_benchmark("data_validation_10k")
            validated = pipeline._validate_data(test_data.copy())
            validation_result = self.end_benchmark("data_validation_10k")
            print(f"  Validate 10k rows: {validation_result['elapsed_seconds']:.4f}s")

            # Benchmark caching
            cache_key = "test_data"

            self.start_benchmark("cache_write")
            pipeline._cache_data(cache_key, test_data)
            write_result = self.end_benchmark("cache_write")
            print(f"  Cache write: {write_result['elapsed_seconds']:.4f}s")

            self.start_benchmark("cache_read")
            cached = pipeline._get_cached(cache_key)
            read_result = self.end_benchmark("cache_read")
            print(f"  Cache read: {read_result['elapsed_seconds']:.4f}s")

    def benchmark_strategy_signals(self):
        """Benchmark strategy signal generation"""
        from bot.strategy.demo_ma import DemoMAStrategy
        from bot.strategy.trend_breakout import TrendBreakoutStrategy

        print("\nBenchmarking strategy signal generation...")

        # Create test data
        dates = pd.date_range(end=datetime.now(), periods=5000, freq="H")
        test_data = pd.DataFrame(
            {
                "open": np.random.randn(5000).cumsum() + 100,
                "high": np.random.randn(5000).cumsum() + 101,
                "low": np.random.randn(5000).cumsum() + 99,
                "close": np.random.randn(5000).cumsum() + 100,
                "volume": np.random.randint(100000, 1000000, 5000),
            },
            index=dates,
        )

        # Ensure high >= low
        test_data["high"] = test_data[["high", "close"]].max(axis=1)
        test_data["low"] = test_data[["low", "close"]].min(axis=1)

        # Benchmark MA strategy
        self.start_benchmark("ma_strategy_5k")
        ma_strategy = DemoMAStrategy(window=20)
        ma_signals = ma_strategy.generate_signals(test_data)
        ma_result = self.end_benchmark("ma_strategy_5k")
        print(f"  MA strategy (5k bars): {ma_result['elapsed_seconds']:.4f}s")

        # Benchmark trend breakout strategy
        self.start_benchmark("trend_strategy_5k")
        trend_strategy = TrendBreakoutStrategy(
            donchian_period=55, atr_period=20, atr_multiplier=2.0
        )
        trend_signals = trend_strategy.generate_signals(test_data)
        trend_result = self.end_benchmark("trend_strategy_5k")
        print(f"  Trend strategy (5k bars): {trend_result['elapsed_seconds']:.4f}s")

    def benchmark_monitoring(self):
        """Benchmark monitoring system"""
        from bot.monitoring.monitor import UnifiedMonitor, MonitorConfig

        print("\nBenchmarking monitoring system...")

        config = MonitorConfig(metrics_interval=1, health_check_interval=1, alert_check_interval=1)

        # Initialize monitor
        self.start_benchmark("monitor_init")
        monitor = UnifiedMonitor(config)
        init_result = self.end_benchmark("monitor_init")
        print(f"  Monitor initialization: {init_result['elapsed_seconds']:.4f}s")

        # Start monitor
        self.start_benchmark("monitor_start")
        monitor.start()
        start_result = self.end_benchmark("monitor_start")
        print(f"  Monitor start: {start_result['elapsed_seconds']:.4f}s")

        # Let it run for a bit
        time.sleep(2)

        # Get metrics
        self.start_benchmark("monitor_metrics")
        metrics = monitor.get_metrics()
        metrics_result = self.end_benchmark("monitor_metrics")
        print(f"  Get metrics: {metrics_result['elapsed_seconds']:.4f}s")

        # Stop monitor
        self.start_benchmark("monitor_stop")
        monitor.stop()
        stop_result = self.end_benchmark("monitor_stop")
        print(f"  Monitor stop: {stop_result['elapsed_seconds']:.4f}s")

    def generate_report(self):
        """Generate benchmark report"""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)

        # Calculate totals
        total_time = sum(r["elapsed_seconds"] for r in self.results.values())
        avg_memory = np.mean([r["memory_mb"] for r in self.results.values()])

        print(f"\nTotal operations: {len(self.results)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average memory: {avg_memory:.2f} MB")

        # Find slowest operations
        sorted_results = sorted(
            self.results.items(), key=lambda x: x[1]["elapsed_seconds"], reverse=True
        )

        print("\nSlowest operations:")
        for name, result in sorted_results[:5]:
            print(f"  {name}: {result['elapsed_seconds']:.4f}s")

        # Performance grades
        print("\nPerformance Assessment:")

        # Database performance
        if "db_insert_1000" in self.results:
            inserts_per_sec = 1000 / self.results["db_insert_1000"]["elapsed_seconds"]
            print(f"  Database inserts: {inserts_per_sec:.0f} ops/sec", end="")
            if inserts_per_sec > 500:
                print(" ✅ Excellent")
            elif inserts_per_sec > 200:
                print(" ✓ Good")
            else:
                print(" ⚠️ Needs optimization")

        # CLI startup
        if "cli_help" in self.results:
            cli_time = self.results["cli_help"]["elapsed_seconds"]
            print(f"  CLI startup: {cli_time:.2f}s", end="")
            if cli_time < 1:
                print(" ✅ Excellent")
            elif cli_time < 2:
                print(" ✓ Good")
            else:
                print(" ⚠️ Slow")

        # Strategy performance
        if "ma_strategy_5k" in self.results:
            strategy_time = self.results["ma_strategy_5k"]["elapsed_seconds"]
            bars_per_sec = 5000 / strategy_time
            print(f"  Strategy signals: {bars_per_sec:.0f} bars/sec", end="")
            if bars_per_sec > 10000:
                print(" ✅ Excellent")
            elif bars_per_sec > 5000:
                print(" ✓ Good")
            else:
                print(" ⚠️ Needs optimization")

        # Save results to file
        report_path = Path("benchmark_results.json")
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {report_path}")

        return self.results


def compare_with_baseline():
    """Compare current performance with baseline (if available)"""
    baseline_path = Path("baseline_benchmark.json")

    if not baseline_path.exists():
        print("\nNo baseline found. Current results will be saved as baseline.")
        return None

    with open(baseline_path) as f:
        baseline = json.load(f)

    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)

    current = PerformanceBenchmark().results

    for name in baseline.keys():
        if name in current:
            baseline_time = baseline[name]["elapsed_seconds"]
            current_time = current[name]["elapsed_seconds"]

            improvement = (baseline_time - current_time) / baseline_time * 100

            print(f"\n{name}:")
            print(f"  Baseline: {baseline_time:.4f}s")
            print(f"  Current:  {current_time:.4f}s")

            if improvement > 0:
                print(f"  ✅ {improvement:.1f}% faster")
            else:
                print(f"  ⚠️ {abs(improvement):.1f}% slower")


def main():
    """Run complete benchmark suite"""
    print("=" * 80)
    print("GPT-TRADER PERFORMANCE BENCHMARK")
    print("Phase 1.5: Consolidated Architecture")
    print("=" * 80)

    benchmark = PerformanceBenchmark()

    try:
        # Run benchmarks
        benchmark.benchmark_database_operations()
        benchmark.benchmark_cli_startup()
        benchmark.benchmark_data_pipeline()
        benchmark.benchmark_strategy_signals()
        benchmark.benchmark_monitoring()

        # Generate report
        results = benchmark.generate_report()

        # Compare with baseline if available
        compare_with_baseline()

        print("\n" + "=" * 80)
        print("✅ BENCHMARK COMPLETE")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
