#!/usr/bin/env python3
"""
GPT-Trader System Performance Benchmark
Comprehensive performance analysis including:
1. Import times for key modules
2. Memory usage of main components
3. Backtest execution speed
4. Data pipeline throughput
5. Bottleneck identification
"""

import gc
import importlib
import json
import os
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import numpy as np
    import pandas as pd
    import psutil

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: psutil/pandas not available, limited monitoring")


class PerformanceBenchmark:
    """Comprehensive system performance benchmark"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "import_times": {},
            "memory_usage": {},
            "component_performance": {},
            "bottlenecks": [],
            "recommendations": [],
        }

        if MONITORING_AVAILABLE:
            tracemalloc.start()

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information"""
        info = {
            "python_version": sys.version,
            "platform": sys.platform,
        }

        if MONITORING_AVAILABLE:
            info.update(
                {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                    "disk_usage": psutil.disk_usage("/")._asdict(),
                    "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
                }
            )

        return info

    def measure_import_performance(self):
        """Measure import times and memory usage of key modules"""
        print("=== IMPORT PERFORMANCE BENCHMARK ===")

        # Core modules to test
        modules_to_test = [
            ("bot.config", "Configuration System"),
            ("bot.core.base", "Core Base Module"),
            ("bot.dataflow.pipeline", "Data Pipeline"),
            ("bot.strategy.base", "Strategy Base"),
            ("bot.strategy.demo_ma", "Demo MA Strategy"),
            ("bot.strategy.trend_breakout", "Trend Breakout Strategy"),
            ("bot.backtest.engine_portfolio", "Portfolio Backtest Engine"),
            ("bot.risk.integration", "Risk Management"),
            ("bot.integration.orchestrator", "Integration Orchestrator"),
            ("bot.dataflow.sources.yfinance_source", "YFinance Data Source"),
            ("bot.cli.commands", "CLI Commands"),
            ("bot.ml.baseline_models", "ML Baseline Models"),
            ("bot.ml.performance_benchmark", "ML Performance Benchmark"),
        ]

        for module_name, description in modules_to_test:
            self._measure_single_import(module_name, description)

    def _measure_single_import(self, module_name: str, description: str):
        """Measure import time and memory for a single module"""
        print(f"Testing: {description} ({module_name})")

        # Clean up
        gc.collect()

        start_memory = 0
        if MONITORING_AVAILABLE and tracemalloc.is_tracing():
            start_memory = tracemalloc.get_traced_memory()[0] / (1024**2)  # MB

        start_time = time.perf_counter()

        try:
            # Remove from sys.modules if already imported for accurate timing
            if module_name in sys.modules:
                del sys.modules[module_name]

            module = importlib.import_module(module_name)

            end_time = time.perf_counter()

            end_memory = 0
            if MONITORING_AVAILABLE and tracemalloc.is_tracing():
                end_memory = tracemalloc.get_traced_memory()[0] / (1024**2)  # MB

            import_time = (end_time - start_time) * 1000  # Convert to ms
            memory_delta = end_memory - start_memory

            self.results["import_times"][description] = {
                "module": module_name,
                "time_ms": round(import_time, 2),
                "memory_mb": round(memory_delta, 2),
                "success": True,
            }

            print(f"  ✓ {import_time:.2f}ms, {memory_delta:.2f}MB")

            # Flag slow imports as bottlenecks
            if import_time > 1000:  # > 1 second
                self.results["bottlenecks"].append(
                    {
                        "type": "slow_import",
                        "module": module_name,
                        "time_ms": import_time,
                        "description": f"Slow import: {description}",
                    }
                )

        except Exception as e:
            error_msg = str(e)
            self.results["import_times"][description] = {
                "module": module_name,
                "error": error_msg,
                "success": False,
            }
            print(f"  ✗ Failed: {error_msg}")

            self.results["bottlenecks"].append(
                {
                    "type": "import_failure",
                    "module": module_name,
                    "error": error_msg,
                    "description": f"Import failure: {description}",
                }
            )

    def measure_data_pipeline_performance(self):
        """Measure data pipeline throughput"""
        print("\n=== DATA PIPELINE PERFORMANCE ===")

        try:
            from bot.dataflow.sources.yfinance_source import YFinanceSource

            # Test data fetching performance
            print("Testing YFinance data source...")

            start_time = time.perf_counter()

            source = YFinanceSource()
            symbols = ["AAPL", "MSFT", "GOOGL"]

            for symbol in symbols:
                try:
                    # Note: YFinanceSource.get_daily_bars expects start/end dates
                    # For 1 year, calculate start date
                    from datetime import datetime, timedelta

                    end_date = datetime.now().strftime("%Y-%m-%d")
                    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                    data = source.get_daily_bars(symbol, start=start_date, end=end_date)
                    if data is not None and len(data) > 0:
                        print(f"  ✓ {symbol}: {len(data)} records")
                    else:
                        print(f"  ✗ {symbol}: No data")
                except Exception as e:
                    print(f"  ✗ {symbol}: {str(e)}")

            end_time = time.perf_counter()
            fetch_time = end_time - start_time

            self.results["component_performance"]["data_pipeline"] = {
                "symbols_tested": len(symbols),
                "total_time_seconds": round(fetch_time, 2),
                "time_per_symbol_seconds": round(fetch_time / len(symbols), 2),
                "throughput_symbols_per_minute": round(len(symbols) / fetch_time * 60, 2),
            }

            print(
                f"Data pipeline throughput: {fetch_time:.2f}s total, {fetch_time/len(symbols):.2f}s per symbol"
            )

            # Flag slow data fetching
            if fetch_time / len(symbols) > 10:  # > 10 seconds per symbol
                self.results["bottlenecks"].append(
                    {
                        "type": "slow_data_fetch",
                        "time_per_symbol": fetch_time / len(symbols),
                        "description": "Slow data fetching from YFinance",
                    }
                )

        except Exception as e:
            print(f"Data pipeline test failed: {str(e)}")
            self.results["bottlenecks"].append(
                {
                    "type": "data_pipeline_failure",
                    "error": str(e),
                    "description": "Data pipeline test failure",
                }
            )

    def measure_backtest_performance(self):
        """Measure backtest execution speed"""
        print("\n=== BACKTEST PERFORMANCE ===")

        try:
            from bot.integration.orchestrator import IntegratedOrchestrator

            print("Testing integrated backtest performance...")

            start_time = time.perf_counter()
            start_memory = 0
            if MONITORING_AVAILABLE:
                start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB

            # Run a simple backtest
            orchestrator = IntegratedOrchestrator()

            # Test data
            symbols = ["AAPL"]

            # Create a strategy instance
            from bot.strategy.demo_ma import DemoMAStrategy

            strategy = DemoMAStrategy(fast=10, slow=20, atr_period=14)

            result = orchestrator.run_backtest(strategy=strategy, symbols=symbols)

            end_time = time.perf_counter()
            end_memory = 0
            if MONITORING_AVAILABLE:
                end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB

            backtest_time = end_time - start_time
            memory_usage = end_memory - start_memory

            self.results["component_performance"]["backtest"] = {
                "strategy": strategy_name,
                "symbols": len(symbols),
                "timeframe": timeframe,
                "execution_time_seconds": round(backtest_time, 2),
                "memory_usage_mb": round(memory_usage, 2),
                "success": result is not None,
            }

            print(f"Backtest performance: {backtest_time:.2f}s, {memory_usage:.2f}MB")

            # Flag slow backtests
            if backtest_time > 30:  # > 30 seconds
                self.results["bottlenecks"].append(
                    {
                        "type": "slow_backtest",
                        "time_seconds": backtest_time,
                        "description": "Slow backtest execution",
                    }
                )

        except Exception as e:
            print(f"Backtest test failed: {str(e)}")
            self.results["bottlenecks"].append(
                {
                    "type": "backtest_failure",
                    "error": str(e),
                    "description": "Backtest execution failure",
                }
            )

    def measure_strategy_performance(self):
        """Measure strategy calculation performance"""
        print("\n=== STRATEGY PERFORMANCE ===")

        try:
            from bot.strategy.demo_ma import DemoMAStrategy
            from bot.strategy.trend_breakout import TrendBreakoutStrategy

            # Create test data
            if MONITORING_AVAILABLE:
                dates = pd.date_range("2023-01-01", "2024-01-01", freq="D")
                test_data = pd.DataFrame(
                    {
                        "Open": np.random.rand(len(dates)) * 100 + 100,
                        "High": np.random.rand(len(dates)) * 100 + 110,
                        "Low": np.random.rand(len(dates)) * 100 + 90,
                        "Close": np.random.rand(len(dates)) * 100 + 100,
                        "Volume": np.random.randint(1000000, 10000000, len(dates)),
                    },
                    index=dates,
                )

                strategies = [
                    ("DemoMA", DemoMAStrategy()),
                    ("TrendBreakout", TrendBreakoutStrategy()),
                ]

                for name, strategy in strategies:
                    print(f"Testing {name} strategy...")

                    start_time = time.perf_counter()

                    try:
                        signals = strategy.generate_signals(test_data)

                        end_time = time.perf_counter()
                        calculation_time = end_time - start_time

                        self.results["component_performance"][f"strategy_{name.lower()}"] = {
                            "name": name,
                            "data_points": len(test_data),
                            "calculation_time_seconds": round(calculation_time, 4),
                            "signals_generated": len(signals) if signals is not None else 0,
                            "throughput_points_per_second": round(
                                len(test_data) / calculation_time, 2
                            ),
                        }

                        print(f"  ✓ {calculation_time:.4f}s for {len(test_data)} points")

                        # Flag slow strategies
                        if calculation_time > 1.0:  # > 1 second for 1 year of daily data
                            self.results["bottlenecks"].append(
                                {
                                    "type": "slow_strategy",
                                    "strategy": name,
                                    "time_seconds": calculation_time,
                                    "description": f"Slow strategy calculation: {name}",
                                }
                            )

                    except Exception as e:
                        print(f"  ✗ {name} failed: {str(e)}")

            else:
                print("Pandas not available, skipping strategy performance tests")

        except Exception as e:
            print(f"Strategy performance test failed: {str(e)}")

    def measure_memory_usage(self):
        """Measure memory usage of key components"""
        print("\n=== MEMORY USAGE ANALYSIS ===")

        if not MONITORING_AVAILABLE:
            print("Memory monitoring not available")
            return

        # Get current memory usage
        process = psutil.Process()
        memory_info = process.memory_info()

        self.results["memory_usage"] = {
            "current_rss_mb": round(memory_info.rss / (1024**2), 2),
            "current_vms_mb": round(memory_info.vms / (1024**2), 2),
            "peak_memory_mb": 0,
        }

        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            self.results["memory_usage"]["peak_memory_mb"] = round(peak / (1024**2), 2)

        print(f"Current RSS: {self.results['memory_usage']['current_rss_mb']}MB")
        print(f"Current VMS: {self.results['memory_usage']['current_vms_mb']}MB")
        print(f"Peak traced memory: {self.results['memory_usage']['peak_memory_mb']}MB")

        # Flag high memory usage
        if self.results["memory_usage"]["current_rss_mb"] > 1000:  # > 1GB
            self.results["bottlenecks"].append(
                {
                    "type": "high_memory_usage",
                    "memory_mb": self.results["memory_usage"]["current_rss_mb"],
                    "description": "High memory usage detected",
                }
            )

    def analyze_bottlenecks(self):
        """Analyze identified bottlenecks and generate recommendations"""
        print("\n=== BOTTLENECK ANALYSIS ===")

        if not self.results["bottlenecks"]:
            print("No major bottlenecks detected")
            return

        print(f"Found {len(self.results['bottlenecks'])} potential bottlenecks:")

        for i, bottleneck in enumerate(self.results["bottlenecks"], 1):
            print(f"{i}. {bottleneck['description']}")
            if "time_ms" in bottleneck:
                print(f"   Time: {bottleneck['time_ms']}ms")
            elif "time_seconds" in bottleneck:
                print(f"   Time: {bottleneck['time_seconds']}s")
            if "error" in bottleneck:
                print(f"   Error: {bottleneck['error']}")

        # Generate recommendations
        self._generate_recommendations()

    def _generate_recommendations(self):
        """Generate performance improvement recommendations"""
        recommendations = []

        # Analyze bottlenecks and suggest fixes
        for bottleneck in self.results["bottlenecks"]:
            if bottleneck["type"] == "slow_import":
                recommendations.append(
                    {
                        "priority": "medium",
                        "area": "imports",
                        "issue": f"Slow import: {bottleneck['module']}",
                        "recommendation": "Consider lazy imports or module restructuring",
                    }
                )

            elif bottleneck["type"] == "import_failure":
                recommendations.append(
                    {
                        "priority": "high",
                        "area": "dependencies",
                        "issue": f"Import failure: {bottleneck['module']}",
                        "recommendation": "Fix missing dependencies or import errors",
                    }
                )

            elif bottleneck["type"] == "slow_data_fetch":
                recommendations.append(
                    {
                        "priority": "high",
                        "area": "data_pipeline",
                        "issue": "Slow data fetching",
                        "recommendation": "Implement caching, parallel fetching, or data source optimization",
                    }
                )

            elif bottleneck["type"] == "slow_backtest":
                recommendations.append(
                    {
                        "priority": "medium",
                        "area": "backtesting",
                        "issue": "Slow backtest execution",
                        "recommendation": "Optimize data processing, use vectorization, or parallel execution",
                    }
                )

            elif bottleneck["type"] == "high_memory_usage":
                recommendations.append(
                    {
                        "priority": "medium",
                        "area": "memory",
                        "issue": "High memory usage",
                        "recommendation": "Optimize data structures, implement memory-efficient algorithms",
                    }
                )

        # Add general recommendations based on system analysis
        if MONITORING_AVAILABLE:
            cpu_count = psutil.cpu_count()
            if cpu_count > 1:
                recommendations.append(
                    {
                        "priority": "medium",
                        "area": "parallelization",
                        "issue": f"Underutilized CPU cores ({cpu_count} available)",
                        "recommendation": "Implement parallel processing for backtests and data fetching",
                    }
                )

        self.results["recommendations"] = recommendations

    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report = f"""
# Performance Report – GPT-Trader System (2025-08-15)

## Executive Summary
| Metric | Value | Status |
|--------|-------|--------|
| Total Bottlenecks | {len(self.results['bottlenecks'])} | {'⚠️' if self.results['bottlenecks'] else '✅'} |
| Import Failures | {sum(1 for _, data in self.results['import_times'].items() if not data.get('success', True))} | {'❌' if any(not data.get('success', True) for data in self.results['import_times'].values()) else '✅'} |
| System Memory | {self.results['system_info'].get('memory_available_gb', 'N/A'):.1f}GB available | ℹ️ |

## Import Performance Analysis
"""

        # Import times table
        successful_imports = [
            (desc, data)
            for desc, data in self.results["import_times"].items()
            if data.get("success")
        ]
        failed_imports = [
            (desc, data)
            for desc, data in self.results["import_times"].items()
            if not data.get("success")
        ]

        if successful_imports:
            report += "### Successful Imports\n"
            report += "| Module | Time (ms) | Memory (MB) |\n"
            report += "|--------|-----------|-------------|\n"

            for desc, data in sorted(
                successful_imports, key=lambda x: x[1]["time_ms"], reverse=True
            ):
                report += f"| {desc} | {data['time_ms']} | {data['memory_mb']} |\n"

        if failed_imports:
            report += "\n### Failed Imports\n"
            for desc, data in failed_imports:
                report += f"- **{desc}**: {data['error']}\n"

        # Component performance
        if self.results["component_performance"]:
            report += "\n## Component Performance\n"
            for component, metrics in self.results["component_performance"].items():
                report += f"\n### {component.replace('_', ' ').title()}\n"
                for key, value in metrics.items():
                    report += f"- **{key.replace('_', ' ').title()}**: {value}\n"

        # Bottlenecks
        if self.results["bottlenecks"]:
            report += "\n## Bottlenecks Identified\n"
            for i, bottleneck in enumerate(self.results["bottlenecks"], 1):
                report += f"{i}. **{bottleneck['type'].replace('_', ' ').title()}**: {bottleneck['description']}\n"

        # Recommendations
        if self.results["recommendations"]:
            report += "\n## Recommendations\n"

            high_priority = [r for r in self.results["recommendations"] if r["priority"] == "high"]
            medium_priority = [
                r for r in self.results["recommendations"] if r["priority"] == "medium"
            ]

            if high_priority:
                report += "\n### High Priority\n"
                for rec in high_priority:
                    report += f"- **{rec['area'].title()}**: {rec['recommendation']}\n"

            if medium_priority:
                report += "\n### Medium Priority\n"
                for rec in medium_priority:
                    report += f"- **{rec['area'].title()}**: {rec['recommendation']}\n"

        report += f"\n---\n**Report generated**: {self.results['timestamp']}\n"

        return report

    def save_results(self, filepath: str = None):
        """Save benchmark results to file"""
        if filepath is None:
            filepath = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"Results saved to: {filepath}")

    def run_full_benchmark(self):
        """Run complete performance benchmark"""
        print("🚀 Starting GPT-Trader Performance Benchmark")
        print("=" * 50)

        self.measure_import_performance()
        self.measure_data_pipeline_performance()
        self.measure_backtest_performance()
        self.measure_strategy_performance()
        self.measure_memory_usage()
        self.analyze_bottlenecks()

        print("\n" + "=" * 50)
        print("📊 Benchmark Complete!")

        # Generate and save report
        report = self.generate_report()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_results(f"benchmark_results_{timestamp}.json")

        with open(f"benchmark_report_{timestamp}.md", "w") as f:
            f.write(report)

        print(f"📄 Report saved: benchmark_report_{timestamp}.md")
        print(f"📊 Data saved: benchmark_results_{timestamp}.json")

        return report


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    report = benchmark.run_full_benchmark()
    print("\n" + report)
