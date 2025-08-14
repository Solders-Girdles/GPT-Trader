"""
Serialization format benchmark for data storage optimization.

Compares performance of different serialization formats:
- Parquet
- Feather
- CSV
- JSON
- Pickle/Joblib
- HDF5
"""

import time
import os
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import psutil
from datetime import datetime, timedelta


class SerializationBenchmark:
    """Benchmark different serialization formats for market data."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def generate_test_data(self, 
                          n_rows: int = 10000,
                          n_cols: int = 20) -> pd.DataFrame:
        """Generate test market data of specified size."""
        np.random.seed(42)
        
        # Create realistic market data
        dates = pd.date_range(
            start="2020-01-01",
            periods=n_rows,
            freq="1min"
        )
        
        data = {
            "timestamp": dates,
            "open": np.random.uniform(100, 200, n_rows),
            "high": np.random.uniform(100, 200, n_rows),
            "low": np.random.uniform(100, 200, n_rows),
            "close": np.random.uniform(100, 200, n_rows),
            "volume": np.random.randint(1000, 1000000, n_rows),
        }
        
        # Add technical indicators
        for i in range(n_cols - 6):
            data[f"indicator_{i}"] = np.random.randn(n_rows)
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def measure_performance(self, 
                          func, 
                          *args, 
                          **kwargs) -> Tuple[float, float]:
        """Measure execution time and memory usage of a function."""
        # Memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        
        return execution_time, memory_used, result
    
    def benchmark_csv(self, df: pd.DataFrame, filepath: Path) -> Dict[str, float]:
        """Benchmark CSV format."""
        results = {}
        
        # Write benchmark
        write_time, write_mem, _ = self.measure_performance(
            df.to_csv, filepath
        )
        results["write_time"] = write_time
        results["write_memory"] = write_mem
        
        # Read benchmark
        read_time, read_mem, _ = self.measure_performance(
            pd.read_csv, filepath, index_col=0, parse_dates=True
        )
        results["read_time"] = read_time
        results["read_memory"] = read_mem
        
        # File size
        results["file_size_mb"] = filepath.stat().st_size / 1024 / 1024
        
        return results
    
    def benchmark_parquet(self, df: pd.DataFrame, filepath: Path) -> Dict[str, float]:
        """Benchmark Parquet format."""
        results = {}
        
        # Write benchmark
        write_time, write_mem, _ = self.measure_performance(
            df.to_parquet, filepath, compression="snappy"
        )
        results["write_time"] = write_time
        results["write_memory"] = write_mem
        
        # Read benchmark
        read_time, read_mem, _ = self.measure_performance(
            pd.read_parquet, filepath
        )
        results["read_time"] = read_time
        results["read_memory"] = read_mem
        
        # File size
        results["file_size_mb"] = filepath.stat().st_size / 1024 / 1024
        
        return results
    
    def benchmark_feather(self, df: pd.DataFrame, filepath: Path) -> Dict[str, float]:
        """Benchmark Feather format."""
        results = {}
        
        # Reset index for feather (doesn't support datetime index)
        df_reset = df.reset_index()
        
        # Write benchmark
        write_time, write_mem, _ = self.measure_performance(
            df_reset.to_feather, filepath
        )
        results["write_time"] = write_time
        results["write_memory"] = write_mem
        
        # Read benchmark
        read_time, read_mem, df_read = self.measure_performance(
            pd.read_feather, filepath
        )
        results["read_time"] = read_time
        results["read_memory"] = read_mem
        
        # File size
        results["file_size_mb"] = filepath.stat().st_size / 1024 / 1024
        
        return results
    
    def benchmark_hdf5(self, df: pd.DataFrame, filepath: Path) -> Dict[str, float]:
        """Benchmark HDF5 format."""
        results = {}
        
        # Write benchmark
        write_time, write_mem, _ = self.measure_performance(
            df.to_hdf, filepath, key="data", mode="w", complevel=5
        )
        results["write_time"] = write_time
        results["write_memory"] = write_mem
        
        # Read benchmark
        read_time, read_mem, _ = self.measure_performance(
            pd.read_hdf, filepath, key="data"
        )
        results["read_time"] = read_time
        results["read_memory"] = read_mem
        
        # File size
        results["file_size_mb"] = filepath.stat().st_size / 1024 / 1024
        
        return results
    
    def benchmark_pickle(self, df: pd.DataFrame, filepath: Path) -> Dict[str, float]:
        """Benchmark Pickle format."""
        results = {}
        
        # Write benchmark
        def write_pickle():
            with open(filepath, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        write_time, write_mem, _ = self.measure_performance(write_pickle)
        results["write_time"] = write_time
        results["write_memory"] = write_mem
        
        # Read benchmark
        def read_pickle():
            with open(filepath, "rb") as f:
                return pickle.load(f)
        
        read_time, read_mem, _ = self.measure_performance(read_pickle)
        results["read_time"] = read_time
        results["read_memory"] = read_mem
        
        # File size
        results["file_size_mb"] = filepath.stat().st_size / 1024 / 1024
        
        return results
    
    def benchmark_joblib(self, df: pd.DataFrame, filepath: Path) -> Dict[str, float]:
        """Benchmark Joblib format."""
        results = {}
        
        # Write benchmark
        write_time, write_mem, _ = self.measure_performance(
            joblib.dump, df, filepath, compress=3
        )
        results["write_time"] = write_time
        results["write_memory"] = write_mem
        
        # Read benchmark
        read_time, read_mem, _ = self.measure_performance(
            joblib.load, filepath
        )
        results["read_time"] = read_time
        results["read_memory"] = read_mem
        
        # File size
        results["file_size_mb"] = filepath.stat().st_size / 1024 / 1024
        
        return results
    
    def benchmark_json(self, df: pd.DataFrame, filepath: Path) -> Dict[str, float]:
        """Benchmark JSON format."""
        results = {}
        
        # Write benchmark
        write_time, write_mem, _ = self.measure_performance(
            df.to_json, filepath, orient="split", date_format="iso"
        )
        results["write_time"] = write_time
        results["write_memory"] = write_mem
        
        # Read benchmark
        read_time, read_mem, _ = self.measure_performance(
            pd.read_json, filepath, orient="split"
        )
        results["read_time"] = read_time
        results["read_memory"] = read_mem
        
        # File size
        results["file_size_mb"] = filepath.stat().st_size / 1024 / 1024
        
        return results
    
    def run_benchmark(self, 
                     data_sizes: list = None,
                     formats: list = None) -> pd.DataFrame:
        """Run complete benchmark suite."""
        if data_sizes is None:
            data_sizes = [
                (1000, 10),    # Small
                (10000, 20),   # Medium
                (100000, 30),  # Large
                (500000, 40),  # Extra large
            ]
        
        if formats is None:
            formats = ["csv", "parquet", "feather", "hdf5", "pickle", "joblib", "json"]
        
        results = []
        
        for n_rows, n_cols in data_sizes:
            print(f"\nBenchmarking data size: {n_rows} rows x {n_cols} columns")
            
            # Generate test data
            df = self.generate_test_data(n_rows, n_cols)
            data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            for format_name in formats:
                print(f"  Testing {format_name}...", end=" ")
                
                filepath = self.output_dir / f"test.{format_name}"
                
                try:
                    # Run appropriate benchmark
                    if format_name == "csv":
                        metrics = self.benchmark_csv(df, filepath)
                    elif format_name == "parquet":
                        metrics = self.benchmark_parquet(df, filepath)
                    elif format_name == "feather":
                        metrics = self.benchmark_feather(df, filepath)
                    elif format_name == "hdf5":
                        metrics = self.benchmark_hdf5(df, filepath.with_suffix(".h5"))
                    elif format_name == "pickle":
                        metrics = self.benchmark_pickle(df, filepath.with_suffix(".pkl"))
                    elif format_name == "joblib":
                        metrics = self.benchmark_joblib(df, filepath.with_suffix(".joblib"))
                    elif format_name == "json":
                        metrics = self.benchmark_json(df, filepath)
                    else:
                        continue
                    
                    # Calculate derived metrics
                    metrics["format"] = format_name
                    metrics["n_rows"] = n_rows
                    metrics["n_cols"] = n_cols
                    metrics["data_size_mb"] = data_size_mb
                    metrics["compression_ratio"] = data_size_mb / metrics["file_size_mb"]
                    metrics["total_time"] = metrics["read_time"] + metrics["write_time"]
                    metrics["throughput_mb_s"] = data_size_mb / metrics["total_time"]
                    
                    results.append(metrics)
                    print(f"✓ (Total: {metrics['total_time']:.3f}s, Size: {metrics['file_size_mb']:.1f}MB)")
                    
                    # Clean up
                    if filepath.exists():
                        filepath.unlink()
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / "benchmark_results.csv", index=False)
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze benchmark results and provide recommendations."""
        analysis = {}
        
        # Group by format
        format_stats = results_df.groupby("format").agg({
            "read_time": "mean",
            "write_time": "mean",
            "total_time": "mean",
            "file_size_mb": "mean",
            "compression_ratio": "mean",
            "throughput_mb_s": "mean"
        }).round(3)
        
        # Find best performers
        analysis["fastest_read"] = format_stats["read_time"].idxmin()
        analysis["fastest_write"] = format_stats["write_time"].idxmin()
        analysis["smallest_size"] = format_stats["file_size_mb"].idxmin()
        analysis["best_compression"] = format_stats["compression_ratio"].idxmax()
        analysis["best_throughput"] = format_stats["throughput_mb_s"].idxmax()
        
        # Overall recommendation based on use case
        analysis["recommendations"] = {
            "real_time_trading": "feather",  # Fastest read/write
            "historical_storage": "parquet",  # Good compression and performance
            "data_archival": "parquet",       # Best compression
            "temporary_cache": "feather",     # Fastest access
            "data_exchange": "csv",           # Universal compatibility
        }
        
        # Performance summary
        analysis["summary"] = format_stats.to_dict()
        
        return analysis
    
    def create_report(self, results_df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Create a detailed benchmark report."""
        report = []
        report.append("# Serialization Format Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n## Executive Summary")
        report.append(f"- **Fastest Read**: {analysis['fastest_read']}")
        report.append(f"- **Fastest Write**: {analysis['fastest_write']}")
        report.append(f"- **Smallest Size**: {analysis['smallest_size']}")
        report.append(f"- **Best Compression**: {analysis['best_compression']}")
        report.append(f"- **Best Throughput**: {analysis['best_throughput']}")
        
        report.append("\n## Recommendations by Use Case")
        for use_case, format_name in analysis["recommendations"].items():
            report.append(f"- **{use_case.replace('_', ' ').title()}**: {format_name}")
        
        report.append("\n## Detailed Performance Metrics")
        report.append("```")
        for format_name, stats in analysis["summary"].items():
            report.append(f"\n{format_name.upper()}:")
            for metric, values in stats.items():
                report.append(f"  {metric}: {values:.3f}")
        report.append("```")
        
        report.append("\n## Performance by Data Size")
        for size in results_df["n_rows"].unique():
            size_data = results_df[results_df["n_rows"] == size]
            report.append(f"\n### {size:,} rows")
            
            best_format = size_data.loc[size_data["total_time"].idxmin(), "format"]
            best_time = size_data["total_time"].min()
            
            report.append(f"Best performer: **{best_format}** ({best_time:.3f}s total)")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / "benchmark_report.md"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        return report_text


def main():
    """Run serialization benchmarks."""
    print("=" * 60)
    print("SERIALIZATION FORMAT BENCHMARK")
    print("=" * 60)
    
    benchmark = SerializationBenchmark()
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    results = benchmark.run_benchmark()
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = benchmark.analyze_results(results)
    
    # Create report
    print("\nGenerating report...")
    report = benchmark.create_report(results, analysis)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    # Print summary
    print("\nTop Recommendations:")
    for use_case, format_name in analysis["recommendations"].items():
        print(f"  {use_case.replace('_', ' ').title()}: {format_name}")
    
    print(f"\nFull report saved to: benchmark_results/benchmark_report.md")
    
    return results, analysis


if __name__ == "__main__":
    results, analysis = main()