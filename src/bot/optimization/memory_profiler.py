"""
Memory profiling and optimization utilities.

Provides tools for:
- Memory usage tracking
- Memory leak detection
- Object size analysis
- Memory optimization recommendations
"""

import gc
import os
import sys
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psutil


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    available_mb: float
    percent_used: float
    python_objects: int
    gc_stats: dict[str, int]
    top_types: list[tuple[str, int, int]]  # (type, count, size)


class MemoryProfiler:
    """Profile and analyze memory usage."""

    def __init__(self, track_allocations: bool = False):
        """Initialize memory profiler."""
        self.snapshots = []
        self.track_allocations = track_allocations

        if track_allocations:
            tracemalloc.start()

        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024

    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        # System memory
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()

        # Python objects
        gc.collect()
        python_objects = len(gc.get_objects())

        # GC stats
        gc_stats = {
            f"generation_{i}": gc.get_count()[i]
            for i in range(gc.get_count().__len__())
        }

        # Top object types
        type_counts = defaultdict(int)
        type_sizes = defaultdict(int)

        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            type_counts[obj_type] += 1
            try:
                type_sizes[obj_type] += sys.getsizeof(obj)
            except:
                pass

        top_types = sorted(
            [(k, v, type_sizes[k]) for k, v in type_counts.items()],
            key=lambda x: x[2],
            reverse=True
        )[:10]

        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            available_mb=psutil.virtual_memory().available / 1024 / 1024,
            percent_used=mem_percent,
            python_objects=python_objects,
            gc_stats=gc_stats,
            top_types=top_types
        )

        self.snapshots.append((label, snapshot))

        return snapshot

    def compare_snapshots(self, idx1: int = -2, idx2: int = -1) -> dict[str, Any]:
        """Compare two snapshots to find memory changes."""
        if len(self.snapshots) < 2:
            return {"error": "Need at least 2 snapshots"}

        label1, snap1 = self.snapshots[idx1]
        label2, snap2 = self.snapshots[idx2]

        comparison = {
            "from": label1,
            "to": label2,
            "time_diff": (snap2.timestamp - snap1.timestamp).total_seconds(),
            "memory_change_mb": snap2.rss_mb - snap1.rss_mb,
            "object_change": snap2.python_objects - snap1.python_objects,
            "percent_change": ((snap2.rss_mb - snap1.rss_mb) / snap1.rss_mb) * 100,
        }

        # Type changes
        type_changes = {}
        types1 = {t[0]: (t[1], t[2]) for t in snap1.top_types}
        types2 = {t[0]: (t[1], t[2]) for t in snap2.top_types}

        all_types = set(types1.keys()) | set(types2.keys())

        for type_name in all_types:
            count1, size1 = types1.get(type_name, (0, 0))
            count2, size2 = types2.get(type_name, (0, 0))

            if count2 - count1 != 0:
                type_changes[type_name] = {
                    "count_change": count2 - count1,
                    "size_change": size2 - size1
                }

        comparison["type_changes"] = type_changes

        return comparison

    def find_memory_leaks(self, threshold_mb: float = 10) -> list[dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []

        if len(self.snapshots) < 2:
            return leaks

        # Check for consistent memory growth
        memory_values = [snap.rss_mb for _, snap in self.snapshots]

        # Simple leak detection: consistent growth
        growth_count = sum(
            1 for i in range(1, len(memory_values))
            if memory_values[i] > memory_values[i-1]
        )

        if growth_count > len(memory_values) * 0.8:  # 80% growth
            total_growth = memory_values[-1] - memory_values[0]

            if total_growth > threshold_mb:
                leaks.append({
                    "type": "consistent_growth",
                    "total_growth_mb": total_growth,
                    "snapshots": len(self.snapshots),
                    "severity": "high" if total_growth > threshold_mb * 5 else "medium"
                })

        # Check for specific object accumulation
        if len(self.snapshots) >= 3:
            # Compare object counts
            for type_name in set(t[0] for _, snap in self.snapshots for t in snap.top_types):
                counts = []

                for _, snap in self.snapshots:
                    type_data = [t for t in snap.top_types if t[0] == type_name]
                    if type_data:
                        counts.append(type_data[0][1])

                if len(counts) >= 3:
                    # Check for consistent increase
                    increases = sum(1 for i in range(1, len(counts)) if counts[i] > counts[i-1])

                    if increases >= len(counts) - 1:
                        leaks.append({
                            "type": "object_leak",
                            "object_type": type_name,
                            "count_increase": counts[-1] - counts[0],
                            "severity": "medium"
                        })

        return leaks

    def get_memory_usage_df(self) -> pd.DataFrame:
        """Get memory usage as DataFrame."""
        data = []

        for label, snap in self.snapshots:
            data.append({
                "label": label,
                "timestamp": snap.timestamp,
                "rss_mb": snap.rss_mb,
                "vms_mb": snap.vms_mb,
                "available_mb": snap.available_mb,
                "percent_used": snap.percent_used,
                "python_objects": snap.python_objects,
            })

        return pd.DataFrame(data)

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Optimize DataFrame memory usage."""
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        optimizations = {
            "original_memory_mb": original_memory,
            "optimizations": []
        }

        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            # Downcast integers
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                    optimizations["optimizations"].append(f"{col}: int64 -> uint8")
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                    optimizations["optimizations"].append(f"{col}: int64 -> uint16")
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                    optimizations["optimizations"].append(f"{col}: int64 -> uint32")
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                    optimizations["optimizations"].append(f"{col}: int64 -> int8")
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                    optimizations["optimizations"].append(f"{col}: int64 -> int16")
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)
                    optimizations["optimizations"].append(f"{col}: int64 -> int32")

        # Optimize float columns
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            optimizations["optimizations"].append(f"{col}: float64 -> float32")

        # Convert object columns to category if appropriate
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])

            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
                optimizations["optimizations"].append(f"{col}: object -> category")

        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        optimizations["final_memory_mb"] = final_memory
        optimizations["savings_mb"] = original_memory - final_memory
        optimizations["savings_percent"] = ((original_memory - final_memory) / original_memory) * 100

        return df, optimizations

    def profile_function(self, func, *args, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Profile memory usage of a function."""
        # Take initial snapshot
        self.take_snapshot("before_function")

        # Run function
        result = func(*args, **kwargs)

        # Take final snapshot
        self.take_snapshot("after_function")

        # Compare snapshots
        comparison = self.compare_snapshots()

        profile = {
            "function": func.__name__,
            "memory_used_mb": comparison["memory_change_mb"],
            "objects_created": comparison["object_change"],
            "execution_time": comparison["time_diff"],
            "memory_per_second": comparison["memory_change_mb"] / max(comparison["time_diff"], 0.001),
        }

        return result, profile

    def get_optimization_recommendations(self) -> list[str]:
        """Get memory optimization recommendations."""
        recommendations = []

        if not self.snapshots:
            return ["No snapshots available. Take snapshots first."]

        latest_snap = self.snapshots[-1][1]

        # Check overall memory usage
        if latest_snap.percent_used > 80:
            recommendations.append(
                "⚠️ High memory usage (>80%). Consider reducing batch sizes or using generators."
            )

        # Check for large object types
        for type_name, count, size in latest_snap.top_types[:5]:
            if size > 100 * 1024 * 1024:  # > 100 MB
                recommendations.append(
                    f"Large {type_name} objects ({size / 1024 / 1024:.1f} MB). "
                    f"Consider using more memory-efficient data structures."
                )

            if type_name == "DataFrame" and count > 10:
                recommendations.append(
                    f"Multiple DataFrames ({count}). Consider consolidating or using views."
                )

        # Check for memory leaks
        leaks = self.find_memory_leaks()
        if leaks:
            for leak in leaks:
                if leak["type"] == "consistent_growth":
                    recommendations.append(
                        f"⚠️ Potential memory leak detected: {leak['total_growth_mb']:.1f} MB growth"
                    )
                elif leak["type"] == "object_leak":
                    recommendations.append(
                        f"Object accumulation: {leak['object_type']} (+{leak['count_increase']} objects)"
                    )

        # GC recommendations
        if latest_snap.gc_stats.get("generation_2", 0) > 100:
            recommendations.append(
                "High Gen 2 GC count. Consider manual gc.collect() after large operations."
            )

        return recommendations if recommendations else ["✓ No major memory issues detected"]

    def create_report(self) -> str:
        """Create memory profiling report."""
        report = []
        report.append("# Memory Profiling Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not self.snapshots:
            report.append("\nNo snapshots available.")
            return "\n".join(report)

        # Summary
        first_snap = self.snapshots[0][1]
        last_snap = self.snapshots[-1][1]

        report.append("\n## Summary")
        report.append(f"- Initial Memory: {first_snap.rss_mb:.1f} MB")
        report.append(f"- Final Memory: {last_snap.rss_mb:.1f} MB")
        report.append(f"- Total Change: {last_snap.rss_mb - first_snap.rss_mb:+.1f} MB")
        report.append(f"- Peak Memory: {max(s.rss_mb for _, s in self.snapshots):.1f} MB")

        # Snapshots
        report.append("\n## Snapshots")
        for label, snap in self.snapshots:
            report.append(f"\n### {label}")
            report.append(f"- Memory: {snap.rss_mb:.1f} MB")
            report.append(f"- Objects: {snap.python_objects:,}")
            report.append(f"- Top Types: {', '.join(t[0] for t in snap.top_types[:3])}")

        # Leak Detection
        leaks = self.find_memory_leaks()
        if leaks:
            report.append("\n## Potential Memory Leaks")
            for leak in leaks:
                report.append(f"- {leak}")

        # Recommendations
        report.append("\n## Recommendations")
        for rec in self.get_optimization_recommendations():
            report.append(f"- {rec}")

        return "\n".join(report)


def demo_memory_profiling():
    """Demonstrate memory profiling capabilities."""
    print("Memory Profiling Demo")
    print("=" * 50)

    profiler = MemoryProfiler(track_allocations=True)

    # Take initial snapshot
    profiler.take_snapshot("initial")

    # Create some data
    print("\nCreating large DataFrame...")
    data = {
        f"col_{i}": np.random.randn(100000)
        for i in range(50)
    }
    df = pd.DataFrame(data)
    profiler.take_snapshot("after_dataframe")

    # Optimize DataFrame
    print("Optimizing DataFrame memory...")
    df_optimized, optimization_stats = profiler.optimize_dataframe_memory(df)
    profiler.take_snapshot("after_optimization")

    print("\nOptimization Results:")
    print(f"- Original: {optimization_stats['original_memory_mb']:.1f} MB")
    print(f"- Optimized: {optimization_stats['final_memory_mb']:.1f} MB")
    print(f"- Savings: {optimization_stats['savings_percent']:.1f}%")

    # Create more data (simulate leak)
    print("\nSimulating memory accumulation...")
    leaked_data = []
    for i in range(5):
        leaked_data.append(np.random.randn(10000, 100))
        profiler.take_snapshot(f"iteration_{i}")

    # Check for leaks
    print("\nChecking for memory leaks...")
    leaks = profiler.find_memory_leaks(threshold_mb=5)

    if leaks:
        print("⚠️ Potential leaks detected:")
        for leak in leaks:
            print(f"  - {leak['type']}: {leak.get('total_growth_mb', 'N/A')} MB")
    else:
        print("✓ No significant leaks detected")

    # Get recommendations
    print("\nMemory Optimization Recommendations:")
    for rec in profiler.get_optimization_recommendations():
        print(f"  {rec}")

    # Generate report
    report = profiler.create_report()

    # Save report
    with open("memory_profile_report.md", "w") as f:
        f.write(report)

    print("\n✓ Memory profiling complete. Report saved to memory_profile_report.md")

    return profiler


if __name__ == "__main__":
    profiler = demo_memory_profiling()
