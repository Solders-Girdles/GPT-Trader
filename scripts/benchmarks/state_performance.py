#!/usr/bin/env python3
"""
Performance benchmarks for State Management system.

Compares StateManager batch operations vs direct repository access
to validate optimization improvements for backup/checkpoint/recovery workloads.

Usage:
    python scripts/benchmarks/state_performance.py
"""

import asyncio
import json
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Mock adapters for in-memory testing
class MockRedisAdapter:
    """In-memory mock of Redis for benchmarking."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        self._data[key] = value
        return True

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def keys(self, pattern: str) -> list[str]:
        # Simple pattern matching for benchmark
        if pattern == "*":
            return list(self._data.keys())
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self._data.keys() if k.startswith(prefix)]
        return []

    def dbsize(self) -> int:
        return len(self._data)

    def close(self) -> None:
        pass


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    duration_ms: float
    memory_mb: float
    keys_processed: int
    cache_hits: int
    cache_misses: int

    @property
    def keys_per_second(self) -> float:
        """Calculate throughput."""
        if self.duration_ms == 0:
            return 0.0
        return (self.keys_processed / self.duration_ms) * 1000

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Duration: {self.duration_ms:.2f}ms\n"
            f"  Memory: {self.memory_mb:.2f}MB\n"
            f"  Keys: {self.keys_processed}\n"
            f"  Throughput: {self.keys_per_second:.2f} keys/sec\n"
            f"  Cache hits: {self.cache_hits} ({self.cache_hit_rate:.1f}%)\n"
            f"  Cache misses: {self.cache_misses}"
        )


class StateBenchmarkSuite:
    """Benchmark suite for state management operations."""

    def __init__(self, num_keys: int = 500):
        self.num_keys = num_keys
        self.results: list[BenchmarkResult] = []

    async def setup_test_data(self) -> tuple[Any, MockRedisAdapter]:
        """Create test StateManager with realistic data."""
        from bot_v2.state.state_manager import StateConfig, StateManager

        # Create mock adapter with test data
        redis_adapter = MockRedisAdapter()

        # Pre-populate with test data
        for i in range(self.num_keys):
            key = f"position:{i}"
            value = json.dumps(
                {
                    "symbol": f"BTC-USD-{i}",
                    "size": 1.5 + (i * 0.1),
                    "entry_price": 50000 + (i * 100),
                    "unrealized_pnl": (i * 10.5),
                    "timestamp": "2025-10-01T12:00:00Z",
                }
            )
            redis_adapter.setex(key, 3600, value)

        # Create StateManager with performance tracking
        config = StateConfig(
            cache_size_mb=10,  # Small cache to force some misses
            enable_performance_tracking=True,
        )

        manager = StateManager(
            config=config,
            redis_adapter=redis_adapter,
            postgres_adapter=None,
            s3_adapter=None,
        )

        return manager, redis_adapter

    async def benchmark_state_manager_batch_read(self) -> BenchmarkResult:
        """Benchmark batch reading via StateManager (current pattern)."""
        manager, _ = await self.setup_test_data()

        tracemalloc.start()
        start_time = time.perf_counter()

        # Typical batch operation pattern
        keys = await manager.get_keys_by_pattern("position:*")
        results = {}
        for key in keys:
            value = await manager.get_state(key)
            if value:
                results[key] = value

        duration_ms = (time.perf_counter() - start_time) * 1000
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Get cache stats
        cache_stats = manager._cache_manager.get_cache_stats()
        cache_keys = cache_stats.get("cache_keys", 0)

        return BenchmarkResult(
            name="StateManager Batch Read (via get_state loop)",
            duration_ms=duration_ms,
            memory_mb=peak_mem / (1024 * 1024),
            keys_processed=len(results),
            cache_hits=cache_keys,
            cache_misses=len(results) - cache_keys,
        )

    async def benchmark_direct_repository_batch_read(self) -> BenchmarkResult:
        """Benchmark batch reading via direct repository access (optimized)."""
        manager, _ = await self.setup_test_data()
        repos = manager.get_repositories()

        tracemalloc.start()
        start_time = time.perf_counter()

        # Optimized pattern - direct repository access
        results = {}
        if repos.redis:
            keys = await repos.redis.keys("position:*")
            for key in keys:
                value = await repos.redis.fetch(key)
                if value:
                    results[key] = value

        duration_ms = (time.perf_counter() - start_time) * 1000
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            name="Direct Repository Batch Read (optimized)",
            duration_ms=duration_ms,
            memory_mb=peak_mem / (1024 * 1024),
            keys_processed=len(results),
            cache_hits=0,  # Doesn't use cache
            cache_misses=len(results),
        )

    async def benchmark_state_manager_with_cache_warmup(self) -> BenchmarkResult:
        """Benchmark StateManager with warm cache (best case)."""
        manager, _ = await self.setup_test_data()

        # Warm up cache
        keys = await manager.get_keys_by_pattern("position:*")
        for key in keys[: self.num_keys // 2]:  # Warm half the cache
            await manager.get_state(key)

        tracemalloc.start()
        start_time = time.perf_counter()

        # Now do the batch read
        results = {}
        for key in keys:
            value = await manager.get_state(key)
            if value:
                results[key] = value

        duration_ms = (time.perf_counter() - start_time) * 1000
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Get cache stats
        cache_stats = manager._cache_manager.get_cache_stats()
        cache_keys = cache_stats.get("cache_keys", 0)

        return BenchmarkResult(
            name="StateManager with Warm Cache (50% pre-cached)",
            duration_ms=duration_ms,
            memory_mb=peak_mem / (1024 * 1024),
            keys_processed=len(results),
            cache_hits=cache_keys,
            cache_misses=len(results) - cache_keys,
        )

    async def run_all_benchmarks(self) -> None:
        """Run all benchmarks and print results."""
        print(f"\n{'='*70}")
        print(f"State Management Performance Benchmarks")
        print(f"{'='*70}")
        print(f"Dataset: {self.num_keys} keys (position:*)\n")

        # Run benchmarks
        print("Running benchmarks...")

        result1 = await self.benchmark_state_manager_batch_read()
        self.results.append(result1)
        print(f"\n✓ Completed: {result1.name}")

        result2 = await self.benchmark_direct_repository_batch_read()
        self.results.append(result2)
        print(f"✓ Completed: {result2.name}")

        result3 = await self.benchmark_state_manager_with_cache_warmup()
        self.results.append(result3)
        print(f"✓ Completed: {result3.name}")

        # Print detailed results
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}\n")

        for result in self.results:
            print(result)
            print()

        # Print comparison
        self.print_comparison()

    def print_comparison(self) -> None:
        """Print performance comparison."""
        if len(self.results) < 2:
            return

        state_manager_result = self.results[0]
        repository_result = self.results[1]

        time_improvement = (
            (state_manager_result.duration_ms - repository_result.duration_ms)
            / state_manager_result.duration_ms
            * 100
        )

        memory_improvement = (
            (state_manager_result.memory_mb - repository_result.memory_mb)
            / state_manager_result.memory_mb
            * 100
        )

        print(f"{'='*70}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*70}\n")

        print(f"StateManager vs Direct Repository Access:\n")
        print(
            f"  Time improvement: {time_improvement:+.1f}% "
            f"({repository_result.duration_ms:.2f}ms vs {state_manager_result.duration_ms:.2f}ms)"
        )
        print(
            f"  Memory improvement: {memory_improvement:+.1f}% "
            f"({repository_result.memory_mb:.2f}MB vs {state_manager_result.memory_mb:.2f}MB)"
        )
        print(
            f"  Throughput: {repository_result.keys_per_second:.0f} vs "
            f"{state_manager_result.keys_per_second:.0f} keys/sec\n"
        )

        # Interpretation
        if time_improvement > 50:
            print(f"✅ VALIDATED: {time_improvement:.0f}% faster matches 60-80% estimate")
        elif time_improvement > 30:
            print(f"⚠️  MODERATE: {time_improvement:.0f}% improvement (lower than 60-80% estimate)")
        else:
            print(f"❌ UNEXPECTED: Only {time_improvement:.0f}% improvement (below estimate)")


async def main():
    """Run benchmark suite."""
    # Test with different dataset sizes
    for num_keys in [100, 500, 1000]:
        suite = StateBenchmarkSuite(num_keys=num_keys)
        await suite.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
