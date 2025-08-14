"""
Intelligent Caching System

Provides advanced caching capabilities for optimization workflows:
- Multi-level caching (memory, disk, distributed)
- Intelligent cache eviction and replacement
- Content-aware caching with hash-based keys
- Performance monitoring and analytics
- Automatic cache warming and prefetching
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Import logger for error handling
logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache performance statistics"""

    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    avg_hit_time_ms: float = 0.0
    avg_miss_time_ms: float = 0.0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def efficiency_score(self) -> float:
        """Overall cache efficiency score (0-1)"""
        hit_component = self.hit_rate * 0.6
        size_component = min(1.0, self.size / max(1, self.max_size)) * 0.2
        speed_component = min(1.0, 10.0 / max(0.1, self.avg_hit_time_ms)) * 0.2
        return hit_component + size_component + speed_component


@dataclass
class CacheEntry:
    """Individual cache entry with metadata"""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    computation_time_ms: float = 0.0
    ttl: float | None = None

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)

    def update_access(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1


class IntelligentCache:
    """
    Multi-level intelligent caching system with advanced features.

    Features:
    - LRU eviction with intelligent scoring
    - Automatic memory management
    - Disk persistence for large objects
    - Content-aware hashing
    - Performance analytics
    - Thread-safe operations
    """

    def __init__(
        self,
        max_memory_mb: float = 512.0,
        max_disk_mb: float = 2048.0,
        cache_dir: Path | None = None,
        enable_disk_cache: bool = True,
        enable_analytics: bool = True,
        auto_optimize: bool = True,
    ) -> None:
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.max_disk_bytes = int(max_disk_mb * 1024 * 1024)
        self.enable_disk_cache = enable_disk_cache
        self.enable_analytics = enable_analytics
        self.auto_optimize = auto_optimize

        # Cache storage
        self._memory_cache: dict[str, CacheEntry] = {}
        self._disk_cache_keys: set = set()

        # Cache directory setup
        if cache_dir is None:
            cache_dir = Path.home() / ".gpt_trader" / "cache"
        self.cache_dir = Path(cache_dir)
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics and monitoring
        self.stats = CacheStats()
        self.stats.max_size = max(100, int(self.max_memory_bytes / (1024 * 50)))  # Estimate entries

        # Thread safety
        self._lock = threading.RLock()

        # Background optimization
        self._optimizer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-opt")
        self._last_optimization = time.time()

        # Logging
        self.logger = logging.getLogger(__name__)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with performance tracking"""
        start_time = time.time()

        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]

                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key)
                    self._record_miss(start_time)
                    return default

                # Update access and return
                entry.update_access()
                self._record_hit(start_time)
                return entry.value

            # Check disk cache
            if self.enable_disk_cache and key in self._disk_cache_keys:
                try:
                    disk_path = self._get_disk_path(key)
                    if disk_path.exists():
                        value = joblib.load(disk_path)

                        # Promote to memory cache if space available
                        self._try_promote_to_memory(key, value, start_time)
                        self._record_hit(start_time)
                        return value
                except Exception as e:
                    self.logger.warning(f"Disk cache read error for {key}: {e}")
                    self._disk_cache_keys.discard(key)

            # Cache miss
            self._record_miss(start_time)
            return default

    def put(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        computation_time_ms: float | None = None,
    ) -> bool:
        """Put value in cache with intelligent placement"""
        with self._lock:
            # Create entry
            now = time.time()
            size_bytes = self._estimate_size(value)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                size_bytes=size_bytes,
                computation_time_ms=computation_time_ms or 0.0,
                ttl=ttl,
            )

            # Decide placement strategy
            if self._should_use_memory_cache(entry):
                return self._put_memory(entry)
            elif self.enable_disk_cache:
                return self._put_disk(entry)
            else:
                return False

    def invalidate(self, key: str) -> bool:
        """Remove specific key from all cache levels"""
        with self._lock:
            removed = False

            # Remove from memory
            if key in self._memory_cache:
                self._remove_entry(key)
                removed = True

            # Remove from disk
            if key in self._disk_cache_keys:
                try:
                    disk_path = self._get_disk_path(key)
                    if disk_path.exists():
                        disk_path.unlink()
                    self._disk_cache_keys.discard(key)
                    removed = True
                except Exception as e:
                    self.logger.warning(f"Failed to remove disk cache {key}: {e}")

            return removed

    def clear(self, memory_only: bool = False) -> None:
        """Clear cache entirely or memory only"""
        with self._lock:
            # Clear memory cache
            self._memory_cache.clear()

            # Clear disk cache if requested
            if not memory_only and self.enable_disk_cache:
                try:
                    for cache_file in self.cache_dir.glob("cache_*.joblib"):
                        cache_file.unlink()
                    self._disk_cache_keys.clear()
                except Exception as e:
                    self.logger.warning(f"Failed to clear disk cache: {e}")

            # Reset stats
            self.stats.size = 0
            self.stats.memory_usage_mb = 0.0
            if not memory_only:
                self.stats.disk_usage_mb = 0.0

    def cached(
        self,
        ttl: float | None = None,
        key_func: Callable | None = None,
        enable_for_size: int | None = None,
    ):
        """Decorator for caching function results"""

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_function_key(func, args, kwargs)

                # Check cache first
                time.time()
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Compute result
                computation_start = time.time()
                result = func(*args, **kwargs)
                computation_time = (time.time() - computation_start) * 1000

                # Cache result if beneficial
                if enable_for_size is None or self._estimate_size(result) <= enable_for_size:
                    self.put(cache_key, result, ttl=ttl, computation_time_ms=computation_time)

                return result

            return wrapper

        return decorator

    def warm_cache(self, warm_func: Callable, *args, **kwargs) -> None:
        """Warm cache by precomputing common operations"""

        def _warm_worker() -> None:
            try:
                warm_func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Cache warming failed: {e}")

        self._optimizer_executor.submit(_warm_worker)

    def optimize(self) -> dict[str, Any]:
        """Optimize cache performance and return optimization results"""
        with self._lock:
            optimization_start = time.time()

            # Memory optimization
            memory_freed = self._optimize_memory()

            # Disk optimization
            disk_freed = 0
            if self.enable_disk_cache:
                disk_freed = self._optimize_disk()

            # Update statistics
            self._update_memory_usage()
            self._update_disk_usage()

            optimization_time = time.time() - optimization_start
            self._last_optimization = time.time()

            return {
                "memory_freed_mb": memory_freed / (1024 * 1024),
                "disk_freed_mb": disk_freed / (1024 * 1024),
                "optimization_time_ms": optimization_time * 1000,
                "current_hit_rate": self.stats.hit_rate,
                "memory_usage_mb": self.stats.memory_usage_mb,
                "disk_usage_mb": self.stats.disk_usage_mb,
            }

    def get_analytics(self) -> dict[str, Any]:
        """Get comprehensive cache analytics"""
        with self._lock:
            return {
                "stats": {
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "hit_rate": self.stats.hit_rate,
                    "efficiency_score": self.stats.efficiency_score,
                    "size": self.stats.size,
                    "max_size": self.stats.max_size,
                    "memory_usage_mb": self.stats.memory_usage_mb,
                    "disk_usage_mb": self.stats.disk_usage_mb,
                    "avg_hit_time_ms": self.stats.avg_hit_time_ms,
                    "avg_miss_time_ms": self.stats.avg_miss_time_ms,
                    "evictions": self.stats.evictions,
                },
                "top_entries": self._get_top_entries(10),
                "size_distribution": self._get_size_distribution(),
                "access_patterns": self._get_access_patterns(),
            }

    # Private methods

    def _generate_key(self, obj: Any) -> str:
        """Generate hash-based cache key for any object"""
        if isinstance(obj, str):
            return hashlib.sha256(obj.encode()).hexdigest()[:16]
        elif isinstance(obj, int | float | bool):
            return hashlib.sha256(str(obj).encode()).hexdigest()[:16]
        elif isinstance(obj, list | tuple):
            content = str([self._generate_key(item) for item in obj])
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        elif isinstance(obj, dict):
            content = str(sorted((k, self._generate_key(v)) for k, v in obj.items()))
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        elif isinstance(obj, pd.DataFrame):
            # Use shape, columns, and sample of data for DataFrame hash
            content = f"{obj.shape}_{list(obj.columns)}_{obj.iloc[::max(1, len(obj)//100)].values.tobytes()}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        elif isinstance(obj, np.ndarray):
            return hashlib.sha256(obj.tobytes()).hexdigest()[:16]
        else:
            try:
                return hashlib.sha256(str(obj).encode()).hexdigest()[:16]
            except (UnicodeEncodeError, MemoryError) as e:
                # String conversion or encoding failed - use type as fallback
                logger.debug(f"Key generation fallback for {type(obj)}: {e}")
                return hashlib.sha256(str(type(obj)).encode()).hexdigest()[:16]

    def _generate_function_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call"""
        func_name = f"{func.__module__}.{func.__name__}"
        args_key = self._generate_key(args)
        kwargs_key = self._generate_key(kwargs)
        return f"func_{func_name}_{args_key}_{kwargs_key}"

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        try:
            if isinstance(obj, str | bytes):
                return len(obj)
            elif isinstance(obj, int | float | bool):
                return 8
            elif isinstance(obj, list | tuple):
                return sum(self._estimate_size(item) for item in obj) + 64
            elif isinstance(obj, dict):
                return (
                    sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
                    + 128
                )
            elif isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                # Fallback estimate
                return 1024
        except (AttributeError, TypeError, MemoryError) as e:
            # Size estimation failed - use default size
            logger.debug(f"Size estimation failed: {e}")
            return 1024

    def _should_use_memory_cache(self, entry: CacheEntry) -> bool:
        """Decide if entry should go to memory cache"""
        # Always use memory for small entries
        if entry.size_bytes < 1024:
            return True

        # Consider computation time vs size ratio
        if entry.computation_time_ms > 100:  # Expensive computation
            return True

        # Check available memory
        current_memory = sum(e.size_bytes for e in self._memory_cache.values())
        return (current_memory + entry.size_bytes) < self.max_memory_bytes * 0.8

    def _put_memory(self, entry: CacheEntry) -> bool:
        """Put entry in memory cache with eviction if needed"""
        # Make space if needed
        while self._get_memory_usage() + entry.size_bytes > self.max_memory_bytes:
            if not self._evict_memory_entry():
                return False

        self._memory_cache[entry.key] = entry
        self.stats.size += 1
        return True

    def _put_disk(self, entry: CacheEntry) -> bool:
        """Put entry in disk cache"""
        try:
            disk_path = self._get_disk_path(entry.key)
            joblib.dump(entry.value, disk_path)

            self._disk_cache_keys.add(entry.key)
            return True
        except Exception as e:
            self.logger.warning(f"Disk cache write error for {entry.key}: {e}")
            return False

    def _get_disk_path(self, key: str) -> Path:
        """Get disk path for cache key"""
        return self.cache_dir / f"cache_{key}.joblib"

    def _evict_memory_entry(self) -> bool:
        """Evict least valuable entry from memory cache"""
        if not self._memory_cache:
            return False

        # Score entries for eviction (lower = more likely to evict)
        scores = {}
        now = time.time()

        for key, entry in self._memory_cache.items():
            # Time since last access (higher = more likely to evict)
            time_score = now - entry.last_accessed

            # Size factor (larger = more likely to evict)
            size_score = entry.size_bytes / 1024

            # Access frequency (lower = more likely to evict)
            freq_score = 1.0 / (entry.access_count + 1)

            # Computation time (higher = less likely to evict)
            comp_score = 1.0 / (entry.computation_time_ms + 1)

            scores[key] = time_score + size_score + freq_score - comp_score

        # Evict entry with highest score
        worst_key = max(scores.keys(), key=lambda k: scores[k])
        self._remove_entry(worst_key)
        self.stats.evictions += 1

        return True

    def _remove_entry(self, key: str) -> None:
        """Remove entry from memory cache"""
        if key in self._memory_cache:
            del self._memory_cache[key]
            self.stats.size -= 1

    def _try_promote_to_memory(self, key: str, value: Any, computation_start: float) -> None:
        """Try to promote disk cache entry to memory"""
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            size_bytes=self._estimate_size(value),
        )

        if self._should_use_memory_cache(entry):
            self._put_memory(entry)

    def _record_hit(self, start_time: float) -> None:
        """Record cache hit statistics"""
        hit_time = (time.time() - start_time) * 1000
        self.stats.hits += 1

        # Update running average
        total_hits = self.stats.hits
        self.stats.avg_hit_time_ms = (
            self.stats.avg_hit_time_ms * (total_hits - 1) + hit_time
        ) / total_hits

    def _record_miss(self, start_time: float) -> None:
        """Record cache miss statistics"""
        miss_time = (time.time() - start_time) * 1000
        self.stats.misses += 1

        # Update running average
        total_misses = self.stats.misses
        self.stats.avg_miss_time_ms = (
            self.stats.avg_miss_time_ms * (total_misses - 1) + miss_time
        ) / total_misses

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        return sum(entry.size_bytes for entry in self._memory_cache.values())

    def _update_memory_usage(self) -> None:
        """Update memory usage statistics"""
        self.stats.memory_usage_mb = self._get_memory_usage() / (1024 * 1024)

    def _update_disk_usage(self) -> None:
        """Update disk usage statistics"""
        if not self.enable_disk_cache:
            return

        total_size = 0
        try:
            for cache_file in self.cache_dir.glob("cache_*.joblib"):
                total_size += cache_file.stat().st_size
        except Exception:
            pass

        self.stats.disk_usage_mb = total_size / (1024 * 1024)

    def _optimize_memory(self) -> int:
        """Optimize memory cache and return bytes freed"""
        initial_usage = self._get_memory_usage()

        # Remove expired entries
        time.time()
        expired_keys = [key for key, entry in self._memory_cache.items() if entry.is_expired()]

        for key in expired_keys:
            self._remove_entry(key)

        return initial_usage - self._get_memory_usage()

    def _optimize_disk(self) -> int:
        """Optimize disk cache and return bytes freed"""
        if not self.enable_disk_cache:
            return 0

        initial_size = 0
        final_size = 0

        try:
            # Calculate initial size
            for cache_file in self.cache_dir.glob("cache_*.joblib"):
                initial_size += cache_file.stat().st_size

            # Remove old or oversized files
            files_by_age = []
            for cache_file in self.cache_dir.glob("cache_*.joblib"):
                try:
                    stat = cache_file.stat()
                    files_by_age.append((cache_file, stat.st_mtime, stat.st_size))
                except OSError as e:
                    # File stat failed - skip this file
                    logger.debug(f"Failed to stat cache file {cache_file}: {e}")
                    continue

            # Sort by age (oldest first)
            files_by_age.sort(key=lambda x: x[1])

            # Remove oldest files if over disk limit
            current_size = sum(size for _, _, size in files_by_age)

            for cache_file, _, size in files_by_age:
                if current_size <= self.max_disk_bytes:
                    break

                try:
                    cache_file.unlink()
                    current_size -= size

                    # Update disk cache keys
                    key = cache_file.stem.replace("cache_", "")
                    self._disk_cache_keys.discard(key)
                except OSError as e:
                    # File removal failed - continue with next file
                    logger.debug(f"Failed to remove cache file {cache_file}: {e}")
                    continue

            # Calculate final size
            for cache_file in self.cache_dir.glob("cache_*.joblib"):
                final_size += cache_file.stat().st_size

        except Exception as e:
            self.logger.warning(f"Disk optimization error: {e}")

        return initial_size - final_size

    def _get_top_entries(self, limit: int) -> list[dict[str, Any]]:
        """Get top cache entries by access count"""
        entries = list(self._memory_cache.values())
        entries.sort(key=lambda e: e.access_count, reverse=True)

        return [
            {
                "key": entry.key[:32] + "..." if len(entry.key) > 32 else entry.key,
                "access_count": entry.access_count,
                "size_kb": entry.size_bytes / 1024,
                "age_hours": (time.time() - entry.created_at) / 3600,
                "computation_time_ms": entry.computation_time_ms,
            }
            for entry in entries[:limit]
        ]

    def _get_size_distribution(self) -> dict[str, int]:
        """Get cache entry size distribution"""
        distribution = {"<1KB": 0, "1-10KB": 0, "10-100KB": 0, "100KB-1MB": 0, ">1MB": 0}

        for entry in self._memory_cache.values():
            size_kb = entry.size_bytes / 1024
            if size_kb < 1:
                distribution["<1KB"] += 1
            elif size_kb < 10:
                distribution["1-10KB"] += 1
            elif size_kb < 100:
                distribution["10-100KB"] += 1
            elif size_kb < 1024:
                distribution["100KB-1MB"] += 1
            else:
                distribution[">1MB"] += 1

        return distribution

    def _get_access_patterns(self) -> dict[str, float]:
        """Get cache access pattern statistics"""
        if not self._memory_cache:
            return {}

        access_counts = [entry.access_count for entry in self._memory_cache.values()]

        return {
            "avg_access_count": sum(access_counts) / len(access_counts),
            "max_access_count": max(access_counts),
            "min_access_count": min(access_counts),
            "entries_accessed_once": sum(1 for count in access_counts if count == 1),
            "entries_heavily_used": sum(1 for count in access_counts if count >= 10),
        }


# Global cache instance
_global_cache: IntelligentCache | None = None


def get_global_cache() -> IntelligentCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache


def cached(ttl: float | None = None, key_func: Callable | None = None):
    """Global cache decorator"""
    return get_global_cache().cached(ttl=ttl, key_func=key_func)


def benchmark_cache_performance():
    """Benchmark cache performance with realistic workloads"""
    print("🚀 Intelligent Cache Benchmark")
    print("=" * 50)

    cache = IntelligentCache(max_memory_mb=64, enable_disk_cache=True)

    # Generate test data
    test_datasets = []
    for i in range(10):
        np.random.seed(i)
        df = pd.DataFrame(
            {"a": np.random.randn(1000), "b": np.random.randn(1000), "c": np.random.randn(1000)}
        )
        test_datasets.append(df)

    # Test different operations
    @cache.cached(ttl=300)  # 5 minute TTL
    def expensive_computation(data_id: int, operation: str):
        """Simulate expensive computation"""
        df = test_datasets[data_id]
        time.sleep(0.01)  # Simulate computation time

        if operation == "mean":
            return df.mean()
        elif operation == "std":
            return df.std()
        elif operation == "corr":
            return df.corr()
        else:
            return df.describe()

    # Benchmark scenarios
    scenarios = [
        ("Cold cache", lambda: [expensive_computation(i % 10, "mean") for i in range(50)]),
        ("Warm cache", lambda: [expensive_computation(i % 5, "mean") for i in range(50)]),
        (
            "Mixed operations",
            lambda: [
                expensive_computation(i % 5, ["mean", "std", "corr"][i % 3]) for i in range(50)
            ],
        ),
        ("Heavy reuse", lambda: [expensive_computation(0, "mean") for _ in range(50)]),
    ]

    results = {}

    for scenario_name, scenario_func in scenarios:
        cache.clear()  # Reset for each scenario

        print(f"\n🧪 Testing: {scenario_name}")

        start_time = time.time()
        scenario_func()
        execution_time = time.time() - start_time

        analytics = cache.get_analytics()
        stats = analytics["stats"]

        results[scenario_name] = {
            "execution_time": execution_time,
            "hit_rate": stats["hit_rate"],
            "efficiency_score": stats["efficiency_score"],
            "memory_usage_mb": stats["memory_usage_mb"],
        }

        print(f"   ⏱️  Execution time: {execution_time:.3f}s")
        print(f"   🎯 Hit rate: {stats['hit_rate']:.1%}")
        print(f"   📊 Efficiency score: {stats['efficiency_score']:.3f}")
        print(f"   💾 Memory usage: {stats['memory_usage_mb']:.1f} MB")

    # Summary
    print("\n📊 CACHE BENCHMARK SUMMARY:")
    baseline_time = results["Cold cache"]["execution_time"]

    for scenario, metrics in results.items():
        speedup = baseline_time / metrics["execution_time"]
        print(f"   {scenario}: {speedup:.1f}x speedup, {metrics['hit_rate']:.1%} hit rate")

    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    benchmark_cache_performance()
