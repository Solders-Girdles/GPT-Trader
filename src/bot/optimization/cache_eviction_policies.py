"""
Advanced cache eviction policies for performance optimization.

Implements various cache eviction strategies:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- FIFO (First In First Out)
- Adaptive Replacement Cache (ARC)
- Clock Algorithm
- Segmented LRU (SLRU)
"""

import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    size: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    priority: float = 0.0

    def update_access(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""

    @abstractmethod
    def add(self, key: str, entry: CacheEntry) -> str | None:
        """Add entry and return key to evict if needed."""
        pass

    @abstractmethod
    def access(self, key: str) -> None:
        """Update metadata when entry is accessed."""
        pass

    @abstractmethod
    def evict(self) -> str | None:
        """Select and return key to evict."""
        pass

    @abstractmethod
    def remove(self, key: str) -> None:
        """Remove entry from policy tracking."""
        pass


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""

    def __init__(self, max_size: int):
        """Initialize LRU policy."""
        self.max_size = max_size
        self.access_order = OrderedDict()
        self.current_size = 0

    def add(self, key: str, entry: CacheEntry) -> str | None:
        """Add entry to LRU tracking."""
        evict_key = None

        if key in self.access_order:
            # Update existing
            self.access_order.move_to_end(key)
        else:
            # Add new
            if self.current_size >= self.max_size:
                evict_key = self.evict()

            self.access_order[key] = entry
            self.current_size += 1

        return evict_key

    def access(self, key: str) -> None:
        """Move accessed item to end (most recent)."""
        if key in self.access_order:
            self.access_order.move_to_end(key)

    def evict(self) -> str | None:
        """Evict least recently used item."""
        if self.access_order:
            key, _ = self.access_order.popitem(last=False)
            self.current_size -= 1
            return key
        return None

    def remove(self, key: str) -> None:
        """Remove entry from tracking."""
        if key in self.access_order:
            del self.access_order[key]
            self.current_size -= 1


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""

    def __init__(self, max_size: int):
        """Initialize LFU policy."""
        self.max_size = max_size
        self.freq_map = defaultdict(int)
        self.freq_lists = defaultdict(OrderedDict)
        self.key_freq = {}
        self.min_freq = 0
        self.current_size = 0

    def add(self, key: str, entry: CacheEntry) -> str | None:
        """Add entry to LFU tracking."""
        evict_key = None

        if key in self.key_freq:
            # Update existing
            self._update_freq(key)
        else:
            # Add new
            if self.current_size >= self.max_size:
                evict_key = self.evict()

            self.key_freq[key] = 1
            self.freq_lists[1][key] = entry
            self.min_freq = 1
            self.current_size += 1

        return evict_key

    def access(self, key: str) -> None:
        """Increment frequency for accessed item."""
        if key in self.key_freq:
            self._update_freq(key)

    def _update_freq(self, key: str) -> None:
        """Update frequency of a key."""
        freq = self.key_freq[key]
        entry = self.freq_lists[freq][key]

        # Remove from current frequency list
        del self.freq_lists[freq][key]
        if not self.freq_lists[freq] and freq == self.min_freq:
            self.min_freq += 1

        # Add to next frequency list
        self.key_freq[key] = freq + 1
        self.freq_lists[freq + 1][key] = entry

    def evict(self) -> str | None:
        """Evict least frequently used item."""
        if self.min_freq in self.freq_lists and self.freq_lists[self.min_freq]:
            key, _ = self.freq_lists[self.min_freq].popitem(last=False)
            del self.key_freq[key]
            self.current_size -= 1
            return key
        return None

    def remove(self, key: str) -> None:
        """Remove entry from tracking."""
        if key in self.key_freq:
            freq = self.key_freq[key]
            del self.freq_lists[freq][key]
            del self.key_freq[key]
            self.current_size -= 1


class FIFOPolicy(EvictionPolicy):
    """First In First Out eviction policy."""

    def __init__(self, max_size: int):
        """Initialize FIFO policy."""
        self.max_size = max_size
        self.queue = OrderedDict()
        self.current_size = 0

    def add(self, key: str, entry: CacheEntry) -> str | None:
        """Add entry to FIFO queue."""
        evict_key = None

        if key not in self.queue:
            if self.current_size >= self.max_size:
                evict_key = self.evict()

            self.queue[key] = entry
            self.current_size += 1

        return evict_key

    def access(self, key: str) -> None:
        """FIFO doesn't change on access."""
        pass

    def evict(self) -> str | None:
        """Evict oldest item."""
        if self.queue:
            key, _ = self.queue.popitem(last=False)
            self.current_size -= 1
            return key
        return None

    def remove(self, key: str) -> None:
        """Remove entry from queue."""
        if key in self.queue:
            del self.queue[key]
            self.current_size -= 1


class ARCPolicy(EvictionPolicy):
    """Adaptive Replacement Cache policy."""

    def __init__(self, max_size: int):
        """Initialize ARC policy."""
        self.max_size = max_size
        self.p = 0  # Target size for T1

        # Recent cache entries
        self.t1 = OrderedDict()  # Recent once
        self.t2 = OrderedDict()  # Recent twice

        # Ghost entries (only metadata)
        self.b1 = OrderedDict()  # Ghost for T1
        self.b2 = OrderedDict()  # Ghost for T2

    def add(self, key: str, entry: CacheEntry) -> str | None:
        """Add entry using ARC algorithm."""
        evict_key = None

        if key in self.t1 or key in self.t2:
            # Cache hit - move to T2
            if key in self.t1:
                del self.t1[key]
            else:
                self.t2.move_to_end(key)
            self.t2[key] = entry

        elif key in self.b1:
            # Ghost hit in B1 - increase P
            self.p = min(self.p + max(1, len(self.b2) // len(self.b1)), self.max_size)
            evict_key = self._replace(key, entry, from_b2=False)

        elif key in self.b2:
            # Ghost hit in B2 - decrease P
            self.p = max(self.p - max(1, len(self.b1) // len(self.b2)), 0)
            evict_key = self._replace(key, entry, from_b2=True)

        else:
            # Cache miss
            if len(self.t1) + len(self.t2) >= self.max_size:
                if len(self.t1) == self.max_size:
                    # T1 is full, evict from T1
                    evict_key, _ = self.t1.popitem(last=False)
                    self.b1[evict_key] = None
                else:
                    evict_key = self._replace(key, entry, from_b2=False)

            # Add to T1
            self.t1[key] = entry

        return evict_key

    def _replace(self, key: str, entry: CacheEntry, from_b2: bool) -> str | None:
        """Replace entry in cache."""
        evict_key = None

        if len(self.t1) > 0 and (len(self.t1) > self.p or (from_b2 and len(self.t1) == self.p)):
            # Evict from T1
            evict_key, _ = self.t1.popitem(last=False)
            self.b1[evict_key] = None
        else:
            # Evict from T2
            evict_key, _ = self.t2.popitem(last=False)
            self.b2[evict_key] = None

        return evict_key

    def access(self, key: str) -> None:
        """Update on access."""
        if key in self.t1:
            # Move from T1 to T2
            entry = self.t1[key]
            del self.t1[key]
            self.t2[key] = entry
        elif key in self.t2:
            # Move to end of T2
            self.t2.move_to_end(key)

    def evict(self) -> str | None:
        """Evict based on ARC policy."""
        if len(self.t1) > self.p:
            key, _ = self.t1.popitem(last=False)
            return key
        elif self.t2:
            key, _ = self.t2.popitem(last=False)
            return key
        elif self.t1:
            key, _ = self.t1.popitem(last=False)
            return key
        return None

    def remove(self, key: str) -> None:
        """Remove entry from all lists."""
        for cache_dict in [self.t1, self.t2, self.b1, self.b2]:
            if key in cache_dict:
                del cache_dict[key]


class ClockPolicy(EvictionPolicy):
    """Clock (Second Chance) eviction policy."""

    def __init__(self, max_size: int):
        """Initialize Clock policy."""
        self.max_size = max_size
        self.entries = {}
        self.reference_bits = {}
        self.clock_hand = 0
        self.keys = []

    def add(self, key: str, entry: CacheEntry) -> str | None:
        """Add entry to clock."""
        evict_key = None

        if key not in self.entries:
            if len(self.entries) >= self.max_size:
                evict_key = self.evict()

            self.entries[key] = entry
            self.reference_bits[key] = True
            self.keys.append(key)
        else:
            self.reference_bits[key] = True

        return evict_key

    def access(self, key: str) -> None:
        """Set reference bit on access."""
        if key in self.reference_bits:
            self.reference_bits[key] = True

    def evict(self) -> str | None:
        """Evict using clock algorithm."""
        if not self.keys:
            return None

        while True:
            if self.clock_hand >= len(self.keys):
                self.clock_hand = 0

            key = self.keys[self.clock_hand]

            if self.reference_bits.get(key, False):
                # Give second chance
                self.reference_bits[key] = False
                self.clock_hand = (self.clock_hand + 1) % len(self.keys)
            else:
                # Evict this entry
                del self.entries[key]
                del self.reference_bits[key]
                self.keys.pop(self.clock_hand)
                return key

    def remove(self, key: str) -> None:
        """Remove entry from clock."""
        if key in self.entries:
            del self.entries[key]
            del self.reference_bits[key]
            self.keys.remove(key)


class SLRUPolicy(EvictionPolicy):
    """Segmented LRU policy (protected and probationary segments)."""

    def __init__(self, max_size: int, protected_ratio: float = 0.8):
        """Initialize SLRU policy."""
        self.max_size = max_size
        self.protected_size = int(max_size * protected_ratio)
        self.probation_size = max_size - self.protected_size

        self.protected = OrderedDict()
        self.probation = OrderedDict()

    def add(self, key: str, entry: CacheEntry) -> str | None:
        """Add entry to probation segment."""
        evict_key = None

        if key in self.protected:
            # Already protected, move to end
            self.protected.move_to_end(key)
        elif key in self.probation:
            # Promote to protected
            self._promote(key)
        else:
            # New entry goes to probation
            if len(self.probation) >= self.probation_size:
                evict_key = self._evict_from_probation()

            self.probation[key] = entry

        return evict_key

    def _promote(self, key: str) -> None:
        """Promote entry from probation to protected."""
        entry = self.probation[key]
        del self.probation[key]

        # Make room in protected if needed
        if len(self.protected) >= self.protected_size:
            # Demote oldest protected to probation
            demote_key, demote_entry = self.protected.popitem(last=False)

            # Make room in probation if needed
            if len(self.probation) >= self.probation_size:
                self._evict_from_probation()

            self.probation[demote_key] = demote_entry

        self.protected[key] = entry

    def access(self, key: str) -> None:
        """Handle access in SLRU."""
        if key in self.protected:
            self.protected.move_to_end(key)
        elif key in self.probation:
            self._promote(key)

    def _evict_from_probation(self) -> str | None:
        """Evict from probation segment."""
        if self.probation:
            key, _ = self.probation.popitem(last=False)
            return key
        return None

    def evict(self) -> str | None:
        """Evict from probation first, then protected."""
        evict_key = self._evict_from_probation()

        if evict_key is None and self.protected:
            evict_key, _ = self.protected.popitem(last=False)

        return evict_key

    def remove(self, key: str) -> None:
        """Remove entry from segments."""
        if key in self.protected:
            del self.protected[key]
        elif key in self.probation:
            del self.probation[key]


class AdaptivePolicy(EvictionPolicy):
    """Adaptive policy that switches between strategies based on workload."""

    def __init__(self, max_size: int):
        """Initialize adaptive policy."""
        self.max_size = max_size

        # Multiple policies
        self.lru = LRUPolicy(max_size)
        self.lfu = LFUPolicy(max_size)
        self.current_policy = self.lru

        # Metrics for adaptation
        self.hit_count = 0
        self.miss_count = 0
        self.access_pattern = []
        self.pattern_window = 100

        # Thresholds
        self.recency_threshold = 0.7
        self.frequency_threshold = 0.3

    def add(self, key: str, entry: CacheEntry) -> str | None:
        """Add using current policy."""
        self._analyze_pattern()
        return self.current_policy.add(key, entry)

    def access(self, key: str) -> None:
        """Track access and use current policy."""
        self.hit_count += 1
        self.access_pattern.append(key)

        if len(self.access_pattern) > self.pattern_window:
            self.access_pattern.pop(0)

        self.current_policy.access(key)

    def _analyze_pattern(self) -> None:
        """Analyze access pattern and switch policy if needed."""
        if len(self.access_pattern) < self.pattern_window:
            return

        # Calculate recency vs frequency characteristics
        unique_accesses = len(set(self.access_pattern))
        recency_score = unique_accesses / len(self.access_pattern)

        # Switch policy based on pattern
        if recency_score > self.recency_threshold:
            self.current_policy = self.lru
        else:
            self.current_policy = self.lfu

    def evict(self) -> str | None:
        """Evict using current policy."""
        return self.current_policy.evict()

    def remove(self, key: str) -> None:
        """Remove from both policies."""
        self.lru.remove(key)
        self.lfu.remove(key)


def benchmark_eviction_policies(data_size: int = 1000,
                               cache_size: int = 100,
                               access_pattern: str = "zipf") -> dict[str, dict[str, float]]:
    """Benchmark different eviction policies."""
    import numpy as np

    # Generate access pattern
    if access_pattern == "zipf":
        # Zipfian distribution (some items much more popular)
        accesses = np.random.zipf(1.5, data_size * 10) % data_size
    elif access_pattern == "uniform":
        # Uniform distribution
        accesses = np.random.randint(0, data_size, data_size * 10)
    elif access_pattern == "sequential":
        # Sequential with loops
        accesses = list(range(data_size)) * 10
    else:
        accesses = np.random.randint(0, data_size, data_size * 10)

    policies = {
        "LRU": LRUPolicy(cache_size),
        "LFU": LFUPolicy(cache_size),
        "FIFO": FIFOPolicy(cache_size),
        "ARC": ARCPolicy(cache_size),
        "Clock": ClockPolicy(cache_size),
        "SLRU": SLRUPolicy(cache_size),
        "Adaptive": AdaptivePolicy(cache_size),
    }

    results = {}

    for name, policy in policies.items():
        hits = 0
        misses = 0
        evictions = 0

        cache = {}

        start_time = time.time()

        for item_id in accesses:
            key = f"item_{item_id}"

            if key in cache:
                hits += 1
                policy.access(key)
            else:
                misses += 1
                entry = CacheEntry(
                    key=key,
                    value=f"value_{item_id}",
                    size=1,
                    created_at=time.time(),
                    last_accessed=time.time()
                )

                evict_key = policy.add(key, entry)
                if evict_key:
                    del cache[evict_key]
                    evictions += 1

                cache[key] = entry

        end_time = time.time()

        total_accesses = hits + misses
        hit_rate = hits / total_accesses if total_accesses > 0 else 0

        results[name] = {
            "hit_rate": hit_rate,
            "hits": hits,
            "misses": misses,
            "evictions": evictions,
            "time": end_time - start_time,
        }

    return results


if __name__ == "__main__":
    print("Benchmarking Cache Eviction Policies")
    print("=" * 50)

    for pattern in ["zipf", "uniform", "sequential"]:
        print(f"\nAccess Pattern: {pattern.upper()}")
        print("-" * 30)

        results = benchmark_eviction_policies(
            data_size=1000,
            cache_size=100,
            access_pattern=pattern
        )

        # Sort by hit rate
        sorted_results = sorted(results.items(),
                              key=lambda x: x[1]["hit_rate"],
                              reverse=True)

        for policy, metrics in sorted_results:
            print(f"{policy:10} Hit Rate: {metrics['hit_rate']:.3f} "
                  f"Time: {metrics['time']:.4f}s")

    print("\n✓ Eviction policy benchmarks complete")
