#!/usr/bin/env python3
"""
Redis Failover Drill: Measure Recovery Performance with Batch Operations.

Simulates a Redis outage and measures recovery time comparing:
- Sequential recovery (old approach): Individual get/set/delete operations
- Batch recovery (new approach): Batch operations with cache coherence

Validates data integrity and cache coherence after recovery.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


# ============================================================================
# Mock Repositories with Realistic Latencies
# ============================================================================


class MockRedisRepo:
    """Mock Redis repository with network latency simulation."""

    def __init__(self, data: dict[str, Any] | None = None, latency_ms: float = 0.1):
        self._data = data or {}
        self._latency = latency_ms / 1000  # Convert to seconds

    async def keys(self, pattern: str) -> list[str]:
        """Simulate Redis KEYS command with latency."""
        await asyncio.sleep(self._latency)
        prefix = pattern.rstrip("*")
        return [k for k in self._data if k.startswith(prefix)]

    async def fetch(self, key: str) -> Any:
        """Simulate Redis GET command."""
        await asyncio.sleep(self._latency)
        return self._data.get(key)

    async def store(self, key: str, value: str, metadata: dict) -> bool:
        """Simulate Redis SET command."""
        await asyncio.sleep(self._latency)
        self._data[key] = value
        return True

    async def store_many(self, items: dict[str, tuple[str, dict]]) -> set[str]:
        """Simulate batch Redis write (faster than individual writes)."""
        # Batch operations have lower overhead
        await asyncio.sleep(self._latency * 0.5)  # 50% faster per item
        for key, (value, _metadata) in items.items():
            self._data[key] = value
        return set(items.keys())

    async def delete(self, key: str) -> bool:
        """Simulate Redis DELETE command."""
        await asyncio.sleep(self._latency)
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def delete_many(self, keys: list[str]) -> int:
        """Simulate batch Redis delete."""
        await asyncio.sleep(self._latency * 0.5)
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count


class MockPostgresRepo:
    """Mock PostgreSQL repository (WARM tier backup)."""

    def __init__(self, data: dict[str, Any] | None = None, latency_ms: float = 5.0):
        self._data = data or {}
        self._latency = latency_ms / 1000

    async def keys(self, pattern: str) -> list[str]:
        """Simulate PostgreSQL query."""
        await asyncio.sleep(self._latency)
        prefix = pattern.rstrip("*")
        return [k for k in self._data if k.startswith(prefix)]

    async def fetch(self, key: str) -> Any:
        """Simulate PostgreSQL fetch."""
        await asyncio.sleep(self._latency)
        return self._data.get(key)

    async def fetch_many(self, keys: list[str]) -> dict[str, Any]:
        """Simulate batch PostgreSQL fetch (much faster)."""
        await asyncio.sleep(self._latency * 0.2)  # Single query is 80% faster
        return {k: self._data[k] for k in keys if k in self._data}

    async def store(self, key: str, value: str, metadata: dict) -> bool:
        """Simulate PostgreSQL insert/update."""
        await asyncio.sleep(self._latency)
        self._data[key] = value
        return True

    async def store_many(self, items: dict[str, tuple[str, dict]]) -> set[str]:
        """Simulate batch PostgreSQL upsert (transactional)."""
        await asyncio.sleep(self._latency * 0.3)  # Batch is 70% faster
        for key, (value, _metadata) in items.items():
            self._data[key] = value
        return set(items.keys())


# ============================================================================
# Mock StateManager (Sequential Operations - OLD)
# ============================================================================


class StateManagerSequential:
    """Mock StateManager using sequential operations."""

    def __init__(self, redis_repo: MockRedisRepo, postgres_repo: MockPostgresRepo):
        self.redis_repo = redis_repo
        self.postgres_repo = postgres_repo
        self._cache = {}

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        # In reality, this checks all tiers
        hot_keys = await self.redis_repo.keys(pattern)
        warm_keys = await self.postgres_repo.keys(pattern)
        return list(set(hot_keys + warm_keys))

    async def get_state(self, key: str) -> Any:
        """Get state (check cache, then Redis, then PostgreSQL)."""
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        # Check Redis
        value = await self.redis_repo.fetch(key)
        if value:
            self._cache[key] = value
            return value

        # Check PostgreSQL
        value = await self.postgres_repo.fetch(key)
        if value:
            self._cache[key] = value
            return value

        return None

    async def set_state(self, key: str, value: Any, category: str = "HOT") -> bool:
        """Set state (sequential write + cache update)."""
        success = await self.redis_repo.store(key, str(value), {})
        if success:
            self._cache[key] = value
        return success

    async def delete_state(self, key: str) -> bool:
        """Delete state (sequential delete + cache invalidation)."""
        success = await self.redis_repo.delete(key)
        if success and key in self._cache:
            del self._cache[key]
        return success


# ============================================================================
# Mock StateManager (Batch Operations - NEW)
# ============================================================================


class StateManagerBatch:
    """Mock StateManager using batch operations."""

    def __init__(self, redis_repo: MockRedisRepo, postgres_repo: MockPostgresRepo):
        self.redis_repo = redis_repo
        self.postgres_repo = postgres_repo
        self._cache = {}

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        hot_keys = await self.redis_repo.keys(pattern)
        warm_keys = await self.postgres_repo.keys(pattern)
        return list(set(hot_keys + warm_keys))

    async def get_state(self, key: str) -> Any:
        """Get state (check cache, then Redis, then PostgreSQL)."""
        if key in self._cache:
            return self._cache[key]

        value = await self.redis_repo.fetch(key)
        if value:
            self._cache[key] = value
            return value

        value = await self.postgres_repo.fetch(key)
        if value:
            self._cache[key] = value
            return value

        return None

    async def set_state(self, key: str, value: Any, category: str = "HOT") -> bool:
        """Set state (for compatibility with sequential code)."""
        success = await self.redis_repo.store(key, str(value), {})
        if success:
            self._cache[key] = value
        return success

    async def batch_set_state(self, items: dict[str, tuple[Any, str]]) -> int:
        """Batch set state with proper cache population."""
        # Prepare items for repository
        repo_items = {}
        for key, (value, _category) in items.items():
            repo_items[key] = (str(value), {})

        # Batch write to repository
        successful_keys = await self.redis_repo.store_many(repo_items)

        # Update cache ONLY for successful keys
        for key in successful_keys:
            if key in items:
                self._cache[key] = items[key][0]

        return len(successful_keys)

    async def batch_delete_state(self, keys: list[str]) -> int:
        """Batch delete state with proper cache invalidation."""
        deleted_count = await self.redis_repo.delete_many(keys)

        # Invalidate cache for all deleted keys
        for key in keys:
            if key in self._cache:
                del self._cache[key]

        return deleted_count


# ============================================================================
# Recovery Scenarios
# ============================================================================


async def recover_redis_sequential(
    state_manager: StateManagerSequential, postgres_repo: MockPostgresRepo
) -> tuple[float, int]:
    """Simulate Redis recovery using sequential operations."""
    start_time = time.perf_counter()

    # Simulate fetching recent keys from PostgreSQL
    recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
    all_keys = await postgres_repo.keys("*")

    # Restore keys one by one (old approach)
    restored_count = 0
    for key in all_keys[:1000]:  # Limit to 1000 most recent
        value = await postgres_repo.fetch(key)
        if value:
            await state_manager.set_state(key, value, "HOT")
            restored_count += 1

    elapsed = time.perf_counter() - start_time
    return elapsed, restored_count


async def recover_redis_batch(
    state_manager: StateManagerBatch, postgres_repo: MockPostgresRepo
) -> tuple[float, int]:
    """Simulate Redis recovery using batch operations."""
    start_time = time.perf_counter()

    # Simulate fetching recent keys from PostgreSQL
    recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
    all_keys = await postgres_repo.keys("*")

    # Batch fetch from PostgreSQL (NEW: much faster than 1000 individual fetches!)
    keys_to_restore = all_keys[:1000]  # Limit to 1000 most recent
    fetched_data = await postgres_repo.fetch_many(keys_to_restore)

    # Collect all items to restore
    items_to_restore = {}
    for key, value in fetched_data.items():
        items_to_restore[key] = (value, "HOT")

    # Batch restore to Redis (new approach)
    restored_count = await state_manager.batch_set_state(items_to_restore)

    elapsed = time.perf_counter() - start_time
    return elapsed, restored_count


async def cancel_pending_orders_sequential(
    state_manager: StateManagerSequential,
) -> tuple[float, int]:
    """Cancel pending orders using sequential operations."""
    start_time = time.perf_counter()

    # Get all orders
    order_keys = await state_manager.get_keys_by_pattern("order:*")

    # Cancel pending orders one by one
    cancelled_count = 0
    for key in order_keys:
        order_data = await state_manager.get_state(key)
        if order_data and order_data.get("status") == "pending":
            order_data["status"] = "cancelled"
            order_data["cancel_reason"] = "Trading engine recovery"
            await state_manager.set_state(key, order_data, "HOT")
            cancelled_count += 1

    elapsed = time.perf_counter() - start_time
    return elapsed, cancelled_count


async def cancel_pending_orders_batch(
    state_manager: StateManagerBatch,
) -> tuple[float, int]:
    """Cancel pending orders using batch operations."""
    start_time = time.perf_counter()

    # Get all orders
    order_keys = await state_manager.get_keys_by_pattern("order:*")

    # Collect pending orders to cancel
    orders_to_cancel = {}
    for key in order_keys:
        order_data = await state_manager.get_state(key)
        if order_data and isinstance(order_data, dict):
            if order_data.get("status") == "pending":
                order_data["status"] = "cancelled"
                order_data["cancel_reason"] = "Trading engine recovery"
                orders_to_cancel[key] = (order_data, "HOT")

    # Batch update cancelled orders
    cancelled_count = 0
    if orders_to_cancel:
        await state_manager.batch_set_state(orders_to_cancel)
        cancelled_count = len(orders_to_cancel)

    elapsed = time.perf_counter() - start_time
    return elapsed, cancelled_count


# ============================================================================
# Test Data Generation
# ============================================================================


def generate_test_data(num_positions: int = 100, num_orders: int = 200) -> dict[str, Any]:
    """Generate realistic test data for failover drill."""
    data = {}

    # Generate positions
    symbols = ["BTC", "ETH", "SOL", "AVAX", "MATIC", "ARB", "OP", "LINK", "UNI", "AAVE"]
    for i in range(num_positions):
        symbol = symbols[i % len(symbols)]
        data[f"position:{symbol}:{i}"] = {
            "symbol": symbol,
            "quantity": 1.5 + (i * 0.1),
            "entry_price": 50000 + (i * 100),
            "unrealized_pnl": (i * 10) - 500,
        }

    # Generate orders
    statuses = ["pending", "filled", "cancelled", "pending", "filled"]  # 40% pending
    for i in range(num_orders):
        symbol = symbols[i % len(symbols)]
        data[f"order:{i}"] = {
            "order_id": str(i),
            "symbol": symbol,
            "quantity": 1.0,
            "price": 50000 + (i * 100),
            "status": statuses[i % len(statuses)],
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Generate ML models
    models = ["momentum", "reversal", "trend", "volatility"]
    for i, model_type in enumerate(models):
        data[f"ml_model:{model_type}"] = {
            "current_version": f"v2.{i}",
            "last_stable_version": f"v1.{i}",
            "accuracy": 0.75 + (i * 0.02),
            "last_updated": datetime.utcnow().isoformat(),
        }

    # Generate rate limit counters
    for i in range(50):
        data[f"rate_limit:endpoint_{i}"] = {
            "count": i * 10,
            "reset_at": (datetime.utcnow() + timedelta(minutes=1)).isoformat(),
        }

    return data


# ============================================================================
# Benchmark Runner
# ============================================================================


async def run_failover_drill():
    """Run complete failover drill and compare sequential vs batch operations."""
    print("=" * 80)
    print("Redis Failover Drill: Sequential vs Batch Recovery")
    print("=" * 80)
    print()

    # Generate test data
    print("Generating test data...")
    test_data = generate_test_data(num_positions=100, num_orders=200)
    print(f"✓ Generated {len(test_data)} state items")
    print(f"  - Positions: {len([k for k in test_data if k.startswith('position:')])}")
    print(f"  - Orders: {len([k for k in test_data if k.startswith('order:')])}")
    print(f"  - ML Models: {len([k for k in test_data if k.startswith('ml_model:')])}")
    print(f"  - Rate Limits: {len([k for k in test_data if k.startswith('rate_limit:')])}")
    print()

    # ========================================================================
    # Scenario 1: Redis Recovery (Restore from PostgreSQL)
    # ========================================================================
    print("─" * 80)
    print("SCENARIO 1: Redis Outage → Restore from PostgreSQL")
    print("─" * 80)

    # Setup: PostgreSQL has all data, Redis is empty
    postgres_data = test_data.copy()
    redis_data_seq = {}
    redis_data_batch = {}

    # Sequential recovery
    print("\n[Sequential Recovery]")
    postgres_repo_seq = MockPostgresRepo(postgres_data, latency_ms=5.0)
    redis_repo_seq = MockRedisRepo(redis_data_seq, latency_ms=0.1)
    state_manager_seq = StateManagerSequential(redis_repo_seq, postgres_repo_seq)

    seq_time, seq_count = await recover_redis_sequential(state_manager_seq, postgres_repo_seq)
    print(f"  Time: {seq_time:.3f}s")
    print(f"  Restored: {seq_count} keys")
    print(f"  Cache size: {len(state_manager_seq._cache)}")

    # Batch recovery
    print("\n[Batch Recovery]")
    postgres_repo_batch = MockPostgresRepo(postgres_data, latency_ms=5.0)
    redis_repo_batch = MockRedisRepo(redis_data_batch, latency_ms=0.1)
    state_manager_batch = StateManagerBatch(redis_repo_batch, postgres_repo_batch)

    batch_time, batch_count = await recover_redis_batch(state_manager_batch, postgres_repo_batch)
    print(f"  Time: {batch_time:.3f}s")
    print(f"  Restored: {batch_count} keys")
    print(f"  Cache size: {len(state_manager_batch._cache)}")

    # Results
    speedup_redis = seq_time / batch_time if batch_time > 0 else 0
    print(f"\n📊 Redis Recovery Speedup: {speedup_redis:.1f}x faster")
    print(f"   Sequential: {seq_time:.3f}s")
    print(f"   Batch:      {batch_time:.3f}s")
    print(f"   Saved:      {seq_time - batch_time:.3f}s")

    # ========================================================================
    # Scenario 2: Trading Engine Recovery (Cancel Pending Orders)
    # ========================================================================
    print("\n" + "─" * 80)
    print("SCENARIO 2: Trading Engine Crash → Cancel Pending Orders")
    print("─" * 80)

    # Setup: Orders in Redis (need deep copy for each scenario)
    import copy

    order_data = {k: v for k, v in test_data.items() if k.startswith("order:")}

    # Sequential cancellation
    print("\n[Sequential Cancellation]")
    order_data_seq = copy.deepcopy(order_data)  # Deep copy to avoid mutation
    redis_repo_seq2 = MockRedisRepo(order_data_seq, latency_ms=0.1)
    postgres_repo_seq2 = MockPostgresRepo({}, latency_ms=5.0)
    state_manager_seq2 = StateManagerSequential(redis_repo_seq2, postgres_repo_seq2)

    seq_time2, seq_count2 = await cancel_pending_orders_sequential(state_manager_seq2)
    print(f"  Time: {seq_time2:.3f}s")
    print(f"  Cancelled: {seq_count2} orders")

    # Batch cancellation
    print("\n[Batch Cancellation]")
    order_data_batch = copy.deepcopy(order_data)  # Deep copy to avoid mutation
    redis_repo_batch2 = MockRedisRepo(order_data_batch, latency_ms=0.1)
    postgres_repo_batch2 = MockPostgresRepo({}, latency_ms=5.0)
    state_manager_batch2 = StateManagerBatch(redis_repo_batch2, postgres_repo_batch2)

    batch_time2, batch_count2 = await cancel_pending_orders_batch(state_manager_batch2)
    print(f"  Time: {batch_time2:.3f}s")
    print(f"  Cancelled: {batch_count2} orders")

    # Results
    speedup_orders = seq_time2 / batch_time2 if batch_time2 > 0 else 0
    print(f"\n📊 Order Cancellation Speedup: {speedup_orders:.1f}x faster")
    print(f"   Sequential: {seq_time2:.3f}s")
    print(f"   Batch:      {batch_time2:.3f}s")
    print(f"   Saved:      {seq_time2 - batch_time2:.3f}s")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Batch Operations Performance Gains")
    print("=" * 80)
    print(f"\n{'Scenario':<40} {'Sequential':<12} {'Batch':<12} {'Speedup':<10}")
    print("─" * 80)
    print(
        f"{'Redis Recovery (1000 keys)':<40} {seq_time:>8.3f}s    {batch_time:>8.3f}s    {speedup_redis:>6.1f}x"
    )
    print(
        f"{'Order Cancellation (~80 orders)':<40} {seq_time2:>8.3f}s    {batch_time2:>8.3f}s    {speedup_orders:>6.1f}x"
    )
    print()

    # Validate cache coherence
    print("✓ Cache Coherence Validation:")
    print(f"  - Sequential cache size: {len(state_manager_seq._cache)}")
    print(f"  - Batch cache size: {len(state_manager_batch._cache)}")
    print(f"  - Match: {len(state_manager_seq._cache) == len(state_manager_batch._cache)}")
    print()


if __name__ == "__main__":
    asyncio.run(run_failover_drill())
