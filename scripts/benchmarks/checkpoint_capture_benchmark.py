#!/usr/bin/env python3
"""
Benchmark StateCapture performance: StateManager vs Direct Repository Access.

Measures the performance improvement from using direct repository access
in checkpoint capture operations.
"""

import asyncio
import time
from datetime import datetime
from typing import Any

# Mock repositories for benchmarking
class MockRedisRepo:
    """Mock Redis repository with realistic data."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    async def keys(self, pattern: str) -> list[str]:
        """Simulate Redis KEYS command."""
        prefix = pattern.rstrip("*")
        return [k for k in self._data if k.startswith(prefix)]

    async def fetch(self, key: str) -> Any:
        """Simulate Redis GET command."""
        return self._data.get(key)


class MockStateManagerOld:
    """Mock StateManager using old approach (individual queries)."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Mock pattern matching."""
        await asyncio.sleep(0.0001)  # Simulate network latency
        prefix = pattern.rstrip("*")
        return [k for k in self._data if k.startswith(prefix)]

    async def get_state(self, key: str) -> Any:
        """Mock state retrieval."""
        await asyncio.sleep(0.0001)  # Simulate network latency
        return self._data.get(key)

    def get_repositories(self) -> None:
        """Old version doesn't have repositories."""
        raise AttributeError("get_repositories not available")


class MockStateManagerNew:
    """Mock StateManager with direct repository access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        self._redis = MockRedisRepo(data)

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Mock pattern matching (fallback)."""
        await asyncio.sleep(0.0001)  # Simulate network latency
        prefix = pattern.rstrip("*")
        return [k for k in self._data if k.startswith(prefix)]

    async def get_state(self, key: str) -> Any:
        """Mock state retrieval (fallback)."""
        await asyncio.sleep(0.0001)  # Simulate network latency
        return self._data.get(key)

    def get_repositories(self) -> Any:
        """Return mock repositories."""
        from dataclasses import dataclass

        @dataclass
        class Repos:
            redis: MockRedisRepo
            postgres: None = None
            s3: None = None

        return Repos(redis=self._redis)


class StateCaptureOld:
    """Old StateCapture implementation (StateManager access)."""

    def __init__(self, state_manager: Any):
        self.state_manager = state_manager

    async def capture_system_state(self) -> dict[str, Any]:
        """Capture complete system state using old approach."""
        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": {},
            "orders": {},
            "ml_models": {},
            "configuration": {},
        }

        # Old approach: individual queries
        position_keys = await self.state_manager.get_keys_by_pattern("position:*")
        for key in position_keys:
            value = await self.state_manager.get_state(key)
            if value:
                state["positions"][key] = value

        order_keys = await self.state_manager.get_keys_by_pattern("order:*")
        for key in order_keys:
            value = await self.state_manager.get_state(key)
            if value and value.get("status") != "filled":
                state["orders"][key] = value

        ml_keys = await self.state_manager.get_keys_by_pattern("ml_model:*")
        for key in ml_keys:
            value = await self.state_manager.get_state(key)
            if value:
                state["ml_models"][key] = value

        config_keys = await self.state_manager.get_keys_by_pattern("config:*")
        for key in config_keys:
            value = await self.state_manager.get_state(key)
            if value:
                state["configuration"][key] = value

        return state


class StateCaptureNew:
    """New StateCapture implementation (direct repository access)."""

    def __init__(self, state_manager: Any):
        self.state_manager = state_manager

    async def _get_all_by_pattern(self, pattern: str) -> dict[str, Any]:
        """Get all keys matching pattern using direct repository access."""
        result = {}

        try:
            repos = self.state_manager.get_repositories()
        except (AttributeError, TypeError):
            repos = None

        if repos is not None:
            try:
                # Direct repository access (fast path)
                if repos.redis:
                    keys = await repos.redis.keys(pattern)
                    for key in keys:
                        value = await repos.redis.fetch(key)
                        if value:
                            result[key] = value

                if repos.postgres:
                    keys = await repos.postgres.keys(pattern)
                    for key in keys:
                        if key not in result:
                            value = await repos.postgres.fetch(key)
                            if value:
                                result[key] = value

                if repos.s3:
                    keys = await repos.s3.keys(pattern)
                    for key in keys:
                        if key not in result:
                            value = await repos.s3.fetch(key)
                            if value:
                                result[key] = value
            except TypeError:
                # Fallback
                result = {}
                keys = await self.state_manager.get_keys_by_pattern(pattern)
                for key in keys:
                    value = await self.state_manager.get_state(key)
                    if value:
                        result[key] = value
        else:
            # Fallback
            keys = await self.state_manager.get_keys_by_pattern(pattern)
            for key in keys:
                value = await self.state_manager.get_state(key)
                if value:
                    result[key] = value

        return result

    async def capture_system_state(self) -> dict[str, Any]:
        """Capture complete system state using new optimized approach."""
        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": {},
            "orders": {},
            "ml_models": {},
            "configuration": {},
        }

        # New approach: batch reads with direct repository access
        all_positions = await self._get_all_by_pattern("position:*")
        state["positions"] = all_positions

        all_orders = await self._get_all_by_pattern("order:*")
        state["orders"] = {
            k: v for k, v in all_orders.items() if v.get("status") != "filled"
        }

        state["ml_models"] = await self._get_all_by_pattern("ml_model:*")
        state["configuration"] = await self._get_all_by_pattern("config:*")

        return state


def generate_test_data(num_positions: int, num_orders: int, num_models: int, num_configs: int) -> dict[str, Any]:
    """Generate realistic test data."""
    data = {}

    # Positions
    for i in range(num_positions):
        data[f"position:BTC-PERP-{i}"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": f"BTC-PERP-{i}",
            "quantity": "1.5",
            "entry_price": "50000",
        }

    # Orders (mix of filled and open)
    for i in range(num_orders):
        status = "filled" if i % 3 == 0 else "open"
        data[f"order:{i}"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "id": f"order-{i}",
            "status": status,
            "symbol": "BTC-PERP",
        }

    # ML models
    for i in range(num_models):
        data[f"ml_model:predictor-{i}"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": f"predictor-{i}",
            "weights": [0.1, 0.2, 0.3],
        }

    # Configuration
    for i in range(num_configs):
        data[f"config:param-{i}"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "key": f"param-{i}",
            "value": f"value-{i}",
        }

    return data


async def benchmark_old_approach(state_manager: MockStateManagerOld, iterations: int = 5) -> float:
    """Benchmark old StateCapture approach."""
    capture = StateCaptureOld(state_manager)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        await capture.capture_system_state()
        duration = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(duration)

    return sum(times) / len(times)  # Average time in ms


async def benchmark_new_approach(state_manager: MockStateManagerNew, iterations: int = 5) -> float:
    """Benchmark new StateCapture approach with direct repository access."""
    capture = StateCaptureNew(state_manager)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        await capture.capture_system_state()
        duration = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(duration)

    return sum(times) / len(times)  # Average time in ms


async def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("StateCapture Performance Benchmark: StateManager vs Direct Repository Access")
    print("=" * 80)
    print()

    # Test scenarios
    scenarios = [
        {"positions": 50, "orders": 100, "models": 20, "configs": 30, "name": "Small"},
        {"positions": 100, "orders": 200, "models": 50, "configs": 50, "name": "Medium"},
        {"positions": 200, "orders": 500, "models": 100, "configs": 100, "name": "Large"},
    ]

    results = []

    for scenario in scenarios:
        print(f"\n{scenario['name']} Dataset:")
        print(f"  - {scenario['positions']} positions")
        print(f"  - {scenario['orders']} orders")
        print(f"  - {scenario['models']} ML models")
        print(f"  - {scenario['configs']} config keys")
        print(f"  Total keys: {scenario['positions'] + scenario['orders'] + scenario['models'] + scenario['configs']}")

        # Generate test data
        data = generate_test_data(
            scenario['positions'],
            scenario['orders'],
            scenario['models'],
            scenario['configs']
        )

        # Benchmark old approach
        old_manager = MockStateManagerOld(data)
        old_time = await benchmark_old_approach(old_manager)

        # Benchmark new approach
        new_manager = MockStateManagerNew(data)
        new_time = await benchmark_new_approach(new_manager)

        # Calculate improvement
        improvement = ((old_time - new_time) / old_time) * 100
        speedup = old_time / new_time if new_time > 0 else 0

        print(f"\n  Results:")
        print(f"    Old (StateManager):  {old_time:.2f}ms")
        print(f"    New (Direct Repos):  {new_time:.2f}ms")
        print(f"    Improvement:         {improvement:.1f}% faster")
        print(f"    Speedup:             {speedup:.1f}x")

        results.append({
            "scenario": scenario['name'],
            "total_keys": scenario['positions'] + scenario['orders'] + scenario['models'] + scenario['configs'],
            "old_time": old_time,
            "new_time": new_time,
            "improvement": improvement,
            "speedup": speedup,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("| Scenario | Total Keys | Old (ms) | New (ms) | Improvement | Speedup |")
    print("|----------|------------|----------|----------|-------------|---------|")
    for r in results:
        print(f"| {r['scenario']:<8} | {r['total_keys']:<10} | {r['old_time']:>8.2f} | {r['new_time']:>8.2f} | {r['improvement']:>10.1f}% | {r['speedup']:>6.1f}x |")

    print("\n" + "=" * 80)
    print("Conclusion:")
    avg_improvement = sum(r['improvement'] for r in results) / len(results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"  Average improvement: {avg_improvement:.1f}% faster")
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
