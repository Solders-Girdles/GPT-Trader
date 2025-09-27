"""
Memory Usage Tests for Phase 5 Production Integration.

This module tests system memory usage patterns including:
- Memory usage patterns over time
- Memory efficiency under different loads
- Memory leak detection
- Memory allocation patterns
- Memory cleanup and garbage collection
- Memory optimization validation
"""

import asyncio
import gc
import os
import sys
import time
import tracemalloc
from typing import Any
from unittest.mock import Mock, patch

import psutil
import pytest
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.knowledge.strategy_knowledge_base import StrategyKnowledgeBase
from bot.live.production_orchestrator import (
    OrchestrationMode,
    OrchestratorConfig,
    ProductionOrchestrator,
)
from bot.live.strategy_selector import SelectionMethod
from bot.portfolio.optimizer import OptimizationMethod


class TestMemoryUsage:
    """Test system memory usage patterns and efficiency."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for memory testing."""
        return OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=10,  # Moderate intervals for memory testing
            risk_check_interval=5,
            performance_check_interval=8,
            max_strategies=5,
            min_strategy_confidence=0.6,
            selection_method=SelectionMethod.HYBRID,
            optimization_method=OptimizationMethod.SHARPE_MAXIMIZATION,
            max_position_weight=0.4,
            target_volatility=0.15,
            max_portfolio_var=0.02,
            max_drawdown=0.15,
            stop_loss_pct=0.05,
            min_sharpe_ratio=0.5,
            max_drawdown_threshold=0.15,
            enable_alerts=True,
            alert_cooldown_minutes=1,
        )

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for memory testing."""
        broker = Mock(spec=AlpacaPaperBroker)

        # Mock account
        account = Mock()
        account.equity = 100000.0
        account.cash = 50000.0
        account.buying_power = 50000.0
        broker.get_account.return_value = account

        # Mock positions
        positions = []
        for i, symbol in enumerate(
            ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        ):
            position = Mock()
            position.symbol = symbol
            position.qty = 100 + i * 50
            position.market_value = 15000.0 + i * 5000.0
            position.current_price = 150.0 + i * 10.0
            position.avg_entry_price = 145.0 + i * 8.0
            positions.append(position)

        broker.get_positions.return_value = positions
        return broker

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base for memory testing."""
        kb = Mock(spec=StrategyKnowledgeBase)
        kb.find_strategies.return_value = []
        return kb

    @pytest.fixture
    def orchestrator(self, orchestrator_config, mock_broker, mock_knowledge_base):
        """Create a production orchestrator for memory testing."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

        # Patch the data manager to avoid real data fetching
        with patch("bot.live.production_orchestrator.LiveDataManager"):
            orchestrator = ProductionOrchestrator(
                config=orchestrator_config,
                broker=mock_broker,
                knowledge_base=mock_knowledge_base,
                symbols=symbols,
            )
            return orchestrator

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def get_memory_info(self) -> dict[str, float]:
        """Get detailed memory information."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }

    def start_memory_tracking(self):
        """Start memory tracking with tracemalloc."""
        tracemalloc.start()

    def get_memory_snapshot(self) -> tracemalloc.Snapshot:
        """Get current memory snapshot."""
        return tracemalloc.take_snapshot()

    def compare_memory_snapshots(
        self, snapshot1: tracemalloc.Snapshot, snapshot2: tracemalloc.Snapshot
    ) -> dict[str, Any]:
        """Compare two memory snapshots."""
        stats = snapshot2.compare_to(snapshot1, "lineno")
        return {
            "top_stats": stats[:10],  # Top 10 memory differences
            "total_diff": sum(stat.size_diff for stat in stats),
            "count_diff": sum(stat.count_diff for stat in stats),
        }

    @pytest.mark.asyncio
    async def test_memory_usage_patterns_over_time(self, orchestrator):
        """Test memory usage patterns over time."""
        logger.info("Testing memory usage patterns over time")

        # Start memory tracking
        self.start_memory_tracking()
        initial_snapshot = self.get_memory_snapshot()

        # Record initial memory
        initial_memory = self.get_memory_usage()
        memory_readings = [initial_memory]
        timestamps = [time.time()]

        # Start the system
        await orchestrator.start()

        try:
            # Monitor memory usage over time
            for i in range(20):  # 20 readings over time
                # Execute some operations
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()

                # Record memory usage
                current_memory = self.get_memory_usage()
                memory_readings.append(current_memory)
                timestamps.append(time.time())

                # Small delay between readings
                await asyncio.sleep(1)

            # Take final snapshot
            final_snapshot = self.get_memory_snapshot()

            # Analyze memory patterns
            memory_increases = [mr - initial_memory for mr in memory_readings]
            max_increase = max(memory_increases)
            final_increase = memory_increases[-1]

            # Verify memory patterns are reasonable
            assert max_increase < 200.0, f"Maximum memory increase too high: {max_increase:.2f}MB"
            assert final_increase < 150.0, f"Final memory increase too high: {final_increase:.2f}MB"

            # Verify memory stabilizes (doesn't continuously grow)
            if len(memory_readings) > 5:
                recent_readings = memory_readings[-5:]
                memory_variance = max(recent_readings) - min(recent_readings)
                assert (
                    memory_variance < 50.0
                ), f"Memory not stabilizing: variance {memory_variance:.2f}MB"

            # Analyze memory snapshot comparison
            snapshot_comparison = self.compare_memory_snapshots(initial_snapshot, final_snapshot)

            # Verify no excessive memory allocation
            assert (
                snapshot_comparison["total_diff"] < 50 * 1024 * 1024
            ), f"Excessive memory allocation: {snapshot_comparison['total_diff'] / 1024 / 1024:.2f}MB"

            logger.info(
                f"Memory patterns - Max increase: {max_increase:.2f}MB, "
                f"Final increase: {final_increase:.2f}MB, "
                f"Total allocation: {snapshot_comparison['total_diff'] / 1024 / 1024:.2f}MB"
            )

        finally:
            await orchestrator.stop()
            tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_memory_efficiency_under_different_loads(self, orchestrator):
        """Test memory efficiency under different load conditions."""
        logger.info("Testing memory efficiency under different loads")

        # Start memory tracking
        self.start_memory_tracking()

        # Record baseline memory
        baseline_memory = self.get_memory_usage()
        baseline_snapshot = self.get_memory_snapshot()

        # Start the system
        await orchestrator.start()

        try:
            efficiency_metrics = []

            # Test different load levels
            for load_level in range(1, 6):
                # Clear operation history for clean measurement
                orchestrator.operation_history.clear()

                # Create load
                load_tasks = []
                for i in range(load_level * 5):
                    task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                    load_tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                    load_tasks.append(task)

                # Wait for load to complete
                await asyncio.gather(*load_tasks, return_exceptions=True)

                # Record memory usage
                current_memory = self.get_memory_usage()
                current_snapshot = self.get_memory_snapshot()

                # Calculate efficiency metrics
                memory_increase = current_memory - baseline_memory
                operations = len(orchestrator.get_operation_history())
                memory_per_operation = memory_increase / operations if operations > 0 else 0

                efficiency_metrics.append(
                    {
                        "load_level": load_level,
                        "memory_increase_mb": memory_increase,
                        "operations": operations,
                        "memory_per_operation_mb": memory_per_operation,
                    }
                )

                # Small delay between load levels
                await asyncio.sleep(2)

            # Analyze efficiency metrics
            memory_increases = [em["memory_increase_mb"] for em in efficiency_metrics]
            memory_per_ops = [em["memory_per_operation_mb"] for em in efficiency_metrics]

            # Verify memory efficiency
            assert all(
                mi < 100.0 for mi in memory_increases
            ), f"Memory increases too high: {memory_increases}"
            assert all(
                mpo < 1.0 for mpo in memory_per_ops if mpo > 0
            ), f"Memory per operation too high: {memory_per_ops}"

            # Verify efficiency doesn't degrade too much with load
            if len(memory_per_ops) > 1:
                # Guard against tiny baseline causing inflated ratios
                baseline = min(mpo for mpo in memory_per_ops if mpo > 0)
                efficiency_degradation = (
                    max(memory_per_ops) / baseline if baseline > 0 else float("inf")
                )
                if sys.platform == "darwin":
                    assert (
                        efficiency_degradation < 200.0
                    ), f"Memory efficiency degrades too much (macOS): {efficiency_degradation:.2f}x"
                else:
                    assert (
                        efficiency_degradation < 10.0
                    ), f"Memory efficiency degrades too much: {efficiency_degradation:.2f}x"

            logger.info(
                f"Memory efficiency - Max increase: {max(memory_increases):.2f}MB, "
                f"Max per operation: {max(memory_per_ops):.3f}MB"
            )

        finally:
            await orchestrator.stop()
            tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, orchestrator):
        """Test for memory leaks during operation."""
        logger.info("Testing memory leak detection")

        # Start memory tracking
        self.start_memory_tracking()

        # Record initial memory
        initial_memory = self.get_memory_usage()
        initial_snapshot = self.get_memory_snapshot()

        # Start the system
        await orchestrator.start()

        try:
            # Run operations that might cause memory leaks
            for cycle in range(10):  # 10 cycles
                # Execute operations
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()

                # Force garbage collection
                gc.collect()

                # Record memory after GC
                current_memory = self.get_memory_usage()
                current_snapshot = self.get_memory_snapshot()

                # Check for memory leaks
                memory_increase = current_memory - initial_memory

                # Verify no significant memory leaks
                assert (
                    memory_increase < 50.0
                ), f"Memory leak detected at cycle {cycle}: {memory_increase:.2f}MB"

                # Small delay between cycles
                await asyncio.sleep(1)

            # Final memory check
            final_memory = self.get_memory_usage()
            final_snapshot = self.get_memory_snapshot()

            # Analyze final memory state
            total_memory_increase = final_memory - initial_memory
            snapshot_comparison = self.compare_memory_snapshots(initial_snapshot, final_snapshot)

            # Verify no memory leaks
            assert (
                total_memory_increase < 100.0
            ), f"Total memory leak: {total_memory_increase:.2f}MB"
            assert (
                snapshot_comparison["total_diff"] < 100 * 1024 * 1024
            ), f"Excessive memory allocation: {snapshot_comparison['total_diff'] / 1024 / 1024:.2f}MB"

            logger.info(
                f"Memory leak detection - Total increase: {total_memory_increase:.2f}MB, "
                f"Total allocation: {snapshot_comparison['total_diff'] / 1024 / 1024:.2f}MB"
            )

        finally:
            await orchestrator.stop()
            tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_memory_allocation_patterns(self, orchestrator):
        """Test memory allocation patterns during operations."""
        logger.info("Testing memory allocation patterns")

        # Start memory tracking
        self.start_memory_tracking()

        # Start the system
        await orchestrator.start()

        try:
            allocation_patterns = []

            # Monitor memory allocation during different operations
            for i in range(5):
                # Take snapshot before operation
                pre_snapshot = self.get_memory_snapshot()

                # Execute operation
                await orchestrator._execute_strategy_selection_cycle()

                # Take snapshot after operation
                post_snapshot = self.get_memory_snapshot()

                # Analyze allocation pattern
                comparison = self.compare_memory_snapshots(pre_snapshot, post_snapshot)

                allocation_patterns.append(
                    {
                        "operation": "strategy_selection",
                        "cycle": i,
                        "total_diff": comparison["total_diff"],
                        "count_diff": comparison["count_diff"],
                        "top_allocations": comparison["top_stats"][:3],  # Top 3 allocations
                    }
                )

                # Small delay between operations
                await asyncio.sleep(1)

            # Analyze allocation patterns
            total_allocations = [ap["total_diff"] for ap in allocation_patterns]
            count_changes = [ap["count_diff"] for ap in allocation_patterns]

            # Verify allocation patterns are reasonable
            assert all(
                ta < 10 * 1024 * 1024 for ta in total_allocations
            ), f"Excessive allocations: {total_allocations}"

            # Verify allocation patterns are consistent
            if len(total_allocations) > 1:
                allocation_variance = max(total_allocations) - min(total_allocations)
                assert (
                    allocation_variance < 5 * 1024 * 1024
                ), f"Allocation patterns too variable: {allocation_variance / 1024 / 1024:.2f}MB"

            logger.info(
                f"Memory allocation patterns - Max allocation: {max(total_allocations) / 1024 / 1024:.2f}MB, "
                f"Average allocation: {sum(total_allocations) / len(total_allocations) / 1024 / 1024:.2f}MB"
            )

        finally:
            await orchestrator.stop()
            tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_memory_cleanup_and_garbage_collection(self, orchestrator):
        """Test memory cleanup and garbage collection effectiveness."""
        logger.info("Testing memory cleanup and garbage collection")

        # Start memory tracking
        self.start_memory_tracking()

        # Record initial memory
        initial_memory = self.get_memory_usage()
        initial_snapshot = self.get_memory_snapshot()

        # Start the system
        await orchestrator.start()

        try:
            # Create memory pressure
            for i in range(20):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()

            # Record memory before cleanup
            pre_cleanup_memory = self.get_memory_usage()
            pre_cleanup_snapshot = self.get_memory_snapshot()

            # Force garbage collection
            gc.collect()

            # Record memory after cleanup
            post_cleanup_memory = self.get_memory_usage()
            post_cleanup_snapshot = self.get_memory_snapshot()

            # Analyze cleanup effectiveness
            pre_cleanup_increase = pre_cleanup_memory - initial_memory
            post_cleanup_increase = post_cleanup_memory - initial_memory
            cleanup_effectiveness = pre_cleanup_increase - post_cleanup_increase

            # Verify garbage collection is effective (allow zero-change on macOS due to RSS granularity)
            if sys.platform == "darwin":
                assert (
                    cleanup_effectiveness >= 0.0
                ), f"Garbage collection had no effect: {cleanup_effectiveness:.2f}MB"
            else:
                assert (
                    cleanup_effectiveness > 0
                ), f"Garbage collection had no effect: {cleanup_effectiveness:.2f}MB"
            assert (
                post_cleanup_increase < 100.0
            ), f"Memory still too high after cleanup: {post_cleanup_increase:.2f}MB"

            # Analyze snapshot comparison
            pre_cleanup_comparison = self.compare_memory_snapshots(
                initial_snapshot, pre_cleanup_snapshot
            )
            post_cleanup_comparison = self.compare_memory_snapshots(
                initial_snapshot, post_cleanup_snapshot
            )

            # Verify cleanup reduces memory allocation
            assert (
                post_cleanup_comparison["total_diff"] < pre_cleanup_comparison["total_diff"]
            ), "Cleanup didn't reduce memory allocation"

            logger.info(
                f"Memory cleanup - Pre-cleanup increase: {pre_cleanup_increase:.2f}MB, "
                f"Post-cleanup increase: {post_cleanup_increase:.2f}MB, "
                f"Cleanup effectiveness: {cleanup_effectiveness:.2f}MB"
            )

        finally:
            await orchestrator.stop()
            tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_memory_optimization_validation(self, orchestrator):
        """Test memory optimization validation."""
        logger.info("Testing memory optimization validation")

        # Start memory tracking
        self.start_memory_tracking()

        # Record initial memory
        initial_memory = self.get_memory_usage()
        initial_snapshot = self.get_memory_snapshot()

        # Start the system
        await orchestrator.start()

        try:
            # Run optimized operations
            optimization_metrics = []

            for i in range(10):
                # Execute operations
                start_time = time.time()
                await orchestrator._execute_strategy_selection_cycle()
                operation_time = time.time() - start_time

                # Record memory usage
                current_memory = self.get_memory_usage()
                current_snapshot = self.get_memory_snapshot()

                # Calculate optimization metrics
                memory_increase = current_memory - initial_memory
                memory_per_second = memory_increase / (operation_time + 1)  # Avoid division by zero

                optimization_metrics.append(
                    {
                        "cycle": i,
                        "memory_increase_mb": memory_increase,
                        "operation_time_s": operation_time,
                        "memory_per_second_mb": memory_per_second,
                    }
                )

                # Force garbage collection periodically
                if i % 3 == 0:
                    gc.collect()

                # Small delay
                await asyncio.sleep(0.5)

            # Analyze optimization metrics
            memory_increases = [om["memory_increase_mb"] for om in optimization_metrics]
            operation_times = [om["operation_time_s"] for om in optimization_metrics]
            memory_per_second = [om["memory_per_second_mb"] for om in optimization_metrics]

            # Verify memory optimization
            assert all(
                mi < 50.0 for mi in memory_increases
            ), f"Memory increases too high: {memory_increases}"
            assert all(
                ot < 5.0 for ot in operation_times
            ), f"Operation times too high: {operation_times}"
            assert all(
                mps < 10.0 for mps in memory_per_second
            ), f"Memory per second too high: {memory_per_second}"

            # Verify optimization consistency
            if len(memory_increases) > 1:
                memory_stability = max(memory_increases) - min(memory_increases)
                assert (
                    memory_stability < 20.0
                ), f"Memory not stable: variance {memory_stability:.2f}MB"

            # Verify performance doesn't degrade
            if len(operation_times) > 1:
                time_degradation = max(operation_times) / min(operation_times)
                if sys.platform == "darwin":
                    assert (
                        time_degradation < 8.0
                    ), f"Performance degrades too much: {time_degradation:.2f}x"
                else:
                    assert (
                        time_degradation < 3.0
                    ), f"Performance degrades too much: {time_degradation:.2f}x"

            logger.info(
                f"Memory optimization - Max increase: {max(memory_increases):.2f}MB, "
                f"Max operation time: {max(operation_times):.3f}s, "
                f"Max memory per second: {max(memory_per_second):.2f}MB"
            )

        finally:
            await orchestrator.stop()
            tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_memory_fragmentation_analysis(self, orchestrator):
        """Test memory fragmentation analysis."""
        logger.info("Testing memory fragmentation analysis")

        # Start memory tracking
        self.start_memory_tracking()

        # Record initial memory info
        initial_memory_info = self.get_memory_info()

        # Start the system
        await orchestrator.start()

        try:
            fragmentation_metrics = []

            # Monitor memory fragmentation over time
            for i in range(15):
                # Execute operations
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()

                # Record memory info
                current_memory_info = self.get_memory_info()

                # Calculate fragmentation metrics
                rss_vms_ratio = (
                    current_memory_info["rss_mb"] / current_memory_info["vms_mb"]
                    if current_memory_info["vms_mb"] > 0
                    else 0
                )
                memory_efficiency = current_memory_info["rss_mb"] / (
                    current_memory_info["rss_mb"] + 1
                )  # Avoid division by zero

                fragmentation_metrics.append(
                    {
                        "cycle": i,
                        "rss_mb": current_memory_info["rss_mb"],
                        "vms_mb": current_memory_info["vms_mb"],
                        "rss_vms_ratio": rss_vms_ratio,
                        "memory_efficiency": memory_efficiency,
                    }
                )

                # Force garbage collection periodically
                if i % 5 == 0:
                    gc.collect()

                # Small delay
                await asyncio.sleep(1)

            # Analyze fragmentation metrics
            rss_values = [fm["rss_mb"] for fm in fragmentation_metrics]
            vms_values = [fm["vms_mb"] for fm in fragmentation_metrics]
            rss_vms_ratios = [fm["rss_vms_ratio"] for fm in fragmentation_metrics]

            # Verify memory fragmentation is reasonable
            assert all(rss < 500.0 for rss in rss_values), f"RSS too high: {rss_values}"
            # VMS limits and RSS/VMS ratio are not meaningful on macOS due to OS accounting
            if sys.platform != "darwin":
                assert all(vms < 1000.0 for vms in vms_values), f"VMS too high: {vms_values}"
                assert all(
                    ratio > 0.1 for ratio in rss_vms_ratios
                ), f"RSS/VMS ratio too low: {rss_vms_ratios}"

            # Verify fragmentation doesn't worsen significantly
            if len(rss_vms_ratios) > 1:
                fragmentation_degradation = min(rss_vms_ratios) / max(rss_vms_ratios)
                assert (
                    fragmentation_degradation > 0.5
                ), f"Memory fragmentation worsens too much: {fragmentation_degradation:.3f}"

            logger.info(
                f"Memory fragmentation - Max RSS: {max(rss_values):.2f}MB, "
                f"Max VMS: {max(vms_values):.2f}MB, "
                f"Min RSS/VMS ratio: {min(rss_vms_ratios):.3f}"
            )

        finally:
            await orchestrator.stop()
            tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, orchestrator):
        """Test system behavior under memory pressure."""
        logger.info("Testing memory pressure handling")

        # Start memory tracking
        self.start_memory_tracking()

        # Record initial memory
        initial_memory = self.get_memory_usage()

        # Start the system
        await orchestrator.start()

        try:
            # Create memory pressure
            pressure_tasks = []

            # Run many operations to create memory pressure
            for i in range(50):
                task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                pressure_tasks.append(task)
                task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                pressure_tasks.append(task)

            # Wait for pressure to build
            await asyncio.gather(*pressure_tasks, return_exceptions=True)

            # Record memory under pressure
            pressure_memory = self.get_memory_usage()
            pressure_increase = pressure_memory - initial_memory

            # Force garbage collection to handle pressure
            gc.collect()

            # Record memory after pressure handling
            post_pressure_memory = self.get_memory_usage()
            post_pressure_increase = post_pressure_memory - initial_memory

            # Verify memory pressure handling
            assert pressure_increase < 300.0, f"Memory pressure too high: {pressure_increase:.2f}MB"
            assert (
                post_pressure_increase < 150.0
            ), f"Memory not recovered from pressure: {post_pressure_increase:.2f}MB"

            # Verify system is still functional under pressure
            operations = orchestrator.get_operation_history()
            assert len(operations) > 0, "No operations completed under memory pressure"

            # Test system functionality after pressure
            post_pressure_start = time.time()
            await orchestrator._execute_strategy_selection_cycle()
            post_pressure_time = time.time() - post_pressure_start

            assert (
                post_pressure_time < 5.0
            ), f"System too slow after memory pressure: {post_pressure_time:.2f}s"

            logger.info(
                f"Memory pressure handling - Pressure increase: {pressure_increase:.2f}MB, "
                f"Post-pressure increase: {post_pressure_increase:.2f}MB, "
                f"Post-pressure time: {post_pressure_time:.2f}s"
            )

        finally:
            await orchestrator.stop()
            tracemalloc.stop()


# Add logger for test output
import logging

logger = logging.getLogger(__name__)
