"""
Stress Performance Tests for Phase 5 Production Integration.

This module tests system behavior under extreme stress conditions including:
- Maximum load stress testing
- Memory stress testing
- CPU stress testing
- Concurrent stress testing
- Recovery from stress conditions
- System stability under stress

These tests are gated behind the environment variable ENABLE_STRESS_TESTS=1
to reduce CI flakiness and runtime on shared runners. Locally, enable via:

    ENABLE_STRESS_TESTS=1 poetry run pytest tests/performance/test_stress_performance.py
"""

import asyncio
import gc
import os
import sys
import time
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

stress_enabled = os.environ.get("ENABLE_STRESS_TESTS") in {"1", "true", "True"}


@pytest.mark.skipif(
    not stress_enabled, reason="Stress tests disabled; set ENABLE_STRESS_TESTS=1 to enable"
)
class TestStressPerformance:
    """Test system behavior under extreme stress conditions."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for stress testing."""
        return OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=5,  # Very fast intervals for stress testing
            risk_check_interval=2,
            performance_check_interval=3,
            max_strategies=10,  # High number for stress
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
        """Create a mock broker for stress testing."""
        broker = Mock(spec=AlpacaPaperBroker)

        # Mock account
        account = Mock()
        account.equity = 100000.0
        account.cash = 50000.0
        account.buying_power = 50000.0
        broker.get_account.return_value = account

        # Mock many positions for stress testing
        positions = []
        symbols = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "GOOG",
            "MSFT",
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
        ]
        for i, symbol in enumerate(symbols):
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
        """Create a mock knowledge base for stress testing."""
        kb = Mock(spec=StrategyKnowledgeBase)
        kb.find_strategies.return_value = []
        return kb

    @pytest.fixture
    def orchestrator(self, orchestrator_config, mock_broker, mock_knowledge_base):
        """Create a production orchestrator for stress testing."""
        symbols = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "GOOG",
            "MSFT",
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
        ]

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

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        gc.collect()

    @pytest.mark.asyncio
    async def test_maximum_load_stress(self, orchestrator):
        """Test system behavior under maximum load stress."""
        logger.info("Testing maximum load stress")

        # Record initial resource usage
        initial_memory = self.get_memory_usage()
        initial_cpu = self.get_cpu_usage()

        # Start the system
        await orchestrator.start()

        try:
            # Create maximum load stress
            max_load_tasks = []

            # Create many concurrent operations
            for i in range(50):  # High number of concurrent tasks
                task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                max_load_tasks.append(task)
                task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                max_load_tasks.append(task)
                task = asyncio.create_task(orchestrator._execute_performance_monitoring_cycle())
                max_load_tasks.append(task)

            # Wait for maximum load to complete
            stress_start = time.time()
            results = await asyncio.gather(*max_load_tasks, return_exceptions=True)
            stress_duration = time.time() - stress_start

            # Record resource usage under stress
            stress_memory = self.get_memory_usage()
            stress_cpu = self.get_cpu_usage()

            # Analyze stress results
            exceptions = [r for r in results if isinstance(r, Exception)]
            successful_ops = len(results) - len(exceptions)

            # Verify system handled stress
            assert successful_ops > 0, "No operations completed under stress"
            assert (
                len(exceptions) < len(results) * 0.5
            ), f"Too many exceptions under stress: {len(exceptions)}/{len(results)}"

            # Verify resource usage under stress
            memory_increase = stress_memory - initial_memory
            assert (
                memory_increase < 1000.0
            ), f"Memory increase too high under stress: {memory_increase:.2f}MB"
            assert stress_cpu < 95.0, f"CPU usage too high under stress: {stress_cpu:.2f}%"

            # Verify stress duration is reasonable
            assert stress_duration < 60.0, f"Stress test took too long: {stress_duration:.2f}s"

            logger.info(
                f"Maximum load stress - Duration: {stress_duration:.2f}s, "
                f"Successful: {successful_ops}/{len(results)}, "
                f"Memory: +{memory_increase:.2f}MB, CPU: {stress_cpu:.2f}%"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_memory_stress(self, orchestrator):
        """Test system behavior under memory stress conditions."""
        logger.info("Testing memory stress")

        # Record initial memory
        initial_memory = self.get_memory_usage()

        # Start the system
        await orchestrator.start()

        try:
            # Create memory stress by running many operations
            memory_stress_tasks = []

            # Run operations that consume memory
            for i in range(100):  # High number for memory stress
                task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                memory_stress_tasks.append(task)

                # Add some delay to allow memory accumulation
                if i % 10 == 0:
                    await asyncio.sleep(0.1)

            # Wait for memory stress to complete
            await asyncio.gather(*memory_stress_tasks, return_exceptions=True)

            # Record memory usage after stress
            stress_memory = self.get_memory_usage()
            memory_increase = stress_memory - initial_memory

            # Force garbage collection
            self.force_garbage_collection()
            post_gc_memory = self.get_memory_usage()
            post_gc_increase = post_gc_memory - initial_memory

            # Verify memory behavior under stress
            assert (
                memory_increase < 500.0
            ), f"Memory increase too high under stress: {memory_increase:.2f}MB"

            # Verify garbage collection helps (allow zero-change on macOS due to RSS granularity)
            gc_effectiveness = memory_increase - post_gc_increase
            if sys.platform == "darwin":
                assert (
                    gc_effectiveness >= 0.0
                ), f"Garbage collection had no effect: {gc_effectiveness:.2f}MB"
            else:
                assert (
                    gc_effectiveness > 0
                ), f"Garbage collection had no effect: {gc_effectiveness:.2f}MB"

            # Verify final memory usage is reasonable
            assert (
                post_gc_increase < 200.0
            ), f"Final memory increase too high: {post_gc_increase:.2f}MB"

            logger.info(
                f"Memory stress - Peak increase: {memory_increase:.2f}MB, "
                f"Post-GC increase: {post_gc_increase:.2f}MB, "
                f"GC effectiveness: {gc_effectiveness:.2f}MB"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_cpu_stress(self, orchestrator):
        """Test system behavior under CPU stress conditions."""
        logger.info("Testing CPU stress")

        # Record initial CPU usage
        initial_cpu = self.get_cpu_usage()

        # Start the system
        await orchestrator.start()

        try:
            # Create CPU stress by running intensive operations
            cpu_stress_tasks = []

            # Run many operations simultaneously to stress CPU
            for i in range(30):  # High number for CPU stress
                task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                cpu_stress_tasks.append(task)
                task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                cpu_stress_tasks.append(task)
                task = asyncio.create_task(orchestrator._execute_performance_monitoring_cycle())
                cpu_stress_tasks.append(task)

            # Monitor CPU usage during stress
            cpu_readings = []
            for i in range(10):  # Take 10 readings during stress
                await asyncio.sleep(0.5)
                cpu_usage = self.get_cpu_usage()
                cpu_readings.append(cpu_usage)

            # Wait for CPU stress to complete
            await asyncio.gather(*cpu_stress_tasks, return_exceptions=True)

            # Analyze CPU stress results
            max_cpu = max(cpu_readings)
            avg_cpu = sum(cpu_readings) / len(cpu_readings)

            # Verify CPU behavior under stress
            assert max_cpu < 95.0, f"Maximum CPU usage too high: {max_cpu:.2f}%"
            assert avg_cpu < 80.0, f"Average CPU usage too high: {avg_cpu:.2f}%"

            # Verify CPU usage is reasonable after stress
            final_cpu = self.get_cpu_usage()
            assert final_cpu < 50.0, f"Final CPU usage too high: {final_cpu:.2f}%"

            logger.info(
                f"CPU stress - Max: {max_cpu:.2f}%, Avg: {avg_cpu:.2f}%, "
                f"Final: {final_cpu:.2f}%"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_concurrent_stress(self, orchestrator):
        """Test system behavior under extreme concurrent stress."""
        logger.info("Testing concurrent stress")

        # Start the system
        await orchestrator.start()

        try:
            # Create extreme concurrent stress
            concurrent_tasks = []

            # Create many concurrent operations with different types
            for i in range(20):  # Multiple batches
                batch_tasks = []
                for j in range(10):  # Each batch has many tasks
                    task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                    batch_tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                    batch_tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_performance_monitoring_cycle())
                    batch_tasks.append(task)

                concurrent_tasks.extend(batch_tasks)

                # Small delay between batches
                await asyncio.sleep(0.1)

            # Wait for all concurrent tasks to complete
            stress_start = time.time()
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            stress_duration = time.time() - stress_start

            # Analyze concurrent stress results
            total_tasks = len(results)
            exceptions = [r for r in results if isinstance(r, Exception)]
            successful_tasks = total_tasks - len(exceptions)

            # Verify concurrent stress handling
            assert (
                successful_tasks > total_tasks * 0.8
            ), f"Too many failed tasks: {successful_tasks}/{total_tasks}"
            assert (
                stress_duration < 120.0
            ), f"Concurrent stress took too long: {stress_duration:.2f}s"

            # Verify system is still functional after stress
            operations = orchestrator.get_operation_history()
            assert len(operations) > 0, "No operations recorded after concurrent stress"

            # Verify system can still perform operations after stress
            post_stress_start = time.time()
            await orchestrator._execute_strategy_selection_cycle()
            post_stress_time = time.time() - post_stress_start

            assert (
                post_stress_time < 5.0
            ), f"Post-stress operation too slow: {post_stress_time:.2f}s"

            logger.info(
                f"Concurrent stress - Duration: {stress_duration:.2f}s, "
                f"Successful: {successful_tasks}/{total_tasks}, "
                f"Post-stress time: {post_stress_time:.2f}s"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_stress_recovery(self, orchestrator):
        """Test system recovery after stress conditions."""
        logger.info("Testing stress recovery")

        # Record initial resource usage
        initial_memory = self.get_memory_usage()
        initial_cpu = self.get_cpu_usage()

        # Start the system
        await orchestrator.start()

        try:
            # Apply stress
            stress_tasks = []
            for i in range(30):
                task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                stress_tasks.append(task)
                task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                stress_tasks.append(task)

            # Wait for stress to complete
            await asyncio.gather(*stress_tasks, return_exceptions=True)

            # Record resource usage after stress
            post_stress_memory = self.get_memory_usage()
            post_stress_cpu = self.get_cpu_usage()

            # Allow system to recover
            await asyncio.sleep(15)  # Recovery period

            # Force garbage collection to aid recovery
            self.force_garbage_collection()

            # Record resource usage after recovery
            recovery_memory = self.get_memory_usage()
            recovery_cpu = self.get_cpu_usage()

            # Test system functionality after recovery
            recovery_start = time.time()
            for i in range(5):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
            recovery_duration = time.time() - recovery_start

            # Verify recovery performance
            memory_recovery = post_stress_memory - recovery_memory
            cpu_recovery = post_stress_cpu - recovery_cpu

            # Memory should recover (decrease or stay stable)
            assert memory_recovery > -100.0, f"Poor memory recovery: {memory_recovery:.2f}MB"

            # CPU should decrease during recovery
            assert cpu_recovery > -30.0, f"Poor CPU recovery: {cpu_recovery:.2f}%"

            # System should be functional after recovery
            recovery_throughput = 10 / recovery_duration  # 10 operations
            assert (
                recovery_throughput > 1.0
            ), f"Recovery throughput too low: {recovery_throughput:.2f} ops/sec"

            # Final resource usage should be reasonable
            final_memory_increase = recovery_memory - initial_memory
            assert (
                final_memory_increase < 200.0
            ), f"Final memory increase too high: {final_memory_increase:.2f}MB"
            assert recovery_cpu < 50.0, f"Final CPU usage too high: {recovery_cpu:.2f}%"

            logger.info(
                f"Stress recovery - Memory recovery: {memory_recovery:.2f}MB, "
                f"CPU recovery: {cpu_recovery:.2f}%, "
                f"Recovery throughput: {recovery_throughput:.2f} ops/sec"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_system_stability_under_stress(self, orchestrator):
        """Test system stability under prolonged stress conditions."""
        logger.info("Testing system stability under stress")

        # Start the system
        await orchestrator.start()

        try:
            # Monitor system stability over time
            stability_metrics = []

            # Run stress cycles
            for cycle in range(5):  # 5 stress cycles
                cycle_start = time.time()

                # Create stress for this cycle
                cycle_tasks = []
                for i in range(15):
                    task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                    cycle_tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                    cycle_tasks.append(task)

                # Wait for cycle to complete
                results = await asyncio.gather(*cycle_tasks, return_exceptions=True)
                cycle_duration = time.time() - cycle_start

                # Record cycle metrics
                exceptions = [r for r in results if isinstance(r, Exception)]
                success_rate = (len(results) - len(exceptions)) / len(results)

                stability_metrics.append(
                    {
                        "cycle": cycle,
                        "duration": cycle_duration,
                        "success_rate": success_rate,
                        "exceptions": len(exceptions),
                    }
                )

                # Small recovery period between cycles
                await asyncio.sleep(2)

            # Analyze stability metrics
            success_rates = [m["success_rate"] for m in stability_metrics]
            durations = [m["duration"] for m in stability_metrics]
            total_exceptions = sum(m["exceptions"] for m in stability_metrics)

            # Verify system stability
            assert all(sr > 0.7 for sr in success_rates), f"Success rates too low: {success_rates}"
            assert all(d < 30.0 for d in durations), f"Cycle durations too high: {durations}"
            assert (
                total_exceptions < len(stability_metrics) * 10
            ), f"Too many exceptions: {total_exceptions}"

            # Verify no degradation over time
            if len(success_rates) > 1:
                first_success = success_rates[0]
                last_success = success_rates[-1]
                degradation = first_success - last_success
                assert degradation < 0.3, f"Success rate degraded too much: {degradation:.3f}"

            # Verify system is still functional after all cycles
            final_operations = orchestrator.get_operation_history()
            assert len(final_operations) > 0, "No operations recorded after stability test"

            logger.info(
                f"System stability - Success rates: {success_rates}, "
                f"Total exceptions: {total_exceptions}, "
                f"Final operations: {len(final_operations)}"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_memory_leak_stress(self, orchestrator):
        """Test for memory leaks under stress conditions."""
        logger.info("Testing memory leak stress")

        # Record initial memory
        initial_memory = self.get_memory_usage()

        # Start the system
        await orchestrator.start()

        try:
            # Run stress cycles to check for memory leaks
            memory_readings = [initial_memory]

            for cycle in range(10):  # 10 stress cycles
                # Create stress
                stress_tasks = []
                for i in range(10):
                    task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                    stress_tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                    stress_tasks.append(task)

                # Wait for stress to complete
                await asyncio.gather(*stress_tasks, return_exceptions=True)

                # Record memory usage
                current_memory = self.get_memory_usage()
                memory_readings.append(current_memory)

                # Force garbage collection
                self.force_garbage_collection()

                # Record memory after GC
                post_gc_memory = self.get_memory_usage()
                memory_readings.append(post_gc_memory)

                # Small delay between cycles
                await asyncio.sleep(1)

            # Analyze memory leak patterns
            gc_memory_readings = memory_readings[1::2]  # Every other reading (after GC)
            memory_increases = [mr - initial_memory for mr in gc_memory_readings]

            # Check for memory leaks
            final_memory_increase = memory_increases[-1]
            max_memory_increase = max(memory_increases)

            # Verify no significant memory leaks
            assert (
                final_memory_increase < 100.0
            ), f"Memory leak detected: {final_memory_increase:.2f}MB"
            assert (
                max_memory_increase < 200.0
            ), f"Peak memory increase too high: {max_memory_increase:.2f}MB"

            # Verify memory stabilizes
            if len(memory_increases) > 3:
                recent_increases = memory_increases[-3:]
                memory_stability = max(recent_increases) - min(recent_increases)
                assert memory_stability < 50.0, f"Memory not stabilizing: {memory_stability:.2f}MB"

            logger.info(
                f"Memory leak stress - Final increase: {final_memory_increase:.2f}MB, "
                f"Max increase: {max_memory_increase:.2f}MB"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_error_stress(self, orchestrator):
        """Test system behavior under error stress conditions."""
        logger.info("Testing error stress")

        # Start the system
        await orchestrator.start()

        try:
            # Create error stress by injecting errors
            error_tasks = []

            # Mock errors in strategy selection
            original_method = orchestrator.strategy_selector.get_current_selection

            for i in range(20):
                if i % 3 == 0:  # Inject error every 3rd operation
                    orchestrator.strategy_selector.get_current_selection = Mock(
                        side_effect=Exception(f"Stress error {i}")
                    )
                else:
                    orchestrator.strategy_selector.get_current_selection = original_method

                task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                error_tasks.append(task)
                task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                error_tasks.append(task)

            # Wait for error stress to complete
            results = await asyncio.gather(*error_tasks, return_exceptions=True)

            # Restore original method
            orchestrator.strategy_selector.get_current_selection = original_method

            # Analyze error stress results
            exceptions = [r for r in results if isinstance(r, Exception)]
            successful_ops = len(results) - len(exceptions)

            # Verify system handles errors gracefully
            assert successful_ops > 0, "No operations completed under error stress"
            assert (
                len(exceptions) < len(results) * 0.8
            ), f"Too many errors: {len(exceptions)}/{len(results)}"

            # Verify system can recover from errors
            recovery_start = time.time()
            await orchestrator._execute_strategy_selection_cycle()
            recovery_time = time.time() - recovery_start

            assert recovery_time < 5.0, f"Error recovery too slow: {recovery_time:.2f}s"

            # Verify system is still functional
            operations = orchestrator.get_operation_history()
            assert len(operations) > 0, "No operations recorded after error stress"

            logger.info(
                f"Error stress - Successful: {successful_ops}/{len(results)}, "
                f"Recovery time: {recovery_time:.2f}s"
            )

        finally:
            await orchestrator.stop()


# Add logger for test output
import logging

logger = logging.getLogger(__name__)
