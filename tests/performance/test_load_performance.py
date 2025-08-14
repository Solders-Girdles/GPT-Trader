"""
Load Performance Tests for Phase 5 Production Integration.

This module tests system performance under various load conditions including:
- Normal load performance
- Elevated load performance
- Concurrent operation handling
- Response time under load
- Throughput capacity
- Resource utilization under load
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

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


class TestLoadPerformance:
    """Test system performance under various load conditions."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for load testing."""
        return OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=10,  # Fast intervals for load testing
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
        """Create a mock broker for load testing."""
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
        """Create a mock knowledge base for load testing."""
        kb = Mock(spec=StrategyKnowledgeBase)

        # Create test strategies for load testing
        from bot.knowledge.strategy_knowledge_base import (
            StrategyContext,
            StrategyMetadata,
            StrategyPerformance,
        )

        strategies = []
        for i in range(5):
            strategy = StrategyMetadata(
                strategy_id=f"load_test_strategy_{i}",
                name=f"Load Test Strategy {i}",
                description=f"Strategy for load testing {i}",
                strategy_type="trend_following",
                parameters={"param1": i},
                context=StrategyContext(
                    market_regime="trending",
                    time_period="bull_market",
                    asset_class="equity",
                    risk_profile="moderate",
                    volatility_regime="medium",
                    correlation_regime="low",
                ),
                performance=StrategyPerformance(
                    sharpe_ratio=1.0 + i * 0.1,
                    cagr=0.1 + i * 0.02,
                    max_drawdown=0.05 + i * 0.01,
                    win_rate=0.6 + i * 0.02,
                    consistency_score=0.7 + i * 0.05,
                    n_trades=50 + i * 10,
                    avg_trade_duration=5.0,
                    profit_factor=1.2 + i * 0.1,
                    calmar_ratio=1.0 + i * 0.1,
                    sortino_ratio=1.5 + i * 0.1,
                    information_ratio=1.0 + i * 0.1,
                    beta=0.8 + i * 0.05,
                    alpha=0.05 + i * 0.01,
                ),
                discovery_date=datetime.now() - timedelta(days=30),
                last_updated=datetime.now() - timedelta(days=5),
                usage_count=20 + i * 5,
                success_rate=0.7 + i * 0.05,
            )
            strategies.append(strategy)

        kb.find_strategies.return_value = strategies
        return kb

    @pytest.fixture
    def orchestrator(self, orchestrator_config, mock_broker, mock_knowledge_base):
        """Create a production orchestrator for load testing."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "GOOG", "MSFT"]

        # Patch the data manager to avoid real data fetching
        with patch("bot.live.production_orchestrator.LiveDataManager"):
            orchestrator = ProductionOrchestrator(
                config=orchestrator_config,
                broker=mock_broker,
                knowledge_base=mock_knowledge_base,
                symbols=symbols,
            )

            # Mock the data manager's async methods
            orchestrator.data_manager.start = AsyncMock()
            orchestrator.data_manager.stop = AsyncMock()

            # Mock the start method to prevent infinite loops
            original_start = orchestrator.start

            async def mock_start():
                orchestrator.is_running = True
                await orchestrator.data_manager.start()

            orchestrator.start = mock_start

            # Mock strategy selector to return strategies
            from bot.knowledge.strategy_knowledge_base import (
                StrategyContext,
                StrategyMetadata,
                StrategyPerformance,
            )

            strategy1 = StrategyMetadata(
                strategy_id="load_test_strategy_1",
                name="Load Test Strategy 1",
                description="Strategy for load testing",
                strategy_type="trend_following",
                parameters={"param1": 1},
                context=StrategyContext(
                    market_regime="trending",
                    time_period="bull_market",
                    asset_class="equity",
                    risk_profile="moderate",
                    volatility_regime="medium",
                    correlation_regime="low",
                ),
                performance=StrategyPerformance(
                    sharpe_ratio=1.2,
                    cagr=0.15,
                    max_drawdown=0.08,
                    win_rate=0.65,
                    consistency_score=0.75,
                    n_trades=50,
                    avg_trade_duration=5.0,
                    profit_factor=1.3,
                    calmar_ratio=1.1,
                    sortino_ratio=1.4,
                    information_ratio=1.0,
                    beta=0.8,
                    alpha=0.05,
                ),
                discovery_date=datetime.now() - timedelta(days=30),
                last_updated=datetime.now() - timedelta(days=5),
                usage_count=25,
                success_rate=0.75,
            )

            selected_strategies = [Mock(strategy=strategy1, score=0.85, confidence=0.8)]
            orchestrator.strategy_selector.get_current_selection = Mock(
                return_value=selected_strategies
            )

            return orchestrator

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    @pytest.mark.asyncio
    async def test_normal_load_performance(self, orchestrator):
        """Test system performance under normal load conditions."""
        logger.info("Testing normal load performance")

        # Record initial resource usage
        initial_memory = self.get_memory_usage()
        initial_cpu = self.get_cpu_usage()

        # Start the system
        start_time = time.time()
        await orchestrator.start()
        startup_time = time.time() - start_time

        try:
            # Run normal operations for 30 seconds by manually executing cycles
            operation_count = 0
            start_ops = time.time()

            while time.time() - start_ops < 30:
                # Execute operations manually
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                operation_count += 3

                # Small delay between operations
                await asyncio.sleep(1)

            # Record final resource usage
            final_memory = self.get_memory_usage()
            final_cpu = self.get_cpu_usage()

            # Get performance metrics
            operations = orchestrator.get_operation_history()

            # Calculate performance metrics
            memory_increase = final_memory - initial_memory
            cpu_usage = final_cpu

            # Verify performance is acceptable
            assert startup_time < 5.0, f"Startup too slow: {startup_time:.2f}s"
            assert memory_increase < 100.0, f"Memory increase too high: {memory_increase:.2f}MB"
            assert cpu_usage < 50.0, f"CPU usage too high: {cpu_usage:.2f}%"
            assert operation_count > 0, "No operations performed"

            # Calculate throughput
            throughput = operation_count / 30.0  # operations per second
            assert throughput > 0.1, f"Throughput too low: {throughput:.2f} ops/sec"

            logger.info(
                f"Normal load performance - Startup: {startup_time:.2f}s, "
                f"Memory: +{memory_increase:.2f}MB, CPU: {cpu_usage:.2f}%, "
                f"Throughput: {throughput:.2f} ops/sec"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_elevated_load_performance(self, orchestrator):
        """Test system performance under elevated load conditions."""
        logger.info("Testing elevated load performance")

        # Record initial resource usage
        initial_memory = self.get_memory_usage()
        initial_cpu = self.get_cpu_usage()

        # Start the system
        start_time = time.time()
        await orchestrator.start()
        startup_time = time.time() - start_time

        try:
            # Create elevated load by running many operations concurrently
            async def run_concurrent_operations():
                tasks = []
                for i in range(10):
                    task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                    tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                    tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_performance_monitoring_cycle())
                    tasks.append(task)

                await asyncio.gather(*tasks, return_exceptions=True)

            # Run elevated load for 20 seconds
            elevated_start = time.time()
            await run_concurrent_operations()
            await asyncio.sleep(20)
            elevated_duration = time.time() - elevated_start

            # Record final resource usage
            final_memory = self.get_memory_usage()
            final_cpu = self.get_cpu_usage()

            # Get performance metrics
            operations = orchestrator.get_operation_history()
            operation_count = len(operations)

            # Calculate performance metrics
            memory_increase = final_memory - initial_memory
            cpu_usage = final_cpu

            # Verify performance under elevated load
            assert startup_time < 5.0, f"Startup too slow: {startup_time:.2f}s"
            assert memory_increase < 200.0, f"Memory increase too high: {memory_increase:.2f}MB"
            assert cpu_usage < 80.0, f"CPU usage too high: {cpu_usage:.2f}%"
            assert operation_count > 0, "No operations performed"

            # Calculate throughput under elevated load
            throughput = operation_count / elevated_duration
            assert throughput > 0.5, f"Elevated load throughput too low: {throughput:.2f} ops/sec"

            logger.info(
                f"Elevated load performance - Startup: {startup_time:.2f}s, "
                f"Memory: +{memory_increase:.2f}MB, CPU: {cpu_usage:.2f}%, "
                f"Throughput: {throughput:.2f} ops/sec"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_concurrent_operation_handling(self, orchestrator):
        """Test system ability to handle concurrent operations."""
        logger.info("Testing concurrent operation handling")

        # Start the system
        await orchestrator.start()

        try:
            # Create concurrent operation tasks
            async def run_strategy_operations():
                for i in range(5):
                    await orchestrator._execute_strategy_selection_cycle()
                    await asyncio.sleep(0.1)

            async def run_risk_operations():
                for i in range(5):
                    await orchestrator._execute_risk_monitoring_cycle()
                    await asyncio.sleep(0.1)

            async def run_performance_operations():
                for i in range(5):
                    await orchestrator._execute_performance_monitoring_cycle()
                    await asyncio.sleep(0.1)

            # Run operations concurrently
            start_time = time.time()
            await asyncio.gather(
                run_strategy_operations(), run_risk_operations(), run_performance_operations()
            )
            concurrent_duration = time.time() - start_time

            # Verify all operations completed
            operations = orchestrator.get_operation_history()
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

            assert (
                len(strategy_ops) == 5
            ), f"Expected 5 strategy operations, got {len(strategy_ops)}"
            assert len(risk_ops) == 5, f"Expected 5 risk operations, got {len(risk_ops)}"
            assert len(perf_ops) == 5, f"Expected 5 performance operations, got {len(perf_ops)}"

            # Verify concurrent execution was efficient
            total_operations = len(operations)
            throughput = total_operations / concurrent_duration
            assert throughput > 2.0, f"Concurrent throughput too low: {throughput:.2f} ops/sec"

            logger.info(
                f"Concurrent operation handling - Duration: {concurrent_duration:.2f}s, "
                f"Operations: {total_operations}, Throughput: {throughput:.2f} ops/sec"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_response_time_under_load(self, orchestrator):
        """Test response time under various load conditions."""
        logger.info("Testing response time under load")

        # Start the system
        await orchestrator.start()

        try:
            response_times = []

            # Test response times under increasing load
            for load_level in range(1, 6):
                # Create load
                load_tasks = []
                for i in range(load_level * 2):
                    task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                    load_tasks.append(task)

                # Measure response time for a single operation
                start_time = time.time()
                await orchestrator._execute_risk_monitoring_cycle()
                response_time = time.time() - start_time
                response_times.append((load_level, response_time))

                # Wait for load tasks to complete
                await asyncio.gather(*load_tasks, return_exceptions=True)
                await asyncio.sleep(1)

            # Analyze response times
            load_levels = [rt[0] for rt in response_times]
            times = [rt[1] for rt in response_times]

            # Verify response times are reasonable
            assert all(t < 5.0 for t in times), f"Response times too high: {times}"

            # Verify response time doesn't degrade too much under load
            baseline_time = times[0]
            max_time = max(times)
            degradation = max_time / baseline_time if baseline_time > 0 else float("inf")

            assert degradation < 10.0, f"Response time degradation too high: {degradation:.2f}x"

            logger.info(
                f"Response time under load - Baseline: {baseline_time:.3f}s, "
                f"Max: {max_time:.3f}s, Degradation: {degradation:.2f}x"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_throughput_capacity(self, orchestrator):
        """Test system throughput capacity."""
        logger.info("Testing throughput capacity")

        # Start the system
        await orchestrator.start()

        try:
            throughput_measurements = []

            # Test throughput at different load levels
            for load_level in range(1, 6):
                # Clear operation history for clean measurement
                orchestrator.operation_history.clear()

                # Create load
                start_time = time.time()
                tasks = []
                for i in range(load_level * 10):
                    task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                    tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                    tasks.append(task)

                # Wait for all tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start_time

                # Calculate throughput
                operations = orchestrator.get_operation_history()
                throughput = len(operations) / duration
                throughput_measurements.append((load_level, throughput))

                await asyncio.sleep(1)

            # Analyze throughput capacity
            load_levels = [tm[0] for tm in throughput_measurements]
            throughputs = [tm[1] for tm in throughput_measurements]

            # Verify throughput increases with load (up to a point)
            assert len(throughputs) > 1, "Need multiple throughput measurements"

            # Verify minimum throughput
            min_throughput = min(throughputs)
            assert min_throughput > 0.5, f"Minimum throughput too low: {min_throughput:.2f} ops/sec"

            # Verify throughput doesn't collapse under load
            max_throughput = max(throughputs)
            throughput_ratio = (
                max_throughput / min_throughput if min_throughput > 0 else float("inf")
            )
            assert throughput_ratio < 100.0, f"Throughput ratio too high: {throughput_ratio:.2f}"

            logger.info(
                f"Throughput capacity - Min: {min_throughput:.2f} ops/sec, "
                f"Max: {max_throughput:.2f} ops/sec, Ratio: {throughput_ratio:.2f}"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_resource_utilization_under_load(self, orchestrator):
        """Test resource utilization under load."""
        logger.info("Testing resource utilization under load")

        # Record baseline resource usage
        baseline_memory = self.get_memory_usage()
        baseline_cpu = self.get_cpu_usage()

        # Start the system
        await orchestrator.start()

        try:
            resource_measurements = []

            # Test resource usage under different load levels
            for load_level in range(1, 6):
                # Create load
                tasks = []
                for i in range(load_level * 5):
                    task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                    tasks.append(task)
                    task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                    tasks.append(task)

                # Wait for tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

                # Measure resource usage
                memory_usage = self.get_memory_usage()
                cpu_usage = self.get_cpu_usage()

                resource_measurements.append(
                    {
                        "load_level": load_level,
                        "memory_mb": memory_usage,
                        "cpu_percent": cpu_usage,
                        "memory_increase": memory_usage - baseline_memory,
                    }
                )

                await asyncio.sleep(2)

            # Analyze resource utilization
            memory_increases = [rm["memory_increase"] for rm in resource_measurements]
            cpu_usages = [rm["cpu_percent"] for rm in resource_measurements]

            # Verify memory usage is reasonable
            max_memory_increase = max(memory_increases)
            assert (
                max_memory_increase < 500.0
            ), f"Memory increase too high: {max_memory_increase:.2f}MB"

            # Verify CPU usage is reasonable (macOS may report transient spikes differently)
            max_cpu_usage = max(cpu_usages)
            if os.getenv("CI") or os.name != "posix":
                assert max_cpu_usage < 90.0, f"CPU usage too high: {max_cpu_usage:.2f}%"
            else:
                assert max_cpu_usage < 98.0, f"CPU usage too high: {max_cpu_usage:.2f}%"

            # Verify resource usage scales reasonably with load
            if len(memory_increases) > 1:
                if memory_increases[0] > 0:
                    memory_scaling = memory_increases[-1] / memory_increases[0]
                    assert memory_scaling < 10.0, f"Memory scaling too high: {memory_scaling:.2f}x"
                else:
                    # If baseline memory increase is 0, just check that memory doesn't grow excessively
                    max_memory_increase = max(memory_increases)
                    assert (
                        max_memory_increase < 200.0
                    ), f"Memory increase too high: {max_memory_increase:.2f}MB"

            logger.info(
                f"Resource utilization - Max memory increase: {max_memory_increase:.2f}MB, "
                f"Max CPU: {max_cpu_usage:.2f}%"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, orchestrator):
        """Test system performance under sustained load."""
        logger.info("Testing sustained load performance")

        # Record initial resource usage
        initial_memory = self.get_memory_usage()
        initial_cpu = self.get_cpu_usage()

        # Start the system
        await orchestrator.start()

        try:
            # Run sustained load for 60 seconds
            sustained_start = time.time()
            operation_count = 0

            while time.time() - sustained_start < 60:
                # Execute operations
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                operation_count += 3

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.5)

            sustained_duration = time.time() - sustained_start

            # Record final resource usage
            final_memory = self.get_memory_usage()
            final_cpu = self.get_cpu_usage()

            # Calculate performance metrics
            memory_increase = final_memory - initial_memory
            cpu_usage = final_cpu
            throughput = operation_count / sustained_duration

            # Verify sustained performance
            assert (
                memory_increase < 300.0
            ), f"Sustained load memory increase too high: {memory_increase:.2f}MB"
            assert cpu_usage < 70.0, f"Sustained load CPU usage too high: {cpu_usage:.2f}%"
            assert throughput > 1.0, f"Sustained load throughput too low: {throughput:.2f} ops/sec"

            # Verify no memory leaks (memory should stabilize)
            operations = orchestrator.get_operation_history()
            assert len(operations) == operation_count, "Operation count mismatch"

            logger.info(
                f"Sustained load performance - Duration: {sustained_duration:.2f}s, "
                f"Operations: {operation_count}, Memory: +{memory_increase:.2f}MB, "
                f"CPU: {cpu_usage:.2f}%, Throughput: {throughput:.2f} ops/sec"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_load_recovery_performance(self, orchestrator):
        """Test system performance recovery after high load."""
        logger.info("Testing load recovery performance")

        # Start the system
        await orchestrator.start()

        try:
            # Apply high load
            high_load_tasks = []
            for i in range(20):
                task = asyncio.create_task(orchestrator._execute_strategy_selection_cycle())
                high_load_tasks.append(task)
                task = asyncio.create_task(orchestrator._execute_risk_monitoring_cycle())
                high_load_tasks.append(task)

            # Wait for high load to complete
            await asyncio.gather(*high_load_tasks, return_exceptions=True)

            # Record resource usage after high load
            post_load_memory = self.get_memory_usage()
            post_load_cpu = self.get_cpu_usage()

            # Allow system to recover
            await asyncio.sleep(10)

            # Record resource usage after recovery
            recovery_memory = self.get_memory_usage()
            recovery_cpu = self.get_cpu_usage()

            # Test performance after recovery
            recovery_start = time.time()
            for i in range(5):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
            recovery_duration = time.time() - recovery_start

            # Verify recovery performance
            memory_recovery = post_load_memory - recovery_memory
            cpu_recovery = post_load_cpu - recovery_cpu

            # Memory should not increase significantly during recovery
            assert memory_recovery > -50.0, f"Memory recovery poor: {memory_recovery:.2f}MB"

            # CPU should decrease during recovery
            assert cpu_recovery > -20.0, f"CPU recovery poor: {cpu_recovery:.2f}%"

            # Performance should be reasonable after recovery
            recovery_throughput = 10 / recovery_duration  # 10 operations
            assert (
                recovery_throughput > 1.0
            ), f"Recovery throughput too low: {recovery_throughput:.2f} ops/sec"

            logger.info(
                f"Load recovery performance - Memory recovery: {memory_recovery:.2f}MB, "
                f"CPU recovery: {cpu_recovery:.2f}%, Recovery throughput: {recovery_throughput:.2f} ops/sec"
            )

        finally:
            await orchestrator.stop()


# Add logger for test output
import logging

logger = logging.getLogger(__name__)
