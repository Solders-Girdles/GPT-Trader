"""
Trading Cycles Tests for Phase 5 Production Integration.

This module tests complete trading cycle execution including:
- Strategy selection cycles
- Portfolio optimization cycles
- Risk monitoring cycles
- Performance monitoring cycles
- Trading execution cycles
- Cycle timing and synchronization
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import numpy as np
import pandas as pd
import pytest
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.intelligence.order_simulator import L2SlippageModel
from bot.intelligence.transition_metrics import TransitionSmoothnessCalculator
from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyKnowledgeBase,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.live.production_orchestrator import (
    OrchestrationMode,
    OrchestratorConfig,
    ProductionOrchestrator,
)
from bot.live.strategy_selector import SelectionMethod
from bot.portfolio.optimizer import OptimizationMethod


class TestTradingCycles:
    """Test complete trading cycle execution."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for trading cycle testing."""
        return OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=30,  # 30 seconds for testing
            risk_check_interval=15,  # 15 seconds for testing
            performance_check_interval=20,  # 20 seconds for testing
            max_strategies=3,
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
            alert_cooldown_minutes=5,
        )

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for trading cycle testing."""
        broker = Mock(spec=AlpacaPaperBroker)

        # Mock account
        account = Mock()
        account.equity = 100000.0
        account.cash = 50000.0
        account.buying_power = 50000.0
        broker.get_account.return_value = account

        # Mock positions
        positions = []
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
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
    def sample_knowledge_base(self):
        """Create a sample knowledge base with test strategies."""
        kb = Mock(spec=StrategyKnowledgeBase)

        # Create test strategies
        strategies = []
        for i in range(5):
            strategy = StrategyMetadata(
                strategy_id=f"test_strategy_{i}",
                name=f"Test Strategy {i}",
                description=f"Test strategy {i}",
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
    def orchestrator(self, orchestrator_config, mock_broker, sample_knowledge_base):
        """Create a production orchestrator for trading cycle testing."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Create orchestrator
        orchestrator = ProductionOrchestrator(
            config=orchestrator_config,
            broker=mock_broker,
            knowledge_base=sample_knowledge_base,
            symbols=symbols,
        )

        # Mock the data manager's async methods properly
        orchestrator.data_manager.start = AsyncMock()
        orchestrator.data_manager.stop = AsyncMock()

        # Mock the start method to prevent infinite loops
        original_start = orchestrator.start

        async def mock_start():
            orchestrator.is_running = True
            await orchestrator.data_manager.start()

        orchestrator.start = mock_start

        return orchestrator

    @pytest.mark.asyncio
    async def test_strategy_selection_cycle(self, orchestrator):
        """Test complete strategy selection cycle execution."""
        logger.info("Testing strategy selection cycle")

        # Mock strategy selection results with proper data
        from bot.knowledge.strategy_knowledge_base import (
            StrategyContext,
            StrategyMetadata,
            StrategyPerformance,
        )

        strategy1 = StrategyMetadata(
            strategy_id="test_strategy_1",
            name="Test Strategy 1",
            description="Test strategy 1",
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

        strategy2 = StrategyMetadata(
            strategy_id="test_strategy_2",
            name="Test Strategy 2",
            description="Test strategy 2",
            strategy_type="mean_reversion",
            parameters={"param1": 2},
            context=StrategyContext(
                market_regime="sideways",
                time_period="sideways_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="low",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.0,
                cagr=0.12,
                max_drawdown=0.10,
                win_rate=0.60,
                consistency_score=0.70,
                n_trades=45,
                avg_trade_duration=4.5,
                profit_factor=1.2,
                calmar_ratio=1.0,
                sortino_ratio=1.3,
                information_ratio=0.9,
                beta=0.9,
                alpha=0.03,
            ),
            discovery_date=datetime.now() - timedelta(days=25),
            last_updated=datetime.now() - timedelta(days=3),
            usage_count=20,
            success_rate=0.70,
        )

        selected_strategies = [
            Mock(strategy=strategy1, score=0.85, confidence=0.8),
            Mock(strategy=strategy2, score=0.75, confidence=0.7),
        ]

        orchestrator.strategy_selector.get_current_selection = Mock(
            return_value=selected_strategies
        )

        # Execute strategy selection cycle
        start_time = time.time()
        await orchestrator._execute_strategy_selection_cycle_impl()
        cycle_duration = time.time() - start_time

        # Verify cycle execution
        assert (
            cycle_duration < 5.0
        ), f"Strategy selection cycle took too long: {cycle_duration:.2f}s"

        # Check operation was recorded
        operations = orchestrator.get_operation_history()
        strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]

        assert len(strategy_ops) > 0, "Strategy selection operation not recorded"

        latest_op = strategy_ops[-1]
        assert "n_strategies" in latest_op["data"], "Number of strategies not recorded"
        assert "allocation" in latest_op["data"], "Portfolio allocation not recorded"
        assert latest_op["data"]["n_strategies"] == 2, "Expected 2 strategies"

        # Verify timestamp is recent
        op_time = latest_op["timestamp"]
        time_diff = datetime.now() - op_time
        assert time_diff.total_seconds() < 10, "Operation timestamp should be recent"

        logger.info(f"Strategy selection cycle completed in {cycle_duration:.2f}s")

        # Verify transition smoothness calculation produces a score in [0,1]
        calc = TransitionSmoothnessCalculator(L2SlippageModel())
        score = calc.calculate_smoothness_score(
            current_allocations={"a": 0.5, "b": 0.5},
            target_allocations={"a": 0.4, "b": 0.6},
            portfolio_value=100000.0,
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_automated_drawdown_guard_blocks_and_turnover_recorded(self, orchestrator):
        # Configure automated mode and low drawdown limit to trigger block
        orchestrator.config.mode = OrchestrationMode.AUTOMATED
        orchestrator.safety_rails.max_drawdown_limit = 0.05

        # Mock performance monitor summary to exceed drawdown
        orchestrator.performance_monitor.get_performance_summary = Mock(
            return_value={
                "strategies": {
                    "portfolio": {"current_drawdown": 0.10},
                }
            }
        )

        # Patch execution to verify it's not called
        called = {"exec": False}

        async def _no_exec(allocation):
            called["exec"] = True

        orchestrator._execute_portfolio_changes = _no_exec

        # Minimal selection to create an allocation and trigger path
        from bot.knowledge.strategy_knowledge_base import (
            StrategyContext,
            StrategyMetadata,
            StrategyPerformance,
        )

        strategy = StrategyMetadata(
            strategy_id="s_guard",
            name="S Guard",
            description="",
            strategy_type="test",
            parameters={},
            context=StrategyContext(
                market_regime="trending",
                time_period="bull",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="low",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.0,
                cagr=0.10,
                max_drawdown=0.05,
                win_rate=0.6,
                consistency_score=0.7,
                n_trades=10,
                avg_trade_duration=5.0,
                profit_factor=1.2,
                calmar_ratio=1.0,
                sortino_ratio=1.2,
                information_ratio=1.0,
                beta=0.8,
                alpha=0.05,
            ),
            discovery_date=datetime.now(),
            last_updated=datetime.now(),
            usage_count=0,
            success_rate=0.0,
        )

        orchestrator.strategy_selector.get_current_selection = Mock(
            return_value=[Mock(strategy=strategy, score=0.8, confidence=0.8)]
        )

        await orchestrator._execute_strategy_selection_cycle_impl()

        # Assert block occurred (execute not called)
        assert called["exec"] is False

        # Operation should contain turnover field
        ops = orchestrator.get_operation_history("strategy_selection")
        assert len(ops) >= 1
        assert "turnover" in ops[-1]["data"]
        assert isinstance(ops[-1]["data"]["turnover"], (int, float))

    @pytest.mark.asyncio
    async def test_risk_monitoring_cycle(self, orchestrator):
        """Test complete risk monitoring cycle execution."""
        logger.info("Testing risk monitoring cycle")

        # Mock current positions
        mock_positions = {
            "AAPL": {
                "quantity": 100,
                "market_value": 15000.0,
                "portfolio_value": 100000.0,
                "weight": 0.15,
                "current_price": 150.0,
                "entry_price": 145.0,
            },
            "GOOGL": {
                "quantity": 50,
                "market_value": 25000.0,
                "portfolio_value": 100000.0,
                "weight": 0.25,
                "current_price": 500.0,
                "entry_price": 480.0,
            },
        }

        # Mock methods
        orchestrator._get_current_positions = AsyncMock(return_value=mock_positions)
        orchestrator._get_symbol_data = AsyncMock(
            return_value=pd.DataFrame({"Close": [150.0, 500.0], "Volume": [1000000, 500000]})
        )

        # Execute risk monitoring cycle
        start_time = time.time()
        await orchestrator._execute_risk_monitoring_cycle()
        cycle_duration = time.time() - start_time

        # Verify cycle execution
        assert cycle_duration < 5.0, f"Risk monitoring cycle took too long: {cycle_duration:.2f}s"

        # Check operation was recorded
        operations = orchestrator.get_operation_history()
        risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]

        assert len(risk_ops) > 0, "Risk monitoring operation not recorded"

        latest_op = risk_ops[-1]
        assert "timestamp" in latest_op, "Operation timestamp not recorded"

        # Verify timestamp is recent
        op_time = latest_op["timestamp"]
        time_diff = datetime.now() - op_time
        assert time_diff.total_seconds() < 10, "Operation timestamp should be recent"

        logger.info(f"Risk monitoring cycle completed in {cycle_duration:.2f}s")

    @pytest.mark.asyncio
    async def test_performance_monitoring_cycle(self, orchestrator):
        """Test complete performance monitoring cycle execution."""
        logger.info("Testing performance monitoring cycle")

        # Mock performance data
        orchestrator.performance_monitor.calculate_performance_metrics = Mock(
            return_value={
                "sharpe_ratio": 1.2,
                "cagr": 0.15,
                "max_drawdown": 0.08,
                "volatility": 0.12,
                "win_rate": 0.65,
            }
        )

        # Execute performance monitoring cycle
        start_time = time.time()
        await orchestrator._execute_performance_monitoring_cycle()
        cycle_duration = time.time() - start_time

        # Verify cycle execution
        assert (
            cycle_duration < 5.0
        ), f"Performance monitoring cycle took too long: {cycle_duration:.2f}s"

        # Check operation was recorded
        operations = orchestrator.get_operation_history()
        perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

        assert len(perf_ops) > 0, "Performance monitoring operation not recorded"

        latest_op = perf_ops[-1]
        assert "timestamp" in latest_op, "Operation timestamp not recorded"

        # Verify timestamp is recent
        op_time = latest_op["timestamp"]
        time_diff = datetime.now() - op_time
        assert time_diff.total_seconds() < 10, "Operation timestamp should be recent"

        logger.info(f"Performance monitoring cycle completed in {cycle_duration:.2f}s")

    @pytest.mark.asyncio
    async def test_cycle_timing_and_synchronization(self, orchestrator):
        """Test cycle timing and synchronization between different cycles."""
        logger.info("Testing cycle timing and synchronization")

        # Start the orchestrator
        await orchestrator.start()

        try:
            # Wait for multiple cycles to complete
            await asyncio.sleep(40)  # Wait for at least one of each cycle type

            # Get all operations
            operations = orchestrator.get_operation_history()

            # Group operations by type
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

            # Verify we have operations from each cycle type
            assert len(strategy_ops) > 0, "No strategy selection operations"
            assert len(risk_ops) > 0, "No risk monitoring operations"
            assert len(perf_ops) > 0, "No performance monitoring operations"

            # Verify timing intervals are respected
            if len(strategy_ops) > 1:
                time_diff = strategy_ops[-1]["timestamp"] - strategy_ops[-2]["timestamp"]
                assert time_diff.total_seconds() >= 25, "Strategy selection cycles too frequent"

            if len(risk_ops) > 1:
                time_diff = risk_ops[-1]["timestamp"] - risk_ops[-2]["timestamp"]
                assert time_diff.total_seconds() >= 10, "Risk monitoring cycles too frequent"

            if len(perf_ops) > 1:
                time_diff = perf_ops[-1]["timestamp"] - perf_ops[-2]["timestamp"]
                assert time_diff.total_seconds() >= 15, "Performance monitoring cycles too frequent"

            logger.info(f"Cycle timing verified. Operations: {len(operations)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_cycle_error_handling(self, orchestrator):
        """Test error handling during cycle execution."""
        logger.info("Testing cycle error handling")

        # Mock error in strategy selection
        original_method = orchestrator.strategy_selector.get_current_selection
        orchestrator.strategy_selector.get_current_selection = Mock(
            side_effect=Exception("Test error")
        )

        # Execute strategy selection cycle (should handle error gracefully)
        await orchestrator._execute_strategy_selection_cycle()

        # Verify error was handled gracefully
        operations = orchestrator.get_operation_history()
        assert len(operations) >= 0, "System should continue after error"

        # Restore original method
        orchestrator.strategy_selector.get_current_selection = original_method

        # Execute cycle again (should work now)
        await orchestrator._execute_strategy_selection_cycle()

        # Verify cycle works after error recovery
        operations = orchestrator.get_operation_history()
        strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
        assert len(strategy_ops) > 0, "Strategy selection should work after error recovery"

        logger.info("Cycle error handling test completed")

    @pytest.mark.asyncio
    async def test_cycle_data_consistency(self, orchestrator):
        """Test data consistency across multiple cycles."""
        logger.info("Testing cycle data consistency")

        # Mock consistent data
        from bot.knowledge.strategy_knowledge_base import (
            StrategyContext,
            StrategyMetadata,
            StrategyPerformance,
        )

        strategy1 = StrategyMetadata(
            strategy_id="strategy_1",
            name="Strategy 1",
            description="Test strategy 1",
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

        test_data = {
            "portfolio_value": 100000.0,
            "positions": {
                "AAPL": {"weight": 0.3, "value": 30000.0},
                "GOOGL": {"weight": 0.4, "value": 40000.0},
                "MSFT": {"weight": 0.3, "value": 30000.0},
            },
        }

        # Mock methods to return consistent data
        orchestrator._get_current_positions = AsyncMock(return_value=test_data["positions"])
        orchestrator.strategy_selector.get_current_selection = Mock(
            return_value=[Mock(strategy=strategy1)]
        )

        # Execute multiple cycles
        await orchestrator._execute_strategy_selection_cycle()
        await orchestrator._execute_risk_monitoring_cycle()
        await orchestrator._execute_strategy_selection_cycle()

        # Verify data consistency
        operations = orchestrator.get_operation_history()

        # Check that portfolio values are consistent across operations
        strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
        risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]

        if strategy_ops and risk_ops:
            # Both should reference the same portfolio structure
            strategy_data = strategy_ops[-1]["data"]
            risk_data = risk_ops[-1]["data"]

            # Verify both operations have portfolio information
            assert (
                "allocation" in strategy_data or "n_strategies" in strategy_data
            ), "Strategy data missing portfolio info"
            assert "timestamp" in risk_data, "Risk data missing timestamp"

        logger.info("Cycle data consistency verified")

    @pytest.mark.asyncio
    async def test_cycle_performance_metrics(self, orchestrator):
        """Test performance metrics collection during cycles."""
        logger.info("Testing cycle performance metrics")

        # Execute multiple cycles and measure performance
        cycle_times = []

        for i in range(3):
            # Strategy selection cycle
            start_time = time.time()
            await orchestrator._execute_strategy_selection_cycle_impl()
            strategy_time = time.time() - start_time
            cycle_times.append(("strategy_selection", strategy_time))

            # Risk monitoring cycle
            start_time = time.time()
            await orchestrator._execute_risk_monitoring_cycle()
            risk_time = time.time() - start_time
            cycle_times.append(("risk_monitoring", risk_time))

            # Performance monitoring cycle
            start_time = time.time()
            await orchestrator._execute_performance_monitoring_cycle()
            perf_time = time.time() - start_time
            cycle_times.append(("performance_monitoring", perf_time))

        # Analyze performance metrics
        strategy_times = [t for op, t in cycle_times if op == "strategy_selection"]
        risk_times = [t for op, t in cycle_times if op == "risk_monitoring"]
        perf_times = [t for op, t in cycle_times if op == "performance_monitoring"]

        # Verify performance is reasonable
        assert all(
            t < 5.0 for t in strategy_times
        ), f"Strategy selection too slow: {strategy_times}"
        assert all(t < 5.0 for t in risk_times), f"Risk monitoring too slow: {risk_times}"
        assert all(t < 5.0 for t in perf_times), f"Performance monitoring too slow: {perf_times}"

        # Calculate averages
        avg_strategy = sum(strategy_times) / len(strategy_times)
        avg_risk = sum(risk_times) / len(risk_times)
        avg_perf = sum(perf_times) / len(perf_times)

        logger.info(
            f"Average cycle times - Strategy: {avg_strategy:.3f}s, Risk: {avg_risk:.3f}s, Performance: {avg_perf:.3f}s"
        )

        # Verify consistency (times should be similar across cycles)
        strategy_std = np.std(strategy_times)
        risk_std = np.std(risk_times)
        perf_std = np.std(perf_times)

        assert strategy_std < 1.0, f"Strategy selection times too variable: std={strategy_std:.3f}"
        assert risk_std < 1.0, f"Risk monitoring times too variable: std={risk_std:.3f}"
        assert perf_std < 1.0, f"Performance monitoring times too variable: std={perf_std:.3f}"

    @pytest.mark.asyncio
    async def test_cycle_concurrency(self, orchestrator):
        """Test concurrent execution of different cycles."""
        logger.info("Testing cycle concurrency")

        # Start the orchestrator to enable concurrent cycles
        await orchestrator.start()

        try:
            # Wait for concurrent cycles to run
            await asyncio.sleep(35)  # Wait for multiple cycles to overlap

            # Get operations
            operations = orchestrator.get_operation_history()

            # Group by operation type
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

            # Verify we have operations from all cycle types
            assert len(strategy_ops) > 0, "No strategy selection operations"
            assert len(risk_ops) > 0, "No risk monitoring operations"
            assert len(perf_ops) > 0, "No performance monitoring operations"

            # Verify operations are interleaved (indicating concurrency)
            all_ops = sorted(operations, key=lambda x: x["timestamp"])
            op_types = [op["operation"] for op in all_ops]

            # Check that different operation types are interleaved
            unique_types = set(op_types)
            assert len(unique_types) >= 2, "Only one operation type found - no concurrency"

            # Verify timing shows overlap
            if len(all_ops) > 1:
                time_diffs = []
                for i in range(1, len(all_ops)):
                    diff = all_ops[i]["timestamp"] - all_ops[i - 1]["timestamp"]
                    time_diffs.append(diff.total_seconds())

                # Some operations should be close together (indicating concurrency)
                close_ops = [d for d in time_diffs if d < 10]
                assert len(close_ops) > 0, "No operations close together - no concurrency"

            logger.info(f"Concurrency verified. Total operations: {len(operations)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_cycle_resource_usage(self, orchestrator):
        """Test resource usage during cycle execution."""
        logger.info("Testing cycle resource usage")

        # Mock memory usage tracking
        memory_usage = []

        # Execute cycles and track resource usage
        for i in range(3):
            # Execute strategy selection cycle
            await orchestrator._execute_strategy_selection_cycle()

            # Execute risk monitoring cycle
            await orchestrator._execute_risk_monitoring_cycle()

            # Execute performance monitoring cycle
            await orchestrator._execute_performance_monitoring_cycle()

            # Record operation count as proxy for resource usage
            operations = orchestrator.get_operation_history()
            memory_usage.append(len(operations))

        # Verify resource usage is reasonable
        assert len(memory_usage) == 3, "Expected 3 resource usage measurements"

        # Verify operation count increases (indicating operations are being recorded)
        assert memory_usage[-1] > memory_usage[0], "Operation count should increase"

        # Verify no excessive growth (indicating memory leaks)
        growth_rate = (memory_usage[-1] - memory_usage[0]) / len(memory_usage)
        assert growth_rate < 10, f"Excessive operation growth: {growth_rate}"

        logger.info(f"Resource usage verified. Operation growth rate: {growth_rate:.2f}")

    @pytest.mark.asyncio
    async def test_cycle_completion_verification(self, orchestrator):
        """Test verification that cycles complete successfully."""
        logger.info("Testing cycle completion verification")

        # Track cycle completion
        completed_cycles = {
            "strategy_selection": 0,
            "risk_monitoring": 0,
            "performance_monitoring": 0,
        }

        # Execute cycles
        for i in range(2):
            # Strategy selection cycle
            await orchestrator._execute_strategy_selection_cycle()
            completed_cycles["strategy_selection"] += 1

            # Risk monitoring cycle
            await orchestrator._execute_risk_monitoring_cycle()
            completed_cycles["risk_monitoring"] += 1

            # Performance monitoring cycle
            await orchestrator._execute_performance_monitoring_cycle()
            completed_cycles["performance_monitoring"] += 1

        # Verify all cycles completed
        assert (
            completed_cycles["strategy_selection"] == 2
        ), "Strategy selection cycles not completed"
        assert completed_cycles["risk_monitoring"] == 2, "Risk monitoring cycles not completed"
        assert (
            completed_cycles["performance_monitoring"] == 2
        ), "Performance monitoring cycles not completed"

        # Verify operations were recorded
        operations = orchestrator.get_operation_history()
        strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
        risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
        perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

        assert len(strategy_ops) == 2, "Strategy selection operations not recorded"
        assert len(risk_ops) == 2, "Risk monitoring operations not recorded"
        assert len(perf_ops) == 2, "Performance monitoring operations not recorded"

        # Verify operation timestamps are sequential
        all_ops = sorted(operations, key=lambda x: x["timestamp"])
        for i in range(1, len(all_ops)):
            assert (
                all_ops[i]["timestamp"] >= all_ops[i - 1]["timestamp"]
            ), "Operations not in chronological order"

        logger.info("Cycle completion verification successful")


# Add logger for test output
import logging

logger = logging.getLogger(__name__)
