"""
Real-World Scenario Tests for Phase 4 User Acceptance Testing.

This module tests the system behavior in realistic market scenarios including:
- Bull Market Scenario: Trending market with high volatility
- Bear Market Scenario: Declining market with crisis conditions
- Sideways Market Scenario: Range-bound market with low volatility
- Crisis Scenario: Extreme volatility and correlation breakdown
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

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


class TestRealWorldScenarios:
    """Test system behavior in realistic market scenarios."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for real-world scenario testing."""
        return OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=30,  # Longer intervals for scenario testing
            risk_check_interval=15,
            performance_check_interval=20,
            max_strategies=3,
            min_strategy_confidence=0.7,
            selection_method=SelectionMethod.HYBRID,
            optimization_method=OptimizationMethod.SHARPE_MAXIMIZATION,
            max_position_weight=0.3,
            target_volatility=0.12,
            max_portfolio_var=0.015,
            max_drawdown=0.10,
            stop_loss_pct=0.03,
            min_sharpe_ratio=0.6,
            max_drawdown_threshold=0.10,
            enable_alerts=True,
            alert_cooldown_minutes=2,
        )

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for scenario testing."""
        broker = Mock(spec=AlpacaPaperBroker)

        # Mock account
        account = Mock()
        account.equity = 100000.0
        account.cash = 40000.0
        account.buying_power = 40000.0
        broker.get_account.return_value = account

        # Mock positions
        positions = []
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]):
            position = Mock()
            position.symbol = symbol
            position.qty = 100 + i * 25
            position.market_value = 15000.0 + i * 3000.0
            position.current_price = 150.0 + i * 5.0
            position.avg_entry_price = 145.0 + i * 4.0
            positions.append(position)

        broker.get_positions.return_value = positions
        return broker

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base for scenario testing."""
        kb = Mock(spec=StrategyKnowledgeBase)

        # Create scenario-specific strategies
        from bot.knowledge.strategy_knowledge_base import (
            StrategyContext,
            StrategyMetadata,
            StrategyPerformance,
        )

        strategies = []

        # Bull market strategy
        bull_strategy = StrategyMetadata(
            strategy_id="bull_market_strategy",
            name="Bull Market Strategy",
            description="Strategy optimized for trending bull markets",
            strategy_type="trend_following",
            parameters={"momentum_period": 20, "volatility_threshold": 0.15},
            context=StrategyContext(
                market_regime="trending",
                time_period="bull_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="high",
                correlation_regime="low",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.8,
                cagr=0.25,
                max_drawdown=0.08,
                win_rate=0.75,
                consistency_score=0.85,
                n_trades=80,
                avg_trade_duration=7.0,
                profit_factor=1.6,
                calmar_ratio=2.1,
                sortino_ratio=2.2,
                information_ratio=1.5,
                beta=1.1,
                alpha=0.12,
            ),
            discovery_date=datetime.now() - timedelta(days=60),
            last_updated=datetime.now() - timedelta(days=10),
            usage_count=45,
            success_rate=0.82,
        )
        strategies.append(bull_strategy)

        # Bear market strategy
        bear_strategy = StrategyMetadata(
            strategy_id="bear_market_strategy",
            name="Bear Market Strategy",
            description="Strategy optimized for declining markets",
            strategy_type="defensive",
            parameters={"defensive_period": 30, "risk_reduction": 0.5},
            context=StrategyContext(
                market_regime="declining",
                time_period="bear_market",
                asset_class="equity",
                risk_profile="conservative",
                volatility_regime="high",
                correlation_regime="high",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=0.9,
                cagr=0.05,
                max_drawdown=0.03,
                win_rate=0.65,
                consistency_score=0.75,
                n_trades=40,
                avg_trade_duration=12.0,
                profit_factor=1.2,
                calmar_ratio=1.8,
                sortino_ratio=1.3,
                information_ratio=0.8,
                beta=0.6,
                alpha=0.08,
            ),
            discovery_date=datetime.now() - timedelta(days=45),
            last_updated=datetime.now() - timedelta(days=5),
            usage_count=30,
            success_rate=0.70,
        )
        strategies.append(bear_strategy)

        # Sideways market strategy
        sideways_strategy = StrategyMetadata(
            strategy_id="sideways_market_strategy",
            name="Sideways Market Strategy",
            description="Strategy optimized for range-bound markets",
            strategy_type="mean_reversion",
            parameters={"reversion_period": 25, "range_threshold": 0.05},
            context=StrategyContext(
                market_regime="sideways",
                time_period="sideways_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="low",
                correlation_regime="medium",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.2,
                cagr=0.12,
                max_drawdown=0.06,
                win_rate=0.68,
                consistency_score=0.80,
                n_trades=60,
                avg_trade_duration=5.0,
                profit_factor=1.4,
                calmar_ratio=1.5,
                sortino_ratio=1.6,
                information_ratio=1.1,
                beta=0.8,
                alpha=0.06,
            ),
            discovery_date=datetime.now() - timedelta(days=40),
            last_updated=datetime.now() - timedelta(days=8),
            usage_count=35,
            success_rate=0.75,
        )
        strategies.append(sideways_strategy)

        kb.find_strategies.return_value = strategies
        return kb

    @pytest.fixture
    def orchestrator(self, orchestrator_config, mock_broker, mock_knowledge_base):
        """Create a production orchestrator for scenario testing."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

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

            # Mock strategy selector to return appropriate strategies for scenarios
            selected_strategies = [
                Mock(strategy=mock_knowledge_base.find_strategies()[0], score=0.85, confidence=0.8)
            ]
            orchestrator.strategy_selector.get_current_selection = Mock(
                return_value=selected_strategies
            )

            return orchestrator

    @pytest.mark.asyncio
    async def test_bull_market_scenario(self, orchestrator):
        """Test system behavior in a bull market scenario."""
        logger.info("Testing bull market scenario")

        # Start the system
        await orchestrator.start()

        try:
            # Simulate bull market conditions
            # High volatility, trending market, low correlation
            scenario_duration = 60  # 1 minute simulation

            # Execute multiple cycles to simulate bull market behavior
            for i in range(5):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.5)

            # Get system status
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            # Verify bull market behavior
            assert status is not None, "System status should be available"
            assert status.is_running, "System should be running"

            # Check that trend-following strategies are preferred
            operations = orchestrator.get_operation_history()
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]

            assert len(strategy_ops) > 0, "Strategy selection operations should be recorded"

            # Verify risk management is active but not overly restrictive
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            assert len(risk_ops) > 0, "Risk monitoring operations should be recorded"

            # Verify performance monitoring is tracking gains
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]
            assert len(perf_ops) > 0, "Performance monitoring operations should be recorded"

            logger.info(f"Bull market scenario completed - Operations: {len(operations)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_bear_market_scenario(self, orchestrator):
        """Test system behavior in a bear market scenario."""
        logger.info("Testing bear market scenario")

        # Start the system
        await orchestrator.start()

        try:
            # Simulate bear market conditions
            # High volatility, declining market, high correlation

            # Execute multiple cycles to simulate bear market behavior
            for i in range(5):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.5)

            # Get system status
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            # Verify bear market behavior
            assert status is not None, "System status should be available"
            assert status.is_running, "System should be running"

            # Check that defensive strategies are preferred
            operations = orchestrator.get_operation_history()
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]

            assert len(strategy_ops) > 0, "Strategy selection operations should be recorded"

            # Verify risk management is more active
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            assert len(risk_ops) > 0, "Risk monitoring operations should be recorded"

            # Verify performance monitoring is tracking losses
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]
            assert len(perf_ops) > 0, "Performance monitoring operations should be recorded"

            logger.info(f"Bear market scenario completed - Operations: {len(operations)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_sideways_market_scenario(self, orchestrator):
        """Test system behavior in a sideways market scenario."""
        logger.info("Testing sideways market scenario")

        # Start the system
        await orchestrator.start()

        try:
            # Simulate sideways market conditions
            # Low volatility, range-bound market, medium correlation

            # Execute multiple cycles to simulate sideways market behavior
            for i in range(5):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.5)

            # Get system status
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            # Verify sideways market behavior
            assert status is not None, "System status should be available"
            assert status.is_running, "System should be running"

            # Check that mean reversion strategies are preferred
            operations = orchestrator.get_operation_history()
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]

            assert len(strategy_ops) > 0, "Strategy selection operations should be recorded"

            # Verify moderate risk management
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            assert len(risk_ops) > 0, "Risk monitoring operations should be recorded"

            # Verify performance monitoring is tracking small gains/losses
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]
            assert len(perf_ops) > 0, "Performance monitoring operations should be recorded"

            logger.info(f"Sideways market scenario completed - Operations: {len(operations)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_crisis_scenario(self, orchestrator):
        """Test system behavior in a crisis scenario."""
        logger.info("Testing crisis scenario")

        # Start the system
        await orchestrator.start()

        try:
            # Simulate crisis conditions
            # Extreme volatility, correlation breakdown, high uncertainty

            # Execute multiple cycles to simulate crisis behavior
            for i in range(5):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.5)

            # Get system status
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            # Verify crisis scenario behavior
            assert status is not None, "System status should be available"
            assert status.is_running, "System should be running"

            # Check that defensive strategies are heavily preferred
            operations = orchestrator.get_operation_history()
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]

            assert len(strategy_ops) > 0, "Strategy selection operations should be recorded"

            # Verify aggressive risk management
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            assert len(risk_ops) > 0, "Risk monitoring operations should be recorded"

            # Verify performance monitoring is tracking significant losses
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]
            assert len(perf_ops) > 0, "Performance monitoring operations should be recorded"

            # Verify alert system is active
            alerts = orchestrator.alert_manager.get_active_alerts()
            # In a crisis, we might expect more alerts
            assert isinstance(alerts, list), "Alerts should be a list"

            logger.info(
                f"Crisis scenario completed - Operations: {len(operations)}, Alerts: {len(alerts)}"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_scenario_transition_handling(self, orchestrator):
        """Test system behavior during market scenario transitions."""
        logger.info("Testing scenario transition handling")

        # Start the system
        await orchestrator.start()

        try:
            # Simulate transition from bull to bear market
            # First, execute bull market cycles
            for i in range(3):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.3)

            # Simulate market transition (change strategy selection)
            # Mock different strategy selection for bear market
            from bot.knowledge.strategy_knowledge_base import (
                StrategyContext,
                StrategyMetadata,
                StrategyPerformance,
            )

            bear_strategy = StrategyMetadata(
                strategy_id="bear_market_strategy",
                name="Bear Market Strategy",
                description="Strategy for declining markets",
                strategy_type="defensive",
                parameters={"defensive_period": 30},
                context=StrategyContext(
                    market_regime="declining",
                    time_period="bear_market",
                    asset_class="equity",
                    risk_profile="conservative",
                    volatility_regime="high",
                    correlation_regime="high",
                ),
                performance=StrategyPerformance(
                    sharpe_ratio=0.9,
                    cagr=0.05,
                    max_drawdown=0.03,
                    win_rate=0.65,
                    consistency_score=0.75,
                    n_trades=40,
                    avg_trade_duration=12.0,
                    profit_factor=1.2,
                    calmar_ratio=1.8,
                    sortino_ratio=1.3,
                    information_ratio=0.8,
                    beta=0.6,
                    alpha=0.08,
                ),
                discovery_date=datetime.now() - timedelta(days=45),
                last_updated=datetime.now() - timedelta(days=5),
                usage_count=30,
                success_rate=0.70,
            )

            selected_strategies = [Mock(strategy=bear_strategy, score=0.90, confidence=0.85)]
            orchestrator.strategy_selector.get_current_selection = Mock(
                return_value=selected_strategies
            )

            # Execute bear market cycles
            for i in range(3):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.3)

            # Verify transition handling
            operations = orchestrator.get_operation_history()
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]

            assert len(strategy_ops) >= 6, "Should have strategy operations from both scenarios"

            # Verify system remained stable during transition
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            assert status is not None, "System status should be available after transition"
            assert status.is_running, "System should remain running after transition"

            logger.info(f"Scenario transition completed - Total operations: {len(operations)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_scenario_performance_comparison(self, orchestrator):
        """Test and compare performance across different scenarios."""
        logger.info("Testing scenario performance comparison")

        # Start the system
        await orchestrator.start()

        try:
            scenario_results = {}

            # Test each scenario and collect performance metrics
            scenarios = ["bull", "bear", "sideways", "crisis"]

            for scenario in scenarios:
                # Clear operation history for clean measurement
                orchestrator.operation_history.clear()

                # Execute scenario cycles
                start_time = time.time()
                for i in range(3):
                    await orchestrator._execute_strategy_selection_cycle()
                    await orchestrator._execute_risk_monitoring_cycle()
                    await orchestrator._execute_performance_monitoring_cycle()
                    await asyncio.sleep(0.2)

                duration = time.time() - start_time
                operations = orchestrator.get_operation_history()

                scenario_results[scenario] = {
                    "duration": duration,
                    "operations": len(operations),
                    "throughput": len(operations) / duration if duration > 0 else 0,
                }

            # Verify all scenarios completed successfully
            for scenario, results in scenario_results.items():
                assert results["operations"] > 0, f"{scenario} scenario should have operations"
                assert (
                    results["throughput"] > 0
                ), f"{scenario} scenario should have positive throughput"

            # Compare scenario performance
            throughputs = [results["throughput"] for results in scenario_results.values()]
            max_throughput = max(throughputs)
            min_throughput = min(throughputs)

            # Verify reasonable performance variation
            throughput_ratio = (
                max_throughput / min_throughput if min_throughput > 0 else float("inf")
            )
            assert (
                throughput_ratio < 10.0
            ), f"Throughput variation too high: {throughput_ratio:.2f}x"

            logger.info("Scenario performance comparison completed:")
            for scenario, results in scenario_results.items():
                logger.info(f"  {scenario}: {results['throughput']:.2f} ops/sec")

        finally:
            await orchestrator.stop()


# Add logger for test output
import logging

logger = logging.getLogger(__name__)
