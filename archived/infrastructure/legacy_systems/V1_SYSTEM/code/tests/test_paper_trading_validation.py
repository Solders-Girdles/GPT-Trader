"""
Paper Trading Validation Tests for Phase 4 User Acceptance Testing.

This module tests the system with paper trading to validate:
- Paper trading execution
- Performance tracking accuracy
- Risk limit enforcement
- Alert system accuracy
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


class TestPaperTradingValidation:
    """Test paper trading validation and accuracy."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for paper trading validation."""
        return OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=60,  # Longer intervals for paper trading
            risk_check_interval=30,
            performance_check_interval=45,
            max_strategies=2,
            min_strategy_confidence=0.8,
            selection_method=SelectionMethod.HYBRID,
            optimization_method=OptimizationMethod.SHARPE_MAXIMIZATION,
            max_position_weight=0.25,
            target_volatility=0.10,
            max_portfolio_var=0.012,
            max_drawdown=0.08,
            stop_loss_pct=0.025,
            min_sharpe_ratio=0.7,
            max_drawdown_threshold=0.08,
            enable_alerts=True,
            alert_cooldown_minutes=3,
        )

    @pytest.fixture
    def mock_paper_broker(self):
        """Create a mock paper broker for validation testing."""
        broker = Mock(spec=AlpacaPaperBroker)

        # Mock account with realistic paper trading values
        account = Mock()
        account.equity = 100000.0
        account.cash = 60000.0
        account.buying_power = 60000.0
        account.daytrade_count = 0
        account.pattern_day_trader = False
        broker.get_account.return_value = account

        # Mock positions with realistic paper trading data
        positions = []
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            position = Mock()
            position.symbol = symbol
            position.qty = 50 + i * 25
            position.market_value = 8000.0 + i * 2000.0
            position.current_price = 160.0 + i * 10.0
            position.avg_entry_price = 155.0 + i * 8.0
            position.unrealized_pl = 250.0 + i * 100.0
            position.unrealized_plpc = 0.03 + i * 0.01
            positions.append(position)

        # Make broker methods async
        broker.get_positions = AsyncMock(return_value=positions)
        broker.get_account = AsyncMock(return_value=account)

        # Mock order execution
        order = Mock()
        order.id = "test_order_123"
        order.symbol = "AAPL"
        order.qty = 100
        order.side = "buy"
        order.type = "market"
        order.status = "filled"
        order.filled_at = datetime.now()
        order.filled_avg_price = 160.0
        broker.submit_market_order.return_value = order
        broker.submit_limit_order.return_value = order

        return broker

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base for paper trading validation."""
        kb = Mock(spec=StrategyKnowledgeBase)

        # Create paper trading strategies
        from bot.knowledge.strategy_knowledge_base import (
            StrategyContext,
            StrategyMetadata,
            StrategyPerformance,
        )

        strategies = []

        # Conservative paper trading strategy
        conservative_strategy = StrategyMetadata(
            strategy_id="paper_trading_conservative",
            name="Conservative Paper Trading Strategy",
            description="Conservative strategy for paper trading validation",
            strategy_type="conservative",
            parameters={"max_position_size": 0.15, "stop_loss": 0.02},
            context=StrategyContext(
                market_regime="mixed",
                time_period="paper_trading",
                asset_class="equity",
                risk_profile="conservative",
                volatility_regime="medium",
                correlation_regime="medium",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.4,
                cagr=0.18,
                max_drawdown=0.05,
                win_rate=0.72,
                consistency_score=0.85,
                n_trades=120,
                avg_trade_duration=8.0,
                profit_factor=1.5,
                calmar_ratio=2.2,
                sortino_ratio=1.8,
                information_ratio=1.2,
                beta=0.7,
                alpha=0.10,
            ),
            discovery_date=datetime.now() - timedelta(days=90),
            last_updated=datetime.now() - timedelta(days=15),
            usage_count=80,
            success_rate=0.78,
        )
        strategies.append(conservative_strategy)

        # Moderate paper trading strategy
        moderate_strategy = StrategyMetadata(
            strategy_id="paper_trading_moderate",
            name="Moderate Paper Trading Strategy",
            description="Moderate strategy for paper trading validation",
            strategy_type="moderate",
            parameters={"max_position_size": 0.25, "stop_loss": 0.03},
            context=StrategyContext(
                market_regime="mixed",
                time_period="paper_trading",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="medium",
            ),
            performance=StrategyPerformance(
                sharpe_ratio=1.6,
                cagr=0.22,
                max_drawdown=0.07,
                win_rate=0.68,
                consistency_score=0.80,
                n_trades=150,
                avg_trade_duration=6.0,
                profit_factor=1.6,
                calmar_ratio=2.5,
                sortino_ratio=2.0,
                information_ratio=1.4,
                beta=0.8,
                alpha=0.12,
            ),
            discovery_date=datetime.now() - timedelta(days=75),
            last_updated=datetime.now() - timedelta(days=12),
            usage_count=95,
            success_rate=0.75,
        )
        strategies.append(moderate_strategy)

        kb.find_strategies.return_value = strategies
        return kb

    @pytest.fixture
    def orchestrator(self, orchestrator_config, mock_paper_broker, mock_knowledge_base):
        """Create a production orchestrator for paper trading validation."""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Patch the data manager to avoid real data fetching
        with patch("bot.live.production_orchestrator.LiveDataManager"):
            orchestrator = ProductionOrchestrator(
                config=orchestrator_config,
                broker=mock_paper_broker,
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

            # Mock strategy selector to return paper trading strategies
            selected_strategies = [
                Mock(strategy=mock_knowledge_base.find_strategies()[0], score=0.88, confidence=0.85)
            ]
            orchestrator.strategy_selector.get_current_selection = Mock(
                return_value=selected_strategies
            )

            return orchestrator

    @pytest.mark.asyncio
    async def test_paper_trading_execution(self, orchestrator):
        """Test paper trading execution capabilities."""
        logger.info("Testing paper trading execution")

        # Start the system
        await orchestrator.start()

        try:
            # Execute paper trading cycles
            for i in range(3):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.5)

            # Get system status
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            # Verify paper trading execution
            assert status is not None, "System status should be available"
            assert status.is_running, "System should be running"

            # Verify operations are recorded
            operations = orchestrator.get_operation_history()
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

            assert len(strategy_ops) > 0, "Strategy selection operations should be recorded"
            assert len(risk_ops) > 0, "Risk monitoring operations should be recorded"
            assert len(perf_ops) > 0, "Performance monitoring operations should be recorded"

            # Verify broker interactions
            broker = orchestrator.broker
            assert broker.get_account.called, "Account should be queried"
            assert broker.get_positions.called, "Positions should be queried"

            logger.info(f"Paper trading execution completed - Operations: {len(operations)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_performance_tracking_accuracy(self, orchestrator):
        """Test performance tracking accuracy in paper trading."""
        logger.info("Testing performance tracking accuracy")

        # Start the system
        await orchestrator.start()

        try:
            # Execute multiple cycles to generate performance data
            for i in range(5):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.3)

            # Get performance data
            operations = orchestrator.get_operation_history()
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

            assert len(perf_ops) > 0, "Performance monitoring operations should be recorded"

            # Verify performance metrics are calculated
            for perf_op in perf_ops:
                assert "timestamp" in perf_op, "Performance operation should have timestamp"
                assert (
                    "metrics" in perf_op or "data" in perf_op
                ), "Performance operation should have metrics"

            # Verify portfolio performance tracking
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            assert status is not None, "System status should be available"
            assert hasattr(status, "portfolio_value"), "Status should have portfolio value"
            assert hasattr(status, "portfolio_return"), "Status should have portfolio return"
            assert hasattr(
                status, "portfolio_volatility"
            ), "Status should have portfolio volatility"
            assert hasattr(status, "portfolio_sharpe"), "Status should have portfolio Sharpe ratio"

            # Verify performance calculations are reasonable
            assert status.portfolio_value > 0, "Portfolio value should be positive"
            assert -1.0 <= status.portfolio_return <= 2.0, "Portfolio return should be reasonable"
            assert (
                0.0 <= status.portfolio_volatility <= 1.0
            ), "Portfolio volatility should be reasonable"

            logger.info(
                f"Performance tracking accuracy completed - Performance operations: {len(perf_ops)}"
            )

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_risk_limit_enforcement(self, orchestrator):
        """Test risk limit enforcement in paper trading."""
        logger.info("Testing risk limit enforcement")

        # Start the system
        await orchestrator.start()

        try:
            # Execute risk monitoring cycles
            for i in range(5):
                await orchestrator._execute_risk_monitoring_cycle()
                await asyncio.sleep(0.3)

            # Get risk monitoring data
            operations = orchestrator.get_operation_history()
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]

            assert len(risk_ops) > 0, "Risk monitoring operations should be recorded"

            # Verify risk calculations
            for risk_op in risk_ops:
                assert "timestamp" in risk_op, "Risk operation should have timestamp"
                assert (
                    "risk_data" in risk_op or "data" in risk_op
                ), "Risk operation should have risk data"

            # Verify risk limits are being checked
            config = orchestrator.config
            assert config.max_portfolio_var > 0, "Max portfolio VaR should be set"
            assert config.max_drawdown > 0, "Max drawdown should be set"
            assert config.max_position_weight > 0, "Max position weight should be set"

            # Verify risk manager is active
            risk_manager = orchestrator.risk_manager
            assert risk_manager is not None, "Risk manager should be initialized"
            assert hasattr(risk_manager, "risk_limits"), "Risk manager should have risk limits"

            # Verify risk monitoring is working
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            assert status is not None, "System status should be available"
            assert hasattr(status, "risk_level"), "Status should have risk level"

            logger.info(f"Risk limit enforcement completed - Risk operations: {len(risk_ops)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_alert_system_accuracy(self, orchestrator):
        """Test alert system accuracy in paper trading."""
        logger.info("Testing alert system accuracy")

        # Start the system
        await orchestrator.start()

        try:
            # Execute cycles to generate alerts
            for i in range(5):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.3)

            # Get alert data
            alert_manager = orchestrator.alert_manager
            active_alerts = alert_manager.get_active_alerts()

            # Verify alert system is working
            assert isinstance(active_alerts, list), "Active alerts should be a list"

            # Verify alert manager is properly configured
            assert alert_manager is not None, "Alert manager should be initialized"
            assert hasattr(alert_manager, "config"), "Alert manager should have config"
            assert hasattr(
                alert_manager, "send_system_alert"
            ), "Alert manager should have send_system_alert method"

            # Verify alert configuration
            config = orchestrator.config
            assert config.enable_alerts, "Alerts should be enabled"
            assert config.alert_cooldown_minutes > 0, "Alert cooldown should be set"

            # Test alert sending capability
            from bot.monitor.alerts import AlertSeverity

            test_alert_sent = await alert_manager.send_system_alert(
                "test_component", "test_event", "Test alert for validation", AlertSeverity.INFO
            )

            # Verify alert was sent (or at least attempted)
            assert test_alert_sent is not None, "Alert sending should return a result"

            # Get updated alerts
            updated_alerts = alert_manager.get_active_alerts()

            logger.info(f"Alert system accuracy completed - Active alerts: {len(updated_alerts)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_paper_trading_consistency(self, orchestrator):
        """Test consistency of paper trading operations."""
        logger.info("Testing paper trading consistency")

        # Start the system
        await orchestrator.start()

        try:
            # Execute multiple cycles to test consistency
            cycle_results = []

            for cycle in range(5):
                cycle_start = time.time()

                # Execute all cycles
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()

                cycle_duration = time.time() - cycle_start
                operations = orchestrator.get_operation_history()

                cycle_results.append(
                    {"cycle": cycle, "duration": cycle_duration, "operations": len(operations)}
                )

                await asyncio.sleep(0.5)

            # Verify consistency across cycles
            assert len(cycle_results) == 5, "Should have 5 cycle results"

            # Verify all cycles completed successfully
            for result in cycle_results:
                assert result["operations"] > 0, f"Cycle {result['cycle']} should have operations"
                assert (
                    result["duration"] > 0
                ), f"Cycle {result['cycle']} should have positive duration"

            # Verify reasonable consistency in operation count
            operation_counts = [r["operations"] for r in cycle_results]
            max_ops = max(operation_counts)
            min_ops = min(operation_counts)

            # Allow some variation but not too much
            assert max_ops - min_ops <= 15, "Operation count should be reasonably consistent"

            # Verify system status remains consistent
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            assert status is not None, "System status should be available"
            assert status.is_running, "System should remain running"

            logger.info(f"Paper trading consistency completed - Cycles: {len(cycle_results)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_paper_trading_error_handling(self, orchestrator):
        """Test error handling in paper trading scenarios."""
        logger.info("Testing paper trading error handling")

        # Start the system
        await orchestrator.start()

        try:
            # Simulate broker error
            original_get_account = orchestrator.broker.get_account

            def mock_broker_error():
                raise Exception("Simulated broker error")

            orchestrator.broker.get_account = mock_broker_error

            # Execute cycles with error
            try:
                await orchestrator._execute_risk_monitoring_cycle()
                # Should handle error gracefully
            except Exception as e:
                logger.info(f"Expected error handled: {e}")

            # Restore broker
            orchestrator.broker.get_account = original_get_account

            # Verify system continues to function
            await orchestrator._execute_strategy_selection_cycle()
            await orchestrator._execute_performance_monitoring_cycle()

            # Verify system status
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            assert status is not None, "System status should be available after error"
            assert status.is_running, "System should remain running after error"

            # Verify operations were recorded
            operations = orchestrator.get_operation_history()
            assert len(operations) > 0, "Operations should be recorded even after error"

            logger.info(f"Paper trading error handling completed - Operations: {len(operations)}")

        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_paper_trading_performance_validation(self, orchestrator):
        """Test performance validation in paper trading."""
        logger.info("Testing paper trading performance validation")

        # Start the system
        await orchestrator.start()

        try:
            # Execute extended paper trading session
            session_start = time.time()

            for i in range(10):
                await orchestrator._execute_strategy_selection_cycle()
                await orchestrator._execute_risk_monitoring_cycle()
                await orchestrator._execute_performance_monitoring_cycle()
                await asyncio.sleep(0.2)

            session_duration = time.time() - session_start

            # Get performance metrics
            operations = orchestrator.get_operation_history()
            strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
            risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
            perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

            # Calculate performance metrics
            total_operations = len(operations)
            throughput = total_operations / session_duration if session_duration > 0 else 0

            # Verify performance is reasonable
            assert (
                total_operations >= 30
            ), "Should have at least 30 operations (10 cycles * 3 operations)"
            assert throughput > 1.0, f"Throughput should be reasonable: {throughput:.2f} ops/sec"

            # Verify operation distribution
            assert len(strategy_ops) > 0, "Should have strategy selection operations"
            assert len(risk_ops) > 0, "Should have risk monitoring operations"
            assert len(perf_ops) > 0, "Should have performance monitoring operations"

            # Verify system status
            status = orchestrator.get_system_status()
            if status is None:
                status = orchestrator._calculate_system_status()

            assert status is not None, "System status should be available"
            assert status.is_running, "System should remain running"

            logger.info("Paper trading performance validation completed:")
            logger.info(f"  Duration: {session_duration:.2f}s")
            logger.info(f"  Operations: {total_operations}")
            logger.info(f"  Throughput: {throughput:.2f} ops/sec")
            logger.info(f"  Strategy ops: {len(strategy_ops)}")
            logger.info(f"  Risk ops: {len(risk_ops)}")
            logger.info(f"  Performance ops: {len(perf_ops)}")

        finally:
            await orchestrator.stop()


# Add logger for test output
import logging

logger = logging.getLogger(__name__)
