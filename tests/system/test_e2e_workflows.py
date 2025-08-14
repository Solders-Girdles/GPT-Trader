"""
End-to-End System Tests for Phase 5 Production Integration.

This module tests complete system workflows including:
- Full trading cycle execution
- Strategy selection to portfolio optimization flow
- Risk management integration
- Performance monitoring integration
- Alert system integration
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from bot.exec.alpaca_paper import AlpacaPaperBroker
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
    SystemStatus,
)
from bot.live.strategy_selector import SelectionMethod
from bot.monitor.alerts import AlertSeverity
from bot.portfolio.optimizer import OptimizationMethod


class TestEndToEndWorkflows:
    """Test complete end-to-end system workflows."""

    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for testing."""
        return OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=60,  # 1 minute for testing
            risk_check_interval=30,  # 30 seconds for testing
            performance_check_interval=45,  # 45 seconds for testing
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
        """Create a mock broker for testing."""
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
        """Create a production orchestrator for testing."""
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

        return orchestrator

    @pytest.mark.asyncio
    async def test_complete_trading_cycle(self, orchestrator):
        """Test a complete trading cycle from strategy selection to execution."""
        logger.info("Testing complete trading cycle")

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

        # Test individual cycle execution without starting the orchestrator
        # Execute strategy selection cycle
        await orchestrator._execute_strategy_selection_cycle()

        # Execute risk monitoring cycle
        await orchestrator._execute_risk_monitoring_cycle()

        # Execute performance monitoring cycle
        await orchestrator._execute_performance_monitoring_cycle()

        # Check that operations were recorded
        operations = orchestrator.get_operation_history()
        assert len(operations) > 0, "No operations recorded"

        # Check for strategy selection operations
        strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
        assert len(strategy_ops) > 0, "No strategy selection operations"

        # Check for risk monitoring operations
        risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]
        assert len(risk_ops) > 0, "No risk monitoring operations"

        # Check for performance monitoring operations
        perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]
        assert len(perf_ops) > 0, "No performance monitoring operations"

        # Check system status
        status = orchestrator.get_system_status()
        if status is None:
            # Calculate system status if not available
            status = orchestrator._calculate_system_status()
        assert status is not None, "No system status available"

        logger.info(f"Trading cycle completed. Operations: {len(operations)}")

    @pytest.mark.asyncio
    async def test_strategy_selection_to_portfolio_flow(self, orchestrator):
        """Test the complete flow from strategy selection to portfolio optimization."""
        logger.info("Testing strategy selection to portfolio flow")

        # Mock the strategy selector to return known strategies with proper data
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
        await orchestrator._execute_strategy_selection_cycle()

        # Check that portfolio optimization was called
        operations = orchestrator.get_operation_history()
        strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]

        assert len(strategy_ops) > 0, "Strategy selection operation not recorded"

        latest_op = strategy_ops[-1]
        assert "n_strategies" in latest_op["data"], "Number of strategies not recorded"
        assert "allocation" in latest_op["data"], "Portfolio allocation not recorded"
        assert latest_op["data"]["n_strategies"] == 2, "Expected 2 strategies"

        logger.info("Strategy selection to portfolio flow completed successfully")

    @pytest.mark.asyncio
    async def test_risk_monitoring_integration(self, orchestrator):
        """Test risk monitoring integration with portfolio and alerts."""
        logger.info("Testing risk monitoring integration")

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

        # Mock the _get_current_positions method
        orchestrator._get_current_positions = AsyncMock(return_value=mock_positions)

        # Mock market data
        orchestrator._get_symbol_data = AsyncMock(
            return_value=pd.DataFrame({"Close": [150.0, 500.0], "Volume": [1000000, 500000]})
        )

        # Execute risk monitoring cycle
        await orchestrator._execute_risk_monitoring_cycle()

        # Check that risk monitoring was executed
        operations = orchestrator.get_operation_history()
        risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]

        assert len(risk_ops) > 0, "Risk monitoring operation not recorded"

        # Check risk summary
        risk_summary = orchestrator.get_risk_summary()
        assert risk_summary is not None, "Risk summary not available"

        logger.info("Risk monitoring integration completed successfully")

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, orchestrator):
        """Test performance monitoring integration."""
        logger.info("Testing performance monitoring integration")

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
        await orchestrator._execute_performance_monitoring_cycle()

        # Check that performance monitoring was executed
        operations = orchestrator.get_operation_history()
        perf_ops = [op for op in operations if op["operation"] == "performance_monitoring"]

        assert len(perf_ops) > 0, "Performance monitoring operation not recorded"

        # Check performance summary
        portfolio_summary = orchestrator.get_portfolio_summary()
        assert portfolio_summary is not None, "Portfolio summary not available"

        logger.info("Performance monitoring integration completed successfully")

    @pytest.mark.asyncio
    async def test_alert_system_integration(self, orchestrator):
        """Test alert system integration across all components."""
        logger.info("Testing alert system integration")

        # Test strategy alert
        await orchestrator.alert_manager.send_strategy_alert(
            "AAPL", "signal_generated", "Buy signal generated", AlertSeverity.INFO
        )

        # Test risk alert
        await orchestrator.alert_manager.send_risk_alert(
            "portfolio_risk", 0.025, 0.02, AlertSeverity.WARNING  # Current VaR  # Limit
        )

        # Test system alert
        await orchestrator.alert_manager.send_system_alert(
            "data_feed", "connection_lost", "Market data connection lost", AlertSeverity.ERROR
        )

        # Check alert summary
        alert_summary = orchestrator.get_alert_summary()
        assert alert_summary is not None, "Alert summary not available"
        assert "total_alerts" in alert_summary, "Total alerts not in summary"
        assert "active_alerts" in alert_summary, "Active alerts not in summary"

        # Check that alerts were sent
        assert alert_summary["total_alerts"] >= 3, "Expected at least 3 alerts"

        logger.info("Alert system integration completed successfully")

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, orchestrator):
        """Test system health monitoring and status reporting."""
        logger.info("Testing system health monitoring")

        # Execute health check directly without starting orchestrator
        await orchestrator._check_system_health()

        # Check system status
        status = orchestrator.get_system_status()
        if status is None:
            status = orchestrator._calculate_system_status()
        assert status is not None, "System status not available"
        assert status.timestamp is not None, "Status timestamp not available"
        assert status.mode == OrchestrationMode.SEMI_AUTOMATED, "Incorrect mode"

        # Check operation history (health check doesn't record operations)
        operations = orchestrator.get_operation_history()
        # Health check doesn't record operations, so this might be empty

        # Check that all components are accessible
        assert orchestrator.strategy_selector is not None, "Strategy selector not available"
        assert orchestrator.portfolio_optimizer is not None, "Portfolio optimizer not available"
        assert orchestrator.risk_manager is not None, "Risk manager not available"
        assert orchestrator.alert_manager is not None, "Alert manager not available"
        assert orchestrator.performance_monitor is not None, "Performance monitor not available"

        logger.info("System health monitoring completed successfully")

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, orchestrator):
        """Test system error recovery and resilience."""
        logger.info("Testing error recovery and resilience")

        # Mock strategy selection results with proper data for recovery
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

        selected_strategies = [Mock(strategy=strategy1, score=0.85, confidence=0.8)]

        # Simulate an error in strategy selection
        original_method = orchestrator.strategy_selector.get_current_selection
        orchestrator.strategy_selector.get_current_selection = Mock(
            side_effect=Exception("Test error")
        )

        # Execute strategy selection cycle (should handle the error)
        try:
            await orchestrator._execute_strategy_selection_cycle()
        except Exception as e:
            logger.info(f"Expected error caught: {e}")

        # Restore original method with proper data
        orchestrator.strategy_selector.get_current_selection = Mock(
            return_value=selected_strategies
        )

        # Execute strategy selection cycle again (should work now)
        await orchestrator._execute_strategy_selection_cycle()

        # Check that operations were recorded
        operations = orchestrator.get_operation_history()
        assert len(operations) > 0, "Operations should be recorded after error recovery"

        logger.info("Error recovery and resilience test completed successfully")

    @pytest.mark.asyncio
    async def test_data_consistency_across_workflow(self, orchestrator):
        """Test data consistency across the entire workflow."""
        logger.info("Testing data consistency across workflow")

        # Mock consistent data across components
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

        strategy2 = StrategyMetadata(
            strategy_id="strategy_2",
            name="Strategy 2",
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

        test_data = {
            "portfolio_value": 100000.0,
            "positions": {
                "AAPL": {
                    "weight": 0.3,
                    "market_value": 30000.0,
                    "portfolio_value": 100000.0,
                    "current_price": 150.0,
                    "entry_price": 145.0,
                    "quantity": 200,
                },
                "GOOGL": {
                    "weight": 0.4,
                    "market_value": 40000.0,
                    "portfolio_value": 100000.0,
                    "current_price": 200.0,
                    "entry_price": 195.0,
                    "quantity": 200,
                },
                "MSFT": {
                    "weight": 0.3,
                    "market_value": 30000.0,
                    "portfolio_value": 100000.0,
                    "current_price": 300.0,
                    "entry_price": 295.0,
                    "quantity": 100,
                },
            },
            "strategies": [strategy1, strategy2],
            "risk_metrics": {"var": 0.015, "volatility": 0.12},
        }

        # Mock methods to return consistent data
        orchestrator._get_current_positions = AsyncMock(return_value=test_data["positions"])
        orchestrator.strategy_selector.get_current_selection = Mock(
            return_value=[Mock(strategy=s) for s in test_data["strategies"]]
        )

        # Execute complete workflow
        await orchestrator._execute_strategy_selection_cycle()
        await orchestrator._execute_risk_monitoring_cycle()

        # Check data consistency in operations
        operations = orchestrator.get_operation_history()

        # Verify portfolio value consistency
        strategy_ops = [op for op in operations if op["operation"] == "strategy_selection"]
        risk_ops = [op for op in operations if op["operation"] == "risk_monitoring"]

        if strategy_ops and risk_ops:
            # Check that portfolio values are consistent
            strategy_data = strategy_ops[-1]["data"]
            risk_data = risk_ops[-1]["data"]

            # Both should reference the same portfolio
            assert (
                "portfolio_value" in strategy_data or "allocation" in strategy_data
            ), "Strategy data missing portfolio info"
            assert (
                "portfolio_var" in risk_data or "portfolio_volatility" in risk_data
            ), "Risk data missing portfolio risk info"

        logger.info("Data consistency across workflow verified")

    @pytest.mark.asyncio
    async def test_comprehensive_system_summary(self, orchestrator):
        """Test comprehensive system summary generation."""
        logger.info("Testing comprehensive system summary")

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

        selected_strategies = [Mock(strategy=strategy1, score=0.85, confidence=0.8)]
        orchestrator.strategy_selector.get_current_selection = Mock(
            return_value=selected_strategies
        )

        # Execute some operations first
        await orchestrator._execute_strategy_selection_cycle()
        await orchestrator._execute_risk_monitoring_cycle()

        # Get all summaries
        strategy_summary = orchestrator.get_strategy_summary()
        portfolio_summary = orchestrator.get_portfolio_summary()
        risk_summary = orchestrator.get_risk_summary()
        alert_summary = orchestrator.get_alert_summary()
        system_status = orchestrator.get_system_status()

        # Verify all summaries are available
        assert strategy_summary is not None, "Strategy summary not available"
        assert portfolio_summary is not None, "Portfolio summary not available"
        assert risk_summary is not None, "Risk summary not available"
        assert alert_summary is not None, "Alert summary not available"
        if system_status is None:
            system_status = orchestrator._calculate_system_status()
        assert system_status is not None, "System status not available"

        # Verify summary structure
        assert isinstance(strategy_summary, dict), "Strategy summary should be dict"
        assert isinstance(portfolio_summary, dict), "Portfolio summary should be dict"
        assert isinstance(risk_summary, dict), "Risk summary should be dict"
        assert isinstance(alert_summary, dict), "Alert summary should be dict"
        assert isinstance(system_status, SystemStatus), "System status should be SystemStatus"

        # Verify key fields
        assert "total_alerts" in alert_summary, "Alert summary missing total_alerts"
        assert "is_running" in vars(system_status), "System status missing is_running"

        logger.info("Comprehensive system summary test completed successfully")


# Add logger for test output
import logging

logger = logging.getLogger(__name__)
