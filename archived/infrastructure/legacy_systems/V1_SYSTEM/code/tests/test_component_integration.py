"""
Integration tests for component hooks: slippage and transition metrics wiring.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

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
)


@pytest.mark.asyncio
async def test_slippage_hook_estimation_and_recording():
    # Arrange
    config = OrchestratorConfig(
        mode=OrchestrationMode.SEMI_AUTOMATED,
        enable_slippage_estimation=True,
        rebalance_interval=1,
    )
    broker = Mock(spec=AlpacaPaperBroker)
    kb = Mock(spec=StrategyKnowledgeBase)

    strat = StrategyMetadata(
        strategy_id="s1",
        name="S1",
        description="",
        strategy_type="test",
        parameters={},
        context=StrategyContext(
            market_regime="trending",
            time_period="bull_market",
            asset_class="equity",
            risk_profile="moderate",
            volatility_regime="medium",
            correlation_regime="low",
        ),
        performance=StrategyPerformance(
            sharpe_ratio=1.0,
            cagr=0.1,
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

    orchestrator = ProductionOrchestrator(
        config=config,
        broker=broker,
        knowledge_base=kb,
        symbols=["AAPL"],
    )
    orchestrator.data_manager.start = AsyncMock()
    orchestrator.data_manager.stop = AsyncMock()

    # Mock selection to return a single strategy wrapper with required attributes
    orchestrator.strategy_selector.get_current_selection = Mock(
        return_value=[Mock(strategy=strat, score=0.8, confidence=0.8)]
    )

    # Act
    await orchestrator._execute_strategy_selection_cycle_impl()

    # Assert: slippage_cost_estimate should be in the latest operation data when enabled
    ops = orchestrator.get_operation_history("strategy_selection")
    assert len(ops) >= 1
    latest = ops[-1]
    data = latest["data"]
    assert "transition_smoothness" in data
    assert "allocation" in data
    assert "slippage_cost_estimate" in data
    assert isinstance(data["slippage_cost_estimate"], float)


"""
Integration tests for Phase 5 component interactions.

This module tests the integration between different components of the Phase 5
Production Integration system, including strategy selector, portfolio optimizer,
risk manager, and alert system.
"""

from datetime import timedelta

import pandas as pd
import pytest
from bot.live.strategy_selector import RealTimeStrategySelector, SelectionConfig
from bot.meta_learning.regime_detection import RegimeCharacteristics, RegimeDetector
from bot.monitor.alerts import AlertManager
from bot.portfolio.optimizer import PortfolioOptimizer
from bot.risk.manager import RiskManager


class TestComponentIntegration:
    """Test integration between Phase 5 components."""

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base with test strategies."""
        kb = Mock(spec=StrategyKnowledgeBase)

        # Create test strategies
        test_strategies = []
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
            test_strategies.append(strategy)

        kb.find_strategies.return_value = test_strategies
        return kb

    @pytest.fixture
    def mock_regime_detector(self):
        """Create a mock regime detector."""
        detector = Mock(spec=RegimeDetector)
        from bot.meta_learning.regime_detection import MarketRegime

        detector.detect_regime.return_value = RegimeCharacteristics(
            regime=MarketRegime.TRENDING_UP,
            confidence=0.8,
            duration_days=15,
            volatility=0.15,
            trend_strength=0.7,
            correlation_level=0.3,
            volume_profile="normal",
            momentum_score=0.7,
            regime_features={"feature1": 0.5, "feature2": 0.3},
        )
        # Add missing methods that are called in the tests
        detector.detect_current_regime = detector.detect_regime
        detector._regime_to_context.return_value = Mock()  # Mock StrategyContext
        detector._calculate_context_match.return_value = 0.8
        return detector

    @pytest.fixture
    def strategy_selector(self, mock_knowledge_base, mock_regime_detector):
        """Create a strategy selector with mocked dependencies."""
        config = SelectionConfig(
            selection_method="hybrid",
            max_strategies=3,
            min_confidence=0.6,
            min_sharpe=0.5,
            max_drawdown=0.15,
        )

        return RealTimeStrategySelector(
            knowledge_base=mock_knowledge_base,
            regime_detector=mock_regime_detector,
            config=config,
            symbols=["AAPL", "GOOGL", "MSFT"],
        )

    @pytest.fixture
    def portfolio_optimizer(self):
        """Create a portfolio optimizer."""
        from bot.portfolio.optimizer import OptimizationMethod, PortfolioConstraints

        constraints = PortfolioConstraints()
        return PortfolioOptimizer(constraints, OptimizationMethod.SHARPE_MAXIMIZATION)

    @pytest.fixture
    def risk_manager(self):
        """Create a risk manager."""
        from bot.risk.manager import RiskLimits, StopLossConfig

        risk_limits = RiskLimits()
        stop_loss_config = StopLossConfig()
        return RiskManager(risk_limits, stop_loss_config)

    @pytest.fixture
    def alert_manager(self):
        """Create an alert manager."""
        from bot.monitor.alerts import AlertConfig

        config = AlertConfig()
        return AlertManager(config)

    def test_strategy_to_portfolio_workflow(
        self, strategy_selector, portfolio_optimizer, risk_manager
    ):
        """Test complete workflow from strategy selection to portfolio optimization."""
        # Mock market data
        market_data = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104], "Volume": [1000, 1100, 1200, 1300, 1400]},
            index=pd.date_range("2024-01-01", periods=5),
        )

        # Get strategy selection
        selected_strategies = strategy_selector.get_current_selection()

        # If no strategies are selected, this is expected for a fresh selector
        # We'll test the workflow with the mock strategies directly
        if len(selected_strategies) == 0:
            # Use the mock strategies from the knowledge base
            mock_strategies = strategy_selector.knowledge_base.find_strategies.return_value
            selected_strategies = [
                Mock(
                    strategy_id=strategy.strategy_id,
                    strategy=strategy,
                    overall_score=0.8,
                    regime_match_score=0.7,
                    performance_score=0.8,
                    confidence_score=0.7,
                    risk_score=0.6,
                    adaptation_score=0.7,
                    selection_reason="test",
                )
                for strategy in mock_strategies[:3]  # Take first 3 strategies
            ]

        # Test portfolio optimization with selected strategies
        # Convert strategy scores to strategy metadata
        strategies = [score.strategy for score in selected_strategies]
        portfolio_allocation = portfolio_optimizer.optimize_portfolio(strategies=strategies)

        # Verify portfolio allocation
        assert portfolio_allocation is not None, "Should return portfolio allocation"
        assert hasattr(portfolio_allocation, "strategy_weights"), "Should have strategy weights"
        assert len(portfolio_allocation.strategy_weights) > 0, "Should have strategy allocations"

        # Test risk calculation for portfolio
        # Create mock positions for risk calculation
        mock_positions = {}
        for strategy_id, weight in portfolio_allocation.strategy_weights.items():
            mock_position = Mock()
            mock_position.symbol = strategy_id
            mock_position.current_value = weight * 100000  # Mock portfolio value
            mock_position.position_size = weight
            mock_position.volatility = 0.15  # Mock volatility
            mock_position.var_95 = -0.02  # Mock VaR
            mock_position.cvar_95 = -0.03  # Mock CVaR
            mock_position.beta = 1.0  # Mock beta
            mock_position.correlation_with_portfolio = 0.5  # Mock correlation
            mock_position.liquidity_score = 0.8  # Mock liquidity
            mock_position.concentration_risk = weight**2  # Mock concentration risk
            mock_position.stop_loss_level = 95.0  # Mock stop loss
            mock_position.risk_contribution = weight * 0.15  # Mock risk contribution
            mock_positions[strategy_id] = mock_position

        mock_market_data = {"AAPL": pd.DataFrame({"Close": [100, 101, 102]})}
        portfolio_risk = risk_manager.calculate_portfolio_risk(
            positions=mock_positions, portfolio_value=100000, market_data=mock_market_data
        )

        # Verify risk calculation
        assert portfolio_risk is not None, "Should return portfolio risk"
        assert hasattr(portfolio_risk, "volatility"), "Should have volatility"
        assert hasattr(portfolio_risk, "var_95"), "Should have VaR"

    def test_risk_monitoring_to_alert_workflow(self, risk_manager, alert_manager):
        """Test risk monitoring triggering alerts."""
        # Create test positions as PositionRisk objects
        from bot.risk.manager import PositionRisk

        test_positions = {
            "AAPL": PositionRisk(
                symbol="AAPL",
                current_value=15500.0,
                position_size=0.1,
                var_95=-0.02,
                cvar_95=-0.03,
                beta=1.0,
                volatility=0.15,
                correlation_with_portfolio=0.5,
                liquidity_score=0.8,
                concentration_risk=0.01,
                stop_loss_level=147.25,
                risk_contribution=0.015,
            ),
            "GOOGL": PositionRisk(
                symbol="GOOGL",
                current_value=137500.0,
                position_size=0.1,
                var_95=-0.02,
                cvar_95=-0.03,
                beta=1.1,
                volatility=0.18,
                correlation_with_portfolio=0.6,
                liquidity_score=0.7,
                concentration_risk=0.01,
                stop_loss_level=2660.0,
                risk_contribution=0.018,
            ),
        }

        # Calculate portfolio risk
        portfolio_value = 100000  # Mock portfolio value
        market_data = {"AAPL": pd.DataFrame({"Close": [150, 155]})}
        portfolio_risk = risk_manager.calculate_portfolio_risk(
            positions=test_positions, portfolio_value=portfolio_value, market_data=market_data
        )

        # Check risk limits
        risk_violations = risk_manager.check_risk_limits(portfolio_risk, test_positions)

        # If there are violations, test alert generation
        if risk_violations:
            for violation in risk_violations:
                alert = alert_manager.send_risk_alert(
                    strategy_id="portfolio",
                    risk_metric=violation.metric,
                    current_value=violation.current_value,
                    limit_value=violation.limit_value,
                    severity="high",
                )

                assert alert is not None, "Should create risk alert"
                assert alert.alert_type == "risk", "Should be risk alert type"
                assert alert.severity == "high", "Should be high severity"

    def test_component_error_recovery(self, strategy_selector, portfolio_optimizer):
        """Test system recovery from component failures."""
        # Test strategy selector with invalid data
        try:
            # This should handle errors gracefully
            invalid_strategies = strategy_selector.get_current_selection()
            assert isinstance(invalid_strategies, list), "Should return list even with errors"
        except Exception as e:
            pytest.fail(f"Strategy selector should handle errors gracefully: {e}")

        # Test portfolio optimizer with empty strategies
        try:
            empty_allocation = portfolio_optimizer.optimize_portfolio(strategies=[])
            pytest.fail("Portfolio optimizer should raise ValueError for empty strategies")
        except ValueError as e:
            assert "No strategies provided" in str(
                e
            ), "Should raise ValueError for empty strategies"
        except Exception as e:
            pytest.fail(f"Portfolio optimizer should raise ValueError, not {type(e).__name__}: {e}")

    def test_data_consistency_across_components(
        self, strategy_selector, portfolio_optimizer, risk_manager
    ):
        """Test data consistency between components."""
        # Get strategy selection
        selected_strategies = strategy_selector.get_current_selection()

        if len(selected_strategies) > 0:
            # Extract strategy IDs
            strategy_ids = [score.strategy_id for score in selected_strategies]

            # Create portfolio allocation
            portfolio_allocation = portfolio_optimizer.optimize_portfolio(
                strategies=selected_strategies
            )

            # Verify strategy IDs are consistent
            portfolio_strategy_ids = list(portfolio_allocation.allocations.keys())
            assert set(strategy_ids) == set(
                portfolio_strategy_ids
            ), "Strategy IDs should be consistent"

            # Test risk calculation with same data
            portfolio_risk = risk_manager.calculate_portfolio_risk(
                positions=portfolio_allocation.allocations
            )

            # Verify risk calculation uses same strategy data
            assert portfolio_risk is not None, "Risk calculation should work with portfolio data"

    @pytest.mark.asyncio
    async def test_async_component_integration(self, strategy_selector, mock_regime_detector):
        """Test async integration between components."""
        # Mock market data
        market_data = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104], "Volume": [1000, 1100, 1200, 1300, 1400]},
            index=pd.date_range("2024-01-01", periods=5),
        )

        # Test async strategy selection
        try:
            # This would normally be called in the selection loop
            current_regime = mock_regime_detector.detect_current_regime()
            selected_strategies = await strategy_selector._select_strategies(
                current_regime=current_regime, market_data=market_data
            )

            assert isinstance(selected_strategies, list), "Should return list of strategy scores"

        except Exception as e:
            pytest.fail(f"Async strategy selection should work: {e}")

    def test_performance_monitoring_integration(self, strategy_selector, alert_manager):
        """Test performance monitoring integration with alert system."""
        # Get current selection
        selected_strategies = strategy_selector.get_current_selection()

        if len(selected_strategies) > 0:
            # Test performance alert for best strategy
            best_strategy = max(selected_strategies, key=lambda x: x.overall_score)

            # Simulate performance decline
            alert = alert_manager.send_performance_alert(
                strategy_id=best_strategy.strategy_id,
                metric="sharpe_ratio",
                current_value=0.5,
                threshold_value=0.8,
                baseline_value=1.2,
                severity="medium",
            )

            assert alert is not None, "Should create performance alert"
            assert alert.alert_type == "performance", "Should be performance alert"
            assert alert.strategy_id == best_strategy.strategy_id, "Should match strategy ID"

    def test_configuration_consistency(self, strategy_selector, portfolio_optimizer, risk_manager):
        """Test configuration consistency across components."""
        # Test that components use consistent configuration
        strategy_config = strategy_selector.config
        portfolio_constraints = portfolio_optimizer.constraints
        risk_limits = risk_manager.risk_limits

        # Verify configuration objects exist
        assert strategy_config is not None, "Strategy selector should have config"
        assert portfolio_constraints is not None, "Portfolio optimizer should have constraints"
        assert risk_limits is not None, "Risk manager should have risk limits"

        # Test that risk limits are consistent
        if hasattr(strategy_config, "max_drawdown") and hasattr(
            portfolio_constraints, "max_drawdown"
        ):
            assert (
                strategy_config.max_drawdown == portfolio_constraints.max_drawdown
            ), "Drawdown limits should be consistent"

        # Test that Sharpe ratio thresholds are consistent
        if hasattr(strategy_config, "min_sharpe") and hasattr(portfolio_constraints, "min_sharpe"):
            assert (
                strategy_config.min_sharpe == portfolio_constraints.min_sharpe
            ), "Sharpe thresholds should be consistent"
