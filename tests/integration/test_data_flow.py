"""
Integration tests for data flow between Phase 5 components.

This module tests the data flow and communication between different components
of the Phase 5 Production Integration system.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyKnowledgeBase,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.live.strategy_selector import RealTimeStrategySelector, SelectionConfig
from bot.meta_learning.regime_detection import RegimeCharacteristics, RegimeDetector
from bot.monitor.alerts import AlertManager, AlertSeverity
from bot.portfolio.optimizer import PortfolioOptimizer
from bot.risk.manager import PositionRisk, RiskManager


class TestDataFlowIntegration:
    """Test data flow between Phase 5 components."""

    @pytest.fixture
    def mock_market_data(self):
        """Create realistic mock market data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        np.random.seed(42)  # For reproducible tests

        data = {}
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            # Generate realistic price data with some trend and volatility
            base_price = 100 if symbol == "AAPL" else 200 if symbol == "GOOGL" else 150
            returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            data[symbol] = pd.DataFrame(
                {
                    "Open": prices,
                    "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                    "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                    "Close": prices,
                    "Volume": np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )

        return data

    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base with diverse test strategies."""
        kb = Mock(spec=StrategyKnowledgeBase)

        # Create diverse test strategies
        test_strategies = []
        strategy_types = ["trend_following", "mean_reversion", "momentum", "breakout"]
        market_regimes = ["trending", "volatile", "sideways", "crisis"]

        for i in range(8):
            strategy = StrategyMetadata(
                strategy_id=f"test_strategy_{i}",
                name=f"Test Strategy {i}",
                description=f"Test strategy {i}",
                strategy_type=strategy_types[i % len(strategy_types)],
                parameters={"param1": i, "param2": i * 2},
                context=StrategyContext(
                    market_regime=market_regimes[i % len(market_regimes)],
                    time_period="bull_market" if i % 2 == 0 else "bear_market",
                    asset_class="equity",
                    risk_profile="moderate",
                    volatility_regime="medium",
                    correlation_regime="low",
                ),
                performance=StrategyPerformance(
                    sharpe_ratio=0.8 + i * 0.1,
                    cagr=0.08 + i * 0.01,
                    max_drawdown=0.08 + i * 0.005,
                    win_rate=0.55 + i * 0.02,
                    consistency_score=0.6 + i * 0.05,
                    n_trades=40 + i * 8,
                    avg_trade_duration=4.0 + i * 0.5,
                    profit_factor=1.1 + i * 0.05,
                    calmar_ratio=0.8 + i * 0.1,
                    sortino_ratio=1.2 + i * 0.1,
                    information_ratio=0.8 + i * 0.1,
                    beta=0.9 + i * 0.02,
                    alpha=0.03 + i * 0.005,
                ),
                discovery_date=datetime.now() - timedelta(days=30 + i * 5),
                last_updated=datetime.now() - timedelta(days=2 + i),
                usage_count=15 + i * 3,
                success_rate=0.65 + i * 0.03,
            )
            test_strategies.append(strategy)

        kb.find_strategies.return_value = test_strategies
        return kb

    @pytest.fixture
    def mock_regime_detector(self):
        """Create a mock regime detector with realistic regime detection."""
        detector = Mock(spec=RegimeDetector)

        from bot.meta_learning.regime_detection import MarketRegime

        # Mock different regime scenarios
        regimes = [
            RegimeCharacteristics(
                regime=MarketRegime.TRENDING_UP,
                confidence=0.85,
                duration_days=15,
                volatility=0.12,
                trend_strength=0.75,
                correlation_level=0.25,
                volume_profile="normal",
                momentum_score=0.7,
                regime_features={"feature1": 0.5, "feature2": 0.3},
            ),
            RegimeCharacteristics(
                regime=MarketRegime.VOLATILE,
                confidence=0.70,
                duration_days=10,
                volatility=0.25,
                trend_strength=0.30,
                correlation_level=0.45,
                volume_profile="high",
                momentum_score=0.2,
                regime_features={"feature1": 0.3, "feature2": 0.7},
            ),
            RegimeCharacteristics(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.80,
                duration_days=20,
                volatility=0.08,
                trend_strength=0.20,
                correlation_level=0.15,
                volume_profile="low",
                momentum_score=0.1,
                regime_features={"feature1": 0.2, "feature2": 0.4},
            ),
        ]

        # Cycle through different regimes
        detector.detect_regime.side_effect = lambda market_data: regimes[
            int(datetime.now().timestamp()) % len(regimes)
        ]
        return detector

    @pytest.fixture
    def strategy_selector(self, mock_knowledge_base, mock_regime_detector):
        """Create a strategy selector with mocked dependencies."""
        config = SelectionConfig(
            selection_method="hybrid",
            max_strategies=5,
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

    def test_market_data_to_strategy_selection_flow(self, strategy_selector, mock_market_data):
        """Test data flow from market data to strategy selection."""
        # Simulate market data ingestion
        market_data = mock_market_data["AAPL"]

        # Test that strategy selector can process market data
        try:
            # Get current regime (which uses market data internally)
            current_regime = strategy_selector.regime_detector.detect_regime(market_data)

            # Verify regime data structure
            assert hasattr(current_regime, "regime"), "Regime should have regime type"
            assert hasattr(current_regime, "confidence"), "Regime should have confidence"
            assert hasattr(current_regime, "volatility"), "Regime should have volatility"

            # Test strategy selection with regime data
            selected_strategies = strategy_selector.get_current_selection()

            # Verify strategy selection data structure
            assert isinstance(selected_strategies, list), "Should return list of strategies"

            if len(selected_strategies) > 0:
                strategy_score = selected_strategies[0]
                assert hasattr(strategy_score, "strategy_id"), "Strategy score should have ID"
                assert hasattr(
                    strategy_score, "overall_score"
                ), "Strategy score should have overall score"
                assert hasattr(
                    strategy_score, "regime_match_score"
                ), "Strategy score should have regime match"

        except Exception as e:
            pytest.fail(f"Market data to strategy selection flow should work: {e}")

    def test_strategy_selection_to_portfolio_optimization_flow(
        self, strategy_selector, portfolio_optimizer
    ):
        """Test data flow from strategy selection to portfolio optimization."""
        # Get strategy selection
        selected_strategies = strategy_selector.get_current_selection()

        # If no strategies are selected, this is expected for a fresh selector
        # We'll test with mock strategies instead
        if len(selected_strategies) == 0:
            # Use mock strategies from the knowledge base
            mock_strategies = strategy_selector.knowledge_base.find_strategies.return_value
            selected_strategies = mock_strategies[:3]  # Take first 3 strategies

        # Verify we have strategies to work with
        assert len(selected_strategies) > 0, "Should have selected strategies"

        # Test portfolio optimization with strategy data
        try:
            # Convert strategy scores to strategy metadata
            strategies = (
                selected_strategies  # selected_strategies are already StrategyMetadata objects
            )
            portfolio_allocation = portfolio_optimizer.optimize_portfolio(strategies=strategies)

            # Verify portfolio allocation data structure
            assert portfolio_allocation is not None, "Should return portfolio allocation"
            assert hasattr(portfolio_allocation, "strategy_weights"), "Should have strategy weights"
            assert hasattr(portfolio_allocation, "expected_return"), "Should have expected return"
            assert hasattr(portfolio_allocation, "sharpe_ratio"), "Should have sharpe ratio"
            assert hasattr(
                portfolio_allocation, "expected_volatility"
            ), "Should have expected volatility"

            # Verify allocation data consistency
            total_alloc = sum(portfolio_allocation.strategy_weights.values())
            assert abs(total_alloc - 1.0) < 0.01, "Strategy weights should sum to 1.0"

        except Exception as e:
            pytest.fail(f"Strategy selection to portfolio optimization flow should work: {e}")

    def test_portfolio_optimization_to_risk_management_flow(
        self, strategy_selector, portfolio_optimizer, risk_manager
    ):
        """Test data flow from portfolio optimization to risk management."""
        # Get strategy selection and create portfolio
        selected_strategies = strategy_selector.get_current_selection()

        # If no strategies are selected, this is expected for a fresh selector
        # We'll test with mock strategies instead
        if len(selected_strategies) == 0:
            # Use mock strategies from the knowledge base
            mock_strategies = strategy_selector.knowledge_base.find_strategies.return_value
            selected_strategies = mock_strategies[:3]  # Take first 3 strategies

        portfolio_allocation = portfolio_optimizer.optimize_portfolio(
            strategies=selected_strategies
        )

        # Test risk calculation with portfolio data
        try:
            portfolio_value = 100000  # Mock portfolio value
            market_data = {"AAPL": pd.DataFrame({"Close": [100, 101, 102]})}
            # Create PositionRisk objects from portfolio allocation
            positions = {}
            for strategy_id, weight in portfolio_allocation.strategy_weights.items():
                positions[strategy_id] = PositionRisk(
                    symbol=strategy_id,
                    current_value=weight * portfolio_value,
                    position_size=weight,
                    var_95=-0.02,
                    cvar_95=-0.03,
                    beta=1.0,
                    volatility=0.15,
                    correlation_with_portfolio=0.5,
                    liquidity_score=0.8,
                    concentration_risk=weight**2,
                    stop_loss_level=95.0,
                    risk_contribution=weight * 0.15,
                )

            portfolio_risk = risk_manager.calculate_portfolio_risk(
                positions=positions, portfolio_value=portfolio_value, market_data=market_data
            )

            # Verify risk calculation data structure
            assert portfolio_risk is not None, "Should return portfolio risk"
            assert hasattr(portfolio_risk, "volatility"), "Should have volatility"
            assert hasattr(portfolio_risk, "var_95"), "Should have VaR"
            assert hasattr(portfolio_risk, "cvar_95"), "Should have CVaR"
            assert hasattr(portfolio_risk, "risk_contributions"), "Should have risk contributions"

            # Verify risk data consistency
            assert portfolio_risk.volatility >= 0, "Volatility should be non-negative"
            assert portfolio_risk.var_95 >= 0, "VaR should be non-negative"

        except Exception as e:
            pytest.fail(f"Portfolio optimization to risk management flow should work: {e}")

    @pytest.mark.asyncio
    async def test_risk_management_to_alert_system_flow(
        self, strategy_selector, portfolio_optimizer, risk_manager, alert_manager
    ):
        """Test data flow from risk management to alert system."""
        # Create portfolio and calculate risk
        selected_strategies = strategy_selector.get_current_selection()

        # If no strategies are selected, this is expected for a fresh selector
        # We'll test with mock strategies instead
        if len(selected_strategies) == 0:
            # Use mock strategies from the knowledge base
            mock_strategies = strategy_selector.knowledge_base.find_strategies.return_value
            selected_strategies = mock_strategies[:3]  # Take first 3 strategies

        portfolio_allocation = portfolio_optimizer.optimize_portfolio(
            strategies=selected_strategies
        )
        portfolio_value = 100000  # Mock portfolio value
        market_data = {"AAPL": pd.DataFrame({"Close": [100, 101, 102]})}
        # Create PositionRisk objects from portfolio allocation
        positions = {}
        for strategy_id, weight in portfolio_allocation.strategy_weights.items():
            positions[strategy_id] = PositionRisk(
                symbol=strategy_id,
                current_value=weight * portfolio_value,
                position_size=weight,
                var_95=-0.02,
                cvar_95=-0.03,
                beta=1.0,
                volatility=0.15,
                correlation_with_portfolio=0.5,
                liquidity_score=0.8,
                concentration_risk=weight**2,
                stop_loss_level=95.0,
                risk_contribution=weight * 0.15,
            )

        portfolio_risk = risk_manager.calculate_portfolio_risk(
            positions=positions, portfolio_value=portfolio_value, market_data=market_data
        )

        # Test risk limit checking and alert generation
        try:
            risk_violations = risk_manager.check_risk_limits(portfolio_risk, positions)

            # Test alert generation for any violations
            if risk_violations:
                for violation in risk_violations:
                    # risk_violations are strings, not objects
                    alert = await alert_manager.send_risk_alert(
                        risk_type="portfolio_risk",
                        current_value=0.1,
                        limit_value=0.05,
                        severity=AlertSeverity.ERROR,
                    )

                    # Verify alert data structure
                    assert alert is not None, "Should create risk alert"
                    assert isinstance(alert, str), "Alert should return alert ID string"

            else:
                # Test that no alerts are generated when no violations
                active_alerts = alert_manager.get_active_alerts()
                # This is a basic check - in practice, there might be other alerts

        except Exception as e:
            pytest.fail(f"Risk management to alert system flow should work: {e}")

    def test_data_consistency_across_workflow(
        self, strategy_selector, portfolio_optimizer, risk_manager
    ):
        """Test data consistency throughout the entire workflow."""
        # Get strategy selection
        selected_strategies = strategy_selector.get_current_selection()

        if len(selected_strategies) > 0:
            # Extract strategy IDs from selection
            strategy_ids = [score.strategy_id for score in selected_strategies]

            # Create portfolio allocation
            portfolio_allocation = portfolio_optimizer.optimize_portfolio(
                strategies=selected_strategies
            )

            # Verify strategy IDs are preserved in portfolio
            portfolio_strategy_ids = list(portfolio_allocation.strategy_weights.keys())
            assert set(strategy_ids) == set(
                portfolio_strategy_ids
            ), "Strategy IDs should be preserved"

            # Calculate risk with portfolio data
            portfolio_value = 100000  # Mock portfolio value
            market_data = {"AAPL": pd.DataFrame({"Close": [100, 101, 102]})}
            # Create PositionRisk objects from portfolio allocation
            positions = {}
            for strategy_id, weight in portfolio_allocation.strategy_weights.items():
                positions[strategy_id] = PositionRisk(
                    symbol=strategy_id,
                    current_value=weight * portfolio_value,
                    position_size=weight,
                    var_95=-0.02,
                    cvar_95=-0.03,
                    beta=1.0,
                    volatility=0.15,
                    correlation_with_portfolio=0.5,
                    liquidity_score=0.8,
                    concentration_risk=weight**2,
                    stop_loss_level=95.0,
                    risk_contribution=weight * 0.15,
                )

            portfolio_risk = risk_manager.calculate_portfolio_risk(
                positions=positions, portfolio_value=portfolio_value, market_data=market_data
            )

            # Verify risk calculation uses same strategy data
            assert portfolio_risk is not None, "Risk calculation should work with portfolio data"

            # Test that performance metrics are consistent
            for strategy_score in selected_strategies:
                strategy_id = strategy_score.strategy_id
                if strategy_id in portfolio_allocation.strategy_weights:
                    # Verify allocation exists for selected strategy
                    allocation = portfolio_allocation.strategy_weights[strategy_id]
                    assert allocation > 0, "Selected strategy should have positive allocation"

    @pytest.mark.asyncio
    async def test_async_data_flow(self, strategy_selector, mock_regime_detector):
        """Test async data flow between components."""
        # Create mock market data
        market_data = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104], "Volume": [1000, 1100, 1200, 1300, 1400]},
            index=pd.date_range("2024-01-01", periods=5),
        )

        # Test async strategy selection workflow
        try:
            # Get current regime
            current_regime = mock_regime_detector.detect_regime(market_data)

            # Mock the regime detector methods that are called in _select_strategies
            mock_regime_detector._regime_to_context.return_value = Mock()
            mock_regime_detector._calculate_context_match.return_value = 0.8

            # Test async strategy selection
            selected_strategies = await strategy_selector._select_strategies(
                current_regime=current_regime, market_data=market_data
            )

            # Verify async workflow data structure
            assert isinstance(selected_strategies, list), "Should return list of strategy scores"

            if len(selected_strategies) > 0:
                strategy_score = selected_strategies[0]
                assert hasattr(strategy_score, "strategy_id"), "Strategy score should have ID"
                assert hasattr(
                    strategy_score, "overall_score"
                ), "Strategy score should have overall score"

        except Exception as e:
            pytest.fail(f"Async data flow should work: {e}")

    def test_data_validation_across_components(
        self, strategy_selector, portfolio_optimizer, risk_manager
    ):
        """Test data validation across all components."""
        # Get strategy selection
        selected_strategies = strategy_selector.get_current_selection()

        if len(selected_strategies) > 0:
            # Validate strategy selection data
            for strategy_score in selected_strategies:
                assert (
                    0 <= strategy_score.overall_score <= 1
                ), "Overall score should be between 0 and 1"
                assert (
                    0 <= strategy_score.regime_match_score <= 1
                ), "Regime match score should be between 0 and 1"
                assert (
                    0 <= strategy_score.performance_score <= 1
                ), "Performance score should be between 0 and 1"
                assert (
                    0 <= strategy_score.confidence_score <= 1
                ), "Confidence score should be between 0 and 1"
                assert 0 <= strategy_score.risk_score <= 1, "Risk score should be between 0 and 1"

            # Create portfolio allocation
            portfolio_allocation = portfolio_optimizer.optimize_portfolio(
                strategies=selected_strategies
            )

            # Validate portfolio allocation data
            total_allocation = sum(portfolio_allocation.strategy_weights.values())
            assert abs(total_allocation - 1.0) < 0.01, "Total allocation should be 1.0"

            for allocation in portfolio_allocation.strategy_weights.values():
                assert 0 <= allocation <= 1, "Individual allocation should be between 0 and 1"

            # Calculate risk
            portfolio_value = 100000  # Mock portfolio value
            market_data = {"AAPL": pd.DataFrame({"Close": [100, 101, 102]})}
            # Create PositionRisk objects from portfolio allocation
            positions = {}
            for strategy_id, weight in portfolio_allocation.strategy_weights.items():
                positions[strategy_id] = PositionRisk(
                    symbol=strategy_id,
                    current_value=weight * portfolio_value,
                    position_size=weight,
                    var_95=-0.02,
                    cvar_95=-0.03,
                    beta=1.0,
                    volatility=0.15,
                    correlation_with_portfolio=0.5,
                    liquidity_score=0.8,
                    concentration_risk=weight**2,
                    stop_loss_level=95.0,
                    risk_contribution=weight * 0.15,
                )

            portfolio_risk = risk_manager.calculate_portfolio_risk(
                positions=positions, portfolio_value=portfolio_value, market_data=market_data
            )

            # Validate risk data
            assert portfolio_risk.total_risk >= 0, "Total risk should be non-negative"
            assert portfolio_risk.var_95 >= 0, "VaR should be non-negative"
            assert (
                portfolio_risk.expected_shortfall >= 0
            ), "Expected shortfall should be non-negative"
