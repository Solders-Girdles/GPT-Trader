"""
Unit tests for the Portfolio Optimizer component.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.portfolio.optimizer import (
    OptimizationMethod,
    PortfolioAllocation,
    PortfolioConstraints,
    PortfolioOptimizer,
)


class TestPortfolioOptimizer:
    """Test cases for the PortfolioOptimizer class."""

    @pytest.fixture
    def sample_strategies(self):
        """Create sample strategies for testing."""
        strategies = []
        for i in range(5):
            strategy = StrategyMetadata(
                strategy_id=f"strategy_{i}",
                name=f"Test Strategy {i}",
                description=f"Test strategy {i}",
                strategy_type="trend_following",
                parameters={"param1": i, "param2": i * 2},
                context=StrategyContext(
                    market_regime="trending",
                    time_period="bull_market",
                    asset_class="equity",
                    risk_profile="moderate",
                    volatility_regime="medium",
                    correlation_regime="low",
                ),
                performance=StrategyPerformance(
                    sharpe_ratio=1.0 + i * 0.2,
                    cagr=0.1 + i * 0.02,
                    max_drawdown=0.1 - i * 0.01,
                    win_rate=0.6 + i * 0.02,
                    consistency_score=0.7 + i * 0.02,
                    n_trades=50 + i * 10,
                    avg_trade_duration=5.0,
                    profit_factor=1.3 + i * 0.1,
                    calmar_ratio=1.2 + i * 0.1,
                    sortino_ratio=1.5 + i * 0.1,
                    information_ratio=1.0 + i * 0.1,
                    beta=0.8 + i * 0.05,
                    alpha=0.05 + i * 0.01,
                ),
                discovery_date=datetime.now() - timedelta(days=30),
                last_updated=datetime.now() - timedelta(days=5),
                usage_count=10 + i * 5,
                success_rate=0.7 + i * 0.02,
            )
            strategies.append(strategy)
        return strategies

    @pytest.fixture
    def constraints(self):
        """Create portfolio constraints for testing."""
        return PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.4,
            max_sector_exposure=0.6,
            max_volatility=0.25,
            max_drawdown=0.15,
            target_return=None,
            risk_free_rate=0.02,
        )

    @pytest.fixture
    def optimizer(self, constraints):
        """Create portfolio optimizer instance."""
        return PortfolioOptimizer(
            constraints=constraints, optimization_method=OptimizationMethod.SHARPE_MAXIMIZATION
        )

    def test_initialization(self, optimizer, constraints):
        """Test optimizer initialization."""
        assert optimizer.constraints == constraints
        assert optimizer.optimization_method == OptimizationMethod.SHARPE_MAXIMIZATION
        assert optimizer.last_optimization is None
        assert len(optimizer.optimization_history) == 0

    def test_optimize_portfolio_with_strategies(self, optimizer, sample_strategies):
        """Test portfolio optimization with valid strategies."""
        allocation = optimizer.optimize_portfolio(sample_strategies)

        assert isinstance(allocation, PortfolioAllocation)
        assert len(allocation.strategy_weights) == len(sample_strategies)
        assert abs(sum(allocation.strategy_weights.values()) - 1.0) < 1e-6
        assert allocation.optimization_method == "sharpe_maximization"
        assert allocation.timestamp is not None

        # Check that weights are within bounds
        for weight in allocation.strategy_weights.values():
            assert optimizer.constraints.min_weight <= weight <= optimizer.constraints.max_weight

    def test_optimize_portfolio_empty_strategies(self, optimizer):
        """Test portfolio optimization with empty strategy list."""
        with pytest.raises(ValueError, match="No strategies provided"):
            optimizer.optimize_portfolio([])

    def test_sharpe_maximization_optimization(self, optimizer, sample_strategies):
        """Test Sharpe ratio maximization optimization."""
        optimizer.optimization_method = OptimizationMethod.SHARPE_MAXIMIZATION
        allocation = optimizer.optimize_portfolio(sample_strategies)

        assert allocation.optimization_method == "sharpe_maximization"
        assert allocation.sharpe_ratio > 0  # Should be positive

    def test_risk_parity_optimization(self, optimizer, sample_strategies):
        """Test risk parity optimization."""
        optimizer.optimization_method = OptimizationMethod.RISK_PARITY
        allocation = optimizer.optimize_portfolio(sample_strategies)

        assert allocation.optimization_method == "risk_parity"
        assert allocation.diversification_ratio > 0

    def test_max_diversification_optimization(self, optimizer, sample_strategies):
        """Test maximum diversification optimization."""
        optimizer.optimization_method = OptimizationMethod.MAX_DIVERSIFICATION
        allocation = optimizer.optimize_portfolio(sample_strategies)

        assert allocation.optimization_method == "max_diversification"
        assert allocation.diversification_ratio > 0

    def test_mean_variance_optimization(self, optimizer, sample_strategies):
        """Test mean-variance optimization."""
        optimizer.optimization_method = OptimizationMethod.MEAN_VARIANCE
        allocation = optimizer.optimize_portfolio(sample_strategies)

        assert allocation.optimization_method == "mean_variance"
        assert allocation.expected_volatility > 0

    def test_correlation_matrix_calculation(self, optimizer, sample_strategies):
        """Test correlation matrix calculation."""
        # Mock the _calculate_strategy_statistics method
        with patch.object(optimizer, "_calculate_strategy_statistics") as mock_stats:
            mock_stats.return_value = {
                f"strategy_{i}": {
                    "mean_return": 0.1 + i * 0.01,
                    "volatility": 0.15 + i * 0.02,
                    "beta": 0.8 + i * 0.1,
                    "alpha": 0.05 + i * 0.01,
                    "sharpe_ratio": 1.0 + i * 0.2,
                    "max_drawdown": 0.1 - i * 0.01,
                    "returns": pd.Series(np.random.randn(100) * 0.02 + 0.001),
                }
                for i in range(len(sample_strategies))
            }

            allocation = optimizer.optimize_portfolio(sample_strategies)

            assert isinstance(allocation.correlation_matrix, pd.DataFrame)
            assert allocation.correlation_matrix.shape == (
                len(sample_strategies),
                len(sample_strategies),
            )
            # Check diagonal is 1.0
            np.testing.assert_array_almost_equal(
                np.diag(allocation.correlation_matrix.values), np.ones(len(sample_strategies))
            )

    def test_portfolio_metrics_calculation(self, optimizer, sample_strategies):
        """Test portfolio metrics calculation."""
        allocation = optimizer.optimize_portfolio(sample_strategies)

        assert allocation.expected_return > 0
        assert allocation.expected_volatility > 0
        assert allocation.sharpe_ratio > 0
        assert allocation.max_drawdown > 0
        assert allocation.diversification_ratio > 0
        assert len(allocation.risk_contributions) == len(sample_strategies)

    def test_optimization_history_tracking(self, optimizer, sample_strategies):
        """Test that optimization history is tracked."""
        initial_history_length = len(optimizer.optimization_history)

        allocation = optimizer.optimize_portfolio(sample_strategies)

        assert optimizer.last_optimization == allocation
        assert len(optimizer.optimization_history) == initial_history_length + 1
        assert optimizer.optimization_history[-1] == allocation

    def test_constraint_violation_handling(self, optimizer, sample_strategies):
        """Test constraint violation handling."""
        # Set very restrictive constraints
        optimizer.constraints.max_weight = 0.1  # Very low max weight
        optimizer.constraints.max_volatility = 0.05  # Very low volatility

        allocation = optimizer.optimize_portfolio(sample_strategies)

        # Should still return valid allocation
        assert isinstance(allocation, PortfolioAllocation)
        assert abs(sum(allocation.strategy_weights.values()) - 1.0) < 1e-6

    def test_optimization_with_historical_returns(self, optimizer, sample_strategies):
        """Test optimization with historical returns data."""
        # Create mock historical returns
        historical_returns = pd.DataFrame(
            {
                f"strategy_{i}": np.random.randn(100) * 0.02 + 0.001
                for i in range(len(sample_strategies))
            }
        )

        allocation = optimizer.optimize_portfolio(sample_strategies, historical_returns)

        assert isinstance(allocation, PortfolioAllocation)
        assert len(allocation.strategy_weights) == len(sample_strategies)

    def test_turnover_penalty_and_cap(self, sample_strategies):
        """Verify transaction cost penalty and turnover cap influence objective/solution."""
        constraints = PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.8,
            max_volatility=0.5,
            transaction_cost_bps=50.0,  # penalize turnover
            max_turnover=0.3,  # cap L1 turnover
        )
        optimizer = PortfolioOptimizer(constraints=constraints)

        # First optimize to get a baseline and prev_weights
        base_allocation = optimizer.optimize_portfolio(sample_strategies)
        prev_weights = base_allocation.strategy_weights

        # Force a scenario where equal weights would exceed turnover cap by shifting target
        # Re-run optimization with prev_weights and ensure turnover constraint applied
        new_allocation = optimizer.optimize_portfolio(sample_strategies, prev_weights=prev_weights)

        # Compute realized turnover
        ids = list(prev_weights.keys())
        prev = np.array([prev_weights[i] for i in ids])
        new = np.array([new_allocation.strategy_weights[i] for i in ids])
        turnover = float(np.sum(np.abs(new - prev)))

        # Should not exceed cap significantly (allow tiny numerical tolerance)
        assert turnover <= constraints.max_turnover + 1e-3


class TestPortfolioConstraints:
    """Test cases for the PortfolioConstraints class."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = PortfolioConstraints()

        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 0.4
        assert constraints.max_sector_exposure == 0.6
        assert constraints.max_volatility == 0.25
        assert constraints.max_drawdown == 0.15
        assert constraints.target_return is None
        assert constraints.risk_free_rate == 0.02

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = PortfolioConstraints(
            min_weight=0.05,
            max_weight=0.3,
            max_volatility=0.2,
            target_return=0.1,
            risk_free_rate=0.03,
        )

        assert constraints.min_weight == 0.05
        assert constraints.max_weight == 0.3
        assert constraints.max_volatility == 0.2
        assert constraints.target_return == 0.1
        assert constraints.risk_free_rate == 0.03


class TestPortfolioAllocation:
    """Test cases for the PortfolioAllocation class."""

    def test_allocation_creation(self):
        """Test portfolio allocation creation."""
        strategy_weights = {"strategy_1": 0.5, "strategy_2": 0.5}
        correlation_matrix = pd.DataFrame(
            {"strategy_1": [1.0, 0.3], "strategy_2": [0.3, 1.0]}, index=["strategy_1", "strategy_2"]
        )
        risk_contributions = {"strategy_1": 0.5, "strategy_2": 0.5}

        allocation = PortfolioAllocation(
            strategy_weights=strategy_weights,
            expected_return=0.1,
            expected_volatility=0.15,
            sharpe_ratio=0.67,
            max_drawdown=0.1,
            diversification_ratio=1.2,
            correlation_matrix=correlation_matrix,
            risk_contributions=risk_contributions,
            optimization_method="sharpe_maximization",
            timestamp=datetime.now(),
        )

        assert allocation.strategy_weights == strategy_weights
        assert allocation.expected_return == 0.1
        assert allocation.expected_volatility == 0.15
        assert allocation.sharpe_ratio == 0.67
        assert allocation.max_drawdown == 0.1
        assert allocation.diversification_ratio == 1.2
        assert allocation.optimization_method == "sharpe_maximization"
        assert allocation.timestamp is not None
        assert isinstance(allocation.correlation_matrix, pd.DataFrame)
        assert allocation.risk_contributions == risk_contributions


class TestOptimizationMethod:
    """Test cases for the OptimizationMethod enum."""

    def test_optimization_methods(self):
        """Test all optimization methods."""
        methods = [
            OptimizationMethod.SHARPE_MAXIMIZATION,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.BLACK_LITTERMAN,
            OptimizationMethod.MEAN_VARIANCE,
            OptimizationMethod.MAX_DIVERSIFICATION,
        ]

        for method in methods:
            assert isinstance(method.value, str)
            assert len(method.value) > 0
