"""
Portfolio Optimization System for Phase 5 Production Integration.
Optimizes portfolio of multiple strategies with risk-adjusted returns, correlation minimization, and drawdown control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from bot.analytics.alpha_analysis import AlphaGenerationAnalyzer
from bot.analytics.risk_decomposition import RiskDecompositionAnalyzer
from bot.knowledge.strategy_knowledge_base import StrategyMetadata
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""

    SHARPE_MAXIMIZATION = "sharpe_maximization"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_VARIANCE = "mean_variance"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio optimization."""

    min_weight: float = 0.0
    max_weight: float = 0.4  # No single strategy > 40%
    max_sector_exposure: float = 0.6
    max_volatility: float = 0.25
    max_drawdown: float = 0.15
    target_return: float | None = None
    risk_free_rate: float = 0.02
    # Phase 2 enhancements
    transaction_cost_bps: float = 0.0  # cost per unit turnover (round-trip approx)
    max_turnover: float | None = (
        None  # optional turnover cap per rebalance (L1 norm of weight changes)
    )


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""

    strategy_weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float
    correlation_matrix: pd.DataFrame
    risk_contributions: dict[str, float]
    optimization_method: str
    timestamp: datetime


class PortfolioOptimizer:
    """Portfolio optimization system for multiple strategies."""

    def __init__(
        self,
        constraints: PortfolioConstraints,
        optimization_method: OptimizationMethod = OptimizationMethod.SHARPE_MAXIMIZATION,
    ) -> None:
        self.constraints = constraints
        self.optimization_method = optimization_method

        # Initialize analyzers
        self.risk_analyzer = RiskDecompositionAnalyzer()
        self.alpha_analyzer = AlphaGenerationAnalyzer()

        # Optimization state
        self.last_optimization: PortfolioAllocation | None = None
        self.optimization_history: list[PortfolioAllocation] = []

        logger.info(f"Portfolio optimizer initialized with method: {optimization_method.value}")

    def optimize_portfolio(
        self,
        strategies: list[StrategyMetadata],
        historical_returns: pd.DataFrame | None = None,
        prev_weights: dict[str, float] | None = None,
    ) -> PortfolioAllocation:
        """Optimize portfolio allocation for given strategies."""

        if not strategies:
            raise ValueError("No strategies provided for optimization")

        logger.info(f"Optimizing portfolio for {len(strategies)} strategies")

        # Calculate strategy statistics
        strategy_stats = self._calculate_strategy_statistics(strategies, historical_returns)

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(strategy_stats)

        # Run optimization based on method
        if self.optimization_method == OptimizationMethod.SHARPE_MAXIMIZATION:
            weights = self._optimize_sharpe_maximization(
                strategy_stats, correlation_matrix, prev_weights
            )
        elif self.optimization_method == OptimizationMethod.RISK_PARITY:
            weights = self._optimize_risk_parity(strategy_stats, correlation_matrix)
        elif self.optimization_method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._optimize_max_diversification(strategy_stats, correlation_matrix)
        else:
            weights = self._optimize_mean_variance(strategy_stats, correlation_matrix)

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            weights, strategy_stats, correlation_matrix
        )

        # Create allocation result
        allocation = PortfolioAllocation(
            strategy_weights=weights,
            expected_return=portfolio_metrics["expected_return"],
            expected_volatility=portfolio_metrics["expected_volatility"],
            sharpe_ratio=portfolio_metrics["sharpe_ratio"],
            max_drawdown=portfolio_metrics["max_drawdown"],
            diversification_ratio=portfolio_metrics["diversification_ratio"],
            correlation_matrix=correlation_matrix,
            risk_contributions=portfolio_metrics["risk_contributions"],
            optimization_method=self.optimization_method.value,
            timestamp=datetime.now(),
        )

        # Update state
        self.last_optimization = allocation
        self.optimization_history.append(allocation)

        logger.info(f"Portfolio optimization completed. Sharpe: {allocation.sharpe_ratio:.3f}")

        return allocation

    def _calculate_strategy_statistics(
        self, strategies: list[StrategyMetadata], historical_returns: pd.DataFrame | None
    ) -> dict[str, dict[str, float]]:
        """Calculate statistics for each strategy."""
        strategy_stats = {}

        for strategy in strategies:
            perf = strategy.performance

            # Use historical returns if available, otherwise use performance metrics
            if (
                historical_returns is not None
                and strategy.strategy_id in historical_returns.columns
            ):
                returns = historical_returns[strategy.strategy_id].dropna()
                if len(returns) > 0:
                    strategy_stats[strategy.strategy_id] = {
                        "mean_return": returns.mean() * 252,  # Annualized
                        "volatility": returns.std() * np.sqrt(252),  # Annualized
                        "sharpe_ratio": (returns.mean() * 252 - self.constraints.risk_free_rate)
                        / (returns.std() * np.sqrt(252)),
                        "max_drawdown": self._calculate_max_drawdown(returns),
                        "beta": perf.beta,
                        "alpha": perf.alpha,
                    }
                else:
                    strategy_stats[strategy.strategy_id] = self._extract_from_performance(perf)
            else:
                strategy_stats[strategy.strategy_id] = self._extract_from_performance(perf)

        return strategy_stats

    def _extract_from_performance(self, perf) -> dict[str, float]:
        """Extract statistics from performance metrics."""
        # Estimate volatility from Sharpe ratio and CAGR if not available
        volatility = getattr(perf, "volatility", None)
        if volatility is None:
            # Estimate volatility from Sharpe ratio: Sharpe = (return - risk_free_rate) / volatility
            # Assuming risk-free rate of 2% and solving for volatility
            risk_free_rate = 0.02
            if perf.sharpe_ratio > 0:
                volatility = (perf.cagr - risk_free_rate) / perf.sharpe_ratio
            else:
                volatility = 0.15  # Default volatility

        return {
            "mean_return": perf.cagr,
            "volatility": volatility,
            "sharpe_ratio": perf.sharpe_ratio,
            "max_drawdown": perf.max_drawdown,
            "beta": perf.beta,
            "alpha": perf.alpha,
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _calculate_correlation_matrix(
        self, strategy_stats: dict[str, dict[str, float]]
    ) -> pd.DataFrame:
        """Calculate correlation matrix between strategies."""
        # For now, use a simplified correlation matrix
        # In practice, this would use historical returns
        n_strategies = len(strategy_stats)
        correlation_matrix = pd.DataFrame(
            np.eye(n_strategies), index=strategy_stats.keys(), columns=strategy_stats.keys()
        )

        # Add some realistic correlations (this would be calculated from historical data)
        for i, strategy1 in enumerate(strategy_stats.keys()):
            for j, strategy2 in enumerate(strategy_stats.keys()):
                if i != j:
                    # Simplified correlation based on strategy characteristics
                    corr = self._estimate_correlation(
                        strategy_stats[strategy1], strategy_stats[strategy2]
                    )
                    correlation_matrix.loc[strategy1, strategy2] = corr

        return correlation_matrix

    def _estimate_correlation(self, stats1: dict[str, float], stats2: dict[str, float]) -> float:
        """Estimate correlation between two strategies based on their characteristics."""
        # Simplified correlation estimation
        # In practice, this would use historical return data

        # Strategies with similar betas tend to be more correlated
        beta_diff = abs(stats1["beta"] - stats2["beta"])
        beta_corr = max(0, 1 - beta_diff)

        # Strategies with similar volatilities tend to be more correlated
        vol_diff = abs(stats1["volatility"] - stats2["volatility"])
        vol_corr = max(0, 1 - vol_diff / 0.3)

        # Combine factors
        estimated_corr = 0.3 * beta_corr + 0.2 * vol_corr + 0.5 * 0.1  # Base correlation

        return min(0.8, max(0.0, estimated_corr))

    def _optimize_sharpe_maximization(
        self,
        strategy_stats: dict[str, dict[str, float]],
        correlation_matrix: pd.DataFrame,
        prev_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Optimize portfolio for maximum Sharpe ratio."""

        strategy_ids = list(strategy_stats.keys())
        n_strategies = len(strategy_ids)

        # Extract returns and volatilities
        returns = np.array([strategy_stats[sid]["mean_return"] for sid in strategy_ids])
        volatilities = np.array([strategy_stats[sid]["volatility"] for sid in strategy_ids])

        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

        # Objective function: negative Sharpe ratio (minimize negative = maximize positive)
        def objective(weights):
            portfolio_return = np.sum(weights * returns)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe = (portfolio_return - self.constraints.risk_free_rate) / portfolio_vol
            penalty = 0.0
            # Transaction cost penalty (proportional to turnover from prev_weights)
            if (
                prev_weights is not None
                and self.constraints.transaction_cost_bps
                and self.constraints.transaction_cost_bps > 0
            ):
                prev = np.array([prev_weights.get(sid, 0.0) for sid in strategy_ids])
                turnover = np.sum(np.abs(weights - prev))
                penalty += (self.constraints.transaction_cost_bps / 10_000.0) * turnover
            return -sharpe + penalty  # minimize negative sharpe plus cost

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]
        # Optional turnover cap
        if prev_weights is not None and self.constraints.max_turnover is not None:
            prev = np.array([prev_weights.get(sid, 0.0) for sid in strategy_ids])
            max_turn = float(self.constraints.max_turnover)

            def turnover_constraint(x):
                return max_turn - np.sum(np.abs(x - prev))

            constraints.append({"type": "ineq", "fun": turnover_constraint})

        if self.constraints.max_volatility is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: self.constraints.max_volatility
                    - np.sqrt(x.T @ cov_matrix @ x),
                }
            )

        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_strategies

        # Initial guess: equal weights
        initial_weights = np.ones(n_strategies) / n_strategies

        # Optimize
        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Try different optimization methods as fallback
            try:
                # Try with different initial conditions
                initial_weights = np.random.dirichlet(np.ones(n_strategies))
                result = minimize(
                    objective,
                    initial_weights,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000},
                )

                if result.success:
                    weights = result.x
                else:
                    # Final fallback to equal weights
                    weights = np.ones(n_strategies) / n_strategies
                    logger.warning("All optimization attempts failed, using equal weights")
            except Exception as e:
                logger.warning(f"Fallback optimization failed: {e}, using equal weights")
                weights = np.ones(n_strategies) / n_strategies
        else:
            weights = result.x

        return dict(zip(strategy_ids, weights, strict=False))

    def _optimize_risk_parity(
        self, strategy_stats: dict[str, dict[str, float]], correlation_matrix: pd.DataFrame
    ) -> dict[str, float]:
        """Optimize portfolio for risk parity (equal risk contribution)."""

        strategy_ids = list(strategy_stats.keys())
        n_strategies = len(strategy_ids)

        # Extract volatilities
        volatilities = np.array([strategy_stats[sid]["volatility"] for sid in strategy_ids])

        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

        # Objective function: minimize variance of risk contributions
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            risk_contributions = weights * (cov_matrix @ weights) / portfolio_vol
            return np.var(risk_contributions)  # Minimize variance of risk contributions

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_strategies

        # Initial guess: equal weights
        initial_weights = np.ones(n_strategies) / n_strategies

        # Optimize
        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            # Fall back to equal weights
            weights = np.ones(n_strategies) / n_strategies
        else:
            weights = result.x

        return dict(zip(strategy_ids, weights, strict=False))

    def _optimize_max_diversification(
        self, strategy_stats: dict[str, dict[str, float]], correlation_matrix: pd.DataFrame
    ) -> dict[str, float]:
        """Optimize portfolio for maximum diversification ratio."""

        strategy_ids = list(strategy_stats.keys())
        n_strategies = len(strategy_ids)

        # Extract volatilities
        volatilities = np.array([strategy_stats[sid]["volatility"] for sid in strategy_ids])

        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

        # Objective function: maximize diversification ratio
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            weighted_vol = np.sum(weights * volatilities)
            diversification_ratio = weighted_vol / portfolio_vol
            return -diversification_ratio  # Negative because we minimize

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_strategies

        # Initial guess: equal weights
        initial_weights = np.ones(n_strategies) / n_strategies

        # Optimize
        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if not result.success:
            logger.warning(f"Max diversification optimization failed: {result.message}")
            # Fall back to equal weights
            weights = np.ones(n_strategies) / n_strategies
        else:
            weights = result.x

        return dict(zip(strategy_ids, weights, strict=False))

    def _optimize_mean_variance(
        self, strategy_stats: dict[str, dict[str, float]], correlation_matrix: pd.DataFrame
    ) -> dict[str, float]:
        """Optimize portfolio using mean-variance optimization."""

        strategy_ids = list(strategy_stats.keys())
        n_strategies = len(strategy_ids)

        # Extract returns and volatilities
        returns = np.array([strategy_stats[sid]["mean_return"] for sid in strategy_ids])
        volatilities = np.array([strategy_stats[sid]["volatility"] for sid in strategy_ids])

        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

        # Objective function: minimize portfolio variance
        def objective(weights):
            return weights.T @ cov_matrix @ weights

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]

        if self.constraints.target_return is not None:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.sum(x * returns) - self.constraints.target_return,
                }
            )

        # Bounds
        bounds = [(self.constraints.min_weight, self.constraints.max_weight)] * n_strategies

        # Initial guess: equal weights
        initial_weights = np.ones(n_strategies) / n_strategies

        # Optimize
        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if not result.success:
            logger.warning(f"Mean-variance optimization failed: {result.message}")
            # Fall back to equal weights
            weights = np.ones(n_strategies) / n_strategies
        else:
            weights = result.x

        return dict(zip(strategy_ids, weights, strict=False))

    def _calculate_portfolio_metrics(
        self,
        weights: dict[str, float],
        strategy_stats: dict[str, dict[str, float]],
        correlation_matrix: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate portfolio-level metrics."""

        strategy_ids = list(strategy_stats.keys())

        # Extract returns and volatilities
        returns = np.array([strategy_stats[sid]["mean_return"] for sid in strategy_ids])
        volatilities = np.array([strategy_stats[sid]["volatility"] for sid in strategy_ids])
        weights_array = np.array([weights[sid] for sid in strategy_ids])

        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

        # Calculate portfolio metrics
        expected_return = np.sum(weights_array * returns)
        expected_volatility = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
        sharpe_ratio = (expected_return - self.constraints.risk_free_rate) / expected_volatility

        # Calculate risk contributions
        portfolio_vol = expected_volatility
        risk_contributions = {}
        for i, sid in enumerate(strategy_ids):
            risk_contrib = weights_array[i] * (cov_matrix[i, :] @ weights_array) / portfolio_vol
            risk_contributions[sid] = risk_contrib

        # Calculate diversification ratio
        weighted_vol = np.sum(weights_array * volatilities)
        diversification_ratio = weighted_vol / expected_volatility

        # Estimate max drawdown (simplified)
        max_drawdown = min(0.15, expected_volatility * 2)  # Simplified estimate

        return {
            "expected_return": expected_return,
            "expected_volatility": expected_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "diversification_ratio": diversification_ratio,
            "risk_contributions": risk_contributions,
        }

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get a summary of the last optimization."""
        if self.last_optimization is None:
            return {"status": "no_optimization"}

        return {
            "timestamp": self.last_optimization.timestamp,
            "method": self.last_optimization.optimization_method,
            "n_strategies": len(self.last_optimization.strategy_weights),
            "expected_return": self.last_optimization.expected_return,
            "expected_volatility": self.last_optimization.expected_volatility,
            "sharpe_ratio": self.last_optimization.sharpe_ratio,
            "diversification_ratio": self.last_optimization.diversification_ratio,
            "strategy_weights": self.last_optimization.strategy_weights,
        }

    def get_optimization_history(self) -> list[PortfolioAllocation]:
        """Get the optimization history."""
        return self.optimization_history
