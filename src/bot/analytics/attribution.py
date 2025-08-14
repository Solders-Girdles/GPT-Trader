"""
Performance Attribution Analysis.

This module provides tools to attribute performance to specific factors and decisions:
- Market timing
- Stock selection
- Risk management
- Transaction costs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from bot.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class AttributionFactor:
    """Performance attribution for a specific factor."""

    factor_name: str
    contribution: float
    contribution_pct: float
    risk_contribution: float
    risk_pct: float
    information_ratio: float
    details: dict[str, Any]


@dataclass
class AttributionResult:
    """Results of performance attribution analysis."""

    strategy_name: str
    total_return: float
    benchmark_return: float
    excess_return: float
    factors: list[AttributionFactor]
    timing_contribution: float
    selection_contribution: float
    risk_contribution: float
    cost_contribution: float
    unexplained: float
    attribution_quality: float


class PerformanceAttributionAnalyzer:
    """
    Analyzes strategy performance by attributing returns to specific factors.

    This analyzer helps understand which factors are driving performance
    and provides insights for strategy improvement.
    """

    def __init__(self) -> None:
        self.analysis_history: list[AttributionResult] = []

    def analyze_strategy(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        benchmark: pd.Series | None = None,
        transaction_costs: float = 0.001,
    ) -> AttributionResult:
        """
        Perform comprehensive performance attribution analysis.

        Args:
            strategy: The strategy to analyze
            data: Market data used for analysis
            benchmark: Optional benchmark series for relative analysis
            transaction_costs: Transaction cost rate (default 0.1%)

        Returns:
            AttributionResult with detailed factor analysis
        """
        logger.info(f"Starting performance attribution for strategy: {strategy.__class__.__name__}")

        # Calculate strategy returns
        strategy_returns = self._calculate_strategy_returns(strategy, data)

        # Use market returns as benchmark if not provided
        if benchmark is None:
            benchmark = data["Close"].pct_change().fillna(0)

        # Calculate excess returns
        excess_returns = strategy_returns - benchmark

        # Perform factor attribution
        factors = self._analyze_factors(strategy, data, strategy_returns, benchmark)

        # Calculate specific contributions
        timing_contrib = self._analyze_timing_contribution(strategy, data, benchmark)
        selection_contrib = self._analyze_selection_contribution(strategy, data, benchmark)
        risk_contrib = self._analyze_risk_contribution(strategy, data, strategy_returns)
        cost_contrib = self._analyze_cost_contribution(strategy, data, transaction_costs)

        # Calculate unexplained portion
        total_attributed = sum(f.contribution for f in factors)
        unexplained = excess_returns.mean() - total_attributed

        # Calculate attribution quality
        attribution_quality = self._calculate_attribution_quality(
            factors, excess_returns.mean(), unexplained
        )

        result = AttributionResult(
            strategy_name=strategy.__class__.__name__,
            total_return=strategy_returns.mean(),
            benchmark_return=benchmark.mean(),
            excess_return=excess_returns.mean(),
            factors=factors,
            timing_contribution=timing_contrib,
            selection_contribution=selection_contrib,
            risk_contribution=risk_contrib,
            cost_contribution=cost_contrib,
            unexplained=unexplained,
            attribution_quality=attribution_quality,
        )

        self.analysis_history.append(result)
        return result

    def _calculate_strategy_returns(self, strategy: Strategy, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns."""
        try:
            signals = strategy.generate_signals(data)
            position = signals["position"].fillna(0)
            returns = data["Close"].pct_change().fillna(0)
            strategy_returns = position.shift(1) * returns
            return strategy_returns
        except Exception as e:
            logger.error(f"Error calculating strategy returns: {e}")
            return pd.Series(0.0, index=data.index)

    def _analyze_factors(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        strategy_returns: pd.Series,
        benchmark: pd.Series,
    ) -> list[AttributionFactor]:
        """Analyze performance factors."""
        factors = []

        # Market factor
        market_factor = self._analyze_market_factor(strategy_returns, benchmark)
        factors.append(market_factor)

        # Volatility factor
        volatility_factor = self._analyze_volatility_factor(strategy, data, strategy_returns)
        factors.append(volatility_factor)

        # Momentum factor
        momentum_factor = self._analyze_momentum_factor(strategy, data, strategy_returns)
        factors.append(momentum_factor)

        # Size factor
        size_factor = self._analyze_size_factor(strategy, data, strategy_returns)
        factors.append(size_factor)

        # Quality factor
        quality_factor = self._analyze_quality_factor(strategy, data, strategy_returns)
        factors.append(quality_factor)

        return factors

    def _analyze_market_factor(
        self, strategy_returns: pd.Series, benchmark: pd.Series
    ) -> AttributionFactor:
        """Analyze market factor contribution."""
        # Calculate beta
        covariance = np.cov(strategy_returns, benchmark)[0, 1]
        benchmark_variance = np.var(benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Market contribution
        market_contribution = beta * benchmark.mean()
        market_contribution_pct = (
            market_contribution / strategy_returns.mean() if strategy_returns.mean() != 0 else 0
        )

        # Risk contribution
        market_risk = abs(beta) * benchmark.std()
        market_risk_pct = market_risk / strategy_returns.std() if strategy_returns.std() != 0 else 0

        # Information ratio
        excess_return = strategy_returns.mean() - market_contribution
        tracking_error = (strategy_returns - benchmark).std()
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

        details = {
            "beta": beta,
            "market_correlation": np.corrcoef(strategy_returns, benchmark)[0, 1],
            "tracking_error": tracking_error,
        }

        return AttributionFactor(
            factor_name="Market",
            contribution=market_contribution,
            contribution_pct=market_contribution_pct,
            risk_contribution=market_risk,
            risk_pct=market_risk_pct,
            information_ratio=information_ratio,
            details=details,
        )

    def _analyze_volatility_factor(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> AttributionFactor:
        """Analyze volatility factor contribution."""
        # Calculate rolling volatility
        volatility = data["Close"].pct_change().rolling(20).std()

        # Calculate volatility exposure
        volatility_exposure = np.corrcoef(strategy_returns, volatility)[0, 1]

        # Volatility contribution
        volatility_contribution = volatility_exposure * volatility.mean()
        volatility_contribution_pct = (
            volatility_contribution / strategy_returns.mean() if strategy_returns.mean() != 0 else 0
        )

        # Risk contribution
        volatility_risk = abs(volatility_exposure) * volatility.std()
        volatility_risk_pct = (
            volatility_risk / strategy_returns.std() if strategy_returns.std() != 0 else 0
        )

        # Information ratio
        volatility_ir = volatility_contribution / volatility_risk if volatility_risk > 0 else 0

        details = {
            "volatility_exposure": volatility_exposure,
            "avg_volatility": volatility.mean(),
            "volatility_std": volatility.std(),
        }

        return AttributionFactor(
            factor_name="Volatility",
            contribution=volatility_contribution,
            contribution_pct=volatility_contribution_pct,
            risk_contribution=volatility_risk,
            risk_pct=volatility_risk_pct,
            information_ratio=volatility_ir,
            details=details,
        )

    def _analyze_momentum_factor(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> AttributionFactor:
        """Analyze momentum factor contribution."""
        # Calculate momentum (12-month return)
        momentum = data["Close"].pct_change(252).fillna(0)

        # Calculate momentum exposure
        momentum_exposure = np.corrcoef(strategy_returns, momentum)[0, 1]

        # Momentum contribution
        momentum_contribution = momentum_exposure * momentum.mean()
        momentum_contribution_pct = (
            momentum_contribution / strategy_returns.mean() if strategy_returns.mean() != 0 else 0
        )

        # Risk contribution
        momentum_risk = abs(momentum_exposure) * momentum.std()
        momentum_risk_pct = (
            momentum_risk / strategy_returns.std() if strategy_returns.std() != 0 else 0
        )

        # Information ratio
        momentum_ir = momentum_contribution / momentum_risk if momentum_risk > 0 else 0

        details = {
            "momentum_exposure": momentum_exposure,
            "avg_momentum": momentum.mean(),
            "momentum_std": momentum.std(),
        }

        return AttributionFactor(
            factor_name="Momentum",
            contribution=momentum_contribution,
            contribution_pct=momentum_contribution_pct,
            risk_contribution=momentum_risk,
            risk_pct=momentum_risk_pct,
            information_ratio=momentum_ir,
            details=details,
        )

    def _analyze_size_factor(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> AttributionFactor:
        """Analyze size factor contribution."""
        # For single asset strategies, size factor is minimal
        # This would be more relevant for multi-asset portfolios

        size_contribution = 0.0
        size_contribution_pct = 0.0
        size_risk = 0.0
        size_risk_pct = 0.0
        size_ir = 0.0

        details = {
            "size_exposure": 0.0,
            "note": "Size factor not applicable for single asset strategy",
        }

        return AttributionFactor(
            factor_name="Size",
            contribution=size_contribution,
            contribution_pct=size_contribution_pct,
            risk_contribution=size_risk,
            risk_pct=size_risk_pct,
            information_ratio=size_ir,
            details=details,
        )

    def _analyze_quality_factor(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> AttributionFactor:
        """Analyze quality factor contribution."""
        # Calculate quality proxy (low volatility periods)
        returns = data["Close"].pct_change().fillna(0)
        volatility = returns.rolling(20).std()
        quality = 1 / (1 + volatility)  # Higher quality in low volatility periods

        # Calculate quality exposure
        quality_exposure = np.corrcoef(strategy_returns, quality)[0, 1]

        # Quality contribution
        quality_contribution = quality_exposure * quality.mean()
        quality_contribution_pct = (
            quality_contribution / strategy_returns.mean() if strategy_returns.mean() != 0 else 0
        )

        # Risk contribution
        quality_risk = abs(quality_exposure) * quality.std()
        quality_risk_pct = (
            quality_risk / strategy_returns.std() if strategy_returns.std() != 0 else 0
        )

        # Information ratio
        quality_ir = quality_contribution / quality_risk if quality_risk > 0 else 0

        details = {
            "quality_exposure": quality_exposure,
            "avg_quality": quality.mean(),
            "quality_std": quality.std(),
        }

        return AttributionFactor(
            factor_name="Quality",
            contribution=quality_contribution,
            contribution_pct=quality_contribution_pct,
            risk_contribution=quality_risk,
            risk_pct=quality_risk_pct,
            information_ratio=quality_ir,
            details=details,
        )

    def _analyze_timing_contribution(
        self, strategy: Strategy, data: pd.DataFrame, benchmark: pd.Series
    ) -> float:
        """Analyze market timing contribution."""
        try:
            signals = strategy.generate_signals(data)
            position = signals["position"].fillna(0)

            # Calculate timing contribution
            # This measures how well the strategy times market movements
            timing_contribution = np.corrcoef(position, benchmark)[0, 1] * benchmark.std()

            return timing_contribution
        except Exception as e:
            logger.error(f"Error analyzing timing contribution: {e}")
            return 0.0

    def _analyze_selection_contribution(
        self, strategy: Strategy, data: pd.DataFrame, benchmark: pd.Series
    ) -> float:
        """Analyze stock selection contribution."""
        try:
            # For single asset strategies, selection contribution is minimal
            # This would be more relevant for multi-asset portfolios

            # Calculate selection as residual after timing
            strategy_returns = self._calculate_strategy_returns(strategy, data)
            timing_contrib = self._analyze_timing_contribution(strategy, data, benchmark)

            selection_contribution = strategy_returns.mean() - benchmark.mean() - timing_contrib

            return selection_contribution
        except Exception as e:
            logger.error(f"Error analyzing selection contribution: {e}")
            return 0.0

    def _analyze_risk_contribution(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> float:
        """Analyze risk management contribution."""
        try:
            # Calculate risk-adjusted returns
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            excess_return = strategy_returns.mean() - risk_free_rate
            volatility = strategy_returns.std()

            # Risk contribution is the risk-adjusted return
            risk_contribution = excess_return / volatility if volatility > 0 else 0

            return risk_contribution
        except Exception as e:
            logger.error(f"Error analyzing risk contribution: {e}")
            return 0.0

    def _analyze_cost_contribution(
        self, strategy: Strategy, data: pd.DataFrame, transaction_costs: float
    ) -> float:
        """Analyze transaction cost contribution."""
        try:
            signals = strategy.generate_signals(data)
            position = signals["position"].fillna(0)

            # Calculate position changes
            position_changes = position.diff().abs()

            # Calculate transaction costs
            total_costs = position_changes.sum() * transaction_costs

            # Convert to daily average
            cost_contribution = -total_costs / len(data)

            return cost_contribution
        except Exception as e:
            logger.error(f"Error analyzing cost contribution: {e}")
            return 0.0

    def _calculate_attribution_quality(
        self, factors: list[AttributionFactor], total_excess_return: float, unexplained: float
    ) -> float:
        """Calculate the quality of the attribution analysis."""
        try:
            sum(f.contribution for f in factors)

            if abs(total_excess_return) < 0.001:
                return 1.0

            # Quality is how well factors explain excess returns
            quality = 1 - abs(unexplained) / abs(total_excess_return)
            return min(max(quality, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating attribution quality: {e}")
            return 0.0

    def generate_report(self, result: AttributionResult) -> str:
        """Generate a human-readable report of the attribution analysis."""
        report = f"""
Performance Attribution Analysis Report
======================================

Strategy: {result.strategy_name}
Total Return: {result.total_return:.4f}
Benchmark Return: {result.benchmark_return:.4f}
Excess Return: {result.excess_return:.4f}
Attribution Quality: {result.attribution_quality:.3f}

Factor Contributions:
--------------------
"""

        for factor in result.factors:
            report += f"""
{factor.factor_name}:
  - Contribution: {factor.contribution:.4f} ({factor.contribution_pct:.2%})
  - Risk Contribution: {factor.risk_contribution:.4f} ({factor.risk_pct:.2%})
  - Information Ratio: {factor.information_ratio:.3f}
"""

        report += f"""
Specific Contributions:
----------------------
Market Timing: {result.timing_contribution:.4f}
Stock Selection: {result.selection_contribution:.4f}
Risk Management: {result.risk_contribution:.4f}
Transaction Costs: {result.cost_contribution:.4f}
Unexplained: {result.unexplained:.4f}
"""

        return report

    def get_top_factors(self, result: AttributionResult, n: int = 3) -> list[AttributionFactor]:
        """Get the top contributing factors."""
        sorted_factors = sorted(result.factors, key=lambda x: abs(x.contribution), reverse=True)
        return sorted_factors[:n]

    def get_improvement_opportunities(self, result: AttributionResult) -> list[str]:
        """Identify improvement opportunities based on attribution analysis."""
        opportunities = []

        for factor in result.factors:
            if factor.information_ratio < 0.5:
                opportunities.append(f"Improve {factor.factor_name} factor efficiency")

            if factor.contribution < 0:
                opportunities.append(
                    f"Review negative contribution from {factor.factor_name} factor"
                )

        if result.cost_contribution < -0.001:
            opportunities.append("Reduce transaction costs")

        if result.unexplained > 0.001:
            opportunities.append("Investigate unexplained performance sources")

        return opportunities
