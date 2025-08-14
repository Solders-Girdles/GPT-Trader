"""
Alpha Generation Analysis.

This module provides tools to analyze and optimize alpha generation:
- Alpha persistence
- Alpha decay patterns
- Alpha source identification
- Alpha optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from bot.strategy.base import Strategy
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class AlphaSource:
    """Analysis of an alpha source."""

    source_name: str
    alpha_contribution: float
    alpha_persistence: float
    decay_rate: float
    information_ratio: float
    capacity_limit: float
    details: dict[str, Any]


@dataclass
class AlphaAnalysisResult:
    """Results of alpha generation analysis."""

    strategy_name: str
    total_alpha: float
    alpha_sources: list[AlphaSource]
    alpha_persistence: float
    alpha_decay_rate: float
    capacity_utilization: float
    alpha_quality: float
    optimization_opportunities: list[str]


class AlphaGenerationAnalyzer:
    """
    Analyzes strategy alpha generation and provides optimization insights.

    This analyzer helps understand the sources of alpha, their persistence,
    and opportunities for optimization.
    """

    def __init__(self) -> None:
        self.analysis_history: list[AlphaAnalysisResult] = []

    def analyze_strategy(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        benchmark: pd.Series | None = None,
        lookback_periods: list[int] = None,
    ) -> AlphaAnalysisResult:
        """
        Perform comprehensive alpha generation analysis.

        Args:
            strategy: The strategy to analyze
            data: Market data used for analysis
            benchmark: Optional benchmark series for relative analysis
            lookback_periods: Periods for persistence analysis

        Returns:
            AlphaAnalysisResult with detailed alpha analysis
        """
        if lookback_periods is None:
            lookback_periods = [30, 60, 90, 180, 252]
        logger.info(f"Starting alpha analysis for strategy: {strategy.__class__.__name__}")

        # Calculate strategy returns
        strategy_returns = self._calculate_strategy_returns(strategy, data)

        # Use market returns as benchmark if not provided
        if benchmark is None:
            benchmark = data["Close"].pct_change().fillna(0)

        # Calculate total alpha
        total_alpha = self._calculate_total_alpha(strategy_returns, benchmark)

        # Analyze alpha sources
        alpha_sources = self._analyze_alpha_sources(strategy, data, strategy_returns, benchmark)

        # Analyze alpha persistence
        alpha_persistence = self._analyze_alpha_persistence(
            strategy_returns, benchmark, lookback_periods
        )

        # Analyze alpha decay
        alpha_decay_rate = self._analyze_alpha_decay(strategy_returns, benchmark)

        # Calculate capacity utilization
        capacity_utilization = self._calculate_capacity_utilization(alpha_sources)

        # Calculate alpha quality
        alpha_quality = self._calculate_alpha_quality(
            alpha_sources, alpha_persistence, alpha_decay_rate
        )

        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities(
            alpha_sources, alpha_persistence, alpha_decay_rate
        )

        result = AlphaAnalysisResult(
            strategy_name=strategy.__class__.__name__,
            total_alpha=total_alpha,
            alpha_sources=alpha_sources,
            alpha_persistence=alpha_persistence,
            alpha_decay_rate=alpha_decay_rate,
            capacity_utilization=capacity_utilization,
            alpha_quality=alpha_quality,
            optimization_opportunities=optimization_opportunities,
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

    def _calculate_total_alpha(self, strategy_returns: pd.Series, benchmark: pd.Series) -> float:
        """Calculate total alpha."""
        try:
            # Calculate beta
            covariance = np.cov(strategy_returns, benchmark)[0, 1]
            benchmark_variance = np.var(benchmark)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Alpha = strategy return - beta * benchmark return
            alpha = strategy_returns.mean() - beta * benchmark.mean()

            return alpha
        except Exception as e:
            logger.error(f"Error calculating total alpha: {e}")
            return 0.0

    def _analyze_alpha_sources(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        strategy_returns: pd.Series,
        benchmark: pd.Series,
    ) -> list[AlphaSource]:
        """Analyze different sources of alpha."""
        alpha_sources = []

        # Market timing alpha
        timing_alpha = self._analyze_timing_alpha(strategy, data, benchmark)
        alpha_sources.append(timing_alpha)

        # Volatility alpha
        volatility_alpha = self._analyze_volatility_alpha(strategy, data, strategy_returns)
        alpha_sources.append(volatility_alpha)

        # Momentum alpha
        momentum_alpha = self._analyze_momentum_alpha(strategy, data, strategy_returns)
        alpha_sources.append(momentum_alpha)

        # Mean reversion alpha
        mean_reversion_alpha = self._analyze_mean_reversion_alpha(strategy, data, strategy_returns)
        alpha_sources.append(mean_reversion_alpha)

        # Quality alpha
        quality_alpha = self._analyze_quality_alpha(strategy, data, strategy_returns)
        alpha_sources.append(quality_alpha)

        return alpha_sources

    def _analyze_timing_alpha(
        self, strategy: Strategy, data: pd.DataFrame, benchmark: pd.Series
    ) -> AlphaSource:
        """Analyze market timing alpha."""
        try:
            signals = strategy.generate_signals(data)
            position = signals["position"].fillna(0)

            # Calculate timing alpha
            timing_correlation = np.corrcoef(position, benchmark)[0, 1]
            timing_alpha = timing_correlation * benchmark.std()

            # Calculate persistence
            timing_persistence = self._calculate_source_persistence(position, benchmark)

            # Calculate decay rate
            timing_decay = self._calculate_source_decay(position, benchmark)

            # Calculate information ratio
            timing_ir = (
                timing_alpha / (position.std() * benchmark.std()) if position.std() > 0 else 0
            )

            # Estimate capacity limit
            capacity_limit = 1.0 / (position.std() * benchmark.std()) if position.std() > 0 else 1.0

            details = {
                "timing_correlation": timing_correlation,
                "position_volatility": position.std(),
                "benchmark_volatility": benchmark.std(),
            }

            return AlphaSource(
                source_name="Market Timing",
                alpha_contribution=timing_alpha,
                alpha_persistence=timing_persistence,
                decay_rate=timing_decay,
                information_ratio=timing_ir,
                capacity_limit=capacity_limit,
                details=details,
            )
        except Exception as e:
            logger.error(f"Error analyzing timing alpha: {e}")
            return self._create_default_alpha_source("Market Timing")

    def _analyze_volatility_alpha(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> AlphaSource:
        """Analyze volatility alpha."""
        try:
            # Calculate rolling volatility
            returns = data["Close"].pct_change().fillna(0)
            volatility = returns.rolling(20).std()

            # Calculate volatility alpha
            volatility_correlation = np.corrcoef(strategy_returns, volatility)[0, 1]
            volatility_alpha = volatility_correlation * volatility.mean()

            # Calculate persistence
            volatility_persistence = self._calculate_source_persistence(
                strategy_returns, volatility
            )

            # Calculate decay rate
            volatility_decay = self._calculate_source_decay(strategy_returns, volatility)

            # Calculate information ratio
            volatility_ir = volatility_alpha / volatility.std() if volatility.std() > 0 else 0

            # Estimate capacity limit
            capacity_limit = 1.0 / volatility.std() if volatility.std() > 0 else 1.0

            details = {
                "volatility_correlation": volatility_correlation,
                "avg_volatility": volatility.mean(),
                "volatility_std": volatility.std(),
            }

            return AlphaSource(
                source_name="Volatility",
                alpha_contribution=volatility_alpha,
                alpha_persistence=volatility_persistence,
                decay_rate=volatility_decay,
                information_ratio=volatility_ir,
                capacity_limit=capacity_limit,
                details=details,
            )
        except Exception as e:
            logger.error(f"Error analyzing volatility alpha: {e}")
            return self._create_default_alpha_source("Volatility")

    def _analyze_momentum_alpha(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> AlphaSource:
        """Analyze momentum alpha."""
        try:
            # Calculate momentum
            momentum = data["Close"].pct_change(252).fillna(0)

            # Calculate momentum alpha
            momentum_correlation = np.corrcoef(strategy_returns, momentum)[0, 1]
            momentum_alpha = momentum_correlation * momentum.mean()

            # Calculate persistence
            momentum_persistence = self._calculate_source_persistence(strategy_returns, momentum)

            # Calculate decay rate
            momentum_decay = self._calculate_source_decay(strategy_returns, momentum)

            # Calculate information ratio
            momentum_ir = momentum_alpha / momentum.std() if momentum.std() > 0 else 0

            # Estimate capacity limit
            capacity_limit = 1.0 / momentum.std() if momentum.std() > 0 else 1.0

            details = {
                "momentum_correlation": momentum_correlation,
                "avg_momentum": momentum.mean(),
                "momentum_std": momentum.std(),
            }

            return AlphaSource(
                source_name="Momentum",
                alpha_contribution=momentum_alpha,
                alpha_persistence=momentum_persistence,
                decay_rate=momentum_decay,
                information_ratio=momentum_ir,
                capacity_limit=capacity_limit,
                details=details,
            )
        except Exception as e:
            logger.error(f"Error analyzing momentum alpha: {e}")
            return self._create_default_alpha_source("Momentum")

    def _analyze_mean_reversion_alpha(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> AlphaSource:
        """Analyze mean reversion alpha."""
        try:
            # Calculate mean reversion signal (deviation from moving average)
            ma_short = data["Close"].rolling(20).mean()
            ma_long = data["Close"].rolling(60).mean()
            mean_reversion = (ma_short - ma_long) / ma_long

            # Calculate mean reversion alpha
            mr_correlation = np.corrcoef(strategy_returns, mean_reversion)[0, 1]
            mr_alpha = mr_correlation * mean_reversion.mean()

            # Calculate persistence
            mr_persistence = self._calculate_source_persistence(strategy_returns, mean_reversion)

            # Calculate decay rate
            mr_decay = self._calculate_source_decay(strategy_returns, mean_reversion)

            # Calculate information ratio
            mr_ir = mr_alpha / mean_reversion.std() if mean_reversion.std() > 0 else 0

            # Estimate capacity limit
            capacity_limit = 1.0 / mean_reversion.std() if mean_reversion.std() > 0 else 1.0

            details = {
                "mean_reversion_correlation": mr_correlation,
                "avg_mean_reversion": mean_reversion.mean(),
                "mean_reversion_std": mean_reversion.std(),
            }

            return AlphaSource(
                source_name="Mean Reversion",
                alpha_contribution=mr_alpha,
                alpha_persistence=mr_persistence,
                decay_rate=mr_decay,
                information_ratio=mr_ir,
                capacity_limit=capacity_limit,
                details=details,
            )
        except Exception as e:
            logger.error(f"Error analyzing mean reversion alpha: {e}")
            return self._create_default_alpha_source("Mean Reversion")

    def _analyze_quality_alpha(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> AlphaSource:
        """Analyze quality alpha."""
        try:
            # Calculate quality proxy (low volatility periods)
            returns = data["Close"].pct_change().fillna(0)
            volatility = returns.rolling(20).std()
            quality = 1 / (1 + volatility)

            # Calculate quality alpha
            quality_correlation = np.corrcoef(strategy_returns, quality)[0, 1]
            quality_alpha = quality_correlation * quality.mean()

            # Calculate persistence
            quality_persistence = self._calculate_source_persistence(strategy_returns, quality)

            # Calculate decay rate
            quality_decay = self._calculate_source_decay(strategy_returns, quality)

            # Calculate information ratio
            quality_ir = quality_alpha / quality.std() if quality.std() > 0 else 0

            # Estimate capacity limit
            capacity_limit = 1.0 / quality.std() if quality.std() > 0 else 1.0

            details = {
                "quality_correlation": quality_correlation,
                "avg_quality": quality.mean(),
                "quality_std": quality.std(),
            }

            return AlphaSource(
                source_name="Quality",
                alpha_contribution=quality_alpha,
                alpha_persistence=quality_persistence,
                decay_rate=quality_decay,
                information_ratio=quality_ir,
                capacity_limit=capacity_limit,
                details=details,
            )
        except Exception as e:
            logger.error(f"Error analyzing quality alpha: {e}")
            return self._create_default_alpha_source("Quality")

    def _create_default_alpha_source(self, source_name: str) -> AlphaSource:
        """Create a default alpha source when analysis fails."""
        return AlphaSource(
            source_name=source_name,
            alpha_contribution=0.0,
            alpha_persistence=0.0,
            decay_rate=0.0,
            information_ratio=0.0,
            capacity_limit=1.0,
            details={},
        )

    def _analyze_alpha_persistence(
        self, strategy_returns: pd.Series, benchmark: pd.Series, lookback_periods: list[int]
    ) -> float:
        """Analyze alpha persistence across different time periods."""
        try:
            persistence_scores = []

            for period in lookback_periods:
                if len(strategy_returns) < period:
                    continue

                # Calculate rolling alpha
                rolling_alpha = []
                for i in range(period, len(strategy_returns)):
                    window_returns = strategy_returns.iloc[i - period : i]
                    window_benchmark = benchmark.iloc[i - period : i]

                    # Calculate alpha for this window
                    covariance = np.cov(window_returns, window_benchmark)[0, 1]
                    benchmark_variance = np.var(window_benchmark)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    alpha = window_returns.mean() - beta * window_benchmark.mean()
                    rolling_alpha.append(alpha)

                if rolling_alpha:
                    # Calculate persistence as autocorrelation
                    alpha_series = pd.Series(rolling_alpha)
                    autocorr = alpha_series.autocorr()
                    persistence_scores.append(autocorr if not np.isnan(autocorr) else 0.0)

            # Return average persistence
            return np.mean(persistence_scores) if persistence_scores else 0.0

        except Exception as e:
            logger.error(f"Error analyzing alpha persistence: {e}")
            return 0.0

    def _analyze_alpha_decay(self, strategy_returns: pd.Series, benchmark: pd.Series) -> float:
        """Analyze alpha decay rate."""
        try:
            # Calculate rolling alpha
            window_size = min(60, len(strategy_returns) // 4)
            rolling_alpha = []

            for i in range(window_size, len(strategy_returns)):
                window_returns = strategy_returns.iloc[i - window_size : i]
                window_benchmark = benchmark.iloc[i - window_size : i]

                covariance = np.cov(window_returns, window_benchmark)[0, 1]
                benchmark_variance = np.var(window_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = window_returns.mean() - beta * window_benchmark.mean()
                rolling_alpha.append(alpha)

            if len(rolling_alpha) < 2:
                return 0.0

            # Calculate decay rate using linear regression
            x = np.arange(len(rolling_alpha))
            y = np.array(rolling_alpha)

            # Remove NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0.0

            x_clean = x[mask]
            y_clean = y[mask]

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

            # Decay rate is the negative of the slope
            decay_rate = -slope

            return decay_rate

        except Exception as e:
            logger.error(f"Error analyzing alpha decay: {e}")
            return 0.0

    def _calculate_source_persistence(
        self, strategy_returns: pd.Series, factor: pd.Series
    ) -> float:
        """Calculate persistence for a specific alpha source."""
        try:
            # Calculate rolling correlation
            window_size = min(30, len(strategy_returns) // 4)
            rolling_corr = []

            for i in range(window_size, len(strategy_returns)):
                window_returns = strategy_returns.iloc[i - window_size : i]
                window_factor = factor.iloc[i - window_size : i]

                correlation = np.corrcoef(window_returns, window_factor)[0, 1]
                if not np.isnan(correlation):
                    rolling_corr.append(correlation)

            if len(rolling_corr) < 2:
                return 0.0

            # Calculate autocorrelation
            corr_series = pd.Series(rolling_corr)
            autocorr = corr_series.autocorr()

            return autocorr if not np.isnan(autocorr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating source persistence: {e}")
            return 0.0

    def _calculate_source_decay(self, strategy_returns: pd.Series, factor: pd.Series) -> float:
        """Calculate decay rate for a specific alpha source."""
        try:
            # Calculate rolling correlation
            window_size = min(30, len(strategy_returns) // 4)
            rolling_corr = []

            for i in range(window_size, len(strategy_returns)):
                window_returns = strategy_returns.iloc[i - window_size : i]
                window_factor = factor.iloc[i - window_size : i]

                correlation = np.corrcoef(window_returns, window_factor)[0, 1]
                if not np.isnan(correlation):
                    rolling_corr.append(correlation)

            if len(rolling_corr) < 2:
                return 0.0

            # Calculate decay rate using linear regression
            x = np.arange(len(rolling_corr))
            y = np.array(rolling_corr)

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Decay rate is the negative of the slope
            decay_rate = -slope

            return decay_rate

        except Exception as e:
            logger.error(f"Error calculating source decay: {e}")
            return 0.0

    def _calculate_capacity_utilization(self, alpha_sources: list[AlphaSource]) -> float:
        """Calculate capacity utilization across alpha sources."""
        try:
            total_capacity = sum(source.capacity_limit for source in alpha_sources)
            total_utilization = sum(abs(source.alpha_contribution) for source in alpha_sources)

            if total_capacity > 0:
                return total_utilization / total_capacity
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating capacity utilization: {e}")
            return 0.0

    def _calculate_alpha_quality(
        self, alpha_sources: list[AlphaSource], alpha_persistence: float, alpha_decay_rate: float
    ) -> float:
        """Calculate overall alpha quality score."""
        try:
            # Quality factors
            persistence_score = max(0, alpha_persistence)  # Higher is better
            decay_score = max(0, 1 - alpha_decay_rate)  # Lower decay is better

            # Information ratio score
            avg_ir = np.mean([source.information_ratio for source in alpha_sources])
            ir_score = max(0, min(1, avg_ir / 2))  # Normalize to 0-1

            # Capacity utilization score (moderate utilization is best)
            capacity_score = 1 - abs(0.5 - self._calculate_capacity_utilization(alpha_sources))

            # Combine scores
            quality = (persistence_score + decay_score + ir_score + capacity_score) / 4

            return quality

        except Exception as e:
            logger.error(f"Error calculating alpha quality: {e}")
            return 0.0

    def _identify_optimization_opportunities(
        self, alpha_sources: list[AlphaSource], alpha_persistence: float, alpha_decay_rate: float
    ) -> list[str]:
        """Identify optimization opportunities."""
        opportunities = []

        # Analyze alpha sources
        for source in alpha_sources:
            if source.information_ratio < 0.5:
                opportunities.append(f"Improve {source.source_name} alpha efficiency")

            if source.alpha_persistence < 0.3:
                opportunities.append(f"Enhance {source.source_name} alpha persistence")

            if source.decay_rate > 0.1:
                opportunities.append(f"Reduce {source.source_name} alpha decay")

        # Overall alpha analysis
        if alpha_persistence < 0.5:
            opportunities.append("Improve overall alpha persistence")

        if alpha_decay_rate > 0.05:
            opportunities.append("Reduce overall alpha decay rate")

        # Capacity analysis
        capacity_utilization = self._calculate_capacity_utilization(alpha_sources)
        if capacity_utilization > 0.8:
            opportunities.append("High capacity utilization - consider diversification")
        elif capacity_utilization < 0.2:
            opportunities.append("Low capacity utilization - consider increasing exposure")

        return opportunities

    def optimize_alpha_weights(
        self,
        alpha_sources: list[AlphaSource],
        target_alpha: float = 0.05,
        risk_budget: float = 0.02,
    ) -> dict[str, float]:
        """Optimize weights across alpha sources."""
        try:
            # Define optimization objective
            def objective(weights):
                total_alpha = sum(
                    w * s.alpha_contribution for w, s in zip(weights, alpha_sources, strict=False)
                )
                total_risk = sum(
                    w * s.information_ratio for w, s in zip(weights, alpha_sources, strict=False)
                )

                # Penalize deviation from target alpha and risk budget
                alpha_penalty = (total_alpha - target_alpha) ** 2
                risk_penalty = (total_risk - risk_budget) ** 2

                return alpha_penalty + risk_penalty

            # Constraints: weights sum to 1, all weights >= 0
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            bounds = [(0, 1) for _ in alpha_sources]

            # Initial guess: equal weights
            initial_weights = np.ones(len(alpha_sources)) / len(alpha_sources)

            # Optimize
            result = minimize(
                objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
            )

            if result.success:
                optimized_weights = dict(
                    zip([s.source_name for s in alpha_sources], result.x, strict=False)
                )
                return optimized_weights
            else:
                logger.warning("Alpha optimization failed, using equal weights")
                return {s.source_name: 1.0 / len(alpha_sources) for s in alpha_sources}

        except Exception as e:
            logger.error(f"Error optimizing alpha weights: {e}")
            return {s.source_name: 1.0 / len(alpha_sources) for s in alpha_sources}

    def generate_report(self, result: AlphaAnalysisResult) -> str:
        """Generate a human-readable report of the alpha analysis."""
        report = f"""
Alpha Generation Analysis Report
===============================

Strategy: {result.strategy_name}
Total Alpha: {result.total_alpha:.4f}
Alpha Persistence: {result.alpha_persistence:.3f}
Alpha Decay Rate: {result.alpha_decay_rate:.4f}
Capacity Utilization: {result.capacity_utilization:.3f}
Alpha Quality: {result.alpha_quality:.3f}

Alpha Sources:
--------------
"""

        for source in result.alpha_sources:
            report += f"""
{source.source_name}:
  - Alpha Contribution: {source.alpha_contribution:.4f}
  - Alpha Persistence: {source.alpha_persistence:.3f}
  - Decay Rate: {source.decay_rate:.4f}
  - Information Ratio: {source.information_ratio:.3f}
  - Capacity Limit: {source.capacity_limit:.3f}
"""

        if result.optimization_opportunities:
            report += "\nOptimization Opportunities:\n"
            for opportunity in result.optimization_opportunities:
                report += f"  - {opportunity}\n"

        return report

    def get_top_alpha_sources(self, result: AlphaAnalysisResult, n: int = 3) -> list[AlphaSource]:
        """Get the top alpha sources by contribution."""
        sorted_sources = sorted(
            result.alpha_sources, key=lambda x: abs(x.alpha_contribution), reverse=True
        )
        return sorted_sources[:n]

    def get_alpha_insights(self, result: AlphaAnalysisResult) -> list[str]:
        """Generate alpha insights and recommendations."""
        insights = []

        # Analyze alpha quality
        if result.alpha_quality < 0.5:
            insights.append("Overall alpha quality is low - consider strategy improvements")

        # Analyze persistence
        if result.alpha_persistence < 0.3:
            insights.append("Low alpha persistence - alpha may not be sustainable")

        # Analyze decay
        if result.alpha_decay_rate > 0.05:
            insights.append("High alpha decay rate - alpha may be eroding quickly")

        # Analyze capacity
        if result.capacity_utilization > 0.8:
            insights.append("High capacity utilization - consider diversification")

        # Analyze individual sources
        for source in result.alpha_sources:
            if source.information_ratio > 1.0:
                insights.append(f"{source.source_name} shows strong alpha generation")

            if source.alpha_persistence < 0.2:
                insights.append(f"{source.source_name} has low persistence")

        return insights
