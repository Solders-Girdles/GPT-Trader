"""
Risk Decomposition Analysis.

This module provides tools to understand and decompose risk sources:
- Systematic risk (market risk)
- Idiosyncratic risk (stock-specific)
- Liquidity risk
- Model risk
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from bot.strategy.base import Strategy
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RiskComponent:
    """Risk component analysis."""

    risk_type: str
    risk_contribution: float
    risk_pct: float
    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    details: dict[str, Any]


@dataclass
class RiskDecompositionResult:
    """Results of risk decomposition analysis."""

    strategy_name: str
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    liquidity_risk: float
    model_risk: float
    risk_components: list[RiskComponent]
    risk_attribution: dict[str, float]
    risk_quality: float


class RiskDecompositionAnalyzer:
    """
    Analyzes strategy risk by decomposing it into different risk sources.

    This analyzer helps understand the sources of risk in a strategy
    and provides insights for risk management.
    """

    def __init__(self) -> None:
        self.analysis_history: list[RiskDecompositionResult] = []

    def analyze_strategy(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        benchmark: pd.Series | None = None,
        confidence_level: float = 0.95,
    ) -> RiskDecompositionResult:
        """
        Perform comprehensive risk decomposition analysis.

        Args:
            strategy: The strategy to analyze
            data: Market data used for analysis
            benchmark: Optional benchmark series for relative analysis
            confidence_level: Confidence level for VaR calculations

        Returns:
            RiskDecompositionResult with detailed risk analysis
        """
        logger.info(f"Starting risk decomposition for strategy: {strategy.__class__.__name__}")

        # Calculate strategy returns
        strategy_returns = self._calculate_strategy_returns(strategy, data)

        # Use market returns as benchmark if not provided
        if benchmark is None:
            benchmark = data["Close"].pct_change().fillna(0)

        # Calculate total risk
        total_risk = strategy_returns.std()

        # Analyze different risk components
        systematic_risk = self._analyze_systematic_risk(strategy_returns, benchmark)
        idiosyncratic_risk = self._analyze_idiosyncratic_risk(strategy_returns, benchmark)
        liquidity_risk = self._analyze_liquidity_risk(strategy, data, strategy_returns)
        model_risk = self._analyze_model_risk(strategy, data, strategy_returns)

        # Create risk components
        risk_components = [
            self._create_risk_component(
                "Systematic", systematic_risk, strategy_returns, confidence_level
            ),
            self._create_risk_component(
                "Idiosyncratic", idiosyncratic_risk, strategy_returns, confidence_level
            ),
            self._create_risk_component(
                "Liquidity", liquidity_risk, strategy_returns, confidence_level
            ),
            self._create_risk_component("Model", model_risk, strategy_returns, confidence_level),
        ]

        # Calculate risk attribution
        risk_attribution = {
            "Systematic": systematic_risk / total_risk if total_risk > 0 else 0,
            "Idiosyncratic": idiosyncratic_risk / total_risk if total_risk > 0 else 0,
            "Liquidity": liquidity_risk / total_risk if total_risk > 0 else 0,
            "Model": model_risk / total_risk if total_risk > 0 else 0,
        }

        # Calculate risk quality
        risk_quality = self._calculate_risk_quality(risk_components, total_risk)

        result = RiskDecompositionResult(
            strategy_name=strategy.__class__.__name__,
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            liquidity_risk=liquidity_risk,
            model_risk=model_risk,
            risk_components=risk_components,
            risk_attribution=risk_attribution,
            risk_quality=risk_quality,
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

    def _analyze_systematic_risk(self, strategy_returns: pd.Series, benchmark: pd.Series) -> float:
        """Analyze systematic (market) risk."""
        try:
            # Calculate beta
            covariance = np.cov(strategy_returns, benchmark)[0, 1]
            benchmark_variance = np.var(benchmark)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Systematic risk is beta times market volatility
            systematic_risk = abs(beta) * benchmark.std()

            return systematic_risk
        except Exception as e:
            logger.error(f"Error analyzing systematic risk: {e}")
            return 0.0

    def _analyze_idiosyncratic_risk(
        self, strategy_returns: pd.Series, benchmark: pd.Series
    ) -> float:
        """Analyze idiosyncratic (stock-specific) risk."""
        try:
            # Calculate beta
            covariance = np.cov(strategy_returns, benchmark)[0, 1]
            benchmark_variance = np.var(benchmark)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Idiosyncratic risk is the residual risk after removing systematic risk
            systematic_returns = beta * benchmark
            idiosyncratic_returns = strategy_returns - systematic_returns

            idiosyncratic_risk = idiosyncratic_returns.std()

            return idiosyncratic_risk
        except Exception as e:
            logger.error(f"Error analyzing idiosyncratic risk: {e}")
            return 0.0

    def _analyze_liquidity_risk(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> float:
        """Analyze liquidity risk."""
        try:
            signals = strategy.generate_signals(data)
            position = signals["position"].fillna(0)

            # Calculate position changes
            position_changes = position.diff().abs()

            # Liquidity risk is related to trading frequency and position size
            avg_position_change = position_changes.mean()
            max_position = position.abs().max()

            # Liquidity risk increases with trading frequency and position size
            liquidity_risk = avg_position_change * max_position * 0.1  # Scaling factor

            return liquidity_risk
        except Exception as e:
            logger.error(f"Error analyzing liquidity risk: {e}")
            return 0.0

    def _analyze_model_risk(
        self, strategy: Strategy, data: pd.DataFrame, strategy_returns: pd.Series
    ) -> float:
        """Analyze model risk."""
        try:
            # Model risk is related to parameter uncertainty and model complexity
            # For now, we'll use a simplified approach based on signal variability

            signals = strategy.generate_signals(data)
            position = signals["position"].fillna(0)

            # Model risk increases with signal variability
            signal_volatility = position.std()

            # Also consider the complexity of the strategy
            # This is a simplified proxy - in practice, you'd analyze the actual model
            model_complexity = 1.0  # Placeholder for model complexity score

            model_risk = signal_volatility * model_complexity * 0.05  # Scaling factor

            return model_risk
        except Exception as e:
            logger.error(f"Error analyzing model risk: {e}")
            return 0.0

    def _create_risk_component(
        self,
        risk_type: str,
        risk_contribution: float,
        strategy_returns: pd.Series,
        confidence_level: float,
    ) -> RiskComponent:
        """Create a risk component with detailed metrics."""
        try:
            # Calculate risk metrics
            volatility = risk_contribution
            var_95 = np.percentile(strategy_returns, (1 - confidence_level) * 100)
            cvar_95 = strategy_returns[strategy_returns <= var_95].mean()

            # Calculate max drawdown for this risk component
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            # Calculate risk percentage
            total_volatility = strategy_returns.std()
            risk_pct = risk_contribution / total_volatility if total_volatility > 0 else 0

            details = {
                "var_confidence": confidence_level,
                "cvar_confidence": confidence_level,
                "volatility_annualized": volatility * np.sqrt(252),
                "var_annualized": var_95 * np.sqrt(252),
                "cvar_annualized": cvar_95 * np.sqrt(252),
            }

            return RiskComponent(
                risk_type=risk_type,
                risk_contribution=risk_contribution,
                risk_pct=risk_pct,
                volatility=volatility,
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                details=details,
            )
        except Exception as e:
            logger.error(f"Error creating risk component: {e}")
            return RiskComponent(
                risk_type=risk_type,
                risk_contribution=risk_contribution,
                risk_pct=0.0,
                volatility=0.0,
                var_95=0.0,
                cvar_95=0.0,
                max_drawdown=0.0,
                details={},
            )

    def _calculate_risk_quality(
        self, risk_components: list[RiskComponent], total_risk: float
    ) -> float:
        """Calculate the quality of the risk decomposition."""
        try:
            # Sum of all risk components
            total_attributed_risk = sum(c.risk_contribution for c in risk_components)

            # Risk quality is how well components sum to total risk
            if total_risk < 0.001:
                return 1.0

            quality = 1 - abs(total_attributed_risk - total_risk) / total_risk
            return min(max(quality, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating risk quality: {e}")
            return 0.0

    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95, method: str = "historical"
    ) -> float:
        """Calculate Value at Risk."""
        try:
            if method == "historical":
                var = np.percentile(returns, (1 - confidence_level) * 100)
            elif method == "parametric":
                # Assume normal distribution
                mean_return = returns.mean()
                std_return = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                var = mean_return + z_score * std_return
            else:
                raise ValueError(f"Unknown VaR method: {method}")

            return var
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        try:
            var = self.calculate_var(returns, confidence_level)
            cvar = returns[returns <= var].mean()
            return cvar
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0

    def calculate_stress_test(
        self, strategy: Strategy, data: pd.DataFrame, stress_scenarios: dict[str, float]
    ) -> dict[str, float]:
        """Perform stress testing under different scenarios."""
        try:
            base_returns = self._calculate_strategy_returns(strategy, data)
            stress_results = {}

            for scenario_name, stress_factor in stress_scenarios.items():
                # Apply stress factor to returns
                stressed_returns = base_returns * stress_factor

                # Calculate stressed metrics
                stressed_sharpe = (
                    stressed_returns.mean() / stressed_returns.std()
                    if stressed_returns.std() > 0
                    else 0
                )
                stressed_var = self.calculate_var(stressed_returns)
                stressed_cvar = self.calculate_cvar(stressed_returns)

                stress_results[scenario_name] = {
                    "sharpe_ratio": stressed_sharpe,
                    "var_95": stressed_var,
                    "cvar_95": stressed_cvar,
                    "max_drawdown": self._calculate_max_drawdown(stressed_returns),
                }

            return stress_results
        except Exception as e:
            logger.error(f"Error performing stress test: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return drawdown.min()
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def generate_report(self, result: RiskDecompositionResult) -> str:
        """Generate a human-readable report of the risk decomposition analysis."""
        report = f"""
Risk Decomposition Analysis Report
=================================

Strategy: {result.strategy_name}
Total Risk (Volatility): {result.total_risk:.4f}
Risk Quality: {result.risk_quality:.3f}

Risk Components:
----------------
Systematic Risk: {result.systematic_risk:.4f} ({result.risk_attribution['Systematic']:.2%})
Idiosyncratic Risk: {result.idiosyncratic_risk:.4f} ({result.risk_attribution['Idiosyncratic']:.2%})
Liquidity Risk: {result.liquidity_risk:.4f} ({result.risk_attribution['Liquidity']:.2%})
Model Risk: {result.model_risk:.4f} ({result.risk_attribution['Model']:.2%})

Detailed Risk Analysis:
"""

        for component in result.risk_components:
            report += f"""
{component.risk_type} Risk:
  - Risk Contribution: {component.risk_contribution:.4f} ({component.risk_pct:.2%})
  - Volatility: {component.volatility:.4f}
  - VaR (95%): {component.var_95:.4f}
  - CVaR (95%): {component.cvar_95:.4f}
  - Max Drawdown: {component.max_drawdown:.4f}
"""

        return report

    def get_risk_breakdown(self, result: RiskDecompositionResult) -> dict[str, float]:
        """Get the risk breakdown by component."""
        return result.risk_attribution

    def get_risk_insights(self, result: RiskDecompositionResult) -> list[str]:
        """Generate risk insights and recommendations."""
        insights = []

        # Analyze risk concentration
        max_risk_component = max(result.risk_attribution.items(), key=lambda x: x[1])
        if max_risk_component[1] > 0.5:
            insights.append(
                f"Risk is concentrated in {max_risk_component[0]} component ({max_risk_component[1]:.1%})"
            )

        # Analyze risk quality
        if result.risk_quality < 0.8:
            insights.append("Risk decomposition quality is low - consider additional risk factors")

        # Analyze specific risk components
        if result.systematic_risk > result.total_risk * 0.7:
            insights.append("High systematic risk exposure - consider diversification")

        if result.model_risk > result.total_risk * 0.2:
            insights.append("High model risk - consider model validation and robustness testing")

        if result.liquidity_risk > result.total_risk * 0.1:
            insights.append(
                "Liquidity risk is significant - consider position sizing and trading frequency"
            )

        return insights
