"""
Strategy Decomposition Analysis.

This module provides tools to break down strategy performance into component contributions:
- Entry signal contribution
- Exit signal contribution
- Risk management contribution
- Filter effectiveness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from bot.metrics.report import perf_metrics
from bot.strategy.components import BaseComponent, ComponentBasedStrategy

logger = logging.getLogger(__name__)


@dataclass
class ComponentContribution:
    """Contribution analysis for a single component."""

    component_name: str
    component_type: str
    contribution_score: float
    performance_impact: float
    signal_quality: float
    timing_accuracy: float
    risk_adjustment: float
    details: dict[str, Any]


@dataclass
class DecompositionResult:
    """Results of strategy decomposition analysis."""

    strategy_name: str
    total_performance: float
    component_contributions: list[ComponentContribution]
    entry_contribution: float
    exit_contribution: float
    risk_contribution: float
    filter_contribution: float
    interaction_effects: dict[str, float]
    decomposition_quality: float


class StrategyDecompositionAnalyzer:
    """
    Analyzes strategy performance by decomposing it into component contributions.

    This analyzer helps understand which parts of a strategy are driving performance
    and which components might need improvement.
    """

    def __init__(self) -> None:
        self.analysis_history: list[DecompositionResult] = []

    def analyze_strategy(
        self,
        strategy: ComponentBasedStrategy,
        data: pd.DataFrame,
        benchmark: pd.Series | None = None,
    ) -> DecompositionResult:
        """
        Perform comprehensive decomposition analysis of a strategy.

        Args:
            strategy: The component-based strategy to analyze
            data: Market data used for analysis
            benchmark: Optional benchmark series for relative analysis

        Returns:
            DecompositionResult with detailed component analysis
        """
        logger.info(f"Starting decomposition analysis for strategy: {strategy.__class__.__name__}")

        # Get strategy components
        components = strategy.components

        # Calculate baseline performance
        baseline_performance = self._calculate_baseline_performance(strategy, data)

        # Analyze each component's contribution
        component_contributions = []
        entry_contrib = 0.0
        exit_contrib = 0.0
        risk_contrib = 0.0
        filter_contrib = 0.0

        for component in components:
            contribution = self._analyze_component_contribution(
                component, strategy, data, baseline_performance
            )
            component_contributions.append(contribution)

            # Aggregate contributions by type
            if component.component_type == "entry":
                entry_contrib += contribution.contribution_score
            elif component.component_type == "exit":
                exit_contrib += contribution.contribution_score
            elif component.component_type == "risk":
                risk_contrib += contribution.contribution_score
            elif component.component_type == "filter":
                filter_contrib += contribution.contribution_score

        # Analyze interaction effects
        interaction_effects = self._analyze_interaction_effects(components, strategy, data)

        # Calculate decomposition quality
        decomposition_quality = self._calculate_decomposition_quality(
            component_contributions, baseline_performance
        )

        result = DecompositionResult(
            strategy_name=strategy.__class__.__name__,
            total_performance=baseline_performance,
            component_contributions=component_contributions,
            entry_contribution=entry_contrib,
            exit_contribution=exit_contrib,
            risk_contribution=risk_contrib,
            filter_contribution=filter_contrib,
            interaction_effects=interaction_effects,
            decomposition_quality=decomposition_quality,
        )

        self.analysis_history.append(result)
        return result

    def _calculate_baseline_performance(
        self, strategy: ComponentBasedStrategy, data: pd.DataFrame
    ) -> float:
        """Calculate baseline strategy performance."""
        try:
            # Generate signals using the strategy
            signals = strategy.generate_signals(data)

            # Calculate equity curve (simplified)
            position = signals["position"].fillna(0)
            returns = data["Close"].pct_change().fillna(0)
            strategy_returns = position.shift(1) * returns

            # Calculate performance metrics
            equity_curve = (1 + strategy_returns).cumprod()
            metrics = perf_metrics(equity_curve)

            return metrics["sharpe"]
        except Exception as e:
            logger.error(f"Error calculating baseline performance: {e}")
            return 0.0

    def _analyze_component_contribution(
        self,
        component: BaseComponent,
        strategy: ComponentBasedStrategy,
        data: pd.DataFrame,
        baseline_performance: float,
    ) -> ComponentContribution:
        """Analyze the contribution of a single component."""
        logger.debug(
            f"Analyzing component: {component.component_type} - {component.__class__.__name__}"
        )

        # Create strategy without this component
        components_without = [c for c in strategy.components if c != component]
        strategy_without = ComponentBasedStrategy(components_without)

        # Calculate performance without component
        performance_without = self._calculate_baseline_performance(strategy_without, data)

        # Component contribution is the difference
        contribution_score = baseline_performance - performance_without

        # Analyze component-specific metrics
        signal_quality = self._analyze_signal_quality(component, data)
        timing_accuracy = self._analyze_timing_accuracy(component, data)
        risk_adjustment = self._analyze_risk_adjustment(component, data)

        # Calculate performance impact
        performance_impact = contribution_score / max(abs(baseline_performance), 0.01)

        details = {
            "performance_without": performance_without,
            "signal_quality_score": signal_quality,
            "timing_accuracy_score": timing_accuracy,
            "risk_adjustment_score": risk_adjustment,
        }

        return ComponentContribution(
            component_name=component.__class__.__name__,
            component_type=component.component_type,
            contribution_score=contribution_score,
            performance_impact=performance_impact,
            signal_quality=signal_quality,
            timing_accuracy=timing_accuracy,
            risk_adjustment=risk_adjustment,
            details=details,
        )

    def _analyze_signal_quality(self, component: BaseComponent, data: pd.DataFrame) -> float:
        """Analyze the quality of signals generated by a component."""
        try:
            signals = component.process(data)

            if len(signals) == 0:
                return 0.0

            # Calculate signal quality metrics
            signal_strength = abs(signals).mean()
            signal_consistency = 1 - signals.std() / max(abs(signals.mean()), 0.01)
            signal_clarity = len(signals[abs(signals) > 0.1]) / len(signals)

            # Combine metrics
            quality_score = (signal_strength + signal_consistency + signal_clarity) / 3
            return min(max(quality_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error analyzing signal quality: {e}")
            return 0.0

    def _analyze_timing_accuracy(self, component: BaseComponent, data: pd.DataFrame) -> float:
        """Analyze the timing accuracy of component signals."""
        try:
            signals = component.process(data)
            returns = data["Close"].pct_change().fillna(0)

            if len(signals) == 0 or len(returns) == 0:
                return 0.0

            # Calculate forward returns for timing analysis
            forward_returns = returns.shift(-1).fillna(0)

            # Calculate correlation between signals and forward returns
            correlation = np.corrcoef(signals, forward_returns)[0, 1]

            if np.isnan(correlation):
                return 0.0

            # Convert correlation to accuracy score (0-1)
            accuracy_score = (correlation + 1) / 2
            return min(max(accuracy_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error analyzing timing accuracy: {e}")
            return 0.0

    def _analyze_risk_adjustment(self, component: BaseComponent, data: pd.DataFrame) -> float:
        """Analyze the risk adjustment effectiveness of a component."""
        try:
            if component.component_type != "risk":
                return 0.5  # Neutral score for non-risk components

            signals = component.process(data)
            returns = data["Close"].pct_change().fillna(0)

            if len(signals) == 0:
                return 0.0

            # Calculate risk metrics
            volatility = returns.std()
            adjusted_volatility = (returns * signals).std()

            # Risk reduction effectiveness
            risk_reduction = 1 - (adjusted_volatility / max(volatility, 0.01))

            return min(max(risk_reduction, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error analyzing risk adjustment: {e}")
            return 0.0

    def _analyze_interaction_effects(
        self, components: list[BaseComponent], strategy: ComponentBasedStrategy, data: pd.DataFrame
    ) -> dict[str, float]:
        """Analyze interaction effects between components."""
        interaction_effects = {}

        # Analyze pairwise interactions
        for i, comp1 in enumerate(components):
            for _j, comp2 in enumerate(components[i + 1 :], i + 1):
                interaction_name = f"{comp1.__class__.__name__}_{comp2.__class__.__name__}"

                # Calculate performance with both components
                strategy_both = ComponentBasedStrategy([comp1, comp2])
                perf_both = self._calculate_baseline_performance(strategy_both, data)

                # Calculate performance with each component individually
                strategy_1 = ComponentBasedStrategy([comp1])
                strategy_2 = ComponentBasedStrategy([comp2])
                perf_1 = self._calculate_baseline_performance(strategy_1, data)
                perf_2 = self._calculate_baseline_performance(strategy_2, data)

                # Interaction effect
                interaction_effect = perf_both - (perf_1 + perf_2)
                interaction_effects[interaction_name] = interaction_effect

        return interaction_effects

    def _calculate_decomposition_quality(
        self, component_contributions: list[ComponentContribution], baseline_performance: float
    ) -> float:
        """Calculate the quality of the decomposition analysis."""
        try:
            # Sum of all component contributions
            total_contribution = sum(c.contribution_score for c in component_contributions)

            # Decomposition quality is how well contributions sum to total performance
            if abs(baseline_performance) < 0.01:
                return 1.0

            quality = 1 - abs(total_contribution - baseline_performance) / abs(baseline_performance)
            return min(max(quality, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating decomposition quality: {e}")
            return 0.0

    def generate_report(self, result: DecompositionResult) -> str:
        """Generate a human-readable report of the decomposition analysis."""
        report = f"""
Strategy Decomposition Analysis Report
====================================

Strategy: {result.strategy_name}
Total Performance (Sharpe): {result.total_performance:.3f}
Decomposition Quality: {result.decomposition_quality:.3f}

Component Contributions:
-----------------------
Entry Contribution: {result.entry_contribution:.3f}
Exit Contribution: {result.exit_contribution:.3f}
Risk Contribution: {result.risk_contribution:.3f}
Filter Contribution: {result.filter_contribution:.3f}

Detailed Component Analysis:
"""

        for contrib in result.component_contributions:
            report += f"""
{contrib.component_name} ({contrib.component_type}):
  - Contribution Score: {contrib.contribution_score:.3f}
  - Performance Impact: {contrib.performance_impact:.3f}
  - Signal Quality: {contrib.signal_quality:.3f}
  - Timing Accuracy: {contrib.timing_accuracy:.3f}
  - Risk Adjustment: {contrib.risk_adjustment:.3f}
"""

        if result.interaction_effects:
            report += "\nInteraction Effects:\n"
            for interaction, effect in result.interaction_effects.items():
                report += f"  {interaction}: {effect:.3f}\n"

        return report

    def get_top_contributors(
        self, result: DecompositionResult, n: int = 3
    ) -> list[ComponentContribution]:
        """Get the top contributing components."""
        sorted_contributions = sorted(
            result.component_contributions, key=lambda x: abs(x.contribution_score), reverse=True
        )
        return sorted_contributions[:n]

    def get_improvement_opportunities(self, result: DecompositionResult) -> list[str]:
        """Identify improvement opportunities based on decomposition analysis."""
        opportunities = []

        for contrib in result.component_contributions:
            if contrib.signal_quality < 0.5:
                opportunities.append(f"Improve signal quality for {contrib.component_name}")

            if contrib.timing_accuracy < 0.5:
                opportunities.append(f"Improve timing accuracy for {contrib.component_name}")

            if contrib.contribution_score < 0:
                opportunities.append(f"Review negative contribution from {contrib.component_name}")

        return opportunities
