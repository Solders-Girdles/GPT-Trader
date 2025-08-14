"""
Temporal Strategy Adaptation System for Phase 3 Meta-Learning.
Adapts strategies over time as market conditions change and tracks performance decay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from bot.knowledge.strategy_knowledge_base import StrategyContext, StrategyMetadata
from bot.meta_learning.regime_detection import RegimeCharacteristics, RegimeDetector
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class AdaptationHistory:
    """History of strategy adaptations."""

    strategy_id: str
    adaptation_date: datetime
    original_parameters: dict[str, Any]
    adapted_parameters: dict[str, Any]
    adaptation_reason: str
    performance_before: float
    performance_after: float
    regime_before: str
    regime_after: str
    confidence_score: float


@dataclass
class PerformanceDecay:
    """Analysis of strategy performance decay."""

    strategy_id: str
    decay_start_date: datetime
    decay_rate: float  # Performance decline per day
    decay_confidence: float
    contributing_factors: list[str]
    adaptation_recommendations: list[str]


class TemporalAdaptationEngine:
    """Engine for temporal strategy adaptation and performance tracking."""

    def __init__(self, regime_detector: RegimeDetector) -> None:
        self.regime_detector = regime_detector
        self.adaptation_history: list[AdaptationHistory] = []
        self.performance_tracker = PerformanceTracker()
        self.parameter_drift_analyzer = ParameterDriftAnalyzer()

        # Adaptation thresholds
        self.performance_decay_threshold = 0.1  # 10% performance decline
        self.regime_change_threshold = 0.3  # 30% regime change confidence
        self.adaptation_confidence_threshold = 0.7  # 70% confidence for adaptation

    def adapt_strategy(
        self,
        strategy: StrategyMetadata,
        current_market_data: pd.DataFrame,
        current_regime: RegimeCharacteristics,
    ) -> dict[str, Any]:
        """Adapt a strategy based on current market conditions."""
        try:
            # Analyze current performance
            performance_analysis = self.performance_tracker.analyze_performance_decay(
                strategy, current_market_data
            )

            # Check if adaptation is needed
            adaptation_needed = self._check_adaptation_needed(
                strategy, current_regime, performance_analysis
            )

            if not adaptation_needed["needed"]:
                return {
                    "adapted": False,
                    "reason": "No adaptation needed",
                    "confidence": 1.0,
                    "original_parameters": strategy.parameters,
                }

            # Generate adaptation rules
            adaptation_rules = self._generate_temporal_adaptations(
                strategy, current_regime, performance_analysis
            )

            # Apply adaptations
            adapted_parameters = self._apply_temporal_adaptations(
                strategy.parameters, adaptation_rules
            )

            # Validate adaptations
            validation_score = self._validate_temporal_adaptation(
                adapted_parameters, current_regime
            )

            # Record adaptation
            adaptation_record = AdaptationHistory(
                strategy_id=strategy.strategy_id,
                adaptation_date=datetime.now(),
                original_parameters=strategy.parameters.copy(),
                adapted_parameters=adapted_parameters,
                adaptation_reason=adaptation_needed["reason"],
                performance_before=performance_analysis.get("current_performance", 0.0),
                performance_after=0.0,  # Will be updated after evaluation
                regime_before=strategy.context.market_regime,
                regime_after=current_regime.regime.value,
                confidence_score=validation_score,
            )

            self.adaptation_history.append(adaptation_record)

            return {
                "adapted": True,
                "adapted_parameters": adapted_parameters,
                "adaptation_rules": adaptation_rules,
                "confidence": validation_score,
                "reason": adaptation_needed["reason"],
                "performance_analysis": performance_analysis,
            }

        except Exception as e:
            logger.error(f"Temporal adaptation failed: {e}")
            return {"adapted": False, "error": str(e), "original_parameters": strategy.parameters}

    def update_adaptation_performance(self, strategy_id: str, new_performance: float) -> None:
        """Update performance after adaptation."""
        # Find the most recent adaptation for this strategy
        for adaptation in reversed(self.adaptation_history):
            if adaptation.strategy_id == strategy_id:
                adaptation.performance_after = new_performance
                break

    def get_adaptation_insights(self, strategy_id: str) -> dict[str, Any]:
        """Get insights about strategy adaptations."""
        strategy_adaptations = [a for a in self.adaptation_history if a.strategy_id == strategy_id]

        if not strategy_adaptations:
            return {"message": "No adaptations found for this strategy"}

        # Analyze adaptation patterns
        adaptation_frequency = len(strategy_adaptations) / 30  # per month
        avg_performance_improvement = np.mean(
            [a.performance_after - a.performance_before for a in strategy_adaptations]
        )

        # Most common adaptation reasons
        reasons = [a.adaptation_reason for a in strategy_adaptations]
        common_reasons = pd.Series(reasons).value_counts().head(3).to_dict()

        return {
            "total_adaptations": len(strategy_adaptations),
            "adaptation_frequency": adaptation_frequency,
            "avg_performance_improvement": avg_performance_improvement,
            "common_adaptation_reasons": common_reasons,
            "recent_adaptations": [
                {
                    "date": a.adaptation_date.strftime("%Y-%m-%d"),
                    "reason": a.adaptation_reason,
                    "performance_change": a.performance_after - a.performance_before,
                    "confidence": a.confidence_score,
                }
                for a in strategy_adaptations[-5:]  # Last 5 adaptations
            ],
        }

    def _check_adaptation_needed(
        self,
        strategy: StrategyMetadata,
        current_regime: RegimeCharacteristics,
        performance_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Check if strategy adaptation is needed."""
        # Check for performance decay
        performance_decay = performance_analysis.get("decay_rate", 0.0)
        if performance_decay > self.performance_decay_threshold:
            return {
                "needed": True,
                "reason": f"Performance decay detected: {performance_decay:.3f}",
                "priority": "high",
            }

        # Check for regime change
        regime_change = self._calculate_regime_change(strategy.context, current_regime)
        if regime_change > self.regime_change_threshold:
            return {
                "needed": True,
                "reason": f"Significant regime change: {regime_change:.3f}",
                "priority": "medium",
            }

        # Check for parameter drift
        parameter_drift = self.parameter_drift_analyzer.analyze_drift(strategy, current_regime)
        if parameter_drift["drift_detected"]:
            return {
                "needed": True,
                "reason": f"Parameter drift detected: {parameter_drift['drift_score']:.3f}",
                "priority": "low",
            }

        return {"needed": False, "reason": "No significant changes detected"}

    def _generate_temporal_adaptations(
        self,
        strategy: StrategyMetadata,
        current_regime: RegimeCharacteristics,
        performance_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate temporal adaptation rules."""
        adaptations = []

        # Volatility-based adaptations
        if current_regime.volatility > 0.3:  # High volatility
            adaptations.append(
                {
                    "parameter": "atr_k",
                    "adaptation_type": "multiply",
                    "value": 0.8,  # Tighter stops
                    "reason": "High volatility - reduce risk exposure",
                    "confidence": 0.8,
                }
            )
        elif current_regime.volatility < 0.15:  # Low volatility
            adaptations.append(
                {
                    "parameter": "atr_k",
                    "adaptation_type": "multiply",
                    "value": 1.2,  # Wider stops
                    "reason": "Low volatility - increase position sizing",
                    "confidence": 0.7,
                }
            )

        # Trend-based adaptations
        if current_regime.trend_strength > 0.2:  # Strong trend
            adaptations.append(
                {
                    "parameter": "donchian_lookback",
                    "adaptation_type": "multiply",
                    "value": 0.9,  # Shorter lookback for faster response
                    "reason": "Strong trend - faster signal generation",
                    "confidence": 0.7,
                }
            )
        elif current_regime.trend_strength < 0.05:  # Weak trend
            adaptations.append(
                {
                    "parameter": "donchian_lookback",
                    "adaptation_type": "multiply",
                    "value": 1.1,  # Longer lookback for stability
                    "reason": "Weak trend - more stable signals",
                    "confidence": 0.6,
                }
            )

        # Performance decay adaptations
        decay_rate = performance_analysis.get("decay_rate", 0.0)
        if decay_rate > 0.1:
            adaptations.append(
                {
                    "parameter": "max_risk_per_trade",
                    "adaptation_type": "multiply",
                    "value": 0.8,  # Reduce risk
                    "reason": f"Performance decay detected: {decay_rate:.3f}",
                    "confidence": 0.9,
                }
            )

        # Regime-specific adaptations
        if current_regime.regime.value == "crisis":
            adaptations.extend(
                [
                    {
                        "parameter": "max_risk_per_trade",
                        "adaptation_type": "multiply",
                        "value": 0.5,  # Halve risk in crisis
                        "reason": "Crisis regime - conservative risk management",
                        "confidence": 0.9,
                    },
                    {
                        "parameter": "use_correlation_filter",
                        "adaptation_type": "replace",
                        "value": True,
                        "reason": "Crisis regime - enable correlation filtering",
                        "confidence": 0.8,
                    },
                ]
            )

        return adaptations

    def _apply_temporal_adaptations(
        self, original_parameters: dict[str, Any], adaptations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Apply temporal adaptations to strategy parameters."""
        adapted_parameters = original_parameters.copy()

        for adaptation in adaptations:
            param_name = adaptation["parameter"]
            adaptation_type = adaptation["adaptation_type"]
            value = adaptation["value"]

            if param_name in adapted_parameters:
                if adaptation_type == "multiply":
                    adapted_parameters[param_name] *= value
                elif adaptation_type == "replace":
                    adapted_parameters[param_name] = value
                elif adaptation_type == "add":
                    adapted_parameters[param_name] += value

        return adapted_parameters

    def _validate_temporal_adaptation(
        self, adapted_parameters: dict[str, Any], current_regime: RegimeCharacteristics
    ) -> float:
        """Validate temporal adaptations and return confidence score."""
        confidence = 1.0

        # Validate parameter bounds
        if "atr_k" in adapted_parameters:
            atr_k = adapted_parameters["atr_k"]
            if atr_k < 0.5 or atr_k > 5.0:
                confidence *= 0.8

        if "max_risk_per_trade" in adapted_parameters:
            risk_pct = adapted_parameters["max_risk_per_trade"]
            if risk_pct < 0.001 or risk_pct > 0.05:
                confidence *= 0.7

        if "donchian_lookback" in adapted_parameters:
            lookback = adapted_parameters["donchian_lookback"]
            if lookback < 5 or lookback > 200:
                confidence *= 0.9

        # Regime-specific validations
        if current_regime.regime.value == "crisis":
            risk_pct = adapted_parameters.get("max_risk_per_trade", 0.02)
            if risk_pct > 0.02:  # Should be conservative in crisis
                confidence *= 0.6

        return max(0.0, min(1.0, confidence))

    def _calculate_regime_change(
        self, strategy_context: StrategyContext, current_regime: RegimeCharacteristics
    ) -> float:
        """Calculate the magnitude of regime change."""
        # Simple regime change calculation
        regime_mapping = {"trending": 1.0, "volatile": 2.0, "sideways": 3.0, "crisis": 4.0}

        strategy_regime = regime_mapping.get(strategy_context.market_regime, 3.0)
        current_regime_value = regime_mapping.get(current_regime.regime.value, 3.0)

        return abs(strategy_regime - current_regime_value) / 4.0


class PerformanceTracker:
    """Tracks strategy performance over time and detects decay."""

    def __init__(self) -> None:
        self.performance_history: dict[str, list[dict[str, Any]]] = {}
        self.decay_models: dict[str, LinearRegression] = {}

    def track_performance(self, strategy_id: str, performance: float, date: datetime) -> None:
        """Track strategy performance over time."""
        if strategy_id not in self.performance_history:
            self.performance_history[strategy_id] = []

        self.performance_history[strategy_id].append({"date": date, "performance": performance})

        # Keep only recent history (last 100 points)
        if len(self.performance_history[strategy_id]) > 100:
            self.performance_history[strategy_id] = self.performance_history[strategy_id][-100:]

    def analyze_performance_decay(
        self, strategy: StrategyMetadata, current_market_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Analyze performance decay for a strategy."""
        strategy_id = strategy.strategy_id

        if strategy_id not in self.performance_history:
            return {
                "decay_detected": False,
                "decay_rate": 0.0,
                "current_performance": strategy.performance.sharpe_ratio,
                "confidence": 0.0,
            }

        history = self.performance_history[strategy_id]

        if len(history) < 10:  # Need minimum data points
            return {
                "decay_detected": False,
                "decay_rate": 0.0,
                "current_performance": strategy.performance.sharpe_ratio,
                "confidence": 0.0,
            }

        # Calculate decay rate using linear regression
        dates = [h["date"] for h in history]
        performances = [h["performance"] for h in history]

        # Convert dates to numeric for regression
        date_nums = [(d - dates[0]).days for d in dates]

        # Fit decay model
        X = np.array(date_nums).reshape(-1, 1)
        y = np.array(performances)

        decay_model = LinearRegression()
        decay_model.fit(X, y)

        # Calculate decay rate (negative slope indicates decay)
        decay_rate = -decay_model.coef_[0]  # Convert to positive for decay

        # Calculate confidence based on R-squared
        y_pred = decay_model.predict(X)
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Determine if decay is significant
        decay_detected = decay_rate > 0.001 and r_squared > 0.3  # Minimum threshold

        return {
            "decay_detected": decay_detected,
            "decay_rate": decay_rate,
            "current_performance": performances[-1] if performances else 0.0,
            "confidence": r_squared,
            "trend": "declining" if decay_rate > 0 else "improving",
            "data_points": len(history),
        }


class ParameterDriftAnalyzer:
    """Analyzes parameter drift over time."""

    def __init__(self) -> None:
        self.parameter_history: dict[str, list[dict[str, Any]]] = {}

    def track_parameters(
        self, strategy_id: str, parameters: dict[str, Any], date: datetime
    ) -> None:
        """Track strategy parameters over time."""
        if strategy_id not in self.parameter_history:
            self.parameter_history[strategy_id] = []

        self.parameter_history[strategy_id].append({"date": date, "parameters": parameters.copy()})

        # Keep only recent history
        if len(self.parameter_history[strategy_id]) > 50:
            self.parameter_history[strategy_id] = self.parameter_history[strategy_id][-50:]

    def analyze_drift(
        self, strategy: StrategyMetadata, current_regime: RegimeCharacteristics
    ) -> dict[str, Any]:
        """Analyze parameter drift for a strategy."""
        strategy_id = strategy.strategy_id

        if strategy_id not in self.parameter_history:
            return {"drift_detected": False, "drift_score": 0.0, "drifted_parameters": []}

        history = self.parameter_history[strategy_id]

        if len(history) < 5:  # Need minimum data points
            return {"drift_detected": False, "drift_score": 0.0, "drifted_parameters": []}

        # Analyze drift for each parameter
        drifted_parameters = []
        total_drift_score = 0.0

        current_params = strategy.parameters

        for param_name in current_params:
            if param_name in history[0]["parameters"]:
                initial_value = history[0]["parameters"][param_name]
                current_value = current_params[param_name]

                # Calculate drift as percentage change
                if isinstance(initial_value, int | float) and initial_value != 0:
                    drift_pct = abs(current_value - initial_value) / abs(initial_value)

                    if drift_pct > 0.2:  # 20% drift threshold
                        drifted_parameters.append(
                            {
                                "parameter": param_name,
                                "initial_value": initial_value,
                                "current_value": current_value,
                                "drift_percentage": drift_pct,
                            }
                        )
                        total_drift_score += drift_pct

        drift_detected = len(drifted_parameters) > 0
        avg_drift_score = total_drift_score / len(current_params) if current_params else 0.0

        return {
            "drift_detected": drift_detected,
            "drift_score": avg_drift_score,
            "drifted_parameters": drifted_parameters,
            "total_parameters": len(current_params),
        }
