"""
Meta-Learning and Strategy Transfer System for adapting strategies across different contexts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyKnowledgeBase,
    StrategyMetadata,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class AssetCharacteristics:
    """Characteristics of an asset for strategy adaptation."""

    volatility: float  # Annualized volatility
    correlation: float  # Correlation with market
    volume_profile: str  # "high", "medium", "low"
    price_range: float  # Average price range
    liquidity: str  # "high", "medium", "low"
    market_cap: float | None = None  # Market capitalization
    sector: str | None = None  # Sector classification


@dataclass
class AdaptationRule:
    """Rule for adapting strategy parameters."""

    parameter_name: str
    adaptation_type: str  # "scale", "offset", "multiply", "replace"
    source_value: float
    target_value: float
    confidence: float  # Confidence in this adaptation


class StrategyTransferEngine:
    """Engine for transferring strategies across different assets and contexts."""

    def __init__(self, knowledge_base: StrategyKnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.adaptation_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def transfer_strategy(
        self,
        source_strategy: StrategyMetadata,
        target_context: StrategyContext,
        target_asset: AssetCharacteristics,
    ) -> dict[str, Any]:
        """Transfer a strategy from source context to target context."""
        try:
            # Analyze source strategy performance
            source_analysis = self._analyze_strategy_performance(source_strategy)

            # Generate adaptation rules
            adaptation_rules = self._generate_adaptation_rules(
                source_strategy, target_context, target_asset
            )

            # Apply adaptations
            adapted_parameters = self._apply_adaptations(
                source_strategy.parameters, adaptation_rules
            )

            # Validate adaptations
            validation_score = self._validate_adaptation(
                adapted_parameters, target_context, target_asset
            )

            return {
                "adapted_parameters": adapted_parameters,
                "adaptation_rules": adaptation_rules,
                "confidence_score": validation_score,
                "source_analysis": source_analysis,
                "adaptation_notes": self._generate_adaptation_notes(adaptation_rules),
            }

        except Exception as e:
            logger.error(f"Strategy transfer failed: {e}")
            return {
                "adapted_parameters": source_strategy.parameters,
                "adaptation_rules": [],
                "confidence_score": 0.0,
                "error": str(e),
            }

    def train_adaptation_model(self, training_data: list[dict[str, Any]]) -> None:
        """Train the adaptation model on historical transfer data."""
        if not training_data:
            logger.warning("No training data provided")
            return

        # Prepare training features and targets
        X = []
        y = []

        for transfer_case in training_data:
            features = self._extract_transfer_features(transfer_case)
            target = transfer_case.get("success_score", 0.0)

            X.append(features)
            y.append(target)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.adaptation_model.fit(X_scaled, y)
        self.is_trained = True

        logger.info(f"Trained adaptation model on {len(training_data)} transfer cases")

    def predict_transfer_success(
        self,
        source_strategy: StrategyMetadata,
        target_context: StrategyContext,
        target_asset: AssetCharacteristics,
    ) -> float:
        """Predict the success probability of a strategy transfer."""
        if not self.is_trained:
            return 0.5  # Default confidence

        # Extract features
        features = self._extract_transfer_features(
            {
                "source_strategy": source_strategy,
                "target_context": target_context,
                "target_asset": target_asset,
            }
        )

        # Scale features
        features_scaled = self.scaler.transform([features])

        # Predict success probability
        success_prob = self.adaptation_model.predict(features_scaled)[0]
        return max(0.0, min(1.0, success_prob))  # Clamp to [0, 1]

    def get_similar_strategies(
        self, strategy: StrategyMetadata, n_similar: int = 5
    ) -> list[StrategyMetadata]:
        """Find strategies similar to the given strategy."""
        similar_strategies = []

        for other_strategy in self.knowledge_base.strategies.values():
            if other_strategy.strategy_id == strategy.strategy_id:
                continue

            similarity_score = self._calculate_strategy_similarity(strategy, other_strategy)
            similar_strategies.append((similarity_score, other_strategy))

        # Sort by similarity and return top results
        similar_strategies.sort(key=lambda x: x[0], reverse=True)
        return [strategy for _, strategy in similar_strategies[:n_similar]]

    def _analyze_strategy_performance(self, strategy: StrategyMetadata) -> dict[str, Any]:
        """Analyze the performance characteristics of a strategy."""
        perf = strategy.performance

        return {
            "risk_adjusted_return": perf.sharpe_ratio,
            "risk_level": (
                "high"
                if perf.max_drawdown > 0.2
                else "medium" if perf.max_drawdown > 0.1 else "low"
            ),
            "consistency": perf.consistency_score,
            "trade_frequency": (
                "high" if perf.n_trades > 100 else "medium" if perf.n_trades > 50 else "low"
            ),
            "profitability": (
                "high" if perf.win_rate > 0.6 else "medium" if perf.win_rate > 0.5 else "low"
            ),
        }

    def _generate_adaptation_rules(
        self,
        source_strategy: StrategyMetadata,
        target_context: StrategyContext,
        target_asset: AssetCharacteristics,
    ) -> list[AdaptationRule]:
        """Generate adaptation rules for transferring the strategy."""
        rules = []

        # Volatility-based adaptations
        if target_asset.volatility > 0.3:  # High volatility
            rules.append(
                AdaptationRule(
                    parameter_name="atr_k",
                    adaptation_type="multiply",
                    source_value=source_strategy.parameters.get("atr_k", 2.0),
                    target_value=source_strategy.parameters.get("atr_k", 2.0)
                    * 0.8,  # Tighter stops
                    confidence=0.7,
                )
            )
        elif target_asset.volatility < 0.15:  # Low volatility
            rules.append(
                AdaptationRule(
                    parameter_name="atr_k",
                    adaptation_type="multiply",
                    source_value=source_strategy.parameters.get("atr_k", 2.0),
                    target_value=source_strategy.parameters.get("atr_k", 2.0) * 1.2,  # Wider stops
                    confidence=0.7,
                )
            )

        # Correlation-based adaptations
        if target_asset.correlation > 0.7:  # High correlation
            rules.append(
                AdaptationRule(
                    parameter_name="use_correlation_filter",
                    adaptation_type="replace",
                    source_value=source_strategy.parameters.get("use_correlation_filter", False),
                    target_value=True,
                    confidence=0.8,
                )
            )

        # Market regime adaptations
        if target_context.market_regime == "crisis":
            rules.append(
                AdaptationRule(
                    parameter_name="max_risk_per_trade",
                    adaptation_type="multiply",
                    source_value=source_strategy.parameters.get("max_risk_per_trade", 0.02),
                    target_value=source_strategy.parameters.get("max_risk_per_trade", 0.02)
                    * 0.5,  # Reduce risk
                    confidence=0.9,
                )
            )

        # Liquidity adaptations
        if target_asset.liquidity == "low":
            rules.append(
                AdaptationRule(
                    parameter_name="cooldown_periods",
                    adaptation_type="offset",
                    source_value=source_strategy.parameters.get("cooldown_periods", 0),
                    target_value=source_strategy.parameters.get("cooldown_periods", 0)
                    + 2,  # More cooldown
                    confidence=0.6,
                )
            )

        return rules

    def _apply_adaptations(
        self, original_parameters: dict[str, Any], adaptation_rules: list[AdaptationRule]
    ) -> dict[str, Any]:
        """Apply adaptation rules to strategy parameters."""
        adapted_parameters = original_parameters.copy()

        for rule in adaptation_rules:
            if rule.parameter_name in adapted_parameters:
                if rule.adaptation_type == "multiply":
                    adapted_parameters[rule.parameter_name] = rule.target_value
                elif rule.adaptation_type == "offset":
                    adapted_parameters[rule.parameter_name] = rule.target_value
                elif rule.adaptation_type == "replace":
                    adapted_parameters[rule.parameter_name] = rule.target_value
                elif rule.adaptation_type == "scale":
                    # Scale based on confidence
                    original_val = adapted_parameters[rule.parameter_name]
                    adapted_parameters[rule.parameter_name] = (
                        original_val * (1 - rule.confidence) + rule.target_value * rule.confidence
                    )

        return adapted_parameters

    def _validate_adaptation(
        self,
        adapted_parameters: dict[str, Any],
        target_context: StrategyContext,
        target_asset: AssetCharacteristics,
    ) -> float:
        """Validate the adapted parameters and return confidence score."""
        # Check parameter bounds
        confidence = 1.0

        # Validate ATR multiplier
        atr_k = adapted_parameters.get("atr_k", 2.0)
        if atr_k < 0.5 or atr_k > 5.0:
            confidence *= 0.8

        # Validate risk per trade
        risk_pct = adapted_parameters.get("max_risk_per_trade", 0.02)
        if risk_pct < 0.001 or risk_pct > 0.05:
            confidence *= 0.7

        # Validate lookback periods
        lookback = adapted_parameters.get("donchian_lookback", 55)
        if lookback < 5 or lookback > 200:
            confidence *= 0.9

        # Context-specific validations
        if target_context.market_regime == "crisis":
            if risk_pct > 0.02:  # Should be conservative in crisis
                confidence *= 0.6

        return max(0.0, min(1.0, confidence))

    def _extract_transfer_features(self, transfer_case: dict[str, Any]) -> list[float]:
        """Extract features for the adaptation model."""
        source_strategy = transfer_case["source_strategy"]
        target_context = transfer_case["target_context"]
        target_asset = transfer_case["target_asset"]

        features = [
            # Strategy performance features
            source_strategy.performance.sharpe_ratio,
            source_strategy.performance.max_drawdown,
            source_strategy.performance.consistency_score,
            # Context similarity features
            1.0 if source_strategy.context.market_regime == target_context.market_regime else 0.0,
            1.0 if source_strategy.context.asset_class == target_context.asset_class else 0.0,
            1.0 if source_strategy.context.risk_profile == target_context.risk_profile else 0.0,
            # Asset characteristic features
            target_asset.volatility,
            target_asset.correlation,
            (
                1.0
                if target_asset.liquidity == "high"
                else 0.5 if target_asset.liquidity == "medium" else 0.0
            ),
            # Parameter complexity
            len(source_strategy.parameters),
        ]

        return features

    def _calculate_strategy_similarity(
        self, strategy1: StrategyMetadata, strategy2: StrategyMetadata
    ) -> float:
        """Calculate similarity between two strategies."""
        similarity = 0.0

        # Context similarity
        if strategy1.context.market_regime == strategy2.context.market_regime:
            similarity += 0.3
        if strategy1.context.asset_class == strategy2.context.asset_class:
            similarity += 0.3
        if strategy1.context.risk_profile == strategy2.context.risk_profile:
            similarity += 0.2

        # Strategy type similarity
        if strategy1.strategy_type == strategy2.strategy_type:
            similarity += 0.2

        # Performance similarity (normalized)
        perf_diff = abs(strategy1.performance.sharpe_ratio - strategy2.performance.sharpe_ratio)
        similarity += max(0, 0.1 * (1 - perf_diff / 2))  # Max 0.1 for performance similarity

        return min(1.0, similarity)

    def _generate_adaptation_notes(self, adaptation_rules: list[AdaptationRule]) -> str:
        """Generate human-readable notes about the adaptations."""
        if not adaptation_rules:
            return "No adaptations applied - using original parameters"

        notes = []
        for rule in adaptation_rules:
            if rule.adaptation_type == "multiply":
                notes.append(
                    f"Adjusted {rule.parameter_name} from {rule.source_value:.3f} to {rule.target_value:.3f}"
                )
            elif rule.adaptation_type == "offset":
                notes.append(
                    f"Added offset to {rule.parameter_name}: {rule.target_value - rule.source_value:.3f}"
                )
            elif rule.adaptation_type == "replace":
                notes.append(f"Changed {rule.parameter_name} to {rule.target_value}")

        return "; ".join(notes)
