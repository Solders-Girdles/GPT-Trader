"""
Market Regime Detection System for Phase 3 Meta-Learning.
Automatically detects market regimes and provides regime-specific strategy recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from bot.knowledge.strategy_knowledge_base import StrategyContext
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"


@dataclass
class RegimeCharacteristics:
    """Characteristics of a detected market regime."""

    regime: MarketRegime
    confidence: float
    duration_days: int
    volatility: float
    trend_strength: float
    correlation_level: float
    volume_profile: str
    momentum_score: float
    regime_features: dict[str, float]


class RegimeDetector:
    """Market regime detection system using multiple indicators."""

    def __init__(self, lookback_period: int = 252) -> None:
        self.lookback_period = lookback_period
        self.regime_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

        # Regime detection parameters
        self.volatility_thresholds = {"low": 0.15, "medium": 0.25, "high": 0.35}

        self.trend_thresholds = {"weak": 0.1, "moderate": 0.2, "strong": 0.3}

        # Regime history
        self.regime_history: list[RegimeCharacteristics] = []
        self.current_regime: RegimeCharacteristics | None = None

    def detect_regime(self, market_data: pd.DataFrame) -> RegimeCharacteristics:
        """Detect the current market regime from market data."""
        if len(market_data) < self.lookback_period:
            logger.warning(
                f"Insufficient data for regime detection. Need {self.lookback_period}, got {len(market_data)}"
            )
            return self._get_default_regime()

        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(market_data)

        # Extract features for classification
        features = self._extract_regime_features(indicators)

        # Classify regime
        if self.is_trained:
            regime = self._classify_regime(features)
        else:
            regime = self._rule_based_regime_detection(indicators)

        # Create regime characteristics
        regime_char = RegimeCharacteristics(
            regime=regime,
            confidence=self._calculate_regime_confidence(indicators, regime),
            duration_days=self._calculate_regime_duration(regime),
            volatility=indicators["volatility"],
            trend_strength=indicators["trend_strength"],
            correlation_level=indicators["correlation"],
            volume_profile=indicators["volume_profile"],
            momentum_score=indicators["momentum"],
            regime_features=features,
        )

        # Update regime history
        self._update_regime_history(regime_char)
        self.current_regime = regime_char

        return regime_char

    def train_regime_classifier(self, training_data: list[dict[str, Any]]) -> None:
        """Train the regime classifier on historical data."""
        if not training_data:
            logger.warning("No training data provided for regime classifier")
            return

        X = []
        y = []

        for sample in training_data:
            features = sample["features"]
            regime_label = sample["regime"]

            X.append(features)
            y.append(regime_label)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        self.regime_classifier.fit(X_scaled, y)
        self.is_trained = True

        logger.info(f"Trained regime classifier on {len(training_data)} samples")

    def get_regime_recommendations(
        self, regime: RegimeCharacteristics, knowledge_base
    ) -> list[dict[str, Any]]:
        """Get strategy recommendations for the detected regime."""
        # Convert regime to strategy context
        context = self._regime_to_context(regime)

        # Get recommendations from knowledge base
        recommendations = knowledge_base.get_strategy_recommendations(context, n_recommendations=5)

        return [
            {
                "strategy_id": rec.strategy_id,
                "name": rec.name,
                "sharpe_ratio": rec.performance.sharpe_ratio,
                "max_drawdown": rec.performance.max_drawdown,
                "confidence": rec.success_rate,
                "context_match": self._calculate_context_match(rec.context, context),
            }
            for rec in recommendations
        ]

    def detect_regime_change(self, new_regime: RegimeCharacteristics) -> bool:
        """Detect if there has been a significant regime change."""
        if not self.current_regime:
            return True

        # Check if regime type changed
        if new_regime.regime != self.current_regime.regime:
            return True

        # Check for significant changes in characteristics
        volatility_change = (
            abs(new_regime.volatility - self.current_regime.volatility)
            / self.current_regime.volatility
        )
        trend_change = abs(new_regime.trend_strength - self.current_regime.trend_strength)

        return volatility_change > 0.3 or trend_change > 0.2

    def _calculate_regime_indicators(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Calculate regime detection indicators."""
        # Use the last lookback_period days
        recent_data = market_data.tail(self.lookback_period).copy()

        # Calculate returns
        recent_data["returns"] = recent_data["close"].pct_change()

        # Volatility (annualized)
        volatility = recent_data["returns"].std() * np.sqrt(252)

        # Trend strength (using linear regression slope)
        x = np.arange(len(recent_data))
        y = recent_data["close"].values
        slope = np.polyfit(x, y, 1)[0]
        trend_strength = abs(slope) / recent_data["close"].mean()

        # Momentum (price change over period)
        momentum = (recent_data["close"].iloc[-1] / recent_data["close"].iloc[0]) - 1

        # Correlation (if multiple assets)
        if "returns" in recent_data.columns and len(recent_data) > 20:
            # Calculate rolling correlation if we have multiple assets
            correlation = 0.5  # Default value
        else:
            correlation = 0.5

        # Volume profile
        if "volume" in recent_data.columns:
            avg_volume = recent_data["volume"].mean()
            recent_volume = recent_data["volume"].tail(20).mean()
            volume_ratio = recent_volume / avg_volume

            if volume_ratio > 1.5:
                volume_profile = "high"
            elif volume_ratio < 0.7:
                volume_profile = "low"
            else:
                volume_profile = "medium"
        else:
            volume_profile = "medium"

        return {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "correlation": correlation,
            "volume_profile": volume_profile,
            "price_level": recent_data["close"].iloc[-1],
            "price_range": (recent_data["high"].max() - recent_data["low"].min())
            / recent_data["close"].mean(),
        }

    def _extract_regime_features(self, indicators: dict[str, float]) -> dict[str, float]:
        """Extract features for regime classification."""
        return {
            "volatility": indicators["volatility"],
            "trend_strength": indicators["trend_strength"],
            "momentum": indicators["momentum"],
            "correlation": indicators["correlation"],
            "price_range": indicators["price_range"],
            "volume_high": 1.0 if indicators["volume_profile"] == "high" else 0.0,
            "volume_low": 1.0 if indicators["volume_profile"] == "low" else 0.0,
            "volatility_squared": indicators["volatility"] ** 2,
            "trend_momentum": indicators["trend_strength"] * indicators["momentum"],
        }

    def _rule_based_regime_detection(self, indicators: dict[str, float]) -> MarketRegime:
        """Rule-based regime detection when classifier is not trained."""
        volatility = indicators["volatility"]
        trend_strength = indicators["trend_strength"]
        momentum = indicators["momentum"]

        # Crisis detection
        if volatility > self.volatility_thresholds["high"] and momentum < -0.1:
            return MarketRegime.CRISIS

        # Trending regimes
        if trend_strength > self.trend_thresholds["moderate"]:
            if momentum > 0.1:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN

        # Volatile regime
        if volatility > self.volatility_thresholds["medium"]:
            return MarketRegime.VOLATILE

        # Recovery regime
        if momentum > 0.05 and volatility < self.volatility_thresholds["medium"]:
            return MarketRegime.RECOVERY

        # Default to sideways
        return MarketRegime.SIDEWAYS

    def _classify_regime(self, features: dict[str, float]) -> MarketRegime:
        """Classify regime using trained classifier."""
        feature_vector = list(features.values())
        feature_vector_scaled = self.scaler.transform([feature_vector])

        prediction = self.regime_classifier.predict(feature_vector_scaled)[0]
        return MarketRegime(prediction)

    def _calculate_regime_confidence(
        self, indicators: dict[str, float], regime: MarketRegime
    ) -> float:
        """Calculate confidence in the detected regime."""
        confidence = 0.5  # Base confidence

        # Adjust based on indicator strength
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            confidence += min(0.3, indicators["trend_strength"])
        elif regime == MarketRegime.VOLATILE:
            confidence += min(0.3, indicators["volatility"] / 0.5)
        elif regime == MarketRegime.CRISIS:
            if indicators["volatility"] > 0.4 and indicators["momentum"] < -0.15:
                confidence += 0.3

        return min(1.0, confidence)

    def _calculate_regime_duration(self, regime: MarketRegime) -> int:
        """Calculate how long the current regime has been active."""
        if not self.regime_history:
            return 1

        # Count consecutive occurrences of the same regime
        duration = 1
        for prev_regime in reversed(self.regime_history[:-1]):
            if prev_regime.regime == regime:
                duration += 1
            else:
                break

        return duration

    def _update_regime_history(self, regime_char: RegimeCharacteristics) -> None:
        """Update the regime history."""
        self.regime_history.append(regime_char)

        # Keep only recent history (last 100 regimes)
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]

    def _regime_to_context(self, regime: RegimeCharacteristics) -> StrategyContext:
        """Convert regime characteristics to strategy context."""
        # Map regime to market regime string
        regime_mapping = {
            MarketRegime.TRENDING_UP: "trending",
            MarketRegime.TRENDING_DOWN: "trending",
            MarketRegime.VOLATILE: "volatile",
            MarketRegime.SIDEWAYS: "sideways",
            MarketRegime.CRISIS: "crisis",
            MarketRegime.RECOVERY: "trending",
            MarketRegime.BULL_MARKET: "trending",
            MarketRegime.BEAR_MARKET: "trending",
        }

        market_regime = regime_mapping.get(regime.regime, "sideways")

        # Determine time period
        if regime.momentum_score > 0.1:
            time_period = "bull_market"
        elif regime.momentum_score < -0.1:
            time_period = "bear_market"
        else:
            time_period = "sideways_market"

        # Determine volatility regime
        if regime.volatility > self.volatility_thresholds["high"]:
            volatility_regime = "high"
        elif regime.volatility < self.volatility_thresholds["low"]:
            volatility_regime = "low"
        else:
            volatility_regime = "medium"

        # Determine correlation regime
        if regime.correlation_level > 0.7:
            correlation_regime = "high"
        elif regime.correlation_level < 0.3:
            correlation_regime = "low"
        else:
            correlation_regime = "medium"

        return StrategyContext(
            market_regime=market_regime,
            time_period=time_period,
            asset_class="equity",  # Default, can be overridden
            risk_profile="moderate",  # Default, can be overridden
            volatility_regime=volatility_regime,
            correlation_regime=correlation_regime,
        )

    def _calculate_context_match(
        self, strategy_context: StrategyContext, current_context: StrategyContext
    ) -> float:
        """Calculate how well a strategy context matches the current context."""
        match_score = 0.0

        # Market regime match
        if strategy_context.market_regime == current_context.market_regime:
            match_score += 0.4

        # Volatility regime match
        if strategy_context.volatility_regime == current_context.volatility_regime:
            match_score += 0.3

        # Correlation regime match
        if strategy_context.correlation_regime == current_context.correlation_regime:
            match_score += 0.2

        # Asset class match
        if strategy_context.asset_class == current_context.asset_class:
            match_score += 0.1

        return match_score

    def _get_default_regime(self) -> RegimeCharacteristics:
        """Get default regime when insufficient data."""
        return RegimeCharacteristics(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.5,
            duration_days=1,
            volatility=0.2,
            trend_strength=0.0,
            correlation_level=0.5,
            volume_profile="medium",
            momentum_score=0.0,
            regime_features={},
        )
