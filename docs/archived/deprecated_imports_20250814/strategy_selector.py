"""
Real-Time Strategy Selection System for Phase 5 Production Integration.
Automatically selects optimal strategies for current market conditions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from bot.analytics.attribution import PerformanceAttributionAnalyzer
from bot.analytics.decomposition import StrategyDecompositionAnalyzer
from bot.knowledge.strategy_knowledge_base import StrategyKnowledgeBase, StrategyMetadata
from bot.meta_learning.regime_detection import RegimeCharacteristics, RegimeDetector
from bot.meta_learning.temporal_adaptation import TemporalAdaptationEngine

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Strategy selection methods."""

    REGIME_BASED = "regime_based"
    PERFORMANCE_BASED = "performance_based"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class StrategyScore:
    """Score for a strategy in the selection process."""

    strategy_id: str
    strategy: StrategyMetadata
    overall_score: float
    regime_match_score: float
    performance_score: float
    confidence_score: float
    risk_score: float
    adaptation_score: float
    selection_reason: str


@dataclass
class SelectionConfig:
    """Configuration for strategy selection."""

    selection_method: SelectionMethod = SelectionMethod.HYBRID
    max_strategies: int = 5
    min_confidence: float = 0.7
    min_sharpe: float = 0.5
    max_drawdown: float = 0.15
    regime_weight: float = 0.3
    performance_weight: float = 0.4
    confidence_weight: float = 0.2
    risk_weight: float = 0.1
    adaptation_weight: float = 0.2
    rebalance_interval: int = 3600  # 1 hour
    lookback_days: int = 30


class RealTimeStrategySelector:
    """Real-time strategy selection system."""

    def __init__(
        self,
        knowledge_base: StrategyKnowledgeBase,
        regime_detector: RegimeDetector,
        config: SelectionConfig,
        symbols: list[str],
    ) -> None:
        self.knowledge_base = knowledge_base
        self.regime_detector = regime_detector
        self.config = config
        self.symbols = symbols

        # Initialize components
        self.temporal_adaptation = TemporalAdaptationEngine(regime_detector)
        self.decomposition_analyzer = StrategyDecompositionAnalyzer()
        self.attribution_analyzer = PerformanceAttributionAnalyzer()

        # Selection state
        self.current_selection: list[StrategyScore] = []
        self.last_selection_time: datetime | None = None
        self.selection_history: list[dict[str, Any]] = []

        # Performance tracking
        self.strategy_performance: dict[str, list[float]] = {}
        self.adaptation_history: dict[str, list[dict[str, Any]]] = {}

        logger.info("Real-time strategy selector initialized")

    async def start_selection_loop(self) -> None:
        """Start the continuous strategy selection loop."""
        logger.info("Starting real-time strategy selection loop")

        while True:
            try:
                await self._selection_cycle()
                await asyncio.sleep(self.config.rebalance_interval)
            except Exception as e:
                logger.error(f"Error in selection cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _selection_cycle(self) -> None:
        """Execute one selection cycle."""
        logger.info("Executing strategy selection cycle")

        # Get current market data
        market_data = await self._get_current_market_data()

        # Detect current regime
        current_regime = self.regime_detector.detect_regime(market_data)

        # Select strategies
        selected_strategies = await self._select_strategies(current_regime, market_data)

        # Update selection
        self.current_selection = selected_strategies
        self.last_selection_time = datetime.now()

        # Record selection
        self._record_selection(selected_strategies, current_regime)

        logger.info(
            f"Selected {len(selected_strategies)} strategies for current regime: {current_regime.regime.value}"
        )

    async def _select_strategies(
        self, current_regime: RegimeCharacteristics, market_data: pd.DataFrame
    ) -> list[StrategyScore]:
        """Select optimal strategies for current conditions."""

        # Get candidate strategies from knowledge base
        candidates = self._get_candidate_strategies(current_regime)

        if not candidates:
            logger.warning("No candidate strategies found")
            return []

        # Score strategies
        scored_strategies = []
        for strategy in candidates:
            score = await self._score_strategy(strategy, current_regime, market_data)
            if score.overall_score >= self.config.min_confidence:
                scored_strategies.append(score)

        # Sort by overall score
        scored_strategies.sort(key=lambda x: x.overall_score, reverse=True)

        # Select top strategies
        selected = scored_strategies[: self.config.max_strategies]

        return selected

    def _get_candidate_strategies(
        self, current_regime: RegimeCharacteristics
    ) -> list[StrategyMetadata]:
        """Get candidate strategies from knowledge base."""
        # Convert regime to context
        current_context = self.regime_detector._regime_to_context(current_regime)

        # Find strategies matching current context
        candidates = self.knowledge_base.find_strategies(
            context=current_context,
            min_sharpe=self.config.min_sharpe,
            max_drawdown=self.config.max_drawdown,
            limit=50,  # Get more candidates for scoring
        )

        return candidates

    async def _score_strategy(
        self,
        strategy: StrategyMetadata,
        current_regime: RegimeCharacteristics,
        market_data: pd.DataFrame,
    ) -> StrategyScore:
        """Score a strategy for current conditions."""

        # Calculate regime match score
        regime_match_score = self._calculate_regime_match(strategy, current_regime)

        # Calculate performance score
        performance_score = self._calculate_performance_score(strategy)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(strategy)

        # Calculate risk score
        risk_score = self._calculate_risk_score(strategy)

        # Calculate adaptation score
        adaptation_score = await self._calculate_adaptation_score(strategy, market_data)

        # Calculate overall score
        overall_score = (
            self.config.regime_weight * regime_match_score
            + self.config.performance_weight * performance_score
            + self.config.confidence_weight * confidence_score
            + self.config.risk_weight * risk_score
            + self.config.adaptation_weight * adaptation_score
        )

        # Determine selection reason
        selection_reason = self._determine_selection_reason(
            regime_match_score, performance_score, confidence_score, risk_score, adaptation_score
        )

        return StrategyScore(
            strategy_id=strategy.strategy_id,
            strategy=strategy,
            overall_score=overall_score,
            regime_match_score=regime_match_score,
            performance_score=performance_score,
            confidence_score=confidence_score,
            risk_score=risk_score,
            adaptation_score=adaptation_score,
            selection_reason=selection_reason,
        )

    def _calculate_regime_match(
        self, strategy: StrategyMetadata, current_regime: RegimeCharacteristics
    ) -> float:
        """Calculate how well a strategy matches the current regime."""
        strategy_context = strategy.context
        current_context = self.regime_detector._regime_to_context(current_regime)

        # Calculate context similarity
        context_match = self.regime_detector._calculate_context_match(
            strategy_context, current_context
        )

        # Adjust for regime confidence
        regime_confidence = current_regime.confidence

        return context_match * regime_confidence

    def _calculate_performance_score(self, strategy: StrategyMetadata) -> float:
        """Calculate performance score for a strategy."""
        perf = strategy.performance

        # Normalize Sharpe ratio (0-1 scale, assuming max Sharpe of 3.0)
        sharpe_score = min(perf.sharpe_ratio / 3.0, 1.0)

        # Normalize CAGR (0-1 scale, assuming max CAGR of 50%)
        cagr_score = min(perf.cagr / 0.5, 1.0)

        # Normalize drawdown (0-1 scale, lower is better)
        drawdown_score = max(0, 1.0 - perf.max_drawdown / 0.2)

        # Normalize consistency
        consistency_score = perf.consistency_score

        # Weighted average
        performance_score = (
            0.4 * sharpe_score + 0.3 * cagr_score + 0.2 * drawdown_score + 0.1 * consistency_score
        )

        return performance_score

    def _calculate_confidence_score(self, strategy: StrategyMetadata) -> float:
        """Calculate confidence score based on usage history and success rate."""
        # Usage confidence (more usage = higher confidence)
        usage_confidence = min(strategy.usage_count / 100, 1.0)

        # Success rate confidence
        success_confidence = strategy.success_rate

        # Recency confidence (more recent = higher confidence)
        days_since_update = (datetime.now() - strategy.last_updated).days
        recency_confidence = max(0, 1.0 - days_since_update / 365)

        # Weighted average
        confidence_score = (
            0.4 * usage_confidence + 0.4 * success_confidence + 0.2 * recency_confidence
        )

        return confidence_score

    def _calculate_risk_score(self, strategy: StrategyMetadata) -> float:
        """Calculate risk score (lower is better)."""
        perf = strategy.performance

        # Use sortino ratio as proxy for volatility (higher sortino = lower risk)
        # Normalize sortino ratio (0-1 scale, higher is better)
        sortino_score = min(1.0, perf.sortino_ratio / 2.0)  # Cap at 2.0 for normalization

        # Normalize beta (0-1 scale, closer to 1 is better)
        beta_score = max(0, 1.0 - abs(perf.beta - 1.0))

        # Normalize drawdown (0-1 scale, lower is better)
        drawdown_score = max(0, 1.0 - perf.max_drawdown / 0.2)

        # Weighted average
        risk_score = 0.4 * sortino_score + 0.3 * beta_score + 0.3 * drawdown_score

        return risk_score

    async def _calculate_adaptation_score(
        self, strategy: StrategyMetadata, market_data: pd.DataFrame
    ) -> float:
        """Calculate adaptation score based on temporal adaptation."""
        try:
            # Check if strategy needs adaptation
            adaptation_needed = self.temporal_adaptation.check_adaptation_needed(
                strategy.strategy_id, market_data
            )

            if not adaptation_needed:
                return 1.0  # No adaptation needed = high score

            # Calculate adaptation quality
            adaptation_quality = self.temporal_adaptation.calculate_adaptation_quality(
                strategy.strategy_id
            )

            return adaptation_quality

        except Exception as e:
            logger.warning(f"Error calculating adaptation score: {e}")
            return 0.5  # Default score

    def _determine_selection_reason(
        self,
        regime_match: float,
        performance: float,
        confidence: float,
        risk: float,
        adaptation: float,
    ) -> str:
        """Determine the primary reason for strategy selection."""
        scores = [
            ("regime_match", regime_match),
            ("performance", performance),
            ("confidence", confidence),
            ("risk", risk),
            ("adaptation", adaptation),
        ]

        # Find the highest scoring factor
        best_factor, best_score = max(scores, key=lambda x: x[1])

        if best_score < 0.5:
            return "balanced_selection"

        return f"strong_{best_factor}"

    async def _get_current_market_data(self) -> pd.DataFrame:
        """Get current market data for regime detection.
        
        Note: This method requires integration with LiveDataManager.
        Currently returns empty DataFrame as placeholder.
        See LiveDataManager class for implementation details.
        """
        # Integration point for LiveDataManager
        # When implementing, use: self.data_manager.get_latest_market_data()
        return pd.DataFrame()

    def _record_selection(
        self, selected_strategies: list[StrategyScore], current_regime: RegimeCharacteristics
    ) -> None:
        """Record the current selection for analysis."""
        selection_record = {
            "timestamp": datetime.now(),
            "regime": current_regime.regime.value,
            "regime_confidence": current_regime.confidence,
            "selected_strategies": [
                {
                    "strategy_id": score.strategy_id,
                    "overall_score": score.overall_score,
                    "selection_reason": score.selection_reason,
                }
                for score in selected_strategies
            ],
        }

        self.selection_history.append(selection_record)

    def get_current_selection(self) -> list[StrategyScore]:
        """Get the current strategy selection."""
        return self.current_selection

    def get_selection_summary(self) -> dict[str, Any]:
        """Get a summary of the current selection."""
        if not self.current_selection:
            return {"status": "no_selection"}

        return {
            "timestamp": self.last_selection_time,
            "n_strategies": len(self.current_selection),
            "avg_score": np.mean([s.overall_score for s in self.current_selection]),
            "strategies": [
                {
                    "strategy_id": s.strategy_id,
                    "name": s.strategy.name,
                    "overall_score": s.overall_score,
                    "selection_reason": s.selection_reason,
                }
                for s in self.current_selection
            ],
        }

    def get_selection_history(self) -> list[dict[str, Any]]:
        """Get the selection history."""
        return self.selection_history
