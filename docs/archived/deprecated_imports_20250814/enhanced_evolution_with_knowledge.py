"""
Enhanced Evolution System with Knowledge Base Integration for persistent strategy discovery.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyKnowledgeBase,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.meta_learning.strategy_transfer import AssetCharacteristics, StrategyTransferEngine
from bot.optimization.enhanced_evolution import EnhancedEvolutionEngine

logger = logging.getLogger(__name__)


class KnowledgeEnhancedEvolutionEngine(EnhancedEvolutionEngine):
    """Enhanced evolution engine with knowledge base integration."""

    def __init__(
        self,
        config: Any,
        strategy_config: Any,
        knowledge_base: StrategyKnowledgeBase | None = None,
        transfer_engine: StrategyTransferEngine | None = None,
    ) -> None:
        super().__init__(config, strategy_config)

        # Initialize knowledge components
        self.knowledge_base = knowledge_base or StrategyKnowledgeBase()
        self.transfer_engine = transfer_engine or StrategyTransferEngine(self.knowledge_base)

        # Enhanced tracking
        self.discovered_strategies: list[StrategyMetadata] = []
        self.knowledge_enhanced_individuals: list[dict[str, Any]] = []

    def evolve(
        self,
        evaluate_func: Callable[[dict[str, Any]], dict[str, Any]],
        generations: int = 100,
        population_size: int = 50,
        context: StrategyContext | None = None,
    ) -> dict[str, Any]:
        """Enhanced evolution with knowledge base integration."""

        # Initialize population with knowledge-enhanced individuals
        self.initialize_population(population_size)
        self._inject_knowledge_based_individuals(context)

        logger.info(
            f"Starting knowledge-enhanced evolution: {generations} generations, {population_size} population"
        )
        logger.info(f"Knowledge base contains {len(self.knowledge_base.strategies)} strategies")

        # Run evolution
        results = super().evolve(evaluate_func, generations, population_size)

        # Store discovered strategies in knowledge base
        self._store_discovered_strategies(context)

        # Generate knowledge insights
        knowledge_insights = self._generate_knowledge_insights()
        results["knowledge_insights"] = knowledge_insights

        return results

    def _inject_knowledge_based_individuals(self, context: StrategyContext | None = None) -> None:
        """Inject individuals based on knowledge base strategies."""
        if not self.knowledge_base.strategies:
            logger.info("Knowledge base is empty, skipping knowledge injection")
            return

        # Get relevant strategies from knowledge base
        relevant_strategies = self._get_relevant_strategies(context)

        if not relevant_strategies:
            logger.info("No relevant strategies found in knowledge base")
            return

        # Inject top-performing strategies
        n_injections = min(5, len(relevant_strategies), len(self.population) // 4)

        for i, strategy in enumerate(relevant_strategies[:n_injections]):
            # Create individual from strategy parameters
            individual = strategy.parameters.copy()

            # Add some noise to prevent exact duplicates
            individual = self._add_knowledge_noise(individual)

            # Replace random individual in population
            replace_idx = np.random.randint(0, len(self.population))
            self.population[replace_idx] = individual

            logger.info(
                f"Injected knowledge-based individual {i+1}/{n_injections} from strategy {strategy.strategy_id}"
            )

    def _get_relevant_strategies(
        self, context: StrategyContext | None = None
    ) -> list[StrategyMetadata]:
        """Get relevant strategies from knowledge base."""
        if context:
            # Find strategies for similar context
            relevant_strategies = self.knowledge_base.find_strategies(
                context=context, min_sharpe=1.0, limit=20
            )
        else:
            # Get top-performing strategies
            relevant_strategies = self.knowledge_base.find_strategies(min_sharpe=1.5, limit=20)

        return relevant_strategies

    def _add_knowledge_noise(self, individual: dict[str, Any]) -> dict[str, Any]:
        """Add small noise to knowledge-based individual to encourage exploration."""
        noisy_individual = individual.copy()

        for param_name, value in noisy_individual.items():
            if isinstance(value, int | float):
                # Add 5-15% noise
                noise_factor = np.random.uniform(0.85, 1.15)
                noisy_individual[param_name] = value * noise_factor

        return noisy_individual

    def _store_discovered_strategies(self, context: StrategyContext | None = None) -> None:
        """Store discovered strategies in knowledge base."""
        stored_count = 0

        # Get the best individuals from the final generation
        if not self.generation_history:
            logger.warning("No generation history available for storing strategies")
            return

        # Get results from the last generation
        last_generation = self.generation_history[-1]
        results = last_generation.get("results", [])

        if not results:
            logger.warning("No results available in last generation")
            return

        # Store top performers
        top_performers = sorted(
            results, key=lambda x: x.get("sharpe", float("-inf")), reverse=True
        )[:5]

        for result in top_performers:
            if not isinstance(result, dict) or "sharpe" not in result:
                continue

            # Get the individual parameters
            individual = result.get("params", {})
            if not individual:
                continue

            # Create strategy metadata
            strategy_metadata = self._create_strategy_metadata(individual, result, context)

            # Store in knowledge base
            if self.knowledge_base.add_strategy(strategy_metadata):
                stored_count += 1
                self.discovered_strategies.append(strategy_metadata)

        logger.info(f"Stored {stored_count} new strategies in knowledge base")

    def _create_strategy_metadata(
        self,
        individual: dict[str, Any],
        result: dict[str, Any],
        context: StrategyContext | None = None,
    ) -> StrategyMetadata:
        """Create strategy metadata from individual and result."""
        # Generate strategy ID
        strategy_id = f"strategy_{uuid.uuid4().hex[:8]}"

        # Create performance object
        performance = StrategyPerformance(
            sharpe_ratio=result.get("sharpe", 0.0),
            cagr=result.get("cagr", 0.0),
            max_drawdown=result.get("max_drawdown", 0.0),
            win_rate=result.get("win_rate", 0.0),
            consistency_score=result.get("consistency_score", 0.0),
            n_trades=result.get("n_trades", 0),
            avg_trade_duration=result.get("avg_trade_duration", 0.0),
            profit_factor=result.get("profit_factor", 0.0),
            calmar_ratio=result.get("calmar_ratio", 0.0),
            sortino_ratio=result.get("sortino_ratio", 0.0),
            information_ratio=result.get("information_ratio", 0.0),
            beta=result.get("beta", 0.0),
            alpha=result.get("alpha", 0.0),
        )

        # Create context if not provided
        if context is None:
            context = StrategyContext(
                market_regime="trending",  # Default
                time_period="bull_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="medium",
            )

        # Determine strategy type
        strategy_type = self._classify_strategy_type(individual)

        return StrategyMetadata(
            strategy_id=strategy_id,
            name=f"Discovered Strategy {strategy_id}",
            description=f"Strategy discovered through enhanced evolution with Sharpe {performance.sharpe_ratio:.4f}",
            strategy_type=strategy_type,
            parameters=individual,
            context=context,
            performance=performance,
            discovery_date=datetime.now(),
            last_updated=datetime.now(),
            tags=["enhanced_evolution", "discovered"],
            notes=f"Discovered in generation {len(self.generation_history)}",
        )

    def _classify_strategy_type(self, individual: dict[str, Any]) -> str:
        """Classify strategy type based on parameters."""
        # Simple classification based on key parameters
        if individual.get("use_rsi_filter", False):
            return "mean_reversion"
        elif individual.get("use_volume_filter", False):
            return "volume_breakout"
        elif individual.get("use_correlation_filter", False):
            return "correlation_based"
        elif individual.get("use_time_filter", False):
            return "time_based"
        else:
            return "trend_following"

    def _generate_knowledge_insights(self) -> dict[str, Any]:
        """Generate insights from knowledge base analysis."""
        insights = {
            "total_strategies": len(self.knowledge_base.strategies),
            "new_discoveries": len(self.discovered_strategies),
            "strategy_families": self.knowledge_base.analyze_strategy_families(),
            "performance_trends": self.knowledge_base.get_performance_trends(),
            "best_strategies": self._get_best_strategies(),
            "diversity_metrics": self._calculate_diversity_metrics(),
        }

        return insights

    def _get_best_strategies(self) -> list[dict[str, Any]]:
        """Get information about the best strategies in knowledge base."""
        best_strategies = self.knowledge_base.find_strategies(min_sharpe=1.5, limit=5)

        return [
            {
                "strategy_id": s.strategy_id,
                "strategy_type": s.strategy_type,
                "sharpe_ratio": s.performance.sharpe_ratio,
                "max_drawdown": s.performance.max_drawdown,
                "context": s.context.market_regime,
                "discovery_date": s.discovery_date.strftime("%Y-%m-%d"),
            }
            for s in best_strategies
        ]

    def _calculate_diversity_metrics(self) -> dict[str, Any]:
        """Calculate diversity metrics for the knowledge base."""
        if not self.knowledge_base.strategies:
            return {"diversity_score": 0.0, "strategy_types": {}}

        # Count strategy types
        type_counts = {}
        for strategy in self.knowledge_base.strategies.values():
            strategy_type = strategy.strategy_type
            type_counts[strategy_type] = type_counts.get(strategy_type, 0) + 1

        # Calculate diversity score (Shannon entropy)
        total_strategies = len(self.knowledge_base.strategies)
        diversity_score = 0.0

        for count in type_counts.values():
            p = count / total_strategies
            if p > 0:
                diversity_score -= p * np.log2(p)

        return {
            "diversity_score": diversity_score,
            "strategy_types": type_counts,
            "total_strategies": total_strategies,
        }

    def get_strategy_recommendations(
        self, current_context: StrategyContext, n_recommendations: int = 5
    ) -> list[StrategyMetadata]:
        """Get strategy recommendations for current context."""
        return self.knowledge_base.get_strategy_recommendations(current_context, n_recommendations)

    def transfer_strategy_to_context(
        self,
        source_strategy: StrategyMetadata,
        target_context: StrategyContext,
        target_asset: AssetCharacteristics,
    ) -> dict[str, Any]:
        """Transfer a strategy to a new context."""
        return self.transfer_engine.transfer_strategy(source_strategy, target_context, target_asset)

    def get_knowledge_summary(self) -> dict[str, Any]:
        """Get a summary of the knowledge base."""
        return {
            "total_strategies": len(self.knowledge_base.strategies),
            "strategy_families": self.knowledge_base.analyze_strategy_families(),
            "performance_trends": self.knowledge_base.get_performance_trends(),
            "diversity_metrics": self._calculate_diversity_metrics(),
            "best_performing": self._get_best_strategies()[:3],
        }
