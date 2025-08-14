"""
Strategy Knowledge Base for persistent storage and contextual retrieval of discovered strategies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    """Contextual information for a strategy."""

    market_regime: str  # "trending", "volatile", "sideways", "crisis"
    time_period: str  # "bull_market", "bear_market", "sideways_market"
    asset_class: str  # "equity", "commodity", "forex", "crypto"
    risk_profile: str  # "conservative", "moderate", "aggressive"
    volatility_regime: str  # "low", "medium", "high"
    correlation_regime: str  # "low", "medium", "high"
    seasonality: str | None = None  # "spring", "summer", "fall", "winter"
    day_of_week: str | None = None  # "monday", "friday", etc.


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""

    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    win_rate: float
    consistency_score: float
    n_trades: int
    avg_trade_duration: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float
    beta: float
    alpha: float


@dataclass
class StrategyMetadata:
    """Metadata for a stored strategy."""

    strategy_id: str
    name: str
    description: str
    strategy_type: str  # "trend_following", "mean_reversion", "momentum", etc.
    parameters: dict[str, Any]
    context: StrategyContext
    performance: StrategyPerformance
    discovery_date: datetime
    last_updated: datetime
    usage_count: int = 0
    success_rate: float = 0.0
    tags: list[str] = None
    notes: str = ""


class StrategyKnowledgeBase:
    """Persistent storage and retrieval system for discovered strategies."""

    def __init__(self, storage_path: str = "data/strategy_knowledge") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self.strategies: dict[str, StrategyMetadata] = {}
        self.strategy_fingerprints: dict[str, str] = {}
        self.performance_index: dict[str, list[str]] = {}
        self.context_index: dict[str, list[str]] = {}

        # Load existing data
        self._load_knowledge_base()

    def add_strategy(self, strategy: StrategyMetadata) -> bool:
        """Add a strategy to the knowledge base."""
        try:
            # Generate fingerprint
            fingerprint = self._generate_fingerprint(strategy.parameters)

            # Check if similar strategy exists
            if fingerprint in self.strategy_fingerprints:
                existing_id = self.strategy_fingerprints[fingerprint]
                existing = self.strategies[existing_id]

                # Update if better performance
                if strategy.performance.sharpe_ratio > existing.performance.sharpe_ratio:
                    logger.info(f"Updating strategy {existing_id} with better performance")
                    self._update_strategy(existing_id, strategy)
                else:
                    logger.info(
                        f"Strategy {strategy.strategy_id} has lower performance than existing"
                    )
                    return False
            else:
                # Add new strategy
                self.strategies[strategy.strategy_id] = strategy
                self.strategy_fingerprints[fingerprint] = strategy.strategy_id

                # Update indices
                self._update_indices(strategy)

                # Save to disk
                self._save_strategy(strategy)

                logger.info(
                    f"Added new strategy {strategy.strategy_id} with Sharpe {strategy.performance.sharpe_ratio:.4f}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to add strategy: {e}")
            return False

    def find_strategies(
        self,
        context: StrategyContext | None = None,
        min_sharpe: float = 0.0,
        max_drawdown: float = float("inf"),
        strategy_type: str | None = None,
        limit: int = 10,
    ) -> list[StrategyMetadata]:
        """Find strategies matching criteria."""
        candidates = []

        for strategy in self.strategies.values():
            # Performance filters
            if strategy.performance.sharpe_ratio < min_sharpe:
                continue
            if strategy.performance.max_drawdown > max_drawdown:
                continue

            # Strategy type filter
            if strategy_type and strategy.strategy_type != strategy_type:
                continue

            # Context matching
            if context and not self._context_matches(strategy.context, context):
                continue

            candidates.append(strategy)

        # Sort by Sharpe ratio and return top results
        candidates.sort(key=lambda s: s.performance.sharpe_ratio, reverse=True)
        return candidates[:limit]

    def get_strategy_recommendations(
        self, current_context: StrategyContext, n_recommendations: int = 5
    ) -> list[StrategyMetadata]:
        """Get strategy recommendations for current market context."""
        # Find strategies that performed well in similar contexts
        similar_strategies = self.find_strategies(
            context=current_context, min_sharpe=1.0, limit=n_recommendations * 2
        )

        # If not enough similar strategies, expand search
        if len(similar_strategies) < n_recommendations:
            general_strategies = self.find_strategies(min_sharpe=1.5, limit=n_recommendations)
            similar_strategies.extend(general_strategies)

        # Remove duplicates and return top recommendations
        unique_strategies = list({s.strategy_id: s for s in similar_strategies}.values())
        return unique_strategies[:n_recommendations]

    def analyze_strategy_families(self) -> dict[str, Any]:
        """Analyze patterns in strategy families."""
        families = {}

        for strategy in self.strategies.values():
            family_key = f"{strategy.strategy_type}_{strategy.context.market_regime}"

            if family_key not in families:
                families[family_key] = {
                    "count": 0,
                    "avg_sharpe": 0.0,
                    "best_sharpe": 0.0,
                    "strategies": [],
                }

            family = families[family_key]
            family["count"] += 1
            family["avg_sharpe"] += strategy.performance.sharpe_ratio
            family["best_sharpe"] = max(family["best_sharpe"], strategy.performance.sharpe_ratio)
            family["strategies"].append(strategy.strategy_id)

        # Calculate averages
        for family in families.values():
            family["avg_sharpe"] /= family["count"]

        return families

    def get_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {
            "sharpe_trend": [],
            "discovery_rate": [],
            "strategy_types": {},
            "market_regimes": {},
        }

        # Group strategies by discovery date
        strategies_by_date = {}
        for strategy in self.strategies.values():
            date_key = strategy.discovery_date.strftime("%Y-%m")
            if date_key not in strategies_by_date:
                strategies_by_date[date_key] = []
            strategies_by_date[date_key].append(strategy)

        # Calculate trends
        for date_key in sorted(strategies_by_date.keys()):
            strategies = strategies_by_date[date_key]
            avg_sharpe = np.mean([s.performance.sharpe_ratio for s in strategies])

            trends["sharpe_trend"].append(
                {"date": date_key, "avg_sharpe": avg_sharpe, "count": len(strategies)}
            )

        return trends

    def _generate_fingerprint(self, parameters: dict[str, Any]) -> str:
        """Generate a fingerprint for strategy parameters."""
        # Sort parameters for consistent fingerprinting
        sorted_params = sorted(parameters.items())
        param_string = json.dumps(sorted_params, sort_keys=True)

        # Simple hash (could use more sophisticated methods)
        import hashlib

        return hashlib.md5(param_string.encode()).hexdigest()

    def _context_matches(
        self, strategy_context: StrategyContext, query_context: StrategyContext
    ) -> bool:
        """Check if strategy context matches query context."""
        # Exact matches for critical fields
        if (
            strategy_context.market_regime != query_context.market_regime
            or strategy_context.asset_class != query_context.asset_class
        ):
            return False

        # Flexible matches for other fields
        if (
            query_context.risk_profile
            and strategy_context.risk_profile != query_context.risk_profile
        ):
            return False

        return True

    def _update_indices(self, strategy: StrategyMetadata) -> None:
        """Update search indices."""
        # Performance index
        sharpe_bucket = int(strategy.performance.sharpe_ratio)
        if sharpe_bucket not in self.performance_index:
            self.performance_index[sharpe_bucket] = []
        self.performance_index[sharpe_bucket].append(strategy.strategy_id)

        # Context index
        context_key = f"{strategy.context.market_regime}_{strategy.context.asset_class}"
        if context_key not in self.context_index:
            self.context_index[context_key] = []
        self.context_index[context_key].append(strategy.strategy_id)

    def _update_strategy(self, strategy_id: str, new_strategy: StrategyMetadata) -> None:
        """Update existing strategy with new data."""
        new_strategy.strategy_id = strategy_id
        new_strategy.last_updated = datetime.now()
        new_strategy.usage_count = self.strategies[strategy_id].usage_count

        self.strategies[strategy_id] = new_strategy
        self._save_strategy(new_strategy)

    def _save_strategy(self, strategy: StrategyMetadata) -> None:
        """Save strategy to disk."""
        strategy_file = self.storage_path / f"{strategy.strategy_id}.json"

        # Convert to dict for JSON serialization
        strategy_dict = asdict(strategy)
        strategy_dict["discovery_date"] = strategy.discovery_date.isoformat()
        strategy_dict["last_updated"] = strategy.last_updated.isoformat()

        with open(strategy_file, "w") as f:
            json.dump(strategy_dict, f, indent=2)

    def _load_knowledge_base(self) -> None:
        """Load existing strategies from disk."""
        for strategy_file in self.storage_path.glob("*.json"):
            try:
                with open(strategy_file) as f:
                    data = json.load(f)

                # Reconstruct dataclass objects
                context = StrategyContext(**data["context"])
                performance = StrategyPerformance(**data["performance"])

                strategy = StrategyMetadata(
                    strategy_id=data["strategy_id"],
                    name=data["name"],
                    description=data["description"],
                    strategy_type=data["strategy_type"],
                    parameters=data["parameters"],
                    context=context,
                    performance=performance,
                    discovery_date=datetime.fromisoformat(data["discovery_date"]),
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                    usage_count=data.get("usage_count", 0),
                    success_rate=data.get("success_rate", 0.0),
                    tags=data.get("tags", []),
                    notes=data.get("notes", ""),
                )

                self.strategies[strategy.strategy_id] = strategy

                # Rebuild indices
                fingerprint = self._generate_fingerprint(strategy.parameters)
                self.strategy_fingerprints[fingerprint] = strategy.strategy_id
                self._update_indices(strategy)

            except Exception as e:
                logger.error(f"Failed to load strategy from {strategy_file}: {e}")

        logger.info(f"Loaded {len(self.strategies)} strategies from knowledge base")
