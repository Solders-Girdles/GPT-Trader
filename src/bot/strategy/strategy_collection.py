"""
Strategy Collection System for GPT-Trader

Builds and manages a comprehensive library of validated trading strategies:
- Strategy discovery and registration
- Performance-based strategy ranking
- Strategy metadata and categorization
- Collection management and curation
- Strategy recommendation system

This forms the foundation for multi-strategy portfolio construction in Week 4.
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np

# Week 3 imports
from bot.strategy.base import Strategy

# Week 2 imports
from bot.strategy.persistence import create_filesystem_persistence
from bot.strategy.validation_engine import ValidationResult
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class StrategyCategory(Enum):
    """Strategy categories for organization"""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    MULTI_FACTOR = "multi_factor"


class PerformanceTier(Enum):
    """Performance-based strategy tiers"""

    ELITE = "elite"  # Top 10% performers
    PREMIUM = "premium"  # Top 25% performers
    STANDARD = "standard"  # Top 50% performers
    BASIC = "basic"  # Below median performers
    EXPERIMENTAL = "experimental"  # Unvalidated or poor performers


@dataclass
class StrategyMetrics:
    """Comprehensive strategy performance metrics"""

    strategy_id: str
    strategy_name: str

    # Core performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    total_return: float
    volatility: float

    # Risk metrics
    var_95: float
    cvar_95: float
    beta: float
    alpha: float

    # Trade-based metrics
    win_rate: float
    profit_factor: float
    average_trade: float
    max_consecutive_losses: int

    # Validation metrics
    validation_score: float
    confidence_level: float

    # Metadata
    category: StrategyCategory
    performance_tier: PerformanceTier
    data_period_days: int
    last_validated: datetime
    created_at: datetime


@dataclass
class StrategyCollectionStats:
    """Statistics for the strategy collection"""

    total_strategies: int
    by_category: dict[str, int]
    by_tier: dict[str, int]
    average_sharpe: float
    top_performers: list[str]
    recent_additions: list[str]
    validation_coverage: float


class StrategyCollection:
    """Manages a collection of validated trading strategies"""

    def __init__(self, collection_dir: str = "data/strategy_collection") -> None:
        self.collection_dir = Path(collection_dir)
        self.collection_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.collection_dir / "strategies").mkdir(exist_ok=True)
        (self.collection_dir / "metadata").mkdir(exist_ok=True)
        (self.collection_dir / "performance").mkdir(exist_ok=True)
        (self.collection_dir / "reports").mkdir(exist_ok=True)

        # Initialize database
        self.db_path = self.collection_dir / "strategy_collection.db"
        self._initialize_database()

        # Initialize persistence manager (filesystem backend)
        self.persistence_manager = create_filesystem_persistence(
            base_path=str(self.collection_dir / "strategies")
        )

        # Collection state
        self.strategies_cache: dict[str, StrategyMetrics] = {}
        self.last_refresh = datetime.min

        logger.info(f"Strategy Collection initialized at {self.collection_dir}")

    def _initialize_database(self) -> None:
        """Initialize SQLite database for strategy metadata"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    performance_tier TEXT NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_return REAL,
                    validation_score REAL,
                    confidence_level REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    data_period_days INTEGER,
                    last_validated TEXT,
                    created_at TEXT,
                    metadata_json TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    validation_date TEXT NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    validation_score REAL,
                    FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
                )
            """
            )

            # Create indexes for efficient queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON strategies (category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON strategies (performance_tier)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sharpe ON strategies (sharpe_ratio DESC)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_validation_score ON strategies (validation_score DESC)"
            )

            conn.commit()

    def add_strategy(
        self,
        strategy: Strategy,
        validation_result: ValidationResult,
        category: StrategyCategory = None,
        force_tier: PerformanceTier = None,
    ) -> str:
        """Add a validated strategy to the collection"""

        try:
            # Extract performance metrics
            metrics = self._extract_strategy_metrics(strategy, validation_result, category)

            # Determine performance tier if not forced
            if force_tier is None:
                metrics.performance_tier = self._calculate_performance_tier(metrics)
            else:
                metrics.performance_tier = force_tier

            # Register with persistence manager
            strategy_metadata = {
                "category": metrics.category.value,
                "performance_tier": metrics.performance_tier.value,
                "sharpe_ratio": metrics.sharpe_ratio,
                "validation_score": metrics.validation_score,
                "created_at": metrics.created_at.isoformat(),
            }

            strategy_record = self.persistence_manager.register_strategy(
                strategy=strategy, metadata=strategy_metadata
            )

            metrics.strategy_id = strategy_record.strategy_id

            # Store in database
            self._store_strategy_metrics(metrics)

            # Update cache
            self.strategies_cache[metrics.strategy_id] = metrics

            console.print(f"   ✅ Added {strategy.name} to collection")
            console.print(f"      • Category: {metrics.category.value}")
            console.print(f"      • Tier: {metrics.performance_tier.value}")
            console.print(f"      • Sharpe: {metrics.sharpe_ratio:.3f}")

            logger.info(
                f"Strategy {strategy.name} added to collection with ID {metrics.strategy_id}"
            )

            return metrics.strategy_id

        except Exception as e:
            logger.error(f"Failed to add strategy {strategy.name}: {str(e)}")
            raise

    def _extract_strategy_metrics(
        self,
        strategy: Strategy,
        validation_result: ValidationResult,
        category: StrategyCategory = None,
    ) -> StrategyMetrics:
        """Extract comprehensive metrics from strategy and validation results"""

        # Auto-detect category if not provided
        if category is None:
            category = self._auto_detect_category(strategy)

        # Extract performance metrics
        perf_metrics = validation_result.performance_metrics

        return StrategyMetrics(
            strategy_id="",  # Will be set when registered
            strategy_name=strategy.name,
            # Core performance
            sharpe_ratio=getattr(perf_metrics, "sharpe_ratio", 0.0),
            sortino_ratio=getattr(perf_metrics, "sortino_ratio", 0.0),
            calmar_ratio=getattr(perf_metrics, "calmar_ratio", 0.0),
            max_drawdown=getattr(perf_metrics, "max_drawdown", 0.0),
            total_return=getattr(perf_metrics, "total_return", 0.0),
            volatility=getattr(perf_metrics, "volatility", 0.0),
            # Risk metrics
            var_95=getattr(perf_metrics, "var_95", 0.0),
            cvar_95=getattr(perf_metrics, "cvar_95", 0.0),
            beta=getattr(perf_metrics, "beta", 1.0),
            alpha=getattr(perf_metrics, "alpha", 0.0),
            # Trade metrics
            win_rate=getattr(perf_metrics, "win_rate", 0.0),
            profit_factor=getattr(perf_metrics, "profit_factor", 1.0),
            average_trade=getattr(perf_metrics, "average_trade", 0.0),
            max_consecutive_losses=getattr(perf_metrics, "max_consecutive_losses", 0),
            # Validation metrics
            validation_score=validation_result.overall_score,
            confidence_level=validation_result.confidence_level,
            # Metadata
            category=category,
            performance_tier=PerformanceTier.EXPERIMENTAL,  # Will be calculated
            data_period_days=getattr(validation_result, "data_period_days", 365),
            last_validated=datetime.now(),
            created_at=datetime.now(),
        )

    def _auto_detect_category(self, strategy: Strategy) -> StrategyCategory:
        """Auto-detect strategy category based on name and characteristics"""

        name_lower = strategy.name.lower()

        # Pattern matching for category detection
        if any(word in name_lower for word in ["ma", "moving_average", "sma", "ema", "trend"]):
            return StrategyCategory.TREND_FOLLOWING
        elif any(
            word in name_lower for word in ["bollinger", "mean_reversion", "reversion", "rsi"]
        ):
            return StrategyCategory.MEAN_REVERSION
        elif any(word in name_lower for word in ["momentum", "macd", "stochastic"]):
            return StrategyCategory.MOMENTUM
        elif any(word in name_lower for word in ["breakout", "channel", "donchian"]):
            return StrategyCategory.BREAKOUT
        elif any(word in name_lower for word in ["volatility", "atr", "vix"]):
            return StrategyCategory.VOLATILITY
        else:
            return StrategyCategory.MULTI_FACTOR

    def _calculate_performance_tier(self, metrics: StrategyMetrics) -> PerformanceTier:
        """Calculate performance tier based on comprehensive metrics"""

        # Load all strategies for benchmarking
        all_strategies = self.get_all_strategies()

        if len(all_strategies) < 5:  # Not enough data for proper tiering
            # Use absolute thresholds for small collections
            if metrics.sharpe_ratio >= 1.5 and metrics.validation_score >= 85:
                return PerformanceTier.ELITE
            elif metrics.sharpe_ratio >= 1.0 and metrics.validation_score >= 75:
                return PerformanceTier.PREMIUM
            elif metrics.sharpe_ratio >= 0.5 and metrics.validation_score >= 65:
                return PerformanceTier.STANDARD
            else:
                return PerformanceTier.BASIC

        # Calculate percentiles for relative tiering
        sharpe_values = [s.sharpe_ratio for s in all_strategies]
        validation_values = [s.validation_score for s in all_strategies]

        sharpe_percentile = self._calculate_percentile(metrics.sharpe_ratio, sharpe_values)
        validation_percentile = self._calculate_percentile(
            metrics.validation_score, validation_values
        )

        # Combined score (weighted average)
        combined_percentile = 0.6 * sharpe_percentile + 0.4 * validation_percentile

        # Tier assignment based on percentiles
        if combined_percentile >= 90:
            return PerformanceTier.ELITE
        elif combined_percentile >= 75:
            return PerformanceTier.PREMIUM
        elif combined_percentile >= 50:
            return PerformanceTier.STANDARD
        else:
            return PerformanceTier.BASIC

    def _calculate_percentile(self, value: float, values: list[float]) -> float:
        """Calculate percentile rank of value within values"""
        if not values:
            return 50.0

        below_count = sum(1 for v in values if v < value)
        equal_count = sum(1 for v in values if v == value)

        percentile = (below_count + 0.5 * equal_count) / len(values) * 100
        return percentile

    def _store_strategy_metrics(self, metrics: StrategyMetrics) -> None:
        """Store strategy metrics in database"""

        with sqlite3.connect(self.db_path) as conn:
            # Insert or replace main strategy record
            conn.execute(
                """
                INSERT OR REPLACE INTO strategies (
                    strategy_id, strategy_name, category, performance_tier,
                    sharpe_ratio, max_drawdown, total_return, validation_score,
                    confidence_level, win_rate, profit_factor, data_period_days,
                    last_validated, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.strategy_id,
                    metrics.strategy_name,
                    metrics.category.value,
                    metrics.performance_tier.value,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown,
                    metrics.total_return,
                    metrics.validation_score,
                    metrics.confidence_level,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.data_period_days,
                    metrics.last_validated.isoformat(),
                    metrics.created_at.isoformat(),
                    json.dumps(asdict(metrics), default=str),
                ),
            )

            # Insert performance history record
            conn.execute(
                """
                INSERT INTO performance_history (
                    strategy_id, validation_date, sharpe_ratio, max_drawdown, validation_score
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    metrics.strategy_id,
                    metrics.last_validated.isoformat(),
                    metrics.sharpe_ratio,
                    metrics.max_drawdown,
                    metrics.validation_score,
                ),
            )

            conn.commit()

    def get_strategies_by_category(self, category: StrategyCategory) -> list[StrategyMetrics]:
        """Get all strategies in a specific category"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT metadata_json FROM strategies
                WHERE category = ?
                ORDER BY validation_score DESC
            """,
                (category.value,),
            )

            strategies = []
            for row in cursor:
                strategy_data = json.loads(row[0])
                # Convert back to StrategyMetrics (simplified)
                strategy_data["category"] = StrategyCategory(strategy_data["category"])
                strategy_data["performance_tier"] = PerformanceTier(
                    strategy_data["performance_tier"]
                )
                strategy_data["last_validated"] = datetime.fromisoformat(
                    strategy_data["last_validated"]
                )
                strategy_data["created_at"] = datetime.fromisoformat(strategy_data["created_at"])

                strategies.append(StrategyMetrics(**strategy_data))

            return strategies

    def get_strategies_by_tier(self, tier: PerformanceTier) -> list[StrategyMetrics]:
        """Get all strategies in a performance tier"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT metadata_json FROM strategies
                WHERE performance_tier = ?
                ORDER BY sharpe_ratio DESC
            """,
                (tier.value,),
            )

            strategies = []
            for row in cursor:
                strategy_data = json.loads(row[0])
                strategy_data["category"] = StrategyCategory(strategy_data["category"])
                strategy_data["performance_tier"] = PerformanceTier(
                    strategy_data["performance_tier"]
                )
                strategy_data["last_validated"] = datetime.fromisoformat(
                    strategy_data["last_validated"]
                )
                strategy_data["created_at"] = datetime.fromisoformat(strategy_data["created_at"])

                strategies.append(StrategyMetrics(**strategy_data))

            return strategies

    def get_top_performers(
        self, limit: int = 10, metric: str = "sharpe_ratio"
    ) -> list[StrategyMetrics]:
        """Get top performing strategies by specified metric"""

        valid_metrics = ["sharpe_ratio", "validation_score", "total_return", "calmar_ratio"]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Choose from: {valid_metrics}")

        # Use parameterized query with validated metric (whitelist approach prevents SQL injection)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT metadata_json FROM strategies
                ORDER BY {metric} DESC
                LIMIT ?
            """,
                (limit,),
            )

            strategies = []
            for row in cursor:
                strategy_data = json.loads(row[0])
                strategy_data["category"] = StrategyCategory(strategy_data["category"])
                strategy_data["performance_tier"] = PerformanceTier(
                    strategy_data["performance_tier"]
                )
                strategy_data["last_validated"] = datetime.fromisoformat(
                    strategy_data["last_validated"]
                )
                strategy_data["created_at"] = datetime.fromisoformat(strategy_data["created_at"])

                strategies.append(StrategyMetrics(**strategy_data))

            return strategies

    def get_all_strategies(self) -> list[StrategyMetrics]:
        """Get all strategies in the collection"""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT metadata_json FROM strategies
                ORDER BY validation_score DESC
            """
            )

            strategies = []
            for row in cursor:
                strategy_data = json.loads(row[0])
                strategy_data["category"] = StrategyCategory(strategy_data["category"])
                strategy_data["performance_tier"] = PerformanceTier(
                    strategy_data["performance_tier"]
                )
                strategy_data["last_validated"] = datetime.fromisoformat(
                    strategy_data["last_validated"]
                )
                strategy_data["created_at"] = datetime.fromisoformat(strategy_data["created_at"])

                strategies.append(StrategyMetrics(**strategy_data))

            return strategies

    def get_collection_stats(self) -> StrategyCollectionStats:
        """Get comprehensive statistics about the strategy collection"""

        all_strategies = self.get_all_strategies()

        if not all_strategies:
            return StrategyCollectionStats(
                total_strategies=0,
                by_category={},
                by_tier={},
                average_sharpe=0.0,
                top_performers=[],
                recent_additions=[],
                validation_coverage=0.0,
            )

        # Count by category
        by_category = {}
        for category in StrategyCategory:
            count = len([s for s in all_strategies if s.category == category])
            if count > 0:
                by_category[category.value] = count

        # Count by tier
        by_tier = {}
        for tier in PerformanceTier:
            count = len([s for s in all_strategies if s.performance_tier == tier])
            if count > 0:
                by_tier[tier.value] = count

        # Top performers
        top_performers = [s.strategy_name for s in all_strategies[:5]]

        # Recent additions (last 7 days)
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_additions = [s.strategy_name for s in all_strategies if s.created_at > recent_cutoff]

        # Validation coverage (strategies validated in last 30 days)
        validation_cutoff = datetime.now() - timedelta(days=30)
        recently_validated = len(
            [s for s in all_strategies if s.last_validated > validation_cutoff]
        )
        validation_coverage = recently_validated / len(all_strategies) * 100

        return StrategyCollectionStats(
            total_strategies=len(all_strategies),
            by_category=by_category,
            by_tier=by_tier,
            average_sharpe=np.mean([s.sharpe_ratio for s in all_strategies]),
            top_performers=top_performers,
            recent_additions=recent_additions,
            validation_coverage=validation_coverage,
        )

    def recommend_strategies_for_portfolio(
        self,
        target_categories: list[StrategyCategory] = None,
        min_sharpe: float = 0.5,
        max_correlation: float = 0.7,
        max_strategies: int = 10,
    ) -> list[StrategyMetrics]:
        """Recommend strategies for portfolio construction"""

        # Get all strategies meeting minimum criteria
        candidates = [
            s
            for s in self.get_all_strategies()
            if s.sharpe_ratio >= min_sharpe and s.performance_tier != PerformanceTier.EXPERIMENTAL
        ]

        # Filter by target categories if specified
        if target_categories:
            candidates = [s for s in candidates if s.category in target_categories]

        if not candidates:
            return []

        # Sort by composite score (Sharpe * validation_score)
        candidates.sort(key=lambda s: s.sharpe_ratio * (s.validation_score / 100), reverse=True)

        # Select diversified set (simplified correlation approximation)
        selected = []
        selected_categories = set()

        for candidate in candidates:
            if len(selected) >= max_strategies:
                break

            # Favor category diversity
            if candidate.category not in selected_categories or len(selected) < len(
                StrategyCategory
            ):
                selected.append(candidate)
                selected_categories.add(candidate.category)

        return selected[:max_strategies]

    def display_collection_summary(self) -> None:
        """Display a comprehensive summary of the strategy collection"""

        stats = self.get_collection_stats()

        console.print(
            Panel(
                f"[bold blue]Strategy Collection Summary[/bold blue]\n"
                f"Total Strategies: {stats.total_strategies}\n"
                f"Average Sharpe Ratio: {stats.average_sharpe:.3f}\n"
                f"Validation Coverage: {stats.validation_coverage:.1f}%",
                title="📊 Collection Overview",
            )
        )

        if stats.total_strategies > 0:
            # Category breakdown
            category_table = Table(title="🏷️  Strategies by Category")
            category_table.add_column("Category", style="cyan")
            category_table.add_column("Count", justify="right")
            category_table.add_column("Percentage", justify="right", style="dim")

            for category, count in stats.by_category.items():
                percentage = (count / stats.total_strategies) * 100
                category_table.add_row(
                    category.replace("_", " ").title(), str(count), f"{percentage:.1f}%"
                )

            console.print(category_table)

            # Performance tier breakdown
            tier_table = Table(title="🏆 Strategies by Performance Tier")
            tier_table.add_column("Tier", style="cyan")
            tier_table.add_column("Count", justify="right")
            tier_table.add_column("Percentage", justify="right", style="dim")

            for tier, count in stats.by_tier.items():
                percentage = (count / stats.total_strategies) * 100
                tier_table.add_row(tier.replace("_", " ").title(), str(count), f"{percentage:.1f}%")

            console.print(tier_table)

            # Top performers
            if stats.top_performers:
                console.print("\n🌟 [bold]Top Performers[/bold]")
                for i, strategy_name in enumerate(stats.top_performers[:5], 1):
                    console.print(f"   {i}. {strategy_name}")


def create_strategy_collection(
    collection_dir: str = "data/strategy_collection",
) -> StrategyCollection:
    """Factory function to create strategy collection"""
    return StrategyCollection(collection_dir=collection_dir)


if __name__ == "__main__":
    # Example usage
    collection = create_strategy_collection()
    collection.display_collection_summary()
    print("Strategy Collection system created successfully!")
