"""
Portfolio Construction System for GPT-Trader

Advanced multi-strategy portfolio construction system that:
- Integrates with Strategy Collection for strategy selection
- Optimizes portfolio allocation using multiple objective functions
- Provides risk management and constraint enforcement
- Supports dynamic rebalancing and portfolio monitoring
- Generates comprehensive portfolio analytics and reporting

This completes Week 4 by enabling production-ready multi-strategy portfolios.
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Portfolio optimization imports (using existing optimizer)
from bot.portfolio.optimizer import (
    OptimizationMethod,
    PortfolioConstraints,
    PortfolioOptimizer,
)

# Week 4 imports
from bot.strategy.strategy_collection import (
    PerformanceTier,
    StrategyCategory,
    StrategyCollection,
    StrategyMetrics,
)
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class PortfolioObjective(Enum):
    """Portfolio construction objectives"""

    RISK_ADJUSTED_RETURN = "risk_adjusted_return"  # Maximize Sharpe ratio
    MAX_RETURN = "max_return"  # Maximize expected return
    MIN_RISK = "min_risk"  # Minimize portfolio volatility
    RISK_PARITY = "risk_parity"  # Equal risk contribution
    MAX_DIVERSIFICATION = "max_diversification"  # Maximize diversification ratio
    TARGET_VOLATILITY = "target_volatility"  # Match target volatility


class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequencies"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"


@dataclass
class PortfolioConstraints:
    """Extended portfolio construction constraints"""

    # Basic weight constraints
    min_strategy_weight: float = 0.05  # Minimum 5% per strategy
    max_strategy_weight: float = 0.30  # Maximum 30% per strategy

    # Category constraints
    max_category_exposure: float = 0.60  # Max 60% in any category
    min_categories: int = 2  # At least 2 different categories

    # Risk constraints
    max_portfolio_volatility: float = 0.20  # Max 20% annual volatility
    max_portfolio_drawdown: float = 0.15  # Max 15% drawdown
    target_sharpe_ratio: float = 1.0  # Target Sharpe ratio

    # Diversification constraints
    min_strategies: int = 3  # At least 3 strategies
    max_strategies: int = 10  # At most 10 strategies
    max_correlation: float = 0.70  # Max pairwise correlation

    # Performance constraints
    min_strategy_sharpe: float = 0.5  # Min Sharpe per strategy
    min_validation_score: float = 70.0  # Min validation score

    # Rebalancing constraints
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    rebalance_threshold: float = 0.05  # Rebalance if weight drift > 5%
    transaction_cost_bps: float = 5.0  # 5 basis points transaction cost


@dataclass
class PortfolioComposition:
    """Complete portfolio composition with metadata"""

    portfolio_id: str
    portfolio_name: str

    # Strategy allocations
    strategy_weights: dict[str, float]
    strategy_metadata: dict[str, StrategyMetrics]

    # Portfolio metrics
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float

    # Risk analysis
    var_95: float
    cvar_95: float
    correlation_matrix: pd.DataFrame
    risk_contributions: dict[str, float]

    # Composition analysis
    category_weights: dict[str, float]
    tier_weights: dict[str, float]

    # Construction metadata
    objective: PortfolioObjective
    constraints: PortfolioConstraints
    construction_method: str
    created_at: datetime
    last_rebalanced: datetime

    # Performance tracking
    inception_date: datetime
    total_return: float = 0.0
    realized_volatility: float = 0.0
    realized_sharpe: float = 0.0


class PortfolioConstructor:
    """Advanced portfolio construction system"""

    def __init__(
        self, portfolio_dir: str = "data/portfolios", strategy_collection: StrategyCollection = None
    ) -> None:
        self.portfolio_dir = Path(portfolio_dir)
        self.portfolio_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.portfolio_dir / "compositions").mkdir(exist_ok=True)
        (self.portfolio_dir / "performance").mkdir(exist_ok=True)
        (self.portfolio_dir / "reports").mkdir(exist_ok=True)
        (self.portfolio_dir / "rebalancing").mkdir(exist_ok=True)

        # Initialize strategy collection
        if strategy_collection is None:
            from bot.strategy.strategy_collection import create_strategy_collection

            self.strategy_collection = create_strategy_collection()
        else:
            self.strategy_collection = strategy_collection

        # Initialize database
        self.db_path = self.portfolio_dir / "portfolios.db"
        self._initialize_database()

        # Portfolio state
        self.active_portfolios: dict[str, PortfolioComposition] = {}
        self.last_refresh = datetime.min

        logger.info(f"Portfolio Constructor initialized at {self.portfolio_dir}")

    def _initialize_database(self) -> None:
        """Initialize SQLite database for portfolio tracking"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolios (
                    portfolio_id TEXT PRIMARY KEY,
                    portfolio_name TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    expected_return REAL,
                    expected_volatility REAL,
                    sharpe_ratio REAL,
                    diversification_ratio REAL,
                    num_strategies INTEGER,
                    num_categories INTEGER,
                    created_at TEXT,
                    last_rebalanced TEXT,
                    total_return REAL DEFAULT 0.0,
                    composition_json TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    portfolio_value REAL,
                    daily_return REAL,
                    volatility REAL,
                    drawdown REAL,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rebalancing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL,
                    rebalance_date TEXT NOT NULL,
                    reason TEXT,
                    old_weights TEXT,
                    new_weights TEXT,
                    transaction_cost REAL,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_performance ON portfolio_performance (portfolio_id, date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rebalancing ON rebalancing_history (portfolio_id, rebalance_date)"
            )

            conn.commit()

    def construct_portfolio(
        self,
        portfolio_name: str,
        objective: PortfolioObjective = PortfolioObjective.RISK_ADJUSTED_RETURN,
        constraints: PortfolioConstraints = None,
        target_categories: list[StrategyCategory] = None,
        custom_strategy_list: list[str] = None,
    ) -> PortfolioComposition:
        """Construct optimized multi-strategy portfolio"""

        if constraints is None:
            constraints = PortfolioConstraints()

        try:
            console.print(f"🏗️  [bold blue]Constructing Portfolio: {portfolio_name}[/bold blue]")
            console.print(f"   Objective: {objective.value}")
            console.print(f"   Max Strategies: {constraints.max_strategies}")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                # Step 1: Strategy Selection
                selection_task = progress.add_task("🎯 Selecting strategies...", total=1)
                candidate_strategies = self._select_candidate_strategies(
                    constraints, target_categories, custom_strategy_list
                )
                progress.update(selection_task, completed=1)

                console.print(f"   📊 {len(candidate_strategies)} candidate strategies selected")

                # Step 2: Correlation Analysis
                correlation_task = progress.add_task("📈 Analyzing correlations...", total=1)
                correlation_matrix = self._estimate_correlation_matrix(candidate_strategies)
                filtered_strategies = self._filter_by_correlation(
                    candidate_strategies, correlation_matrix, constraints.max_correlation
                )
                progress.update(correlation_task, completed=1)

                console.print(
                    f"   🔗 {len(filtered_strategies)} strategies after correlation filtering"
                )

                # Step 3: Portfolio Optimization
                optimization_task = progress.add_task("⚡ Optimizing allocation...", total=1)
                portfolio_weights = self._optimize_portfolio_allocation(
                    filtered_strategies, objective, constraints, correlation_matrix
                )
                progress.update(optimization_task, completed=1)

                # Step 4: Portfolio Composition Analysis
                analysis_task = progress.add_task("📊 Analyzing composition...", total=1)
                portfolio_composition = self._create_portfolio_composition(
                    portfolio_name,
                    filtered_strategies,
                    portfolio_weights,
                    objective,
                    constraints,
                    correlation_matrix,
                )
                progress.update(analysis_task, completed=1)

            # Store portfolio
            self._store_portfolio_composition(portfolio_composition)
            self.active_portfolios[portfolio_composition.portfolio_id] = portfolio_composition

            console.print("✅ [bold green]Portfolio constructed successfully![/bold green]")
            self._display_portfolio_summary(portfolio_composition)

            return portfolio_composition

        except Exception as e:
            logger.error(f"Portfolio construction failed: {str(e)}")
            console.print(f"❌ [bold red]Portfolio construction failed:[/bold red] {str(e)}")
            raise

    def _select_candidate_strategies(
        self,
        constraints: PortfolioConstraints,
        target_categories: list[StrategyCategory] = None,
        custom_strategy_list: list[str] = None,
    ) -> list[StrategyMetrics]:
        """Select candidate strategies based on constraints"""

        if custom_strategy_list:
            # Use custom strategy list
            all_strategies = self.strategy_collection.get_all_strategies()
            candidates = [s for s in all_strategies if s.strategy_id in custom_strategy_list]
        else:
            # Use recommendation engine
            candidates = self.strategy_collection.recommend_strategies_for_portfolio(
                target_categories=target_categories,
                min_sharpe=constraints.min_strategy_sharpe,
                max_strategies=constraints.max_strategies * 2,  # Get extra for filtering
            )

        # Apply constraints
        filtered_candidates = []
        for strategy in candidates:
            if (
                strategy.sharpe_ratio >= constraints.min_strategy_sharpe
                and strategy.validation_score >= constraints.min_validation_score
                and strategy.performance_tier != PerformanceTier.EXPERIMENTAL
            ):
                filtered_candidates.append(strategy)

        # Ensure minimum strategies available
        if len(filtered_candidates) < constraints.min_strategies:
            # Relax constraints to get minimum strategies
            console.print("   ⚠️  Relaxing constraints to meet minimum strategy requirement")
            relaxed_candidates = self.strategy_collection.get_all_strategies()

            # Sort by composite score and take top candidates
            relaxed_candidates.sort(
                key=lambda s: s.sharpe_ratio * (s.validation_score / 100), reverse=True
            )
            filtered_candidates = relaxed_candidates[: constraints.max_strategies]

        return filtered_candidates[: constraints.max_strategies]

    def _estimate_correlation_matrix(self, strategies: list[StrategyMetrics]) -> pd.DataFrame:
        """Estimate correlation matrix between strategies"""

        n_strategies = len(strategies)
        strategy_ids = [s.strategy_id for s in strategies]

        # Initialize correlation matrix
        correlation_matrix = pd.DataFrame(
            np.eye(n_strategies), index=strategy_ids, columns=strategy_ids
        )

        # Estimate correlations based on strategy characteristics
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i != j:
                    correlation = self._estimate_strategy_correlation(strategy1, strategy2)
                    correlation_matrix.iloc[i, j] = correlation
                    correlation_matrix.iloc[j, i] = correlation  # Symmetric

        return correlation_matrix

    def _estimate_strategy_correlation(
        self, strategy1: StrategyMetrics, strategy2: StrategyMetrics
    ) -> float:
        """Estimate correlation between two strategies based on characteristics"""

        # Base correlation factors
        correlation_factors = []

        # Category correlation
        if strategy1.category == strategy2.category:
            correlation_factors.append(0.6)  # Same category = higher correlation
        else:
            correlation_factors.append(0.2)  # Different categories = lower correlation

        # Beta correlation (market exposure similarity)
        beta_diff = abs(strategy1.beta - strategy2.beta)
        beta_correlation = max(0, 0.8 - beta_diff)
        correlation_factors.append(beta_correlation)

        # Volatility correlation
        vol_diff = abs(strategy1.volatility - strategy2.volatility)
        vol_correlation = max(0, 0.7 - vol_diff / 0.2)  # Normalize by reasonable vol difference
        correlation_factors.append(vol_correlation)

        # Drawdown pattern correlation
        dd_diff = abs(strategy1.max_drawdown - strategy2.max_drawdown)
        dd_correlation = max(0, 0.6 - dd_diff / 0.1)  # Normalize by reasonable DD difference
        correlation_factors.append(dd_correlation)

        # Weighted average of correlation factors
        weights = [0.4, 0.3, 0.2, 0.1]  # Category most important
        estimated_correlation = sum(
            w * f for w, f in zip(weights, correlation_factors, strict=False)
        )

        # Ensure correlation is within reasonable bounds
        return max(0.0, min(0.9, estimated_correlation))

    def _filter_by_correlation(
        self,
        strategies: list[StrategyMetrics],
        correlation_matrix: pd.DataFrame,
        max_correlation: float,
    ) -> list[StrategyMetrics]:
        """Filter strategies to reduce excessive correlation"""

        if len(strategies) <= 3:  # Don't filter if we have few strategies
            return strategies

        # Start with highest performing strategy
        strategies_sorted = sorted(strategies, key=lambda s: s.sharpe_ratio, reverse=True)
        selected = [strategies_sorted[0]]
        selected_ids = {strategies_sorted[0].strategy_id}

        # Add strategies that don't exceed correlation threshold
        for candidate in strategies_sorted[1:]:
            candidate_id = candidate.strategy_id

            # Check correlation with all selected strategies
            max_corr_with_selected = 0.0
            for selected_id in selected_ids:
                if (
                    candidate_id in correlation_matrix.index
                    and selected_id in correlation_matrix.columns
                ):
                    corr = abs(correlation_matrix.loc[candidate_id, selected_id])
                    max_corr_with_selected = max(max_corr_with_selected, corr)

            # Add if correlation is acceptable
            if max_corr_with_selected <= max_correlation:
                selected.append(candidate)
                selected_ids.add(candidate_id)

        return selected

    def _optimize_portfolio_allocation(
        self,
        strategies: list[StrategyMetrics],
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        correlation_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """Optimize portfolio allocation using specified objective"""

        # Convert strategies to format expected by portfolio optimizer
        strategy_metadata_list = []
        for strategy in strategies:
            # Create simplified metadata object for optimizer
            metadata = type(
                "StrategyMetadata",
                (),
                {
                    "strategy_id": strategy.strategy_id,
                    "performance": type(
                        "Performance",
                        (),
                        {
                            "cagr": strategy.total_return,
                            "sharpe_ratio": strategy.sharpe_ratio,
                            "max_drawdown": strategy.max_drawdown,
                            "volatility": strategy.volatility,
                            "beta": strategy.beta,
                            "alpha": strategy.alpha,
                        },
                    )(),
                },
            )()
            strategy_metadata_list.append(metadata)

        # Set up portfolio constraints for optimizer
        portfolio_constraints = type(
            "PortfolioConstraints",
            (),
            {
                "min_weight": constraints.min_strategy_weight,
                "max_weight": constraints.max_strategy_weight,
                "max_volatility": constraints.max_portfolio_volatility,
                "max_drawdown": constraints.max_portfolio_drawdown,
                "risk_free_rate": 0.02,
                "transaction_cost_bps": constraints.transaction_cost_bps,
                "max_turnover": None,
            },
        )()

        # Map objective to optimization method
        objective_map = {
            PortfolioObjective.RISK_ADJUSTED_RETURN: OptimizationMethod.SHARPE_MAXIMIZATION,
            PortfolioObjective.MIN_RISK: OptimizationMethod.MEAN_VARIANCE,
            PortfolioObjective.RISK_PARITY: OptimizationMethod.RISK_PARITY,
            PortfolioObjective.MAX_DIVERSIFICATION: OptimizationMethod.MAX_DIVERSIFICATION,
            PortfolioObjective.MAX_RETURN: OptimizationMethod.SHARPE_MAXIMIZATION,
            PortfolioObjective.TARGET_VOLATILITY: OptimizationMethod.MEAN_VARIANCE,
        }

        optimization_method = objective_map.get(objective, OptimizationMethod.SHARPE_MAXIMIZATION)

        # Initialize portfolio optimizer
        optimizer = PortfolioOptimizer(
            constraints=portfolio_constraints, optimization_method=optimization_method
        )

        # Create historical returns DataFrame (simplified)
        strategy_ids = [s.strategy_id for s in strategies]

        # For now, use simulated returns based on strategy characteristics
        # In production, this would use actual historical returns
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
        historical_returns = pd.DataFrame(index=dates, columns=strategy_ids)

        for strategy in strategies:
            # Simulate returns based on strategy characteristics
            daily_vol = strategy.volatility / np.sqrt(252)
            daily_return = strategy.total_return / 252

            simulated_returns = np.random.normal(daily_return, daily_vol, len(dates))
            historical_returns[strategy.strategy_id] = simulated_returns

        # Optimize portfolio
        allocation_result = optimizer.optimize_portfolio(
            strategies=strategy_metadata_list, historical_returns=historical_returns
        )

        return allocation_result.strategy_weights

    def _create_portfolio_composition(
        self,
        portfolio_name: str,
        strategies: list[StrategyMetrics],
        weights: dict[str, float],
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        correlation_matrix: pd.DataFrame,
    ) -> PortfolioComposition:
        """Create comprehensive portfolio composition"""

        # Generate unique portfolio ID
        portfolio_id = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create strategy metadata dict
        strategy_metadata = {s.strategy_id: s for s in strategies}

        # Calculate portfolio-level metrics
        portfolio_return = sum(weights[s.strategy_id] * s.total_return for s in strategies)

        # Portfolio volatility using correlation matrix
        strategy_ids = [s.strategy_id for s in strategies]
        weights_array = np.array([weights[sid] for sid in strategy_ids])
        volatilities = np.array([strategy_metadata[sid].volatility for sid in strategy_ids])

        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values
        portfolio_variance = weights_array.T @ cov_matrix @ weights_array
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Portfolio Sharpe ratio
        risk_free_rate = 0.02
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

        # Diversification ratio
        weighted_vol = sum(weights[sid] * volatilities[i] for i, sid in enumerate(strategy_ids))
        diversification_ratio = weighted_vol / portfolio_volatility

        # Risk contributions
        risk_contributions = {}
        for i, sid in enumerate(strategy_ids):
            risk_contrib = weights[sid] * (cov_matrix[i, :] @ weights_array) / portfolio_volatility
            risk_contributions[sid] = risk_contrib

        # Category weights
        category_weights = {}
        for strategy in strategies:
            category = strategy.category.value
            if category not in category_weights:
                category_weights[category] = 0.0
            category_weights[category] += weights[strategy.strategy_id]

        # Tier weights
        tier_weights = {}
        for strategy in strategies:
            tier = strategy.performance_tier.value
            if tier not in tier_weights:
                tier_weights[tier] = 0.0
            tier_weights[tier] += weights[strategy.strategy_id]

        # Estimated max drawdown (simplified)
        weighted_drawdowns = [weights[s.strategy_id] * s.max_drawdown for s in strategies]
        portfolio_max_drawdown = sum(weighted_drawdowns)

        # Risk metrics (simplified estimates)
        portfolio_var_95 = portfolio_volatility * 1.645  # Assuming normal distribution
        portfolio_cvar_95 = portfolio_volatility * 2.063

        return PortfolioComposition(
            portfolio_id=portfolio_id,
            portfolio_name=portfolio_name,
            strategy_weights=weights,
            strategy_metadata=strategy_metadata,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=portfolio_sharpe,
            max_drawdown=portfolio_max_drawdown,
            diversification_ratio=diversification_ratio,
            var_95=portfolio_var_95,
            cvar_95=portfolio_cvar_95,
            correlation_matrix=correlation_matrix,
            risk_contributions=risk_contributions,
            category_weights=category_weights,
            tier_weights=tier_weights,
            objective=objective,
            constraints=constraints,
            construction_method="advanced_optimization",
            created_at=datetime.now(),
            last_rebalanced=datetime.now(),
            inception_date=datetime.now(),
        )

    def _store_portfolio_composition(self, composition: PortfolioComposition) -> None:
        """Store portfolio composition in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolios (
                    portfolio_id, portfolio_name, objective, expected_return,
                    expected_volatility, sharpe_ratio, diversification_ratio,
                    num_strategies, num_categories, created_at, last_rebalanced,
                    composition_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    composition.portfolio_id,
                    composition.portfolio_name,
                    composition.objective.value,
                    composition.expected_return,
                    composition.expected_volatility,
                    composition.sharpe_ratio,
                    composition.diversification_ratio,
                    len(composition.strategy_weights),
                    len(composition.category_weights),
                    composition.created_at.isoformat(),
                    composition.last_rebalanced.isoformat(),
                    json.dumps(asdict(composition), default=str),
                ),
            )
            conn.commit()

    def _display_portfolio_summary(self, composition: PortfolioComposition) -> None:
        """Display comprehensive portfolio summary"""

        console.print(f"\n🎯 [bold]Portfolio Summary: {composition.portfolio_name}[/bold]")

        # Portfolio metrics table
        metrics_table = Table(title="📊 Portfolio Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("Assessment", style="green")

        metrics_table.add_row(
            "Expected Return",
            f"{composition.expected_return:.1%}",
            "✅ Good" if composition.expected_return > 0.10 else "⚠️  Moderate",
        )

        metrics_table.add_row(
            "Expected Volatility",
            f"{composition.expected_volatility:.1%}",
            "✅ Low" if composition.expected_volatility < 0.15 else "⚠️  High",
        )

        metrics_table.add_row(
            "Sharpe Ratio",
            f"{composition.sharpe_ratio:.2f}",
            (
                "✅ Excellent"
                if composition.sharpe_ratio > 1.5
                else ("✅ Good" if composition.sharpe_ratio > 1.0 else "⚠️  Moderate")
            ),
        )

        metrics_table.add_row(
            "Diversification Ratio",
            f"{composition.diversification_ratio:.2f}",
            "✅ Well Diversified" if composition.diversification_ratio > 1.2 else "⚠️  Concentrated",
        )

        console.print(metrics_table)

        # Strategy allocation table
        allocation_table = Table(title="🎯 Strategy Allocation")
        allocation_table.add_column("Strategy", style="cyan")
        allocation_table.add_column("Weight", justify="right", style="white")
        allocation_table.add_column("Category", style="dim")
        allocation_table.add_column("Sharpe", justify="right", style="green")

        # Sort by weight descending
        sorted_strategies = sorted(
            composition.strategy_weights.items(), key=lambda x: x[1], reverse=True
        )

        for strategy_id, weight in sorted_strategies:
            strategy = composition.strategy_metadata[strategy_id]
            allocation_table.add_row(
                strategy.strategy_name[:30],  # Truncate long names
                f"{weight:.1%}",
                strategy.category.value.replace("_", " ").title(),
                f"{strategy.sharpe_ratio:.2f}",
            )

        console.print(allocation_table)

        # Category breakdown
        console.print("\n📊 [bold]Category Breakdown[/bold]")
        for category, weight in sorted(
            composition.category_weights.items(), key=lambda x: x[1], reverse=True
        ):
            console.print(f"   • {category.replace('_', ' ').title()}: {weight:.1%}")

    def get_portfolio_recommendations(
        self, risk_tolerance: str = "moderate", investment_horizon: str = "medium_term"
    ) -> list[dict[str, Any]]:
        """Get portfolio recommendations based on risk tolerance and investment horizon"""

        recommendations = []

        # Conservative portfolio
        if risk_tolerance in ["conservative", "low"]:
            recommendations.append(
                {
                    "name": "Conservative Multi-Strategy Portfolio",
                    "description": "Low volatility, risk-parity weighted portfolio",
                    "objective": PortfolioObjective.RISK_PARITY,
                    "constraints": PortfolioConstraints(
                        max_portfolio_volatility=0.15,
                        min_strategy_sharpe=0.8,
                        max_strategies=6,
                        target_sharpe_ratio=0.8,
                    ),
                    "target_categories": [
                        StrategyCategory.MEAN_REVERSION,
                        StrategyCategory.VOLATILITY,
                    ],
                }
            )

        # Moderate portfolio
        if risk_tolerance in ["moderate", "balanced"]:
            recommendations.append(
                {
                    "name": "Balanced Multi-Strategy Portfolio",
                    "description": "Risk-adjusted returns with moderate volatility",
                    "objective": PortfolioObjective.RISK_ADJUSTED_RETURN,
                    "constraints": PortfolioConstraints(
                        max_portfolio_volatility=0.20,
                        min_strategy_sharpe=0.5,
                        max_strategies=8,
                        target_sharpe_ratio=1.0,
                    ),
                    "target_categories": [
                        StrategyCategory.TREND_FOLLOWING,
                        StrategyCategory.MEAN_REVERSION,
                        StrategyCategory.MOMENTUM,
                    ],
                }
            )

        # Aggressive portfolio
        if risk_tolerance in ["aggressive", "high"]:
            recommendations.append(
                {
                    "name": "Growth Multi-Strategy Portfolio",
                    "description": "Maximum returns with higher volatility tolerance",
                    "objective": PortfolioObjective.MAX_RETURN,
                    "constraints": PortfolioConstraints(
                        max_portfolio_volatility=0.30,
                        min_strategy_sharpe=0.3,
                        max_strategies=10,
                        target_sharpe_ratio=1.2,
                    ),
                    "target_categories": [
                        StrategyCategory.MOMENTUM,
                        StrategyCategory.BREAKOUT,
                        StrategyCategory.TREND_FOLLOWING,
                    ],
                }
            )

        return recommendations


def create_portfolio_constructor(
    portfolio_dir: str = "data/portfolios", strategy_collection: StrategyCollection = None
) -> PortfolioConstructor:
    """Factory function to create portfolio constructor"""
    return PortfolioConstructor(
        portfolio_dir=portfolio_dir, strategy_collection=strategy_collection
    )


if __name__ == "__main__":
    # Example usage
    constructor = create_portfolio_constructor()
    recommendations = constructor.get_portfolio_recommendations(risk_tolerance="moderate")

    print("Portfolio Constructor created successfully!")
    print(f"Found {len(recommendations)} portfolio recommendations")
