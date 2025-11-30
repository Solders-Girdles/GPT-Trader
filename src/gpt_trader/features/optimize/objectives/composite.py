"""Composite objective functions with constraints for multi-objective optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gpt_trader.features.optimize.objectives.single import (
    SharpeRatioObjective,
    TotalReturnObjective,
)

if TYPE_CHECKING:
    from gpt_trader.backtesting.metrics.risk import RiskMetrics
    from gpt_trader.backtesting.metrics.statistics import TradeStatistics
    from gpt_trader.backtesting.types import BacktestResult
    from gpt_trader.features.optimize.objectives.base import ObjectiveFunction


@dataclass(frozen=True)
class Constraint:
    """
    Hard constraint for optimization feasibility.

    Constraints filter out infeasible trials before they contribute
    to objective optimization.

    Attributes:
        name: Human-readable constraint name
        metric: Metric to check (e.g., "max_drawdown_pct", "total_trades")
        operator: Comparison operator ("lt", "le", "gt", "ge", "eq")
        threshold: Threshold value for comparison
    """

    name: str
    metric: str
    operator: str
    threshold: float

    def __post_init__(self) -> None:
        """Validate constraint definition."""
        valid_ops = {"lt", "le", "gt", "ge", "eq"}
        if self.operator not in valid_ops:
            raise ValueError(f"operator must be one of {valid_ops}, got '{self.operator}'")

    def is_satisfied(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """
        Check if constraint is satisfied.

        Args:
            result: BacktestResult from the backtest run
            risk_metrics: RiskMetrics calculated from the broker
            trade_statistics: TradeStatistics calculated from the broker

        Returns:
            True if constraint is satisfied, False otherwise
        """
        value = self._extract_metric(result, risk_metrics, trade_statistics)
        if value is None:
            return False

        ops: dict[str, Callable[[float, float], bool]] = {
            "lt": lambda x, t: x < t,
            "le": lambda x, t: x <= t,
            "gt": lambda x, t: x > t,
            "ge": lambda x, t: x >= t,
            "eq": lambda x, t: x == t,
        }
        return ops[self.operator](value, self.threshold)

    def _extract_metric(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float | None:
        """Extract metric value from results."""
        # Map metric names to their locations
        metric_map: dict[str, tuple[Any, str]] = {
            # From BacktestResult
            "total_return": (result, "total_return"),
            "total_return_usd": (result, "total_return_usd"),
            "realized_pnl": (result, "realized_pnl"),
            "fees_paid": (result, "fees_paid"),
            "funding_pnl": (result, "funding_pnl"),
            "circuit_breaker_triggers": (result, "circuit_breaker_triggers"),
            "reduce_only_periods": (result, "reduce_only_periods"),
            # From RiskMetrics
            "max_drawdown_pct": (risk_metrics, "max_drawdown_pct"),
            "max_drawdown_usd": (risk_metrics, "max_drawdown_usd"),
            "avg_drawdown_pct": (risk_metrics, "avg_drawdown_pct"),
            "drawdown_duration_days": (risk_metrics, "drawdown_duration_days"),
            "sharpe_ratio": (risk_metrics, "sharpe_ratio"),
            "sortino_ratio": (risk_metrics, "sortino_ratio"),
            "calmar_ratio": (risk_metrics, "calmar_ratio"),
            "volatility_annualized": (risk_metrics, "volatility_annualized"),
            "downside_volatility": (risk_metrics, "downside_volatility"),
            "var_95_daily": (risk_metrics, "var_95_daily"),
            "var_99_daily": (risk_metrics, "var_99_daily"),
            "time_in_market_pct": (risk_metrics, "time_in_market_pct"),
            "avg_leverage_used": (risk_metrics, "avg_leverage_used"),
            "max_leverage_used": (risk_metrics, "max_leverage_used"),
            "total_return_pct": (risk_metrics, "total_return_pct"),
            "annualized_return_pct": (risk_metrics, "annualized_return_pct"),
            # From TradeStatistics
            "total_trades": (trade_statistics, "total_trades"),
            "winning_trades": (trade_statistics, "winning_trades"),
            "losing_trades": (trade_statistics, "losing_trades"),
            "win_rate": (trade_statistics, "win_rate"),
            "loss_rate": (trade_statistics, "loss_rate"),
            "profit_factor": (trade_statistics, "profit_factor"),
            "total_pnl": (trade_statistics, "total_pnl"),
            "avg_profit_per_trade": (trade_statistics, "avg_profit_per_trade"),
            "max_consecutive_wins": (trade_statistics, "max_consecutive_wins"),
            "max_consecutive_losses": (trade_statistics, "max_consecutive_losses"),
            "current_streak": (trade_statistics, "current_streak"),
            "avg_hold_time_minutes": (trade_statistics, "avg_hold_time_minutes"),
            "max_hold_time_minutes": (trade_statistics, "max_hold_time_minutes"),
            "avg_slippage_bps": (trade_statistics, "avg_slippage_bps"),
            "total_fees_paid": (trade_statistics, "total_fees_paid"),
            "limit_fill_rate": (trade_statistics, "limit_fill_rate"),
        }

        if self.metric not in metric_map:
            raise ValueError(f"Unknown metric: {self.metric}")

        obj, attr = metric_map[self.metric]
        value = getattr(obj, attr, None)
        if value is None:
            return None
        return float(value)


@dataclass
class WeightedObjective:
    """
    Weighted combination of multiple objectives with constraints.

    Combines multiple single objectives using weighted sum. All component
    objectives are normalized to maximize direction before weighting.

    Attributes:
        name: Human-readable name for this composite objective
        components: List of (objective, weight) tuples
        constraints: List of hard constraints that must be satisfied
        direction: Optimization direction (default: "maximize")
    """

    name: str
    components: list[tuple[ObjectiveFunction, float]]
    constraints: list[Constraint] = field(default_factory=list)
    direction: str = "maximize"

    def __post_init__(self) -> None:
        """Validate composite objective."""
        if not self.components:
            raise ValueError("At least one component objective is required")
        total_weight = sum(w for _, w in self.components)
        if abs(total_weight - 1.0) > 0.01:
            # Warn but don't fail - weights don't need to sum to 1
            pass

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """
        Calculate weighted objective value.

        Args:
            result: BacktestResult from the backtest run
            risk_metrics: RiskMetrics calculated from the broker
            trade_statistics: TradeStatistics calculated from the broker

        Returns:
            Weighted sum of component objectives
        """
        total = 0.0
        for objective, weight in self.components:
            value = objective.calculate(result, risk_metrics, trade_statistics)

            # Handle inf/-inf values
            if value == float("inf") or value == float("-inf"):
                return value

            # Normalize direction: flip sign if component minimizes
            if objective.direction == "minimize":
                value = -value

            total += weight * value

        return total

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """
        Check if all constraints are satisfied.

        Args:
            result: BacktestResult from the backtest run
            risk_metrics: RiskMetrics calculated from the broker
            trade_statistics: TradeStatistics calculated from the broker

        Returns:
            True if all constraints pass, False if any fails
        """
        # Check component feasibility
        for objective, _ in self.components:
            if not objective.is_feasible(result, risk_metrics, trade_statistics):
                return False

        # Check explicit constraints
        for constraint in self.constraints:
            if not constraint.is_satisfied(result, risk_metrics, trade_statistics):
                return False

        return True


def create_sharpe_with_drawdown_constraint(
    max_drawdown_pct: float = 20.0,
    sharpe_weight: float = 0.7,
    return_weight: float = 0.3,
    min_trades: int = 10,
) -> WeightedObjective:
    """
    Create a Sharpe-focused objective with drawdown constraint.

    This is a common multi-objective setup that maximizes risk-adjusted
    returns while ensuring maximum drawdown stays below a threshold.

    Args:
        max_drawdown_pct: Maximum allowed drawdown percentage (default 20%)
        sharpe_weight: Weight for Sharpe ratio in objective (default 0.7)
        return_weight: Weight for total return in objective (default 0.3)
        min_trades: Minimum number of trades required (default 10)

    Returns:
        Configured WeightedObjective

    Example:
        objective = create_sharpe_with_drawdown_constraint(
            max_drawdown_pct=15.0,
            sharpe_weight=0.8,
            return_weight=0.2,
        )
    """
    return WeightedObjective(
        name="sharpe_constrained",
        components=[
            (SharpeRatioObjective(min_trades=min_trades), sharpe_weight),
            (TotalReturnObjective(min_trades=min_trades), return_weight),
        ],
        constraints=[
            Constraint(
                name="max_drawdown",
                metric="max_drawdown_pct",
                operator="le",
                threshold=max_drawdown_pct,
            ),
            Constraint(
                name="min_trades",
                metric="total_trades",
                operator="ge",
                threshold=float(min_trades),
            ),
        ],
    )


def create_balanced_objective(
    sharpe_weight: float = 0.4,
    sortino_weight: float = 0.3,
    calmar_weight: float = 0.3,
    max_drawdown_pct: float = 25.0,
    min_trades: int = 10,
) -> WeightedObjective:
    """
    Create a balanced objective using multiple risk-adjusted metrics.

    Combines Sharpe, Sortino, and Calmar ratios for a well-rounded
    optimization that considers multiple aspects of risk.

    Args:
        sharpe_weight: Weight for Sharpe ratio (default 0.4)
        sortino_weight: Weight for Sortino ratio (default 0.3)
        calmar_weight: Weight for Calmar ratio (default 0.3)
        max_drawdown_pct: Maximum allowed drawdown percentage
        min_trades: Minimum number of trades required

    Returns:
        Configured WeightedObjective
    """
    from gpt_trader.features.optimize.objectives.single import (
        CalmarRatioObjective,
        SortinoRatioObjective,
    )

    return WeightedObjective(
        name="balanced_risk_adjusted",
        components=[
            (SharpeRatioObjective(min_trades=min_trades), sharpe_weight),
            (SortinoRatioObjective(min_trades=min_trades), sortino_weight),
            (CalmarRatioObjective(min_trades=min_trades), calmar_weight),
        ],
        constraints=[
            Constraint(
                name="max_drawdown",
                metric="max_drawdown_pct",
                operator="le",
                threshold=max_drawdown_pct,
            ),
            Constraint(
                name="min_trades",
                metric="total_trades",
                operator="ge",
                threshold=float(min_trades),
            ),
        ],
    )


def create_conservative_objective(
    max_drawdown_pct: float = 10.0,
    min_win_rate: float = 50.0,
    min_trades: int = 20,
) -> WeightedObjective:
    """
    Create a conservative objective focused on capital preservation.

    Prioritizes low drawdown and consistent wins over absolute returns.

    Args:
        max_drawdown_pct: Maximum allowed drawdown percentage (default 10%)
        min_win_rate: Minimum win rate percentage (default 50%)
        min_trades: Minimum number of trades required

    Returns:
        Configured WeightedObjective
    """
    from gpt_trader.features.optimize.objectives.single import (
        ProfitFactorObjective,
        WinRateObjective,
    )

    return WeightedObjective(
        name="conservative",
        components=[
            (SharpeRatioObjective(min_trades=min_trades), 0.4),
            (WinRateObjective(min_trades=min_trades), 0.3),
            (ProfitFactorObjective(min_trades=min_trades), 0.3),
        ],
        constraints=[
            Constraint(
                name="max_drawdown",
                metric="max_drawdown_pct",
                operator="le",
                threshold=max_drawdown_pct,
            ),
            Constraint(
                name="min_win_rate",
                metric="win_rate",
                operator="ge",
                threshold=min_win_rate,
            ),
            Constraint(
                name="min_trades",
                metric="total_trades",
                operator="ge",
                threshold=float(min_trades),
            ),
        ],
    )
