"""Advanced constraint types for sophisticated optimization control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_trader.backtesting.metrics.risk import RiskMetrics
    from gpt_trader.backtesting.metrics.statistics import TradeStatistics
    from gpt_trader.backtesting.types import BacktestResult


def _get_metric_map(
    result: BacktestResult,
    risk_metrics: RiskMetrics,
    trade_statistics: TradeStatistics,
) -> dict[str, tuple[Any, str]]:
    """
    Get the metric mapping for extracting values from results.

    This is shared across all constraint types for consistency.
    """
    return {
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


def _extract_metric(
    metric: str,
    result: BacktestResult,
    risk_metrics: RiskMetrics,
    trade_statistics: TradeStatistics,
) -> float | None:
    """Extract a metric value from results."""
    metric_map = _get_metric_map(result, risk_metrics, trade_statistics)

    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric}")

    obj, attr = metric_map[metric]
    value = getattr(obj, attr, None)
    if value is None:
        return None
    return float(value)


def _compare(value: float, operator: str, threshold: float) -> bool:
    """Apply comparison operator."""
    ops: dict[str, Callable[[float, float], bool]] = {
        "lt": lambda x, t: x < t,
        "le": lambda x, t: x <= t,
        "gt": lambda x, t: x > t,
        "ge": lambda x, t: x >= t,
        "eq": lambda x, t: x == t,
    }
    if operator not in ops:
        raise ValueError(f"Unknown operator: {operator}")
    return ops[operator](value, threshold)


@dataclass(frozen=True)
class ConditionalConstraint:
    """
    Constraint that only applies when a condition is met.

    This allows expressing rules like:
    "If drawdown > 10%, then win_rate must be > 60%"

    The constraint is satisfied if:
    1. The condition is NOT met (constraint doesn't apply), OR
    2. The condition IS met AND the constrained metric satisfies its requirement

    Attributes:
        name: Human-readable constraint name
        condition_metric: Metric to check for condition
        condition_operator: Comparison operator for condition ("lt", "le", "gt", "ge", "eq")
        condition_threshold: Threshold for condition trigger
        constrained_metric: Metric that must satisfy constraint when condition is met
        constraint_operator: Comparison operator for constraint
        constraint_threshold: Threshold for constraint

    Example:
        # If loss_rate > 40%, then profit_factor must be > 2.0
        ConditionalConstraint(
            name="high_loss_rate_requires_high_pf",
            condition_metric="loss_rate",
            condition_operator="gt",
            condition_threshold=40.0,
            constrained_metric="profit_factor",
            constraint_operator="gt",
            constraint_threshold=2.0,
        )
    """

    name: str
    condition_metric: str
    condition_operator: str
    condition_threshold: float
    constrained_metric: str
    constraint_operator: str
    constraint_threshold: float

    def __post_init__(self) -> None:
        """Validate constraint definition."""
        valid_ops = {"lt", "le", "gt", "ge", "eq"}
        if self.condition_operator not in valid_ops:
            raise ValueError(
                f"condition_operator must be one of {valid_ops}, got '{self.condition_operator}'"
            )
        if self.constraint_operator not in valid_ops:
            raise ValueError(
                f"constraint_operator must be one of {valid_ops}, got '{self.constraint_operator}'"
            )

    def is_satisfied(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """
        Check if the conditional constraint is satisfied.

        Returns True if:
        - The condition is not met (constraint doesn't apply), OR
        - The condition is met AND the constraint is satisfied
        """
        # Extract condition metric
        condition_value = _extract_metric(
            self.condition_metric, result, risk_metrics, trade_statistics
        )
        if condition_value is None:
            return True  # If condition can't be evaluated, assume satisfied

        # Check if condition triggers
        condition_met = _compare(condition_value, self.condition_operator, self.condition_threshold)

        if not condition_met:
            return True  # Condition not met, constraint doesn't apply

        # Condition is met, check the constraint
        constraint_value = _extract_metric(
            self.constrained_metric, result, risk_metrics, trade_statistics
        )
        if constraint_value is None:
            return False  # Can't evaluate constraint, consider it failed

        return _compare(constraint_value, self.constraint_operator, self.constraint_threshold)


@dataclass(frozen=True)
class RangeConstraint:
    """
    Constraint that requires a metric to be within a range.

    More expressive than two separate lt/gt constraints and clearer in intent.

    Attributes:
        name: Human-readable constraint name
        metric: Metric to check
        lower_bound: Minimum allowed value
        upper_bound: Maximum allowed value
        inclusive: Whether bounds are inclusive (default True)

    Example:
        # Win rate must be between 40% and 70%
        RangeConstraint(
            name="balanced_win_rate",
            metric="win_rate",
            lower_bound=40.0,
            upper_bound=70.0,
        )
    """

    name: str
    metric: str
    lower_bound: float
    upper_bound: float
    inclusive: bool = True

    def __post_init__(self) -> None:
        """Validate constraint definition."""
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"lower_bound ({self.lower_bound}) must be <= upper_bound ({self.upper_bound})"
            )

    def is_satisfied(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if metric is within the specified range."""
        value = _extract_metric(self.metric, result, risk_metrics, trade_statistics)
        if value is None:
            return False

        if self.inclusive:
            return self.lower_bound <= value <= self.upper_bound
        return self.lower_bound < value < self.upper_bound


@dataclass(frozen=True)
class CircuitBreakerConstraint:
    """
    Constraint that limits circuit breaker triggers.

    Circuit breakers are triggered when risk limits are hit.
    Strategies that frequently trigger circuit breakers may indicate
    poor risk management or unstable behavior.

    This is primarily relevant for perpetual futures strategies.

    Attributes:
        max_triggers: Maximum allowed circuit breaker triggers (default 0)

    Example:
        # Allow no circuit breaker triggers
        CircuitBreakerConstraint()

        # Allow up to 2 triggers
        CircuitBreakerConstraint(max_triggers=2)
    """

    max_triggers: int = 0

    @property
    def name(self) -> str:
        """Constraint name."""
        return "circuit_breaker_limit"

    def is_satisfied(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if circuit breaker triggers are within limit."""
        return result.circuit_breaker_triggers <= self.max_triggers


@dataclass(frozen=True)
class ReduceOnlyConstraint:
    """
    Constraint on reduce-only periods.

    Reduce-only mode is entered when a strategy can only reduce positions,
    typically due to hitting risk limits. Frequent reduce-only periods
    indicate risk management issues.

    This is primarily relevant for perpetual futures strategies.

    Attributes:
        max_periods: Maximum allowed reduce-only periods (default 0)

    Example:
        # Allow no reduce-only periods
        ReduceOnlyConstraint()

        # Allow up to 1 reduce-only period
        ReduceOnlyConstraint(max_periods=1)
    """

    max_periods: int = 0

    @property
    def name(self) -> str:
        """Constraint name."""
        return "reduce_only_limit"

    def is_satisfied(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if reduce-only periods are within limit."""
        return result.reduce_only_periods <= self.max_periods
