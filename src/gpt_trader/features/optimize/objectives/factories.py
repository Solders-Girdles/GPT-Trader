"""Factory methods for common objective and constraint combinations."""

from __future__ import annotations

from gpt_trader.features.optimize.objectives.composite import Constraint, WeightedObjective
from gpt_trader.features.optimize.objectives.perpetuals import (
    FundingAdjustedReturnObjective,
)
from gpt_trader.features.optimize.objectives.single import (
    CalmarRatioObjective,
    CostAdjustedReturnObjective,
    DrawdownRecoveryObjective,
    ExecutionQualityObjective,
    LeverageAdjustedReturnObjective,
    SharpeRatioObjective,
    SortinoRatioObjective,
    StreakConsistencyObjective,
    TailRiskAdjustedReturnObjective,
    TimeEfficiencyObjective,
    TotalReturnObjective,
    ValueAtRisk95Objective,
    WinRateObjective,
)


def create_risk_averse_objective(
    max_drawdown_pct: float = 15.0,
    max_var_95: float = 5.0,
    min_trades: int = 10,
) -> WeightedObjective:
    """
    Create a risk-averse objective focused on capital preservation.

    Combines Sortino ratio with VaR and drawdown constraints for
    strategies that prioritize avoiding large losses over maximizing returns.

    Args:
        max_drawdown_pct: Maximum allowed drawdown percentage (default 15%)
        max_var_95: Maximum 95% daily VaR (default 5%)
        min_trades: Minimum trades required for feasibility

    Returns:
        WeightedObjective with risk-averse configuration

    Example:
        objective = create_risk_averse_objective(
            max_drawdown_pct=10.0,
            max_var_95=3.0,
        )
    """
    return WeightedObjective(
        name="risk_averse",
        components=[
            (SortinoRatioObjective(min_trades=min_trades), 0.4),
            (ValueAtRisk95Objective(min_trades=min_trades), 0.3),
            (DrawdownRecoveryObjective(min_trades=min_trades), 0.3),
        ],
        constraints=[
            Constraint("max_drawdown", "max_drawdown_pct", "le", max_drawdown_pct),
            Constraint("max_var", "var_95_daily", "le", max_var_95),
            Constraint("min_trades", "total_trades", "ge", float(min_trades)),
        ],
    )


def create_execution_quality_objective(
    max_slippage_bps: float = 10.0,
    min_fill_rate: float = 80.0,
    min_trades: int = 20,
) -> WeightedObjective:
    """
    Create an objective for strategies dependent on execution quality.

    Useful for market-making, limit-order, or high-frequency strategies
    where slippage and fill rates significantly impact profitability.

    Args:
        max_slippage_bps: Maximum average slippage in basis points
        min_fill_rate: Minimum limit order fill rate percentage
        min_trades: Minimum trades required for feasibility

    Returns:
        WeightedObjective with execution focus

    Example:
        objective = create_execution_quality_objective(
            max_slippage_bps=5.0,
            min_fill_rate=90.0,
        )
    """
    return WeightedObjective(
        name="execution_focused",
        components=[
            (SharpeRatioObjective(min_trades=min_trades), 0.4),
            (ExecutionQualityObjective(min_trades=min_trades), 0.3),
            (CostAdjustedReturnObjective(min_trades=min_trades), 0.3),
        ],
        constraints=[
            Constraint("max_slippage", "avg_slippage_bps", "le", max_slippage_bps),
            Constraint("min_fill_rate", "limit_fill_rate", "ge", min_fill_rate),
            Constraint("min_trades", "total_trades", "ge", float(min_trades)),
        ],
    )


def create_time_efficient_objective(
    max_time_in_market_pct: float = 50.0,
    max_drawdown_pct: float = 20.0,
    min_trades: int = 10,
) -> WeightedObjective:
    """
    Create an objective for strategies that minimize market exposure.

    Rewards high returns achieved with minimal time in positions.
    Useful for strategies that prefer to be in cash and only trade
    during high-conviction opportunities.

    Args:
        max_time_in_market_pct: Maximum percentage of time with positions
        max_drawdown_pct: Maximum allowed drawdown percentage
        min_trades: Minimum trades required for feasibility

    Returns:
        WeightedObjective with time efficiency focus

    Example:
        objective = create_time_efficient_objective(
            max_time_in_market_pct=30.0,
            max_drawdown_pct=15.0,
        )
    """
    return WeightedObjective(
        name="time_efficient",
        components=[
            (TimeEfficiencyObjective(min_trades=min_trades), 0.5),
            (SharpeRatioObjective(min_trades=min_trades), 0.3),
            (TotalReturnObjective(min_trades=min_trades), 0.2),
        ],
        constraints=[
            Constraint("max_exposure", "time_in_market_pct", "le", max_time_in_market_pct),
            Constraint("max_drawdown", "max_drawdown_pct", "le", max_drawdown_pct),
            Constraint("min_trades", "total_trades", "ge", float(min_trades)),
        ],
    )


def create_streak_resilient_objective(
    max_consecutive_losses: int = 5,
    min_win_rate: float = 45.0,
    max_drawdown_pct: float = 20.0,
    min_trades: int = 30,
) -> WeightedObjective:
    """
    Create an objective for psychologically tradeable strategies.

    Minimizes losing streaks while maintaining profitability.
    Important for discretionary traders or strategies where
    long losing streaks could lead to strategy abandonment.

    Args:
        max_consecutive_losses: Maximum consecutive losing trades allowed
        min_win_rate: Minimum win rate percentage
        max_drawdown_pct: Maximum allowed drawdown percentage
        min_trades: Minimum trades required for feasibility

    Returns:
        WeightedObjective with streak resilience focus

    Example:
        objective = create_streak_resilient_objective(
            max_consecutive_losses=4,
            min_win_rate=50.0,
        )
    """
    return WeightedObjective(
        name="streak_resilient",
        components=[
            (SharpeRatioObjective(min_trades=min_trades), 0.4),
            (StreakConsistencyObjective(min_trades=min_trades), 0.3),
            (WinRateObjective(min_trades=min_trades), 0.3),
        ],
        constraints=[
            Constraint("max_streak", "max_consecutive_losses", "le", float(max_consecutive_losses)),
            Constraint("min_win_rate", "win_rate", "ge", min_win_rate),
            Constraint("max_drawdown", "max_drawdown_pct", "le", max_drawdown_pct),
            Constraint("min_trades", "total_trades", "ge", float(min_trades)),
        ],
    )


def create_perpetuals_objective(
    max_drawdown_pct: float = 20.0,
    max_leverage: float = 5.0,
    allow_circuit_breakers: bool = False,
    min_trades: int = 10,
) -> WeightedObjective:
    """
    Create an objective optimized for perpetual futures trading.

    Accounts for funding costs and leverage constraints unique to
    perpetual futures markets. Suitable for strategies that hold
    positions across funding periods.

    Args:
        max_drawdown_pct: Maximum allowed drawdown percentage
        max_leverage: Maximum allowed leverage
        allow_circuit_breakers: Whether to allow circuit breaker triggers
        min_trades: Minimum trades required for feasibility

    Returns:
        WeightedObjective optimized for perpetuals

    Example:
        objective = create_perpetuals_objective(
            max_drawdown_pct=15.0,
            max_leverage=3.0,
            allow_circuit_breakers=False,
        )
    """
    constraints = [
        Constraint("max_drawdown", "max_drawdown_pct", "le", max_drawdown_pct),
        Constraint("max_leverage", "max_leverage_used", "le", max_leverage),
        Constraint("min_trades", "total_trades", "ge", float(min_trades)),
    ]

    if not allow_circuit_breakers:
        constraints.append(Constraint("no_circuit_breakers", "circuit_breaker_triggers", "eq", 0.0))

    return WeightedObjective(
        name="perpetuals_optimized",
        components=[
            (SharpeRatioObjective(min_trades=min_trades), 0.3),
            (FundingAdjustedReturnObjective(min_trades=min_trades), 0.3),
            (LeverageAdjustedReturnObjective(min_trades=min_trades), 0.2),
            (CalmarRatioObjective(min_trades=min_trades), 0.2),
        ],
        constraints=constraints,
    )


def create_tail_risk_aware_objective(
    max_var_99: float = 8.0,
    max_drawdown_pct: float = 25.0,
    min_trades: int = 15,
) -> WeightedObjective:
    """
    Create an objective focused on tail risk management.

    Uses VaR99 and extreme drawdown constraints to avoid strategies
    that may blow up during market stress events. Suitable for
    risk-conscious traders who prioritize avoiding ruin.

    Args:
        max_var_99: Maximum 99% daily VaR (extreme tail risk)
        max_drawdown_pct: Maximum allowed drawdown percentage
        min_trades: Minimum trades required for feasibility

    Returns:
        WeightedObjective with tail risk awareness

    Example:
        objective = create_tail_risk_aware_objective(
            max_var_99=5.0,
            max_drawdown_pct=20.0,
        )
    """
    return WeightedObjective(
        name="tail_risk_aware",
        components=[
            (TailRiskAdjustedReturnObjective(min_trades=min_trades), 0.4),
            (SortinoRatioObjective(min_trades=min_trades), 0.3),
            (CalmarRatioObjective(min_trades=min_trades), 0.3),
        ],
        constraints=[
            Constraint("max_var_99", "var_99_daily", "le", max_var_99),
            Constraint("max_drawdown", "max_drawdown_pct", "le", max_drawdown_pct),
            Constraint("min_trades", "total_trades", "ge", float(min_trades)),
        ],
    )
