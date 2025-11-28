"""Objective functions for optimization."""

from __future__ import annotations

from gpt_trader.features.optimize.objectives.base import ObjectiveFunction
from gpt_trader.features.optimize.objectives.composite import (
    Constraint,
    WeightedObjective,
    create_balanced_objective,
    create_conservative_objective,
    create_sharpe_with_drawdown_constraint,
)
from gpt_trader.features.optimize.objectives.constraints import (
    CircuitBreakerConstraint,
    ConditionalConstraint,
    RangeConstraint,
    ReduceOnlyConstraint,
)
from gpt_trader.features.optimize.objectives.factories import (
    create_execution_quality_objective,
    create_perpetuals_objective,
    create_risk_averse_objective,
    create_streak_resilient_objective,
    create_tail_risk_aware_objective,
    create_time_efficient_objective,
)
from gpt_trader.features.optimize.objectives.perpetuals import (
    FundingAdjustedReturnObjective,
    FundingEfficiencyObjective,
)
from gpt_trader.features.optimize.objectives.single import (
    CalmarRatioObjective,
    CostAdjustedReturnObjective,
    DownsideVolatilityObjective,
    DrawdownRecoveryObjective,
    ExecutionQualityObjective,
    HoldDurationObjective,
    LeverageAdjustedReturnObjective,
    MaxDrawdownObjective,
    ProfitFactorObjective,
    SharpeRatioObjective,
    SortinoRatioObjective,
    StreakConsistencyObjective,
    TailRiskAdjustedReturnObjective,
    TimeEfficiencyObjective,
    TotalReturnObjective,
    ValueAtRisk95Objective,
    ValueAtRisk99Objective,
    WinRateObjective,
)

__all__ = [
    # Protocol
    "ObjectiveFunction",
    # Original single objectives
    "SharpeRatioObjective",
    "SortinoRatioObjective",
    "TotalReturnObjective",
    "CalmarRatioObjective",
    "WinRateObjective",
    "ProfitFactorObjective",
    "MaxDrawdownObjective",
    # New risk-focused objectives
    "ValueAtRisk95Objective",
    "ValueAtRisk99Objective",
    "DrawdownRecoveryObjective",
    "DownsideVolatilityObjective",
    "TailRiskAdjustedReturnObjective",
    # New trade quality objectives
    "StreakConsistencyObjective",
    "CostAdjustedReturnObjective",
    "ExecutionQualityObjective",
    # New exposure/timing objectives
    "TimeEfficiencyObjective",
    "HoldDurationObjective",
    "LeverageAdjustedReturnObjective",
    # Perpetuals objectives
    "FundingAdjustedReturnObjective",
    "FundingEfficiencyObjective",
    # Composite objectives
    "WeightedObjective",
    "Constraint",
    # Basic constraint types (from composite)
    "create_sharpe_with_drawdown_constraint",
    "create_balanced_objective",
    "create_conservative_objective",
    # Advanced constraint types
    "ConditionalConstraint",
    "RangeConstraint",
    "CircuitBreakerConstraint",
    "ReduceOnlyConstraint",
    # Factory methods
    "create_risk_averse_objective",
    "create_execution_quality_objective",
    "create_time_efficient_objective",
    "create_streak_resilient_objective",
    "create_perpetuals_objective",
    "create_tail_risk_aware_objective",
]
