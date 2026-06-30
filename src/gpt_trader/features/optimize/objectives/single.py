"""Single-metric objective functions for optimization.

Objectives are defined in category modules (return / risk / trade-quality) and
re-exported here for backward compatibility.
"""

from __future__ import annotations

from gpt_trader.features.optimize.objectives.return_objectives import (
    CalmarRatioObjective,
    LeverageAdjustedReturnObjective,
    SharpeRatioObjective,
    SortinoRatioObjective,
    TotalReturnObjective,
)
from gpt_trader.features.optimize.objectives.risk_objectives import (
    DownsideVolatilityObjective,
    DrawdownRecoveryObjective,
    MaxDrawdownObjective,
    TailRiskAdjustedReturnObjective,
    ValueAtRisk95Objective,
    ValueAtRisk99Objective,
)
from gpt_trader.features.optimize.objectives.trade_quality_objectives import (
    CostAdjustedReturnObjective,
    ExecutionQualityObjective,
    HoldDurationObjective,
    ProfitFactorObjective,
    StreakConsistencyObjective,
    TimeEfficiencyObjective,
    WinRateObjective,
)

__all__ = [
    "CalmarRatioObjective",
    "CostAdjustedReturnObjective",
    "DownsideVolatilityObjective",
    "DrawdownRecoveryObjective",
    "ExecutionQualityObjective",
    "HoldDurationObjective",
    "LeverageAdjustedReturnObjective",
    "MaxDrawdownObjective",
    "ProfitFactorObjective",
    "SharpeRatioObjective",
    "SortinoRatioObjective",
    "StreakConsistencyObjective",
    "TailRiskAdjustedReturnObjective",
    "TimeEfficiencyObjective",
    "TotalReturnObjective",
    "ValueAtRisk95Objective",
    "ValueAtRisk99Objective",
    "WinRateObjective",
]
