"""
Strategy Optimization Framework.

This module provides tools for optimizing trading strategies using Bayesian optimization (Optuna).
It supports:
- Flexible parameter space definitions
- Multi-objective optimization (e.g., Sharpe + Drawdown)
- Batch backtesting execution
- Result persistence and analysis
"""

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
from gpt_trader.features.optimize.parameter_space.builder import ParameterSpaceBuilder
from gpt_trader.features.optimize.parameter_space.definitions import (
    all_parameter_space,
    risk_parameter_space,
    simulation_parameter_space,
    strategy_parameter_space,
)
from gpt_trader.features.optimize.persistence.storage import OptimizationRun, OptimizationStorage
from gpt_trader.features.optimize.runner.batch_runner import BatchBacktestRunner, TrialResult
from gpt_trader.features.optimize.study.manager import OptimizationStudyManager
from gpt_trader.features.optimize.types import (
    OptimizationConfig,
    ParameterDefinition,
    ParameterSpace,
    ParameterType,
)

__all__ = [
    # Types
    "ParameterType",
    "ParameterDefinition",
    "ParameterSpace",
    "OptimizationConfig",
    # Parameter Space
    "ParameterSpaceBuilder",
    "strategy_parameter_space",
    "risk_parameter_space",
    "simulation_parameter_space",
    "all_parameter_space",
    # Original single objectives
    "ObjectiveFunction",
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
    # Basic factory functions (from composite)
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
    # Study Management
    "OptimizationStudyManager",
    # Execution
    "BatchBacktestRunner",
    "TrialResult",
    # Persistence
    "OptimizationRun",
    "OptimizationStorage",
]
