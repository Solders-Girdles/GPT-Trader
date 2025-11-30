"""Core types for strategy optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterType(Enum):
    """Types of optimizable parameters."""

    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    LOG_UNIFORM = "log_uniform"


@dataclass(frozen=True)
class ParameterDefinition:
    """
    Definition of a single searchable parameter.

    Attributes:
        name: Parameter name (must match config field name)
        parameter_type: Type of parameter (integer, float, categorical, log_uniform)
        low: Lower bound for numeric parameters
        high: Upper bound for numeric parameters
        choices: List of choices for categorical parameters
        step: Step size for numeric parameters (optional)
        log: Use log scale for float parameters
        default: Default value if not optimized
        description: Human-readable description
    """

    name: str
    parameter_type: ParameterType
    low: float | int | None = None
    high: float | int | None = None
    choices: list[Any] | None = None
    step: float | int | None = None
    log: bool = False
    default: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate parameter definition."""
        if self.parameter_type in (
            ParameterType.INTEGER,
            ParameterType.FLOAT,
            ParameterType.LOG_UNIFORM,
        ):
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter '{self.name}' requires low and high bounds")
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}' low must be less than high")
        elif self.parameter_type == ParameterType.CATEGORICAL:
            if not self.choices or len(self.choices) < 2:
                raise ValueError(f"Parameter '{self.name}' requires at least 2 choices")


@dataclass
class ParameterSpace:
    """
    Complete parameter space for optimization.

    Organizes parameters into three categories:
    - strategy_parameters: Trading strategy parameters (MA periods, RSI, etc.)
    - risk_parameters: Risk management parameters (leverage, exposure limits)
    - simulation_parameters: Backtesting simulation parameters (fees, slippage)
    """

    strategy_parameters: list[ParameterDefinition] = field(default_factory=list)
    risk_parameters: list[ParameterDefinition] = field(default_factory=list)
    simulation_parameters: list[ParameterDefinition] = field(default_factory=list)

    @property
    def all_parameters(self) -> list[ParameterDefinition]:
        """Get all parameters as a flat list."""
        return self.strategy_parameters + self.risk_parameters + self.simulation_parameters

    @property
    def parameter_count(self) -> int:
        """Get total number of parameters."""
        return len(self.all_parameters)

    def get_parameter(self, name: str) -> ParameterDefinition | None:
        """Get a parameter by name."""
        for param in self.all_parameters:
            if param.name == name:
                return param
        return None


@dataclass
class OptimizationConfig:
    """
    Configuration for an optimization run.

    Attributes:
        study_name: Unique name for the Optuna study
        parameter_space: Parameter space to optimize
        objective_name: Name of the objective function
        direction: Optimization direction ("maximize" or "minimize")
        number_of_trials: Maximum number of trials to run
        timeout_seconds: Maximum time for optimization (None = no limit)
        parallel_jobs: Number of parallel workers (1 = sequential)
        sampler_type: Optuna sampler type ("tpe", "cmaes", "random")
        pruner_type: Optuna pruner type ("median", "hyperband", None)
        seed: Random seed for reproducibility
    """

    study_name: str
    parameter_space: ParameterSpace
    objective_name: str
    direction: str = "maximize"
    number_of_trials: int = 100
    timeout_seconds: int | None = None
    parallel_jobs: int = 1
    sampler_type: str = "tpe"
    pruner_type: str | None = "median"
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got '{self.direction}'")
        if self.number_of_trials < 1:
            raise ValueError("number_of_trials must be at least 1")
        if self.parallel_jobs < 1:
            raise ValueError("parallel_jobs must be at least 1")
        if self.sampler_type not in ("tpe", "cmaes", "random"):
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")
        if self.pruner_type is not None and self.pruner_type not in (
            "median",
            "hyperband",
            "percentile",
        ):
            raise ValueError(f"Unknown pruner_type: {self.pruner_type}")
