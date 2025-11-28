"""Fluent builder for creating parameter spaces."""

from __future__ import annotations

from typing import Any

from gpt_trader.features.optimize.parameter_space.definitions import (
    risk_parameter_space,
    simulation_parameter_space,
    strategy_parameter_space,
)
from gpt_trader.features.optimize.types import (
    ParameterDefinition,
    ParameterSpace,
    ParameterType,
)


class ParameterSpaceBuilder:
    """
    Fluent builder for creating customized parameter spaces.

    Usage:
        space = (ParameterSpaceBuilder()
            .with_strategy_defaults()
            .with_risk_defaults()
            .add_integer("custom_param", low=1, high=10, category="strategy")
            .exclude("trailing_stop_pct")
            .build())
    """

    def __init__(self) -> None:
        """Initialize empty builder."""
        self._strategy_params: list[ParameterDefinition] = []
        self._risk_params: list[ParameterDefinition] = []
        self._simulation_params: list[ParameterDefinition] = []
        self._excluded: set[str] = set()

    def with_strategy_defaults(self) -> ParameterSpaceBuilder:
        """Add default strategy parameters from BaseStrategyConfig."""
        self._strategy_params.extend(strategy_parameter_space())
        return self

    def with_risk_defaults(self) -> ParameterSpaceBuilder:
        """Add default risk parameters from PerpsStrategyConfig."""
        self._risk_params.extend(risk_parameter_space())
        return self

    def with_simulation_defaults(self) -> ParameterSpaceBuilder:
        """Add default simulation parameters from SimulationConfig."""
        self._simulation_params.extend(simulation_parameter_space())
        return self

    def with_all_defaults(self) -> ParameterSpaceBuilder:
        """Add all default parameters."""
        return self.with_strategy_defaults().with_risk_defaults().with_simulation_defaults()

    def add_integer(
        self,
        name: str,
        low: int,
        high: int,
        *,
        category: str = "strategy",
        step: int = 1,
        default: int | None = None,
        description: str = "",
    ) -> ParameterSpaceBuilder:
        """
        Add a custom integer parameter.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            category: "strategy", "risk", or "simulation"
            step: Step size
            default: Default value
            description: Human-readable description
        """
        param = ParameterDefinition(
            name=name,
            parameter_type=ParameterType.INTEGER,
            low=low,
            high=high,
            step=step,
            default=default,
            description=description,
        )
        self._add_to_category(param, category)
        return self

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        category: str = "strategy",
        step: float | None = None,
        log: bool = False,
        default: float | None = None,
        description: str = "",
    ) -> ParameterSpaceBuilder:
        """
        Add a custom float parameter.

        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
            category: "strategy", "risk", or "simulation"
            step: Step size (optional)
            log: Use log scale
            default: Default value
            description: Human-readable description
        """
        param = ParameterDefinition(
            name=name,
            parameter_type=ParameterType.LOG_UNIFORM if log else ParameterType.FLOAT,
            low=low,
            high=high,
            step=step,
            log=log,
            default=default,
            description=description,
        )
        self._add_to_category(param, category)
        return self

    def add_categorical(
        self,
        name: str,
        choices: list[Any],
        *,
        category: str = "strategy",
        default: Any = None,
        description: str = "",
    ) -> ParameterSpaceBuilder:
        """
        Add a custom categorical parameter.

        Args:
            name: Parameter name
            choices: List of valid choices
            category: "strategy", "risk", or "simulation"
            default: Default value
            description: Human-readable description
        """
        param = ParameterDefinition(
            name=name,
            parameter_type=ParameterType.CATEGORICAL,
            choices=choices,
            default=default,
            description=description,
        )
        self._add_to_category(param, category)
        return self

    def exclude(self, *parameter_names: str) -> ParameterSpaceBuilder:
        """
        Exclude parameters by name from the final space.

        Args:
            parameter_names: Names of parameters to exclude
        """
        self._excluded.update(parameter_names)
        return self

    def override(
        self,
        name: str,
        *,
        low: float | int | None = None,
        high: float | int | None = None,
        step: float | int | None = None,
        choices: list[Any] | None = None,
    ) -> ParameterSpaceBuilder:
        """
        Override properties of an existing parameter.

        Args:
            name: Parameter name to override
            low: New lower bound
            high: New upper bound
            step: New step size
            choices: New choices (for categorical)
        """
        for param_list in [self._strategy_params, self._risk_params, self._simulation_params]:
            for i, param in enumerate(param_list):
                if param.name == name:
                    # Create new parameter with overrides
                    new_param = ParameterDefinition(
                        name=param.name,
                        parameter_type=param.parameter_type,
                        low=low if low is not None else param.low,
                        high=high if high is not None else param.high,
                        choices=choices if choices is not None else param.choices,
                        step=step if step is not None else param.step,
                        log=param.log,
                        default=param.default,
                        description=param.description,
                    )
                    param_list[i] = new_param
                    return self
        return self

    def _add_to_category(self, param: ParameterDefinition, category: str) -> None:
        """Add parameter to the appropriate category list."""
        if category == "strategy":
            self._strategy_params.append(param)
        elif category == "risk":
            self._risk_params.append(param)
        elif category == "simulation":
            self._simulation_params.append(param)
        else:
            raise ValueError(f"Unknown category: {category}. Use 'strategy', 'risk', or 'simulation'")

    def build(self) -> ParameterSpace:
        """
        Build the final ParameterSpace.

        Returns:
            ParameterSpace with all added parameters (excluding those in exclude list)
        """
        # Filter out excluded parameters
        strategy = [p for p in self._strategy_params if p.name not in self._excluded]
        risk = [p for p in self._risk_params if p.name not in self._excluded]
        simulation = [p for p in self._simulation_params if p.name not in self._excluded]

        return ParameterSpace(
            strategy_parameters=strategy,
            risk_parameters=risk,
            simulation_parameters=simulation,
        )
