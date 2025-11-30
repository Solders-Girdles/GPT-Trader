"""Parameter grid for strategy optimization.

Provides:
- ParameterGrid: Define parameter search spaces
- Grid generation for exhaustive search
- Random sampling for large spaces
"""

import itertools
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParameterRange:
    """Define a range of values for a parameter."""

    name: str
    values: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.values is None and (self.min_value is None or self.max_value is None):
            raise ValueError(
                f"Parameter {self.name}: must provide either values list or min/max range"
            )

    def get_values(self) -> list[Any]:
        """Get all values in this range."""
        if self.values is not None:
            return self.values

        if self.step is None:
            # Default to 10 steps
            step = (self.max_value - self.min_value) / 10
        else:
            step = self.step

        if self.log_scale:
            import math

            log_min = math.log10(self.min_value)
            log_max = math.log10(self.max_value)
            num_steps = int((log_max - log_min) / step) + 1
            return [10 ** (log_min + i * step) for i in range(num_steps)]

        values = []
        current = self.min_value
        while current <= self.max_value:
            values.append(current)
            current += step
        return values

    def sample(self) -> Any:
        """Sample a single value from this range."""
        if self.values is not None:
            return random.choice(self.values)

        if self.log_scale:
            import math

            log_min = math.log10(self.min_value)
            log_max = math.log10(self.max_value)
            return 10 ** random.uniform(log_min, log_max)

        return random.uniform(self.min_value, self.max_value)


@dataclass
class ParameterGrid:
    """Define a grid of parameters for optimization.

    Supports:
    - Exhaustive grid search
    - Random sampling for large spaces
    - Conditional parameters
    - Parameter constraints
    """

    parameters: list[ParameterRange] = field(default_factory=list)
    constraints: list[callable] = field(default_factory=list)

    def add_parameter(
        self,
        name: str,
        values: list[Any] | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        step: float | None = None,
        log_scale: bool = False,
    ) -> "ParameterGrid":
        """Add a parameter to the grid.

        Args:
            name: Parameter name
            values: Explicit list of values
            min_value: Minimum value for range
            max_value: Maximum value for range
            step: Step size for range
            log_scale: Use logarithmic scale

        Returns:
            Self for chaining
        """
        param = ParameterRange(
            name=name,
            values=values,
            min_value=min_value,
            max_value=max_value,
            step=step,
            log_scale=log_scale,
        )
        self.parameters.append(param)
        return self

    def add_constraint(self, constraint: callable) -> "ParameterGrid":
        """Add a constraint function.

        The constraint should accept a parameter dict and return True if valid.

        Args:
            constraint: Function that validates parameter combinations

        Returns:
            Self for chaining
        """
        self.constraints.append(constraint)
        return self

    def _is_valid(self, params: dict[str, Any]) -> bool:
        """Check if parameter combination satisfies all constraints."""
        return all(constraint(params) for constraint in self.constraints)

    def __len__(self) -> int:
        """Get total number of combinations (before constraints)."""
        if not self.parameters:
            return 0

        total = 1
        for param in self.parameters:
            total *= len(param.get_values())
        return total

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all valid parameter combinations."""
        if not self.parameters:
            return

        # Get all value lists
        param_names = [p.name for p in self.parameters]
        value_lists = [p.get_values() for p in self.parameters]

        # Generate all combinations
        for values in itertools.product(*value_lists):
            params = dict(zip(param_names, values))
            if self._is_valid(params):
                yield params

    def sample(self, count: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Random sample from parameter space.

        Args:
            count: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            List of parameter dictionaries
        """
        if seed is not None:
            random.seed(seed)

        samples = []
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops

        while len(samples) < count and attempts < max_attempts:
            params = {p.name: p.sample() for p in self.parameters}
            if self._is_valid(params):
                samples.append(params)
            attempts += 1

        return samples

    def latin_hypercube_sample(self, count: int, seed: int | None = None) -> list[dict[str, Any]]:
        """Latin Hypercube Sampling for better space coverage.

        Args:
            count: Number of samples
            seed: Random seed

        Returns:
            List of parameter dictionaries
        """
        if seed is not None:
            random.seed(seed)

        # For each parameter, divide range into count bins
        samples = []

        # Create bin indices for each parameter
        bin_indices = [list(range(count)) for _ in self.parameters]

        # Shuffle each parameter's bins independently
        for indices in bin_indices:
            random.shuffle(indices)

        # Generate samples from each bin
        for i in range(count):
            params = {}
            for j, param in enumerate(self.parameters):
                values = param.get_values()
                bin_size = len(values) / count
                bin_idx = bin_indices[j][i]

                # Sample from within the bin
                start = int(bin_idx * bin_size)
                end = min(int((bin_idx + 1) * bin_size), len(values))
                if start >= len(values):
                    start = len(values) - 1
                if end <= start:
                    end = start + 1

                idx = random.randint(start, min(end - 1, len(values) - 1))
                params[param.name] = values[idx]

            if self._is_valid(params):
                samples.append(params)

        return samples

    def get_param_names(self) -> list[str]:
        """Get all parameter names."""
        return [p.name for p in self.parameters]

    def get_param_values(self, name: str) -> list[Any]:
        """Get values for a specific parameter."""
        for param in self.parameters:
            if param.name == name:
                return param.get_values()
        raise KeyError(f"Parameter {name} not found")

    def summary(self) -> dict[str, Any]:
        """Get summary of parameter grid."""
        param_info = {}
        for param in self.parameters:
            values = param.get_values()
            param_info[param.name] = {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "values": values if len(values) <= 10 else f"{values[:5]}...{values[-5:]}",
            }

        return {
            "total_combinations": len(self),
            "parameters": param_info,
            "constraint_count": len(self.constraints),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameters": [
                {
                    "name": p.name,
                    "values": p.values,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                    "step": p.step,
                    "log_scale": p.log_scale,
                }
                for p in self.parameters
            ],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterGrid":
        """Create from dictionary."""
        grid = cls()
        for param_data in data.get("parameters", []):
            grid.add_parameter(
                name=param_data["name"],
                values=param_data.get("values"),
                min_value=param_data.get("min_value"),
                max_value=param_data.get("max_value"),
                step=param_data.get("step"),
                log_scale=param_data.get("log_scale", False),
            )
        return grid

    @classmethod
    def from_parameter_dict(cls, param_dict: dict[str, list[Any]]) -> "ParameterGrid":
        """Create from simple parameter dictionary.

        Args:
            param_dict: Dict mapping parameter names to value lists

        Returns:
            ParameterGrid instance
        """
        grid = cls()
        for name, values in param_dict.items():
            grid.add_parameter(name, values=values)
        return grid


def create_common_grids() -> dict[str, ParameterGrid]:
    """Create common parameter grids for trading strategies.

    Returns:
        Dictionary of pre-configured grids
    """
    grids = {}

    # Moving average parameters
    ma_grid = ParameterGrid()
    ma_grid.add_parameter("fast_period", values=[5, 10, 15, 20, 25])
    ma_grid.add_parameter("slow_period", values=[20, 30, 50, 100, 200])
    ma_grid.add_constraint(lambda p: p["fast_period"] < p["slow_period"])
    grids["moving_average"] = ma_grid

    # RSI parameters
    rsi_grid = ParameterGrid()
    rsi_grid.add_parameter("period", values=[7, 10, 14, 21, 28])
    rsi_grid.add_parameter("oversold", values=[20, 25, 30, 35])
    rsi_grid.add_parameter("overbought", values=[65, 70, 75, 80])
    rsi_grid.add_constraint(lambda p: p["oversold"] < p["overbought"])
    grids["rsi"] = rsi_grid

    # Bollinger Bands parameters
    bb_grid = ParameterGrid()
    bb_grid.add_parameter("period", values=[10, 15, 20, 25, 30])
    bb_grid.add_parameter("std_dev", values=[1.5, 2.0, 2.5, 3.0])
    grids["bollinger_bands"] = bb_grid

    # Position sizing parameters
    sizing_grid = ParameterGrid()
    sizing_grid.add_parameter("base_fraction", min_value=0.01, max_value=0.10, step=0.01)
    sizing_grid.add_parameter("max_fraction", min_value=0.05, max_value=0.25, step=0.05)
    sizing_grid.add_constraint(lambda p: p["base_fraction"] < p["max_fraction"])
    grids["position_sizing"] = sizing_grid

    return grids
