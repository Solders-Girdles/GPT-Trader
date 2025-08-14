"""
Configuration management for optimization framework.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from bot.utils.base import BaseConfig
from pydantic import BaseModel, Field, field_validator


class ParameterDefinition(BaseModel):
    """Definition of a single parameter for optimization."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type: int, float, bool, str")
    min_value: int | float | None = Field(None, description="Minimum value")
    max_value: int | float | None = Field(None, description="Maximum value")
    default: Any = Field(None, description="Default value")
    step: int | float | None = Field(None, description="Step size for grid search")
    choices: list[Any] | None = Field(None, description="List of valid choices")
    description: str = Field("", description="Parameter description")

    @field_validator("type")
    def validate_type(cls, v: str) -> str:
        valid_types = ["int", "float", "bool", "str"]
        if v not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
        return v

    def validate_value(self, value: Any) -> bool:
        """Validate a parameter value against this definition."""
        try:
            # Type conversion
            if self.type == "int":
                value = int(value)
            elif self.type == "float":
                value = float(value)
            elif self.type == "bool":
                value = bool(value)
            elif self.type == "str":
                value = str(value)

            # Range validation
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

            # Choices validation
            if self.choices is not None and value not in self.choices:
                return False

            return True
        except (ValueError, TypeError):
            return False


class StrategyConfig(BaseModel):
    """Configuration for a trading strategy."""

    name: str = Field(..., description="Strategy name")
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    description: str = Field("", description="Strategy description")

    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names."""
        return list(self.parameters.keys())

    def validate_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and convert parameter values."""
        validated = {}
        for name, value in params.items():
            if name not in self.parameters:
                raise ValueError(f"Unknown parameter: {name}")

            param_def = self.parameters[name]
            if not param_def.validate_value(value):
                raise ValueError(f"Invalid value for {name}: {value}")

            # Convert to proper type
            if param_def.type == "int":
                validated[name] = int(float(value))  # Handle float-to-int conversion
            elif param_def.type == "float":
                validated[name] = float(value)
            elif param_def.type == "bool":
                validated[name] = bool(value)
            else:
                validated[name] = str(value)

        return validated


class ParameterSpace(BaseModel):
    """Definition of the parameter space for optimization."""

    strategy: StrategyConfig = Field(..., description="Strategy configuration")
    grid_ranges: dict[str, list[Any]] = Field(default_factory=dict)
    evolutionary_bounds: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_grid_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        import itertools

        param_names = list(self.grid_ranges.keys())
        param_values = list(self.grid_ranges.values())

        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo, strict=False)))

        return combinations

    def get_evolutionary_bounds(self, param_name: str) -> dict[str, Any]:
        """Get evolutionary bounds for a parameter."""
        return self.evolutionary_bounds.get(param_name, {})


class OptimizationConfig(BaseConfig):
    """Main configuration for optimization runs."""

    # Basic settings
    name: str = Field(..., description="Optimization run name")
    description: str = Field("", description="Run description")

    # Data settings
    symbols: list[str] = Field(default_factory=list)
    symbol_list_path: str | None = Field(None, description="Path to symbol list CSV")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: str = Field(..., description="End date YYYY-MM-DD")

    # Walk-forward settings
    walk_forward: bool = Field(False, description="Enable walk-forward testing")
    train_months: int = Field(12, description="Training window in months")
    test_months: int = Field(6, description="Test window in months")
    step_months: int = Field(6, description="Step between windows in months")

    # Optimization settings
    method: str = Field("grid", description="Optimization method: grid, evolutionary, or both")
    max_workers: int = Field(1, description="Number of parallel workers")

    # Grid search settings
    grid_search: bool = Field(True, description="Enable grid search")
    grid_sample_size: int | None = Field(None, description="Random sample size for grid")

    # Evolutionary settings
    evolutionary: bool = Field(False, description="Enable evolutionary search")
    generations: int = Field(100, description="Number of generations")
    population_size: int = Field(50, description="Population size")
    elite_size: int = Field(8, description="Elite population size")
    mutation_rate: float = Field(0.2, description="Mutation probability")
    crossover_rate: float = Field(0.8, description="Crossover probability")

    # Early stopping
    early_stopping: bool = Field(True, description="Enable early stopping")
    patience: int = Field(20, description="Generations without improvement")
    min_improvement: float = Field(0.0001, description="Minimum improvement threshold")

    # Evaluation settings
    primary_metric: str = Field("sharpe", description="Primary optimization metric")
    secondary_metrics: list[str] = Field(default_factory=lambda: ["cagr", "max_drawdown"])
    min_trades: int = Field(5, description="Minimum trades required")
    min_sharpe: float = Field(-1.0, description="Minimum Sharpe ratio")
    max_drawdown: float = Field(0.5, description="Maximum drawdown")
    early_stop_min: int = Field(0, description="Minimum evaluations before pruning/refine")

    # Output settings
    output_dir: str = Field("data/optimization", description="Output directory")
    save_intermediate: bool = Field(True, description="Save intermediate results")
    create_plots: bool = Field(True, description="Create visualization plots")
    quiet_bars: bool = Field(True, description="Disable progress bars during sweeps")

    # Seeding convenience
    seed_latest: bool = Field(
        False, description="Load seeds from the most recent run in output_dir"
    )
    seed_from: str | None = Field(
        None, description="Path to a seeds.json file or a run directory containing it"
    )
    seed_mode: str = Field(
        "merge", description="How to apply seeds: merge (default) or replace initial set"
    )
    seed_topk: int = Field(
        5, description="Top-k parameter sets to write to seeds.json at end of run"
    )

    # Parameter space
    parameter_space: ParameterSpace = Field(..., description="Parameter space definition")

    # Coarse-then-refine wrapper
    coarse_then_refine: bool = Field(
        False, description="Enable fast coarse stage then full refine stage"
    )
    coarse_months: int = Field(18, description="Months to use for coarse stage")
    coarse_symbols: int = Field(10, description="Number of symbols for coarse stage (subset)")
    refine_top_pct: float = Field(
        0.02, description="Top fraction to refine on full fidelity (e.g., 0.02 => top 2%)"
    )
    vectorized_phase1: bool = Field(
        False, description="Use simplified close-to-close model in phase 1"
    )
    entry_confirm_phase1: int = Field(3, description="Entry confirmation periods for coarse phase")
    min_rebalance_pct_phase1: float = Field(
        0.01, description="Skip tiny rebalances in coarse phase"
    )

    @field_validator("method")
    def validate_method(cls, v: str) -> str:
        valid_methods = ["grid", "evolutionary", "both"]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v

    @field_validator("primary_metric")
    def validate_primary_metric(cls, v: str) -> str:
        valid_metrics = ["sharpe", "cagr", "sortino", "calmar", "max_drawdown"]
        if v not in valid_metrics:
            raise ValueError(f"Primary metric must be one of {valid_metrics}")
        return v

    def get_symbols(self) -> list[str]:
        """Get the list of symbols to optimize on."""
        if self.symbols:
            return self.symbols
        elif self.symbol_list_path:
            try:
                df = pd.read_csv(self.symbol_list_path)
                if "symbol" in df.columns:
                    return df["symbol"].tolist()
                elif "ticker" in df.columns:
                    return df["ticker"].tolist()
                else:
                    raise ValueError("CSV must have 'symbol' or 'ticker' column")
            except Exception as e:
                raise ValueError(f"Failed to load symbol list: {e}")
        else:
            raise ValueError("Must specify either symbols or symbol_list_path")


# save/load methods inherited from BaseConfig


# Predefined strategy configurations
def get_trend_breakout_config() -> StrategyConfig:
    """Get configuration for trend breakout strategy with expanded parameter ranges."""
    return StrategyConfig(
        name="trend_breakout",
        description="Trend following strategy using Donchian channels and ATR with expanded parameter space",
        parameters={
            "donchian_lookback": ParameterDefinition(
                name="donchian_lookback",
                type="int",
                min_value=5,  # Much lower minimum for very short-term strategies
                max_value=500,  # Much higher maximum for very long-term strategies
                default=55,
                step=5,
                description="Donchian channel lookback period (5-500 days)",
            ),
            "atr_period": ParameterDefinition(
                name="atr_period",
                type="int",
                min_value=2,  # Very short ATR for quick reactions
                max_value=100,  # Very long ATR for stable volatility
                default=20,
                step=2,
                description="ATR calculation period (2-100 days)",
            ),
            "atr_k": ParameterDefinition(
                name="atr_k",
                type="float",
                min_value=0.1,  # Very tight stops
                max_value=10.0,  # Very wide stops
                default=2.0,
                step=0.05,
                description="ATR multiplier for position sizing (0.1-10.0)",
            ),
            "entry_confirm": ParameterDefinition(
                name="entry_confirm",
                type="int",
                min_value=0,  # No confirmation (immediate entry)
                max_value=10,  # Very conservative entry
                default=1,
                description="Entry confirmation periods (0-10 days)",
            ),
            "cooldown": ParameterDefinition(
                name="cooldown",
                type="int",
                min_value=0,  # No cooldown
                max_value=50,  # Very long cooldown
                default=0,
                description="Cooldown periods between trades (0-50 days)",
            ),
            "regime_window": ParameterDefinition(
                name="regime_window",
                type="int",
                min_value=10,  # Very short regime detection
                max_value=1000,  # Very long regime detection
                default=200,
                description="Regime filter window (10-1000 days)",
            ),
            "risk_pct": ParameterDefinition(
                name="risk_pct",
                type="float",
                min_value=0.01,  # Very conservative risk
                max_value=5.0,  # Very aggressive risk
                default=0.5,
                step=0.01,
                description="Risk per trade as percentage (0.01-5.0%)",
            ),
            # New parameters for more strategy diversity
            "trend_strength_threshold": ParameterDefinition(
                name="trend_strength_threshold",
                type="float",
                min_value=0.0,  # No trend requirement
                max_value=2.0,  # Strong trend requirement
                default=0.0,
                step=0.1,
                description="Minimum trend strength for entry (0.0-2.0)",
            ),
            "volume_filter": ParameterDefinition(
                name="volume_filter",
                type="bool",
                default=False,
                description="Enable volume-based filtering",
            ),
            "volatility_filter": ParameterDefinition(
                name="volatility_filter",
                type="bool",
                default=False,
                description="Enable volatility-based filtering",
            ),
            "momentum_lookback": ParameterDefinition(
                name="momentum_lookback",
                type="int",
                min_value=1,  # Very short momentum
                max_value=100,  # Very long momentum
                default=20,
                description="Momentum calculation period (1-100 days)",
            ),
            "profit_target_multiplier": ParameterDefinition(
                name="profit_target_multiplier",
                type="float",
                min_value=0.5,  # Tight profit target
                max_value=5.0,  # Wide profit target
                default=2.0,
                step=0.1,
                description="Profit target as multiple of ATR (0.5-5.0)",
            ),
        },
    )


def get_demo_ma_config() -> StrategyConfig:
    """Get configuration for demo moving average strategy."""
    return StrategyConfig(
        name="demo_ma",
        description="Simple moving average crossover strategy",
        parameters={
            "fast_period": ParameterDefinition(
                name="fast_period",
                type="int",
                min_value=5,
                max_value=50,
                default=10,
                step=5,
                description="Fast moving average period",
            ),
            "slow_period": ParameterDefinition(
                name="slow_period",
                type="int",
                min_value=20,
                max_value=200,
                default=50,
                step=10,
                description="Slow moving average period",
            ),
        },
    )
