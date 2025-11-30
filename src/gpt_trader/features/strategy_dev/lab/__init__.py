"""
Strategy Lab: Experiment tracking and optimization.

Provides tools for:
- Defining and tracking experiments
- A/B testing strategy variations
- Parameter grid search
- Results comparison and analysis
"""

from gpt_trader.features.strategy_dev.lab.experiment import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
)
from gpt_trader.features.strategy_dev.lab.parameter_grid import ParameterGrid
from gpt_trader.features.strategy_dev.lab.tracker import ExperimentTracker

__all__ = [
    "Experiment",
    "ExperimentResult",
    "ExperimentStatus",
    "ExperimentTracker",
    "ParameterGrid",
]
