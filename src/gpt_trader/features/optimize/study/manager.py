"""Optuna study manager for strategy optimization."""

from __future__ import annotations

import logging
from typing import Any, Optional

import optuna
from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, PercentilePruner
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler
from optuna.trial import Trial

from gpt_trader.features.optimize.types import OptimizationConfig, ParameterType

logger = logging.getLogger(__name__)


class OptimizationStudyManager:
    """
    Manages Optuna studies for strategy optimization.

    Handles:
    - Study creation and loading
    - Sampler and pruner configuration
    - Parameter suggestion based on ParameterSpace
    """

    def __init__(self, config: OptimizationConfig, storage_url: Optional[str] = None):
        """
        Initialize study manager.

        Args:
            config: Optimization configuration
            storage_url: Database URL for persistent storage (optional)
        """
        self.config = config
        self.storage_url = storage_url

    def create_or_load_study(self) -> optuna.Study:
        """
        Create a new study or load an existing one.

        Returns:
            Configured Optuna study
        """
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.storage_url,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        
        logger.info(
            f"Study '{self.config.study_name}' loaded with sampler={self.config.sampler_type}, "
            f"pruner={self.config.pruner_type}"
        )
        return study

    def suggest_parameters(self, trial: Trial) -> dict[str, Any]:
        """
        Suggest parameters for a trial based on the parameter space.

        Args:
            trial: Optuna trial instance

        Returns:
            Dictionary of suggested parameter values
        """
        params = {}
        
        for param in self.config.parameter_space.all_parameters:
            if param.parameter_type == ParameterType.INTEGER:
                if param.low is None or param.high is None:
                    continue
                params[param.name] = trial.suggest_int(
                    param.name, 
                    int(param.low), 
                    int(param.high), 
                    step=int(param.step) if param.step else 1,
                    log=param.log
                )
            
            elif param.parameter_type == ParameterType.FLOAT:
                if param.low is None or param.high is None:
                    continue
                params[param.name] = trial.suggest_float(
                    param.name,
                    float(param.low),
                    float(param.high),
                    step=float(param.step) if param.step else None,
                    log=param.log
                )
            
            elif param.parameter_type == ParameterType.CATEGORICAL:
                if not param.choices:
                    continue
                params[param.name] = trial.suggest_categorical(
                    param.name,
                    param.choices
                )
                
            elif param.parameter_type == ParameterType.LOG_UNIFORM:
                # Deprecated in favor of suggest_float(log=True) but kept for compatibility if needed
                # We map it to suggest_float with log=True
                if param.low is None or param.high is None:
                    continue
                params[param.name] = trial.suggest_float(
                    param.name,
                    float(param.low),
                    float(param.high),
                    log=True
                )

        return params

    def _create_sampler(self) -> BaseSampler:
        """Create the configured sampler."""
        seed = self.config.seed
        
        if self.config.sampler_type == "tpe":
            return TPESampler(seed=seed)
        elif self.config.sampler_type == "cmaes":
            return CmaEsSampler(seed=seed)
        elif self.config.sampler_type == "random":
            return RandomSampler(seed=seed)
        else:
            logger.warning(f"Unknown sampler type '{self.config.sampler_type}', defaulting to TPE")
            return TPESampler(seed=seed)

    def _create_pruner(self) -> Optional[BasePruner]:
        """Create the configured pruner."""
        if self.config.pruner_type == "median":
            return MedianPruner()
        elif self.config.pruner_type == "hyperband":
            return HyperbandPruner()
        elif self.config.pruner_type == "percentile":
            return PercentilePruner(25.0)
        elif self.config.pruner_type is None:
            return None
        else:
            logger.warning(f"Unknown pruner type '{self.config.pruner_type}', defaulting to Median")
            return MedianPruner()
