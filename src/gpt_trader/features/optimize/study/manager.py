"""Optuna study manager for strategy optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from gpt_trader.features.optimize.types import OptimizationConfig, ParameterType
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    import optuna
    from optuna.pruners import BasePruner
    from optuna.samplers import BaseSampler
    from optuna.trial import Trial

_OPTUNA_IMPORT_ERROR: ImportError | None
_optuna: Any | None

try:
    import optuna as _loaded_optuna
    from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
except ImportError as exc:
    _optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _optuna = _loaded_optuna
    _OPTUNA_IMPORT_ERROR = None

logger = get_logger(__name__, component="optuna_study")

OPTIMIZE_EXTRA_INSTALL_MESSAGE = (
    "Optuna is required for optimization studies. Install the optimize extra with "
    "`pip install 'gpt-trader[optimize]'` or `uv sync --extra optimize`."
)


class MissingOptimizeDependencyError(ImportError):
    """Raised when optimize study tooling is used without its optional dependency."""


def _require_optuna_extra() -> Any:
    """Return the Optuna module or raise a user-facing install hint."""
    if _OPTUNA_IMPORT_ERROR is not None or _optuna is None:
        raise MissingOptimizeDependencyError(OPTIMIZE_EXTRA_INSTALL_MESSAGE) from (
            _OPTUNA_IMPORT_ERROR
        )
    return _optuna


class OptimizationStudyManager:
    """
    Manages Optuna studies for strategy optimization.

    Handles:
    - Study creation and loading
    - Sampler and pruner configuration
    - Parameter suggestion based on ParameterSpace
    """

    def __init__(self, config: OptimizationConfig, storage_url: str | None = None):
        """
        Initialize study manager.

        Args:
            config: Optimization configuration
            storage_url: Database URL for persistent storage (optional)
        """
        _require_optuna_extra()
        self.config = config
        self.storage_url = storage_url

    def create_or_load_study(self) -> optuna.Study:
        """
        Create a new study or load an existing one.

        Returns:
            Configured Optuna study
        """
        optuna_module = _require_optuna_extra()
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        study = optuna_module.create_study(
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
        return cast("optuna.Study", study)

    def suggest_parameters(self, trial: Trial) -> dict[str, Any]:
        """
        Suggest parameters for a trial based on the parameter space.

        Args:
            trial: Optuna trial instance

        Returns:
            Dictionary of suggested parameter values
        """
        params: dict[str, Any] = {}

        for param in self.config.parameter_space.all_parameters:
            if param.parameter_type == ParameterType.INTEGER:
                if param.low is None or param.high is None:
                    continue
                params[param.name] = trial.suggest_int(
                    param.name,
                    int(param.low),
                    int(param.high),
                    step=int(param.step) if param.step else 1,
                    log=param.log,
                )

            elif param.parameter_type == ParameterType.FLOAT:
                if param.low is None or param.high is None:
                    continue
                params[param.name] = trial.suggest_float(
                    param.name,
                    float(param.low),
                    float(param.high),
                    step=float(param.step) if param.step else None,
                    log=param.log,
                )

            elif param.parameter_type == ParameterType.CATEGORICAL:
                if not param.choices:
                    continue
                params[param.name] = trial.suggest_categorical(param.name, param.choices)

            elif param.parameter_type == ParameterType.LOG_UNIFORM:
                # Deprecated in favor of suggest_float(log=True) but kept for compatibility if needed
                # We map it to suggest_float with log=True
                if param.low is None or param.high is None:
                    continue
                params[param.name] = trial.suggest_float(
                    param.name, float(param.low), float(param.high), log=True
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

    def _create_pruner(self) -> BasePruner | None:
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
