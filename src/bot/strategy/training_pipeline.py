"""
Strategy Training Pipeline for GPT-Trader

Provides comprehensive strategy training, optimization, and validation framework.
Integrates with Historical Data Manager and Quality Framework for systematic strategy development.
"""

import json
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from enum import Enum as _Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from bot.portfolio.optimizer import OptimizationMethod

# Strategy and optimization imports
from bot.strategy.base import Strategy


class VersionType(_Enum):
    """Local VersionType to avoid circular imports with persistence.

    Matches bot.strategy.persistence.VersionType values for interoperability.
    """

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    HOTFIX = "hotfix"


from bot.backtest.engine_portfolio import BacktestConfig, BacktestEngine

# Note: Older docs reference `PortfolioAllocator`/`AllocationMethod`.
# The current codebase provides simpler allocation utilities; import if present.
try:
    from bot.portfolio.allocator import AllocationMethod, PortfolioAllocator  # type: ignore
except Exception:  # pragma: no cover - optional compatibility
    PortfolioAllocator = None  # sentinel for compatibility
    AllocationMethod = None

# Data imports
from bot.dataflow.data_quality_framework import DataQualityFramework
from bot.dataflow.historical_data_manager import DataFrequency, HistoricalDataManager

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phases"""

    PARAMETER_SEARCH = "parameter_search"
    WALK_FORWARD_VALIDATION = "walk_forward_validation"
    OUT_OF_SAMPLE_TESTING = "out_of_sample_testing"
    ROBUSTNESS_TESTING = "robustness_testing"
    FINAL_EVALUATION = "final_evaluation"


class ValidationMethod(Enum):
    """Validation methods"""

    SIMPLE_SPLIT = "simple_split"
    WALK_FORWARD = "walk_forward"
    PURGED_CROSS_VALIDATION = "purged_cross_validation"
    TIME_SERIES_SPLIT = "time_series_split"
    EXPANDING_WINDOW = "expanding_window"


class OptimizationObjective(Enum):
    """Optimization objectives"""

    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    TOTAL_RETURN = "total_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class TrainingConfig:
    """Configuration for strategy training"""

    # Training parameters
    training_start_date: datetime
    training_end_date: datetime
    validation_method: ValidationMethod = ValidationMethod.WALK_FORWARD
    test_split_ratio: float = 0.2  # Percentage for out-of-sample testing

    # Walk-forward validation settings
    training_window_months: int = 12  # Training window size
    validation_window_months: int = 3  # Validation window size
    step_size_months: int = 1  # Step size for rolling window
    min_training_samples: int = 252  # Minimum samples for training

    # Optimization settings
    # Use a valid option from Portfolio OptimizationMethod as default for compatibility
    optimization_method: OptimizationMethod = OptimizationMethod.SHARPE_MAXIMIZATION
    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    max_optimization_iterations: int = 100
    optimization_timeout_minutes: int = 30

    # Cross-validation settings
    cv_folds: int = 5
    purged_days: int = 2  # Days to purge between folds
    embargo_days: int = 1  # Embargo period to prevent leakage

    # Risk management
    max_drawdown_threshold: float = 0.15
    min_sharpe_threshold: float = 0.5
    min_trade_frequency: int = 10  # Minimum trades per year

    # Performance settings
    max_parallel_workers: int = mp.cpu_count() - 1
    use_multiprocessing: bool = True
    memory_limit_gb: float = 8.0

    # Output settings
    save_intermediate_results: bool = True
    detailed_logging: bool = True
    generate_plots: bool = False  # Disabled for server environments

    # Robustness testing
    bootstrap_samples: int = 1000
    noise_levels: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    transaction_cost_bps: list[float] = field(default_factory=lambda: [0, 5, 10])


@dataclass
class TrainingResult:
    """Results from strategy training"""

    strategy_id: str
    strategy_class: str
    best_parameters: dict[str, Any]
    training_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    out_of_sample_metrics: dict[str, float]

    # Walk-forward results
    walk_forward_results: list[dict[str, Any]] = field(default_factory=list)
    parameter_stability: dict[str, float] = field(default_factory=dict)

    # Robustness results
    bootstrap_metrics: dict[str, list[float]] = field(default_factory=dict)
    noise_sensitivity: dict[str, dict[str, float]] = field(default_factory=dict)
    transaction_cost_sensitivity: dict[str, dict[str, float]] = field(default_factory=dict)

    # Metadata
    training_duration: timedelta = field(default=timedelta())
    training_timestamp: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 0.0
    total_samples: int = 0

    @property
    def is_robust(self) -> bool:
        """Check if strategy shows robust performance"""
        # Check validation performance
        validation_sharpe = self.validation_metrics.get("sharpe_ratio", 0)
        validation_drawdown = self.validation_metrics.get("max_drawdown", 1)

        # Check out-of-sample performance
        oos_sharpe = self.out_of_sample_metrics.get("sharpe_ratio", 0)
        oos_drawdown = self.out_of_sample_metrics.get("max_drawdown", 1)

        # Check parameter stability
        param_stability = (
            np.mean(list(self.parameter_stability.values())) if self.parameter_stability else 0
        )

        return (
            validation_sharpe > 0.5
            and validation_drawdown < 0.15
            and oos_sharpe > 0.3  # Allow for some degradation
            and oos_drawdown < 0.20
            and param_stability > 0.7  # Parameters should be relatively stable
        )


@dataclass
class ValidationWindow:
    """Represents a validation window for walk-forward analysis"""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int

    @property
    def train_period_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_period_days(self) -> int:
        return (self.test_end - self.test_start).days


class ParameterOptimizer(ABC):
    """Base class for parameter optimization"""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @abstractmethod
    def optimize(
        self, strategy: Strategy, data: pd.DataFrame, parameter_space: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Optimize strategy parameters"""
        pass


class BayesianOptimizer(ParameterOptimizer):
    """Bayesian optimization for strategy parameters"""

    def optimize(
        self, strategy: Strategy, data: pd.DataFrame, parameter_space: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Optimize using Bayesian optimization"""

        try:
            from skopt import gp_minimize
            from skopt.space import Categorical, Integer, Real
            from skopt.utils import use_named_args
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to grid search")
            return self._fallback_grid_search(strategy, data, parameter_space)

        # Convert parameter space to skopt format
        dimensions = []
        param_names = []

        for param_name, param_config in parameter_space.items():
            param_names.append(param_name)

            if param_config["type"] == "real":
                dimensions.append(Real(param_config["low"], param_config["high"], name=param_name))
            elif param_config["type"] == "integer":
                dimensions.append(
                    Integer(param_config["low"], param_config["high"], name=param_name)
                )
            elif param_config["type"] == "categorical":
                dimensions.append(Categorical(param_config["categories"], name=param_name))

        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            try:
                # Update strategy parameters
                for param_name, param_value in params.items():
                    setattr(strategy, param_name, param_value)

                # Run backtest
                backtest_config = BacktestConfig(
                    start_date=data.index[0], end_date=data.index[-1], initial_capital=100000.0
                )

                engine = BacktestEngine(backtest_config)
                results = engine.run_backtest(strategy, data)

                # Return negative value for minimization
                objective_value = self._calculate_objective(results)
                return -objective_value

            except Exception as e:
                logger.warning(f"Optimization iteration failed: {str(e)}")
                return float("inf")  # Return worst possible value

        # Run optimization
        try:
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=self.config.max_optimization_iterations,
                random_state=42,
                n_random_starts=min(10, self.config.max_optimization_iterations // 4),
            )

            # Extract best parameters
            best_params = dict(zip(param_names, result.x, strict=False))
            best_score = -result.fun

            return best_params, best_score

        except Exception as e:
            logger.error(f"Bayesian optimization failed: {str(e)}")
            return self._fallback_grid_search(strategy, data, parameter_space)

    def _fallback_grid_search(
        self, strategy: Strategy, data: pd.DataFrame, parameter_space: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """Fallback to simple grid search"""
        best_params = {}
        best_score = -np.inf

        # Simple grid search implementation
        param_combinations = self._generate_param_combinations(parameter_space, max_combinations=50)

        for params in param_combinations:
            try:
                # Update strategy parameters
                for param_name, param_value in params.items():
                    setattr(strategy, param_name, param_value)

                # Run backtest
                backtest_config = BacktestConfig(
                    start_date=data.index[0], end_date=data.index[-1], initial_capital=100000.0
                )

                engine = BacktestEngine(backtest_config)
                results = engine.run_backtest(strategy, data)

                score = self._calculate_objective(results)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                logger.debug(f"Grid search iteration failed: {str(e)}")
                continue

        return best_params, best_score

    def _generate_param_combinations(
        self, parameter_space: dict[str, Any], max_combinations: int = 100
    ) -> list[dict[str, Any]]:
        """Generate parameter combinations for grid search"""
        combinations = []

        # Simple sampling approach for now
        import random

        random.seed(42)

        for _ in range(max_combinations):
            params = {}
            for param_name, param_config in parameter_space.items():
                if param_config["type"] == "real":
                    params[param_name] = random.uniform(param_config["low"], param_config["high"])
                elif param_config["type"] == "integer":
                    params[param_name] = random.randint(param_config["low"], param_config["high"])
                elif param_config["type"] == "categorical":
                    params[param_name] = random.choice(param_config["categories"])

            combinations.append(params)

        return combinations

    def _calculate_objective(self, backtest_results: dict[str, Any]) -> float:
        """Calculate optimization objective from backtest results"""
        metrics = backtest_results.get("metrics", {})

        if self.config.optimization_objective == OptimizationObjective.SHARPE_RATIO:
            return metrics.get("sharpe_ratio", -1.0)
        elif self.config.optimization_objective == OptimizationObjective.CALMAR_RATIO:
            return metrics.get("calmar_ratio", -1.0)
        elif self.config.optimization_objective == OptimizationObjective.SORTINO_RATIO:
            return metrics.get("sortino_ratio", -1.0)
        elif self.config.optimization_objective == OptimizationObjective.TOTAL_RETURN:
            return metrics.get("total_return", -1.0)
        elif self.config.optimization_objective == OptimizationObjective.MAXIMUM_DRAWDOWN:
            return -metrics.get("max_drawdown", 1.0)  # Minimize drawdown
        else:
            # Multi-objective: weighted combination
            sharpe = metrics.get("sharpe_ratio", 0)
            calmar = metrics.get("calmar_ratio", 0)
            return 0.6 * sharpe + 0.4 * calmar


class WalkForwardValidator:
    """Walk-forward validation implementation"""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def generate_validation_windows(
        self, start_date: datetime, end_date: datetime
    ) -> list[ValidationWindow]:
        """Generate walk-forward validation windows"""
        windows = []
        window_id = 0

        current_date = start_date
        training_delta = timedelta(days=self.config.training_window_months * 30)
        validation_delta = timedelta(days=self.config.validation_window_months * 30)
        step_delta = timedelta(days=self.config.step_size_months * 30)

        while current_date + training_delta + validation_delta <= end_date:
            train_start = current_date
            train_end = current_date + training_delta
            test_start = train_end
            test_end = test_start + validation_delta

            windows.append(
                ValidationWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id,
                )
            )

            window_id += 1
            current_date += step_delta

        return windows

    def validate_strategy(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        parameter_space: dict[str, Any],
        optimizer: ParameterOptimizer,
    ) -> dict[str, Any]:
        """Run walk-forward validation"""

        windows = self.generate_validation_windows(data.index[0], data.index[-1])
        logger.info(f"Generated {len(windows)} validation windows")

        results = []
        parameter_history = []

        for window in windows:
            logger.info(f"Processing window {window.window_id + 1}/{len(windows)}")

            # Get training data
            train_data = data.loc[window.train_start : window.train_end]
            test_data = data.loc[window.test_start : window.test_end]

            if len(train_data) < self.config.min_training_samples:
                logger.warning(f"Insufficient training data in window {window.window_id}")
                continue

            try:
                # Optimize parameters on training data
                best_params, train_score = optimizer.optimize(strategy, train_data, parameter_space)
                parameter_history.append(best_params.copy())

                # Apply best parameters
                for param_name, param_value in best_params.items():
                    setattr(strategy, param_name, param_value)

                # Test on validation data
                backtest_config = BacktestConfig(
                    start_date=test_data.index[0],
                    end_date=test_data.index[-1],
                    initial_capital=100000.0,
                )

                engine = BacktestEngine(backtest_config)
                test_results = engine.run_backtest(strategy, test_data)

                window_result = {
                    "window_id": window.window_id,
                    "train_period": f"{window.train_start.date()} to {window.train_end.date()}",
                    "test_period": f"{window.test_start.date()} to {window.test_end.date()}",
                    "best_parameters": best_params,
                    "train_score": train_score,
                    "test_metrics": test_results.get("metrics", {}),
                    "train_samples": len(train_data),
                    "test_samples": len(test_data),
                }

                results.append(window_result)

            except Exception as e:
                logger.error(f"Walk-forward window {window.window_id} failed: {str(e)}")
                continue

        # Calculate parameter stability
        parameter_stability = self._calculate_parameter_stability(parameter_history)

        # Aggregate results
        validation_summary = self._aggregate_validation_results(results, parameter_stability)

        return validation_summary

    def _calculate_parameter_stability(
        self, parameter_history: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate stability of parameters across windows"""
        if not parameter_history:
            return {}

        stability = {}

        for param_name in parameter_history[0].keys():
            values = [params[param_name] for params in parameter_history if param_name in params]

            if len(values) < 2:
                stability[param_name] = 1.0
                continue

            # Calculate coefficient of variation for numerical parameters
            if isinstance(values[0], int | float):
                if np.std(values) == 0:
                    stability[param_name] = 1.0
                else:
                    cv = (
                        np.std(values) / np.abs(np.mean(values))
                        if np.mean(values) != 0
                        else float("inf")
                    )
                    stability[param_name] = max(0, 1 - cv)
            else:
                # For categorical parameters, calculate consistency
                unique_values = len(set(str(v) for v in values))
                stability[param_name] = 1.0 - (unique_values - 1) / len(values)

        return stability

    def _aggregate_validation_results(
        self, results: list[dict[str, Any]], parameter_stability: dict[str, float]
    ) -> dict[str, Any]:
        """Aggregate walk-forward validation results"""
        if not results:
            return {"error": "No successful validation windows"}

        # Aggregate test metrics
        test_metrics = {}
        metric_names = results[0]["test_metrics"].keys()

        for metric in metric_names:
            values = [r["test_metrics"][metric] for r in results if metric in r["test_metrics"]]
            if values:
                test_metrics[f"{metric}_mean"] = np.mean(values)
                test_metrics[f"{metric}_std"] = np.std(values)
                test_metrics[f"{metric}_median"] = np.median(values)
                test_metrics[f"{metric}_min"] = np.min(values)
                test_metrics[f"{metric}_max"] = np.max(values)

        return {
            "total_windows": len(results),
            "successful_windows": len(results),
            "aggregated_metrics": test_metrics,
            "parameter_stability": parameter_stability,
            "mean_parameter_stability": (
                np.mean(list(parameter_stability.values())) if parameter_stability else 0.0
            ),
            "window_results": results,
        }


class StrategyTrainingPipeline:
    """Main strategy training pipeline"""

    def __init__(
        self,
        config: TrainingConfig,
        data_manager: HistoricalDataManager,
        quality_framework: DataQualityFramework,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.quality_framework = quality_framework
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.optimizer = BayesianOptimizer(config)
        self.validator = WalkForwardValidator(config)

        # Results storage
        self.training_results: dict[str, TrainingResult] = {}

        logger.info("Strategy Training Pipeline initialized")

    def train_strategy(
        self,
        strategy: Strategy,
        symbols: list[str],
        parameter_space: dict[str, Any],
        strategy_id: str | None = None,
    ) -> TrainingResult:
        """Train a single strategy with comprehensive validation"""

        if strategy_id is None:
            strategy_id = (
                f"{strategy.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        logger.info(f"Starting training for strategy: {strategy_id}")
        start_time = datetime.now()

        try:
            # Step 1: Prepare data
            logger.info("Step 1: Preparing training data...")
            datasets, metadata = self.data_manager.get_training_dataset(
                symbols=symbols,
                start_date=self.config.training_start_date,
                end_date=self.config.training_end_date,
                frequency=DataFrequency.DAILY,
            )

            if not datasets:
                raise ValueError("No datasets available for training")

            # Combine datasets if multiple symbols
            combined_data = self._combine_datasets(datasets)
            logger.info(f"Combined dataset: {len(combined_data)} records")

            # Step 2: Split data for out-of-sample testing
            logger.info("Step 2: Splitting data for validation...")
            train_data, oos_data = self._split_data(combined_data)

            # Step 3: Walk-forward validation
            logger.info("Step 3: Running walk-forward validation...")
            validation_results = self.validator.validate_strategy(
                strategy, train_data, parameter_space, self.optimizer
            )

            if "error" in validation_results:
                raise ValueError(f"Validation failed: {validation_results['error']}")

            # Step 4: Train final model on all training data
            logger.info("Step 4: Training final model...")
            final_params, final_score = self.optimizer.optimize(
                strategy, train_data, parameter_space
            )

            # Apply final parameters
            for param_name, param_value in final_params.items():
                setattr(strategy, param_name, param_value)

            # Step 5: Out-of-sample testing
            logger.info("Step 5: Out-of-sample testing...")
            oos_results = self._run_out_of_sample_test(strategy, oos_data)

            # Step 6: Robustness testing
            logger.info("Step 6: Robustness testing...")
            robustness_results = self._run_robustness_tests(strategy, train_data, final_params)

            # Step 7: Compile results
            end_time = datetime.now()
            training_duration = end_time - start_time

            # Calculate data quality score
            quality_scores = [
                metadata.quality_metrics[symbol].quality_score
                for symbol in metadata.quality_metrics
            ]
            avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0

            result = TrainingResult(
                strategy_id=strategy_id,
                strategy_class=strategy.__class__.__name__,
                best_parameters=final_params,
                training_metrics={"final_score": final_score},
                validation_metrics=validation_results.get("aggregated_metrics", {}),
                out_of_sample_metrics=oos_results.get("metrics", {}),
                walk_forward_results=validation_results.get("window_results", []),
                parameter_stability=validation_results.get("parameter_stability", {}),
                bootstrap_metrics=robustness_results.get("bootstrap_metrics", {}),
                noise_sensitivity=robustness_results.get("noise_sensitivity", {}),
                transaction_cost_sensitivity=robustness_results.get(
                    "transaction_cost_sensitivity", {}
                ),
                training_duration=training_duration,
                training_timestamp=start_time,
                data_quality_score=avg_quality_score,
                total_samples=len(combined_data),
            )

            # Step 8: Save results
            if self.config.save_intermediate_results:
                self._save_training_result(result)

            self.training_results[strategy_id] = result

            logger.info(f"Training completed for {strategy_id}")
            logger.info(f"  Duration: {training_duration}")
            logger.info(f"  Robust: {'Yes' if result.is_robust else 'No'}")
            logger.info(
                f"  Validation Sharpe: {result.validation_metrics.get('sharpe_ratio_mean', 'N/A'):.3f}"
            )
            logger.info(
                f"  Out-of-sample Sharpe: {result.out_of_sample_metrics.get('sharpe_ratio', 'N/A'):.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Training failed for {strategy_id}: {str(e)}")
            raise

    def _combine_datasets(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets for training"""
        if len(datasets) == 1:
            return list(datasets.values())[0]

        # For now, use the first dataset
        # In production, this could implement portfolio-based combination
        primary_symbol = list(datasets.keys())[0]
        return datasets[primary_symbol]

    def _split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and out-of-sample sets"""
        split_idx = int(len(data) * (1 - self.config.test_split_ratio))

        train_data = data.iloc[:split_idx]
        oos_data = data.iloc[split_idx:]

        logger.info(f"Data split: {len(train_data)} training, {len(oos_data)} out-of-sample")
        return train_data, oos_data

    def _run_out_of_sample_test(self, strategy: Strategy, oos_data: pd.DataFrame) -> dict[str, Any]:
        """Run out-of-sample testing"""
        try:
            backtest_config = BacktestConfig(
                start_date=oos_data.index[0], end_date=oos_data.index[-1], initial_capital=100000.0
            )

            engine = BacktestEngine(backtest_config)
            results = engine.run_backtest(strategy, oos_data)

            return results

        except Exception as e:
            logger.error(f"Out-of-sample testing failed: {str(e)}")
            return {"error": str(e), "metrics": {}}

    def _run_robustness_tests(
        self, strategy: Strategy, data: pd.DataFrame, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Run robustness tests with bootstrap and sensitivity analysis"""

        robustness_results = {
            "bootstrap_metrics": {},
            "noise_sensitivity": {},
            "transaction_cost_sensitivity": {},
        }

        try:
            # Bootstrap testing
            if self.config.bootstrap_samples > 0:
                logger.info("Running bootstrap tests...")
                bootstrap_results = self._bootstrap_test(strategy, data, parameters)
                robustness_results["bootstrap_metrics"] = bootstrap_results

            # Noise sensitivity testing
            if self.config.noise_levels:
                logger.info("Running noise sensitivity tests...")
                noise_results = self._noise_sensitivity_test(strategy, data, parameters)
                robustness_results["noise_sensitivity"] = noise_results

            # Transaction cost sensitivity
            if self.config.transaction_cost_bps:
                logger.info("Running transaction cost sensitivity tests...")
                cost_results = self._transaction_cost_sensitivity_test(strategy, data, parameters)
                robustness_results["transaction_cost_sensitivity"] = cost_results

        except Exception as e:
            logger.error(f"Robustness testing failed: {str(e)}")
            robustness_results["error"] = str(e)

        return robustness_results

    def _bootstrap_test(
        self, strategy: Strategy, data: pd.DataFrame, parameters: dict[str, Any]
    ) -> dict[str, list[float]]:
        """Bootstrap testing for robustness"""

        bootstrap_metrics = {
            "sharpe_ratio": [],
            "total_return": [],
            "max_drawdown": [],
            "calmar_ratio": [],
        }

        # Use smaller sample size for bootstrap to keep it manageable
        n_samples = min(self.config.bootstrap_samples, 100)

        for i in range(n_samples):
            try:
                # Bootstrap sample
                sample_data = data.sample(n=len(data), replace=True, random_state=i)
                sample_data = sample_data.sort_index()

                # Apply parameters
                for param_name, param_value in parameters.items():
                    setattr(strategy, param_name, param_value)

                # Run backtest
                backtest_config = BacktestConfig(
                    start_date=sample_data.index[0],
                    end_date=sample_data.index[-1],
                    initial_capital=100000.0,
                )

                engine = BacktestEngine(backtest_config)
                results = engine.run_backtest(strategy, sample_data)

                metrics = results.get("metrics", {})
                for metric_name in bootstrap_metrics:
                    if metric_name in metrics:
                        bootstrap_metrics[metric_name].append(metrics[metric_name])

            except Exception as e:
                logger.debug(f"Bootstrap sample {i} failed: {str(e)}")
                continue

        return bootstrap_metrics

    def _noise_sensitivity_test(
        self, strategy: Strategy, data: pd.DataFrame, parameters: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Test sensitivity to noise in data"""

        noise_results = {}

        for noise_level in self.config.noise_levels:
            noise_results[f"noise_{noise_level}"] = {}

            try:
                # Add noise to price data
                noisy_data = data.copy()
                price_columns = ["Open", "High", "Low", "Close"]

                for col in price_columns:
                    if col in noisy_data.columns:
                        noise = np.random.normal(0, noise_level, len(noisy_data))
                        noisy_data[col] *= 1 + noise

                # Apply parameters
                for param_name, param_value in parameters.items():
                    setattr(strategy, param_name, param_value)

                # Run backtest
                backtest_config = BacktestConfig(
                    start_date=noisy_data.index[0],
                    end_date=noisy_data.index[-1],
                    initial_capital=100000.0,
                )

                engine = BacktestEngine(backtest_config)
                results = engine.run_backtest(strategy, noisy_data)

                noise_results[f"noise_{noise_level}"] = results.get("metrics", {})

            except Exception as e:
                logger.debug(f"Noise sensitivity test {noise_level} failed: {str(e)}")
                noise_results[f"noise_{noise_level}"] = {"error": str(e)}

        return noise_results

    def _transaction_cost_sensitivity_test(
        self, strategy: Strategy, data: pd.DataFrame, parameters: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Test sensitivity to transaction costs"""

        cost_results = {}

        for cost_bps in self.config.transaction_cost_bps:
            cost_results[f"cost_{cost_bps}bps"] = {}

            try:
                # Apply parameters
                for param_name, param_value in parameters.items():
                    setattr(strategy, param_name, param_value)

                # Run backtest with transaction costs
                backtest_config = BacktestConfig(
                    start_date=data.index[0],
                    end_date=data.index[-1],
                    initial_capital=100000.0,
                    commission_rate=cost_bps / 10000.0,  # Convert bps to decimal
                )

                engine = BacktestEngine(backtest_config)
                results = engine.run_backtest(strategy, data)

                cost_results[f"cost_{cost_bps}bps"] = results.get("metrics", {})

            except Exception as e:
                logger.debug(f"Transaction cost test {cost_bps}bps failed: {str(e)}")
                cost_results[f"cost_{cost_bps}bps"] = {"error": str(e)}

        return cost_results

    def _save_training_result(self, result: TrainingResult) -> None:
        """Save training result to disk"""
        try:
            results_dir = self.output_dir / "training_results"
            results_dir.mkdir(exist_ok=True)

            # Save as JSON
            result_dict = {
                "strategy_id": result.strategy_id,
                "strategy_class": result.strategy_class,
                "best_parameters": result.best_parameters,
                "training_metrics": result.training_metrics,
                "validation_metrics": result.validation_metrics,
                "out_of_sample_metrics": result.out_of_sample_metrics,
                "walk_forward_results": result.walk_forward_results,
                "parameter_stability": result.parameter_stability,
                "training_duration": str(result.training_duration),
                "training_timestamp": result.training_timestamp.isoformat(),
                "data_quality_score": result.data_quality_score,
                "total_samples": result.total_samples,
                "is_robust": result.is_robust,
            }

            filename = f"{result.strategy_id}_training_result.json"
            filepath = results_dir / filename

            with open(filepath, "w") as f:
                json.dump(result_dict, f, indent=2, default=str)

            # Save full result object using joblib for safe serialization
            joblib_filename = f"{result.strategy_id}_full_result.joblib"
            joblib_filepath = results_dir / joblib_filename

            joblib.dump(result, joblib_filepath)

            logger.info(f"Training result saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to save training result: {str(e)}")

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of all training results"""
        if not self.training_results:
            return {"message": "No training results available"}

        results = list(self.training_results.values())

        return {
            "total_strategies_trained": len(results),
            "robust_strategies": len([r for r in results if r.is_robust]),
            "average_validation_sharpe": np.mean(
                [r.validation_metrics.get("sharpe_ratio_mean", 0) for r in results]
            ),
            "average_oos_sharpe": np.mean(
                [r.out_of_sample_metrics.get("sharpe_ratio", 0) for r in results]
            ),
            "average_training_duration": str(
                np.mean([r.training_duration.total_seconds() for r in results])
            )
            + " seconds",
            "strategy_summaries": [
                {
                    "strategy_id": r.strategy_id,
                    "strategy_class": r.strategy_class,
                    "is_robust": r.is_robust,
                    "validation_sharpe": r.validation_metrics.get("sharpe_ratio_mean", "N/A"),
                    "oos_sharpe": r.out_of_sample_metrics.get("sharpe_ratio", "N/A"),
                    "parameter_stability": (
                        np.mean(list(r.parameter_stability.values()))
                        if r.parameter_stability
                        else 0.0
                    ),
                }
                for r in results
            ],
        }


# Factory function for easy initialization
def create_strategy_training_pipeline(
    config: TrainingConfig,
    data_manager: HistoricalDataManager,
    quality_framework: DataQualityFramework,
    output_dir: str = "data/strategy_training",
) -> StrategyTrainingPipeline:
    """Factory function to create strategy training pipeline"""

    return StrategyTrainingPipeline(
        config=config,
        data_manager=data_manager,
        quality_framework=quality_framework,
        output_dir=Path(output_dir),
    )


# Example usage and testing
if __name__ == "__main__":

    def main() -> None:
        """Example usage of Strategy Training Pipeline"""
        print("Strategy Training Pipeline Testing")
        print("=" * 40)

        # Create configuration
        config = TrainingConfig(
            training_start_date=datetime(2022, 1, 1),
            training_end_date=datetime(2023, 12, 31),
            validation_method=ValidationMethod.WALK_FORWARD,
            training_window_months=6,
            validation_window_months=2,
            max_optimization_iterations=20,  # Reduced for testing
            bootstrap_samples=50,  # Reduced for testing
        )

        print("Training Configuration:")
        print(
            f"   Date Range: {config.training_start_date.date()} to {config.training_end_date.date()}"
        )
        print(f"   Validation Method: {config.validation_method.value}")
        print(f"   Training Window: {config.training_window_months} months")
        print(f"   Max Iterations: {config.max_optimization_iterations}")

        print("\n🚀 Strategy Training Pipeline ready for production!")
        print("   Next: Create strategy instances and parameter spaces")
        print("   Then: Run training pipeline on validated datasets")

    # Run the example
    main()
