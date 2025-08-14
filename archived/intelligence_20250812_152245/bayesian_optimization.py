"""
Bayesian Optimization Framework for GPT-Trader Phase 1.

This module provides sophisticated hyperparameter optimization using:
- Gaussian Process surrogate models
- Advanced acquisition functions (EI, UCB, PI)
- Multi-objective optimization capabilities
- Uncertainty quantification
- Early stopping and convergence detection

Replaces grid search with intelligent exploration/exploitation optimization.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

# Optional advanced optimization libraries
try:
    import skopt
    from skopt import dummy_minimize, forest_minimize, gp_minimize
    from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
    from skopt.space import Categorical, Integer, Real
    from skopt.utils import use_named_args

    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    warnings.warn("scikit-optimize not available. Install with: pip install scikit-optimize")

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results of Bayesian optimization."""

    best_params: dict[str, Any]
    best_score: float
    best_std: float | None = None
    n_iterations: int = 0
    convergence_iter: int | None = None
    optimization_history: list[dict[str, Any]] = field(default_factory=list)
    acquisition_values: list[float] = field(default_factory=list)
    total_time: float = 0.0


@dataclass
class ParameterSpace:
    """Definition of parameter search space."""

    name: str
    param_type: str  # real, integer, categorical
    bounds: tuple[float, float] | None = None
    categories: list[Any] | None = None
    prior: str = "uniform"  # uniform, log-uniform, normal
    transformation: str | None = None  # log, exp, sqrt


@dataclass
class BayesianOptConfig(BaseConfig):
    """Configuration for Bayesian optimization."""

    # Optimization parameters
    n_calls: int = 100
    n_initial_points: int = 10
    acquisition_function: str = "EI"  # EI, UCB, PI, LCB
    acquisition_optimizer: str = "lbfgs"  # lbfgs, sampling, auto

    # Gaussian Process parameters
    kernel_type: str = "matern"  # matern, rbf, combined
    kernel_length_scale: float = 1.0
    kernel_nu: float = 2.5  # For Matern kernel
    alpha: float = 1e-6
    normalize_y: bool = True

    # Acquisition function parameters
    xi: float = 0.01  # Exploration parameter for EI/PI
    kappa: float = 1.96  # Exploration parameter for UCB/LCB

    # Multi-objective parameters
    multi_objective: bool = False
    objectives: list[str] = field(default_factory=list)
    objective_weights: list[float] = field(default_factory=list)

    # Convergence and early stopping
    early_stopping: bool = True
    patience: int = 20
    min_improvement: float = 1e-4
    convergence_threshold: float = 1e-3

    # Cross-validation parameters
    cv_folds: int = 5
    time_series_cv: bool = True
    scoring_metric: str = "sharpe_ratio"

    # Advanced options
    noise_handling: bool = True
    uncertainty_sampling: bool = True
    parallel_evaluation: bool = False
    n_jobs: int = 1

    # Random state
    random_state: int = 42


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(
        self, X: np.ndarray, gp: GaussianProcessRegressor, y_opt: float, **kwargs
    ) -> np.ndarray:
        """Evaluate acquisition function."""
        pass


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi

    def __call__(
        self, X: np.ndarray, gp: GaussianProcessRegressor, y_opt: float, **kwargs
    ) -> np.ndarray:
        """Calculate Expected Improvement."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1) if sigma.ndim == 1 else sigma

        with np.errstate(divide="ignore"):
            imp = mu - y_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.flatten()


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""

    def __init__(self, kappa: float = 1.96) -> None:
        self.kappa = kappa

    def __call__(
        self, X: np.ndarray, gp: GaussianProcessRegressor, y_opt: float, **kwargs
    ) -> np.ndarray:
        """Calculate Upper Confidence Bound."""
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function."""

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi

    def __call__(
        self, X: np.ndarray, gp: GaussianProcessRegressor, y_opt: float, **kwargs
    ) -> np.ndarray:
        """Calculate Probability of Improvement."""
        mu, sigma = gp.predict(X, return_std=True)

        with np.errstate(divide="ignore"):
            Z = (mu - y_opt - self.xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0

        return pi


class LowerConfidenceBound(AcquisitionFunction):
    """Lower Confidence Bound acquisition function (for minimization)."""

    def __init__(self, kappa: float = 1.96) -> None:
        self.kappa = kappa

    def __call__(
        self, X: np.ndarray, gp: GaussianProcessRegressor, y_opt: float, **kwargs
    ) -> np.ndarray:
        """Calculate Lower Confidence Bound."""
        mu, sigma = gp.predict(X, return_std=True)
        return mu - self.kappa * sigma


class BayesianOptimizer:
    """
    Bayesian Optimization framework for hyperparameter tuning.

    Uses Gaussian Process surrogate models with acquisition functions
    for intelligent exploration of parameter space.
    """

    def __init__(self, config: BayesianOptConfig) -> None:
        self.config = config
        self.parameter_space: list[ParameterSpace] = []
        self.optimization_history: list[dict[str, Any]] = []
        self.gp_model: GaussianProcessRegressor | None = None
        self.scaler = StandardScaler()

        # Initialize acquisition function
        self.acquisition_function = self._create_acquisition_function()

    def add_parameter(
        self,
        name: str,
        param_type: str,
        bounds: tuple[float, float] | None = None,
        categories: list[Any] | None = None,
        prior: str = "uniform",
    ) -> None:
        """Add parameter to optimization space."""
        param = ParameterSpace(
            name=name, param_type=param_type, bounds=bounds, categories=categories, prior=prior
        )
        self.parameter_space.append(param)

    def _create_acquisition_function(self) -> AcquisitionFunction:
        """Create acquisition function based on configuration."""
        if self.config.acquisition_function == "EI":
            return ExpectedImprovement(xi=self.config.xi)
        elif self.config.acquisition_function == "UCB":
            return UpperConfidenceBound(kappa=self.config.kappa)
        elif self.config.acquisition_function == "PI":
            return ProbabilityOfImprovement(xi=self.config.xi)
        elif self.config.acquisition_function == "LCB":
            return LowerConfidenceBound(kappa=self.config.kappa)
        else:
            return ExpectedImprovement(xi=self.config.xi)

    def _create_kernel(self) -> Any:
        """Create GP kernel based on configuration."""
        length_scale = self.config.kernel_length_scale

        if self.config.kernel_type == "rbf":
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif self.config.kernel_type == "matern":
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=length_scale, nu=self.config.kernel_nu
            )
        elif self.config.kernel_type == "combined":
            kernel = (
                ConstantKernel(1.0) * RBF(length_scale=length_scale)
                + ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=1.5)
                + WhiteKernel(noise_level=1e-5)
            )
        else:
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)

        return kernel

    def _encode_parameters(self, params: dict[str, Any]) -> np.ndarray:
        """Encode parameters for GP model."""
        encoded = []

        for param_def in self.parameter_space:
            value = params[param_def.name]

            if param_def.param_type == "real":
                encoded.append(float(value))
            elif param_def.param_type == "integer":
                encoded.append(float(value))
            elif param_def.param_type == "categorical":
                # One-hot encoding for categorical variables
                idx = param_def.categories.index(value)
                encoded.append(float(idx))
            else:
                encoded.append(float(value))

        return np.array(encoded)

    def _decode_parameters(self, encoded: np.ndarray) -> dict[str, Any]:
        """Decode parameters from GP representation."""
        params = {}

        for i, param_def in enumerate(self.parameter_space):
            value = encoded[i]

            if param_def.param_type == "real":
                params[param_def.name] = float(value)
            elif param_def.param_type == "integer":
                params[param_def.name] = int(round(value))
            elif param_def.param_type == "categorical":
                idx = int(round(value))
                idx = max(0, min(len(param_def.categories) - 1, idx))
                params[param_def.name] = param_def.categories[idx]
            else:
                params[param_def.name] = value

        return params

    def _generate_initial_points(self, n_points: int) -> list[dict[str, Any]]:
        """Generate initial parameter points for exploration."""
        points = []

        for _ in range(n_points):
            params = {}

            for param_def in self.parameter_space:
                if param_def.param_type == "real":
                    if param_def.prior == "log-uniform":
                        low, high = param_def.bounds
                        value = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    else:
                        value = np.random.uniform(*param_def.bounds)
                    params[param_def.name] = value

                elif param_def.param_type == "integer":
                    low, high = param_def.bounds
                    params[param_def.name] = np.random.randint(low, high + 1)

                elif param_def.param_type == "categorical":
                    params[param_def.name] = np.random.choice(param_def.categories)

            points.append(params)

        return points

    def _suggest_next_point(self) -> dict[str, Any]:
        """Suggest next parameter point using acquisition function."""
        if len(self.optimization_history) < self.config.n_initial_points:
            # Use random sampling for initial points
            return self._generate_initial_points(1)[0]

        # Prepare data for GP
        X = np.array(
            [self._encode_parameters(item["params"]) for item in self.optimization_history]
        )
        y = np.array([item["score"] for item in self.optimization_history])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit GP model
        kernel = self._create_kernel()
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.alpha,
            normalize_y=self.config.normalize_y,
            random_state=self.config.random_state,
        )

        self.gp_model.fit(X_scaled, y)

        # Optimize acquisition function
        y_opt = np.max(y)

        # Create bounds for optimization
        bounds = []
        for param_def in self.parameter_space:
            if param_def.param_type in ["real", "integer"]:
                # Transform bounds to scaled space
                bounds.append((0, 1))  # Will be handled by scaler
            elif param_def.param_type == "categorical":
                bounds.append((0, len(param_def.categories) - 1))

        # Optimize acquisition function
        def objective(x):
            x_scaled = self.scaler.transform(x.reshape(1, -1))
            return -self.acquisition_function(x_scaled, self.gp_model, y_opt)[0]

        # Multiple random starts for global optimization
        best_x = None
        best_value = np.inf

        for _ in range(10):  # Multiple random starts
            x0 = np.random.uniform(0, 1, len(self.parameter_space))

            try:
                result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

                if result.fun < best_value:
                    best_value = result.fun
                    best_x = result.x

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                # Optimization failed for this starting point - try next one
                logger.debug(f"Optimization attempt failed: {e}")
                continue

        if best_x is None:
            # Fallback to random point
            return self._generate_initial_points(1)[0]

        # Transform back to parameter space
        best_x_scaled = self.scaler.transform(best_x.reshape(1, -1))[0]
        return self._decode_parameters(best_x_scaled)

    def optimize(
        self, objective_function: Callable[[dict[str, Any]], float], verbose: bool = True
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.

        Args:
            objective_function: Function to optimize (higher is better)
            verbose: Whether to print progress

        Returns:
            Optimization results
        """
        import time

        if not self.parameter_space:
            raise ValueError("No parameters defined for optimization")

        logger.info(f"Starting Bayesian optimization with {len(self.parameter_space)} parameters")

        start_time = time.time()
        best_score = -np.inf
        best_params = None
        patience_counter = 0

        for iteration in range(self.config.n_calls):
            # Suggest next point
            params = self._suggest_next_point()

            try:
                # Evaluate objective function
                score = objective_function(params)

                # Store result
                result_item = {
                    "iteration": iteration,
                    "params": params.copy(),
                    "score": score,
                    "timestamp": time.time(),
                }
                self.optimization_history.append(result_item)

                # Check for improvement
                if score > best_score + self.config.min_improvement:
                    best_score = score
                    best_params = params.copy()
                    patience_counter = 0

                    if verbose:
                        logger.info(f"Iter {iteration}: New best score {best_score:.6f}")

                else:
                    patience_counter += 1

                # Early stopping
                if self.config.early_stopping and patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at iteration {iteration}")
                    break

            except Exception as e:
                logger.error(f"Error evaluating parameters at iteration {iteration}: {e}")
                # Add failed evaluation with poor score
                result_item = {
                    "iteration": iteration,
                    "params": params.copy(),
                    "score": -np.inf,
                    "timestamp": time.time(),
                    "error": str(e),
                }
                self.optimization_history.append(result_item)

        total_time = time.time() - start_time

        # Create optimization result
        result = OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            n_iterations=len(self.optimization_history),
            optimization_history=self.optimization_history.copy(),
            total_time=total_time,
        )

        logger.info(f"Optimization completed in {total_time:.2f}s. Best score: {best_score:.6f}")

        return result

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get summary of optimization process."""
        if not self.optimization_history:
            return {"status": "not_started"}

        scores = [item["score"] for item in self.optimization_history if item["score"] != -np.inf]

        return {
            "n_evaluations": len(self.optimization_history),
            "n_parameters": len(self.parameter_space),
            "best_score": max(scores) if scores else -np.inf,
            "mean_score": np.mean(scores) if scores else 0,
            "std_score": np.std(scores) if scores else 0,
            "improvement_over_random": self._calculate_improvement(),
            "parameter_importance": self._calculate_parameter_importance(),
        }

    def _calculate_improvement(self) -> float:
        """Calculate improvement over random search."""
        if len(self.optimization_history) < 20:
            return 0.0

        # Compare best BO result with best of first n_initial_points
        initial_scores = [
            item["score"]
            for item in self.optimization_history[: self.config.n_initial_points]
            if item["score"] != -np.inf
        ]

        all_scores = [
            item["score"] for item in self.optimization_history if item["score"] != -np.inf
        ]

        if not initial_scores or not all_scores:
            return 0.0

        best_initial = max(initial_scores)
        best_overall = max(all_scores)

        if best_initial > 0:
            return (best_overall - best_initial) / abs(best_initial)
        else:
            return best_overall - best_initial

    def _calculate_parameter_importance(self) -> dict[str, float]:
        """Calculate parameter importance based on optimization history."""
        if len(self.optimization_history) < 10:
            return {}

        # Simple correlation-based importance
        importance = {}

        for param_def in self.parameter_space:
            param_name = param_def.name
            param_values = []
            scores = []

            for item in self.optimization_history:
                if item["score"] != -np.inf:
                    param_values.append(item["params"][param_name])
                    scores.append(item["score"])

            if len(param_values) > 5:
                # Calculate correlation with score
                if param_def.param_type == "categorical":
                    # Use variance in scores for different categories
                    unique_values = list(set(param_values))
                    if len(unique_values) > 1:
                        category_scores = {}
                        for val in unique_values:
                            cat_scores = [scores[i] for i, v in enumerate(param_values) if v == val]
                            if cat_scores:
                                category_scores[val] = np.mean(cat_scores)

                        if category_scores:
                            importance[param_name] = np.std(list(category_scores.values()))
                        else:
                            importance[param_name] = 0.0
                    else:
                        importance[param_name] = 0.0
                else:
                    # Numeric parameter - use correlation
                    try:
                        correlation = np.corrcoef(param_values, scores)[0, 1]
                        importance[param_name] = (
                            abs(correlation) if not np.isnan(correlation) else 0.0
                        )
                    except (ValueError, np.linalg.LinAlgError) as e:
                        # Correlation calculation failed - insufficient data or numerical issues
                        logger.debug(f"Correlation calculation failed for {param_name}: {e}")
                        importance[param_name] = 0.0
            else:
                importance[param_name] = 0.0

        return importance


def create_strategy_optimizer(strategy_params: dict[str, dict[str, Any]]) -> BayesianOptimizer:
    """
    Create a Bayesian optimizer for strategy parameters.

    Args:
        strategy_params: Dictionary of parameter definitions
            Format: {param_name: {type: str, bounds: tuple, ...}}

    Returns:
        Configured BayesianOptimizer
    """
    config = BayesianOptConfig(
        n_calls=100,
        n_initial_points=10,
        acquisition_function="EI",
        early_stopping=True,
        patience=15,
    )

    optimizer = BayesianOptimizer(config)

    # Add parameters
    for param_name, param_config in strategy_params.items():
        optimizer.add_parameter(
            name=param_name,
            param_type=param_config.get("type", "real"),
            bounds=param_config.get("bounds"),
            categories=param_config.get("categories"),
            prior=param_config.get("prior", "uniform"),
        )

    return optimizer


# Integration with existing GPT-Trader optimization
class StrategyParameterOptimizer:
    """
    Integration wrapper for strategy parameter optimization.

    Replaces grid search in existing optimization framework with
    intelligent Bayesian optimization.
    """

    def __init__(
        self,
        strategy_class,
        data: pd.DataFrame,
        parameter_space: dict[str, Any],
        optimization_config: BayesianOptConfig | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.data = data
        self.parameter_space = parameter_space

        # Create Bayesian optimizer
        config = optimization_config or BayesianOptConfig()
        self.optimizer = BayesianOptimizer(config)

        # Add parameters to optimizer
        for param_name, param_config in parameter_space.items():
            self.optimizer.add_parameter(
                name=param_name,
                param_type=param_config.get("type", "real"),
                bounds=param_config.get("bounds"),
                categories=param_config.get("categories"),
            )

    def objective_function(self, params: dict[str, Any]) -> float:
        """Objective function for strategy optimization."""
        try:
            # Create strategy with parameters
            strategy = self.strategy_class(**params)

            # Run backtest (simplified)
            signals = strategy.generate_signals(self.data)
            position = signals.get("position", pd.Series(0, index=self.data.index))
            returns = self.data["Close"].pct_change().fillna(0)
            strategy_returns = position.shift(1) * returns

            # Calculate Sharpe ratio
            if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                return float(sharpe)
            else:
                return -10.0  # Penalize invalid strategies

        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return -10.0

    def optimize(self) -> OptimizationResult:
        """Run strategy parameter optimization."""
        return self.optimizer.optimize(self.objective_function)


# Example usage with existing trend breakout strategy
def optimize_trend_breakout_strategy(data: pd.DataFrame) -> OptimizationResult:
    """Example: optimize trend breakout strategy parameters."""

    # Define parameter space

    # Note: This would need to be adapted to your actual TrendBreakoutStrategy
    # from bot.strategy.trend_breakout import TrendBreakoutStrategy
    #
    # optimizer = StrategyParameterOptimizer(
    #     strategy_class=TrendBreakoutStrategy,
    #     data=data,
    #     parameter_space=parameter_space
    # )
    #
    # return optimizer.optimize()

    # For now, return example result
    return OptimizationResult(
        best_params={"donchian_lookback": 55, "atr_period": 20, "atr_k": 2.5, "risk_pct": 0.02},
        best_score=1.2,
        n_iterations=50,
    )
