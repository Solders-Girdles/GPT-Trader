"""
Model Performance Comparison Framework for GPT-Trader Phase 1.

This module provides comprehensive model comparison and evaluation:
- Cross-validation with time series splits
- Statistical significance testing
- Performance attribution analysis
- Model benchmarking and ranking
- Uncertainty quantification
- Risk-adjusted performance metrics

Integrates with ensemble models and provides systematic model evaluation.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Optional statistical testing
try:
    from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon

    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False
    warnings.warn("Advanced statistical tests not available")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Plotting libraries not available")

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""

    model_name: str

    # Regression metrics
    mse: float
    mae: float
    rmse: float
    r2: float
    explained_variance: float
    max_error: float

    # Financial metrics
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None

    # Statistical metrics
    mean_residual: float | None = None
    residual_std: float | None = None
    jarque_bera_pvalue: float | None = None
    ljung_box_pvalue: float | None = None

    # Cross-validation metrics
    cv_score_mean: float | None = None
    cv_score_std: float | None = None
    cv_scores: list[float] | None = None

    # Computational metrics
    training_time: float | None = None
    prediction_time: float | None = None
    memory_usage: float | None = None

    # Uncertainty metrics
    prediction_intervals: dict[str, tuple[float, float]] | None = None
    uncertainty_coverage: float | None = None


@dataclass
class ComparisonResult:
    """Results of model comparison analysis."""

    model_metrics: dict[str, ModelMetrics]
    ranking: list[tuple[str, float]]  # (model_name, composite_score)
    statistical_tests: dict[str, dict[str, Any]]
    best_model: str
    performance_summary: dict[str, Any]
    significance_matrix: pd.DataFrame | None = None


@dataclass
class ModelComparisonConfig(BaseConfig):
    """Configuration for model comparison framework."""

    # Cross-validation parameters
    cv_folds: int = 5
    time_series_cv: bool = True
    test_size: float = 0.2

    # Metrics to compute
    compute_financial_metrics: bool = True
    compute_statistical_tests: bool = True
    compute_uncertainty_metrics: bool = True
    compute_residual_analysis: bool = True

    # Statistical testing parameters
    significance_level: float = 0.05
    correction_method: str = "bonferroni"  # bonferroni, holm, fdr_bh

    # Ranking parameters
    ranking_weights: dict[str, float] = field(
        default_factory=lambda: {
            "r2": 0.3,
            "sharpe_ratio": 0.3,
            "max_drawdown": -0.2,  # Negative weight (lower is better)
            "cv_score_mean": 0.2,
        }
    )

    # Performance attribution
    attribution_analysis: bool = True
    benchmark_model: str | None = None

    # Computational analysis
    track_computational_metrics: bool = True

    # Visualization
    create_plots: bool = True
    plot_residuals: bool = True
    plot_performance_comparison: bool = True


class FinancialMetricsCalculator:
    """Calculator for financial performance metrics."""

    @staticmethod
    def calculate_financial_metrics(
        predictions: np.ndarray, actual: np.ndarray, returns_data: pd.Series | None = None
    ) -> dict[str, float]:
        """Calculate financial performance metrics."""
        metrics = {}

        if returns_data is not None:
            # Use actual returns for financial metrics
            strategy_returns = returns_data
        else:
            # Create synthetic returns from predictions vs actual
            price_changes = predictions - actual
            strategy_returns = pd.Series(price_changes)

        if len(strategy_returns) == 0 or strategy_returns.std() == 0:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        try:
            # Sharpe ratio
            metrics["sharpe_ratio"] = (
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            )

            # Sortino ratio (downside deviation)
            downside_returns = strategy_returns[strategy_returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                metrics["sortino_ratio"] = strategy_returns.mean() / downside_std * np.sqrt(252)
            else:
                metrics["sortino_ratio"] = metrics["sharpe_ratio"]

            # Maximum drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            metrics["max_drawdown"] = abs(drawdown.min())

            # Calmar ratio
            if metrics["max_drawdown"] > 0:
                annualized_return = (
                    cumulative_returns.iloc[-1] ** (252 / len(strategy_returns))
                ) - 1
                metrics["calmar_ratio"] = annualized_return / metrics["max_drawdown"]
            else:
                metrics["calmar_ratio"] = np.inf

            # Win rate
            winning_trades = (strategy_returns > 0).sum()
            total_trades = len(strategy_returns)
            metrics["win_rate"] = winning_trades / total_trades

            # Profit factor
            gross_profit = strategy_returns[strategy_returns > 0].sum()
            gross_loss = abs(strategy_returns[strategy_returns < 0].sum())

            if gross_loss > 0:
                metrics["profit_factor"] = gross_profit / gross_loss
            else:
                metrics["profit_factor"] = np.inf

        except Exception as e:
            logger.warning(f"Error calculating financial metrics: {e}")
            metrics = {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        return metrics


class StatisticalTestsCalculator:
    """Calculator for statistical significance tests."""

    @staticmethod
    def calculate_residual_tests(residuals: np.ndarray) -> dict[str, float]:
        """Calculate statistical tests on model residuals."""
        tests = {}

        if len(residuals) < 10:
            return tests

        try:
            # Jarque-Bera test for normality
            if HAS_SCIPY_STATS:
                jb_stat, jb_pvalue = stats.jarque_bera(residuals)
                tests["jarque_bera_statistic"] = float(jb_stat)
                tests["jarque_bera_pvalue"] = float(jb_pvalue)

                # Ljung-Box test for autocorrelation (simplified)
                from statsmodels.stats.diagnostic import acorr_ljungbox

                lb_result = acorr_ljungbox(
                    residuals, lags=min(10, len(residuals) // 4), return_df=True
                )
                tests["ljung_box_pvalue"] = float(lb_result["lb_pvalue"].iloc[0])

        except Exception as e:
            logger.warning(f"Error in residual tests: {e}")

        return tests

    @staticmethod
    def compare_models_pairwise(
        model_scores: dict[str, list[float]], significance_level: float = 0.05
    ) -> pd.DataFrame:
        """Compare models pairwise using statistical tests."""
        if not HAS_SCIPY_STATS:
            logger.warning("Statistical tests not available")
            return pd.DataFrame()

        model_names = list(model_scores.keys())
        n_models = len(model_names)

        # Initialize significance matrix
        significance_matrix = pd.DataFrame(
            np.ones((n_models, n_models)), index=model_names, columns=model_names
        )

        # Perform pairwise tests
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    try:
                        scores1 = model_scores[model1]
                        scores2 = model_scores[model2]

                        # Use paired t-test
                        if len(scores1) == len(scores2) and len(scores1) >= 3:
                            stat, p_value = ttest_rel(scores1, scores2)
                            significance_matrix.loc[model1, model2] = p_value
                        else:
                            # Use Wilcoxon signed-rank test as alternative
                            stat, p_value = wilcoxon(scores1, scores2, alternative="two-sided")
                            significance_matrix.loc[model1, model2] = p_value

                    except Exception as e:
                        logger.warning(f"Error comparing {model1} vs {model2}: {e}")
                        significance_matrix.loc[model1, model2] = 1.0

        return significance_matrix


class ModelComparison:
    """
    Comprehensive framework for comparing machine learning models.

    Provides statistical evaluation, financial metrics, and ranking
    of multiple models for trading strategy applications.
    """

    def __init__(self, config: ModelComparisonConfig) -> None:
        self.config = config
        self.models: dict[str, Any] = {}
        self.results: ComparisonResult | None = None

    def add_model(self, name: str, model: Any, description: str = "") -> None:
        """Add a model to the comparison."""
        self.models[name] = {"model": model, "description": description, "fitted": False}

    def compare_models(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        returns_data: pd.Series | None = None,
    ) -> ComparisonResult:
        """
        Compare all added models on the given dataset.

        Args:
            X: Feature matrix
            y: Target values
            returns_data: Optional returns data for financial metrics

        Returns:
            Comprehensive comparison results
        """
        if not self.models:
            raise ValueError("No models added for comparison")

        logger.info(f"Comparing {len(self.models)} models...")

        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Split data for evaluation
        split_idx = int(len(X_array) * (1 - self.config.test_size))
        X_train, X_test = X_array[:split_idx], X_array[split_idx:]
        y_train, y_test = y_array[:split_idx], y_array[split_idx:]

        if returns_data is not None:
            returns_test = returns_data.iloc[split_idx:]
        else:
            returns_test = None

        # Evaluate each model
        model_metrics = {}
        cv_scores_dict = {}

        for name, model_info in self.models.items():
            logger.info(f"Evaluating model: {name}")

            try:
                metrics = self._evaluate_single_model(
                    name, model_info["model"], X_train, y_train, X_test, y_test, returns_test
                )
                model_metrics[name] = metrics

                # Store CV scores for statistical tests
                if metrics.cv_scores is not None:
                    cv_scores_dict[name] = metrics.cv_scores

            except Exception as e:
                logger.error(f"Error evaluating model {name}: {e}")
                continue

        if not model_metrics:
            raise ValueError("No models could be evaluated")

        # Perform statistical comparisons
        statistical_tests = {}
        significance_matrix = None

        if self.config.compute_statistical_tests and len(cv_scores_dict) > 1:
            significance_matrix = StatisticalTestsCalculator.compare_models_pairwise(
                cv_scores_dict, self.config.significance_level
            )
            statistical_tests["pairwise_significance"] = significance_matrix

        # Rank models
        ranking = self._rank_models(model_metrics)

        # Create performance summary
        performance_summary = self._create_performance_summary(model_metrics, ranking)

        # Store results
        self.results = ComparisonResult(
            model_metrics=model_metrics,
            ranking=ranking,
            statistical_tests=statistical_tests,
            best_model=ranking[0][0] if ranking else "",
            performance_summary=performance_summary,
            significance_matrix=significance_matrix,
        )

        logger.info("Model comparison completed")
        return self.results

    def _evaluate_single_model(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        returns_test: pd.Series | None = None,
    ) -> ModelMetrics:
        """Evaluate a single model comprehensively."""
        import time

        # Fit model and measure training time
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Error fitting model {name}: {e}")
            raise

        # Make predictions and measure prediction time
        start_time = time.time()
        predictions = model.predict(X_test)
        prediction_time = time.time() - start_time

        # Calculate basic regression metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        explained_var = explained_variance_score(y_test, predictions)
        max_err = max_error(y_test, predictions)

        # Calculate residuals
        residuals = y_test - predictions
        mean_residual = np.mean(residuals)
        residual_std = np.std(residuals)

        # Cross-validation scores
        cv_scores = None
        cv_score_mean = None
        cv_score_std = None

        try:
            if self.config.time_series_cv:
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            else:
                cv = self.config.cv_folds

            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
            cv_score_mean = np.mean(cv_scores)
            cv_score_std = np.std(cv_scores)

        except Exception as e:
            logger.warning(f"Cross-validation failed for {name}: {e}")

        # Financial metrics
        financial_metrics = {}
        if self.config.compute_financial_metrics:
            financial_metrics = FinancialMetricsCalculator.calculate_financial_metrics(
                predictions, y_test, returns_test
            )

        # Statistical tests on residuals
        residual_tests = {}
        if self.config.compute_residual_analysis:
            residual_tests = StatisticalTestsCalculator.calculate_residual_tests(residuals)

        # Create metrics object
        metrics = ModelMetrics(
            model_name=name,
            mse=float(mse),
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
            explained_variance=float(explained_var),
            max_error=float(max_err),
            mean_residual=float(mean_residual),
            residual_std=float(residual_std),
            training_time=float(training_time),
            prediction_time=float(prediction_time),
            cv_score_mean=cv_score_mean,
            cv_score_std=cv_score_std,
            cv_scores=cv_scores.tolist() if cv_scores is not None else None,
            **financial_metrics,
            **residual_tests,
        )

        return metrics

    def _rank_models(self, model_metrics: dict[str, ModelMetrics]) -> list[tuple[str, float]]:
        """Rank models based on composite scoring."""
        if not model_metrics:
            return []

        model_scores = {}

        for name, metrics in model_metrics.items():
            composite_score = 0.0
            total_weight = 0.0

            # Calculate weighted composite score
            for metric_name, weight in self.config.ranking_weights.items():
                value = getattr(metrics, metric_name, None)

                if value is not None and not (np.isnan(value) or np.isinf(value)):
                    # Normalize score (handle negative weights for metrics where lower is better)
                    if weight < 0:  # Lower is better (e.g., max_drawdown)
                        normalized_score = 1.0 / (1.0 + abs(value))
                        composite_score += abs(weight) * normalized_score
                    else:  # Higher is better
                        normalized_score = max(0, value)
                        composite_score += weight * normalized_score

                    total_weight += abs(weight)

            # Normalize by total weight
            if total_weight > 0:
                model_scores[name] = composite_score / total_weight
            else:
                model_scores[name] = 0.0

        # Sort by composite score (descending)
        ranking = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        return ranking

    def _create_performance_summary(
        self, model_metrics: dict[str, ModelMetrics], ranking: list[tuple[str, float]]
    ) -> dict[str, Any]:
        """Create performance summary statistics."""
        if not model_metrics:
            return {}

        # Collect all metrics for summary statistics
        r2_scores = [m.r2 for m in model_metrics.values()]
        mse_scores = [m.mse for m in model_metrics.values()]
        sharpe_ratios = [
            m.sharpe_ratio for m in model_metrics.values() if m.sharpe_ratio is not None
        ]

        summary = {
            "n_models": len(model_metrics),
            "best_model": ranking[0][0] if ranking else None,
            "worst_model": ranking[-1][0] if ranking else None,
            "r2_statistics": {
                "mean": float(np.mean(r2_scores)),
                "std": float(np.std(r2_scores)),
                "min": float(np.min(r2_scores)),
                "max": float(np.max(r2_scores)),
            },
            "mse_statistics": {
                "mean": float(np.mean(mse_scores)),
                "std": float(np.std(mse_scores)),
                "min": float(np.min(mse_scores)),
                "max": float(np.max(mse_scores)),
            },
        }

        if sharpe_ratios:
            summary["sharpe_statistics"] = {
                "mean": float(np.mean(sharpe_ratios)),
                "std": float(np.std(sharpe_ratios)),
                "min": float(np.min(sharpe_ratios)),
                "max": float(np.max(sharpe_ratios)),
            }

        # Performance spread
        if len(ranking) > 1:
            best_score = ranking[0][1]
            worst_score = ranking[-1][1]
            summary["performance_spread"] = best_score - worst_score

        return summary

    def generate_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if self.results is None:
            return "No comparison results available. Run compare_models() first."

        report = []
        report.append("=" * 60)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary
        summary = self.results.performance_summary
        report.append(f"Models Compared: {summary.get('n_models', 0)}")
        report.append(f"Best Model: {summary.get('best_model', 'N/A')}")
        report.append("")

        # Rankings
        report.append("MODEL RANKINGS:")
        report.append("-" * 40)
        for i, (name, score) in enumerate(self.results.ranking, 1):
            report.append(f"{i:2d}. {name:<20} Score: {score:.4f}")
        report.append("")

        # Detailed metrics for top 3 models
        report.append("TOP MODELS DETAILED METRICS:")
        report.append("-" * 40)

        top_models = self.results.ranking[:3]
        for name, _ in top_models:
            metrics = self.results.model_metrics[name]
            report.append(f"\n{name}:")
            report.append(f"  R²: {metrics.r2:.4f}")
            report.append(f"  RMSE: {metrics.rmse:.4f}")
            report.append(f"  MAE: {metrics.mae:.4f}")

            if metrics.sharpe_ratio is not None:
                report.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.4f}")

            if metrics.max_drawdown is not None:
                report.append(f"  Max Drawdown: {metrics.max_drawdown:.4f}")

            if metrics.cv_score_mean is not None:
                report.append(
                    f"  CV Score: {metrics.cv_score_mean:.4f} ± {metrics.cv_score_std:.4f}"
                )

        # Statistical significance
        if self.results.significance_matrix is not None:
            report.append("\nSTATISTICAL SIGNIFICANCE MATRIX (p-values):")
            report.append("-" * 40)
            report.append(str(self.results.significance_matrix.round(4)))

        return "\n".join(report)

    def get_best_model(self) -> Any | None:
        """Get the best performing model."""
        if self.results is None or not self.results.ranking:
            return None

        best_model_name = self.results.ranking[0][0]
        return self.models[best_model_name]["model"]

    def export_results(self, filepath: str) -> None:
        """Export comparison results to file."""
        if self.results is None:
            raise ValueError("No results to export")

        import json

        # Convert results to serializable format
        export_data = {
            "model_metrics": {
                name: {
                    attr: getattr(metrics, attr)
                    for attr in dir(metrics)
                    if not attr.startswith("_") and not callable(getattr(metrics, attr))
                }
                for name, metrics in self.results.model_metrics.items()
            },
            "ranking": self.results.ranking,
            "performance_summary": self.results.performance_summary,
            "statistical_tests": self.results.statistical_tests,
        }

        # Handle significance matrix
        if self.results.significance_matrix is not None:
            export_data["significance_matrix"] = self.results.significance_matrix.to_dict()

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Results exported to {filepath}")


def create_default_comparison_framework() -> ModelComparison:
    """Create a default model comparison framework."""
    config = ModelComparisonConfig(
        cv_folds=5,
        time_series_cv=True,
        compute_financial_metrics=True,
        compute_statistical_tests=True,
        compute_uncertainty_metrics=True,
    )

    return ModelComparison(config)


# Integration example with existing ensemble
def compare_ensemble_models(ensemble_framework, X: pd.DataFrame, y: pd.Series) -> ComparisonResult:
    """Compare individual models from ensemble framework."""

    comparison = create_default_comparison_framework()

    # Add individual models from ensemble
    for name, model in ensemble_framework.models.items():
        comparison.add_model(name, model.model, f"Individual {name} model")

    # Add the ensemble itself
    comparison.add_model("ensemble", ensemble_framework, "Full ensemble model")

    # Run comparison
    results = comparison.compare_models(X, y)

    return results
