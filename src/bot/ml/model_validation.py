"""
Model Validation Framework
Phase 2.5 - Day 5

Implements proper cross-validation, performance tracking, and realistic evaluation.
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_validate,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for model validation"""

    # Cross-validation
    n_splits: int = 5
    test_size: int = 252  # Trading days in a year
    gap: int = 5  # Gap between train and test to avoid lookahead

    # Metrics
    primary_metric: str = "f1"  # Primary metric for optimization
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "roc_auc"]
    )

    # Walk-forward analysis
    walk_forward_window: int = 63  # Quarterly retraining
    min_train_size: int = 252  # Minimum 1 year of training data

    # Performance thresholds
    min_accuracy: float = 0.55  # Better than random
    min_sharpe: float = 0.5  # Minimum Sharpe ratio
    max_drawdown: float = 0.20  # Maximum 20% drawdown


@dataclass
class ModelPerformance:
    """Model performance metrics"""

    model_name: str
    timestamp: datetime

    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

    # Trading metrics
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None

    # Cross-validation scores
    cv_scores: dict[str, list[float]] | None = None

    # Feature importance
    feature_importance: dict[str, float] | None = None

    # Confusion matrix
    confusion_matrix: np.ndarray | None = None

    # Additional metadata
    n_samples_train: int | None = None
    n_samples_test: int | None = None
    training_time: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "cv_scores": self.cv_scores,
            "feature_importance": self.feature_importance,
            "confusion_matrix": (
                self.confusion_matrix.tolist() if self.confusion_matrix is not None else None
            ),
            "n_samples_train": self.n_samples_train,
            "n_samples_test": self.n_samples_test,
            "training_time": self.training_time,
        }


class ModelValidator:
    """
    Comprehensive model validation with proper time series cross-validation.

    Features:
    - Time series cross-validation
    - Walk-forward analysis
    - Trading performance metrics
    - Feature importance analysis
    - Model comparison
    """

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()
        self.performance_history: list[ModelPerformance] = []
        self.best_models: dict[str, Any] = {}

        logger.info(f"ModelValidator initialized with {self.config.n_splits} CV splits")

    def time_series_cross_validation(
        self, model: Any, X: pd.DataFrame, y: pd.Series, scoring: str | None = None
    ) -> dict[str, Any]:
        """
        Perform time series cross-validation.

        Args:
            model: sklearn-compatible model
            X: Feature matrix
            y: Target variable
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        # Use primary metric if not specified
        if scoring is None:
            scoring = self.config.primary_metric

        # Create time series splitter
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits, test_size=self.config.test_size, gap=self.config.gap
        )

        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=tscv, scoring=self.config.metrics, return_train_score=True, n_jobs=-1
        )

        # Calculate statistics
        results = {"metrics": {}}

        for metric in self.config.metrics:
            test_key = f"test_{metric}"
            train_key = f"train_{metric}"

            if test_key in cv_results:
                results["metrics"][metric] = {
                    "test_mean": np.mean(cv_results[test_key]),
                    "test_std": np.std(cv_results[test_key]),
                    "train_mean": np.mean(cv_results[train_key]),
                    "train_std": np.std(cv_results[train_key]),
                    "scores": cv_results[test_key].tolist(),
                }

        # Check for overfitting
        results["overfitting_score"] = self._calculate_overfitting_score(cv_results)

        return results

    def walk_forward_analysis(
        self, model_class: Any, X: pd.DataFrame, y: pd.Series, model_params: dict | None = None
    ) -> list[ModelPerformance]:
        """
        Perform walk-forward analysis.

        Args:
            model_class: Model class to instantiate
            X: Feature matrix
            y: Target variable
            model_params: Model parameters

        Returns:
            List of performance results for each period
        """
        if model_params is None:
            model_params = {}

        results = []

        # Ensure we have enough data
        if len(X) < self.config.min_train_size + self.config.walk_forward_window:
            logger.warning("Insufficient data for walk-forward analysis")
            return results

        # Walk forward through time
        for i in range(
            self.config.min_train_size,
            len(X) - self.config.walk_forward_window,
            self.config.walk_forward_window,
        ):
            # Split data
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i : i + self.config.walk_forward_window]
            y_test = y.iloc[i : i + self.config.walk_forward_window]

            # Train model
            model = model_class(**model_params)

            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None

            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            performance = self._calculate_performance(
                y_test,
                y_pred,
                y_pred_proba,
                model_name=model_class.__name__,
                n_train=len(X_train),
                n_test=len(X_test),
                training_time=training_time,
            )

            # Get feature importance if available
            if hasattr(model, "feature_importances_"):
                performance.feature_importance = dict(
                    zip(X.columns, model.feature_importances_, strict=False)
                )

            results.append(performance)

            logger.debug(f"Walk-forward period {i}: Accuracy={performance.accuracy:.3f}")

        return results

    def validate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Unknown",
    ) -> ModelPerformance:
        """
        Validate a trained model.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_name: Model name

        Returns:
            Model performance metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None

        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate performance
        performance = self._calculate_performance(
            y_test,
            y_pred,
            y_pred_proba,
            model_name=model_name,
            n_train=len(X_train),
            n_test=len(X_test),
        )

        # Cross-validation on training data
        cv_results = self.time_series_cross_validation(model, X_train, y_train)
        performance.cv_scores = cv_results["metrics"]

        # Feature importance
        if hasattr(model, "feature_importances_"):
            performance.feature_importance = dict(
                zip(X_train.columns, model.feature_importances_, strict=False)
            )

        # Store performance
        self.performance_history.append(performance)

        # Check if best model
        if self._is_best_model(performance, model_name):
            self.best_models[model_name] = {
                "model": model,
                "performance": performance,
                "timestamp": datetime.now(),
            }

        return performance

    def compare_models(
        self, models: dict[str, Any], X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models: Dictionary of model_name: model_instance
            X: Feature matrix
            y: Target variable
            test_size: Test set size

        Returns:
            Comparison DataFrame
        """
        # Split data
        split_index = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        results = []

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")

            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # Validate
            performance = self.validate_model(
                model, X_train, y_train, X_test, y_test, model_name=model_name
            )
            performance.training_time = training_time

            # Add to results
            results.append(
                {
                    "Model": model_name,
                    "Accuracy": performance.accuracy,
                    "Precision": performance.precision,
                    "Recall": performance.recall,
                    "F1": performance.f1_score,
                    "ROC-AUC": performance.roc_auc,
                    "Training Time": training_time,
                }
            )

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values("F1", ascending=False)

        return comparison_df

    def calculate_trading_metrics(
        self, returns: pd.Series, predictions: pd.Series
    ) -> dict[str, float]:
        """
        Calculate trading-specific metrics.

        Args:
            returns: Actual returns
            predictions: Predicted signals (1 for long, 0 for no position)

        Returns:
            Trading metrics
        """
        # Calculate strategy returns
        strategy_returns = returns * predictions

        # Remove zeros for calculations
        active_returns = strategy_returns[strategy_returns != 0]

        if len(active_returns) == 0:
            return {"sharpe_ratio": 0, "max_drawdown": 0, "win_rate": 0, "profit_factor": 0}

        # Sharpe ratio (annualized)
        sharpe_ratio = (
            np.sqrt(252) * active_returns.mean() / active_returns.std()
            if active_returns.std() > 0
            else 0
        )

        # Maximum drawdown
        cumulative_returns = (1 + active_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Win rate
        win_rate = (active_returns > 0).sum() / len(active_returns)

        # Profit factor
        gains = active_returns[active_returns > 0].sum()
        losses = abs(active_returns[active_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf

        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    def _calculate_performance(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray | None,
        model_name: str,
        n_train: int,
        n_test: int,
        training_time: float | None = None,
    ) -> ModelPerformance:
        """Calculate performance metrics"""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # ROC-AUC if probabilities available
        roc_auc = 0
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
            except:
                roc_auc = 0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        return ModelPerformance(
            model_name=model_name,
            timestamp=datetime.now(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            n_samples_train=n_train,
            n_samples_test=n_test,
            training_time=training_time,
        )

    def _calculate_overfitting_score(self, cv_results: dict) -> float:
        """Calculate overfitting score (0 = no overfit, 1 = severe overfit)"""
        scores = []

        for metric in self.config.metrics:
            test_key = f"test_{metric}"
            train_key = f"train_{metric}"

            if test_key in cv_results and train_key in cv_results:
                test_mean = np.mean(cv_results[test_key])
                train_mean = np.mean(cv_results[train_key])

                # Calculate relative difference
                if train_mean > 0:
                    overfit = (train_mean - test_mean) / train_mean
                    scores.append(max(0, min(1, overfit)))  # Clip to [0, 1]

        return np.mean(scores) if scores else 0

    def _is_best_model(self, performance: ModelPerformance, model_name: str) -> bool:
        """Check if this is the best model so far"""
        if model_name not in self.best_models:
            return True

        current_best = self.best_models[model_name]["performance"]

        # Compare primary metric
        if self.config.primary_metric == "accuracy":
            return performance.accuracy > current_best.accuracy
        elif self.config.primary_metric == "f1":
            return performance.f1_score > current_best.f1_score
        elif self.config.primary_metric == "roc_auc":
            return performance.roc_auc > current_best.roc_auc
        else:
            return performance.accuracy > current_best.accuracy

    def save_performance_report(self, filepath: str):
        """Save performance report to file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_splits": self.config.n_splits,
                "test_size": self.config.test_size,
                "primary_metric": self.config.primary_metric,
            },
            "performance_history": [p.to_dict() for p in self.performance_history],
            "best_models": {
                name: {
                    "performance": data["performance"].to_dict(),
                    "timestamp": data["timestamp"].isoformat(),
                }
                for name, data in self.best_models.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report saved to {filepath}")

    def load_performance_report(self, filepath: str):
        """Load performance report from file"""
        with open(filepath) as f:
            report = json.load(f)

        # Reconstruct performance history
        self.performance_history = []
        for p_dict in report.get("performance_history", []):
            # Create ModelPerformance instance
            # Note: This is simplified, you might need more complex reconstruction
            pass

        logger.info(f"Performance report loaded from {filepath}")

        return report


def create_validator(config: ValidationConfig | None = None) -> ModelValidator:
    """Create model validator instance"""
    return ModelValidator(config)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # Generate sample data
    X, y = make_classification(n_samples=2000, n_features=50, n_informative=30, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(50)])
    y = pd.Series(y)

    # Add time index
    X.index = pd.date_range("2020-01-01", periods=len(X), freq="D")
    y.index = X.index

    # Create validator
    validator = create_validator()

    # Define models to compare
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    }

    # Compare models
    comparison = validator.compare_models(models, X, y)
    print("Model Comparison:")
    print(comparison)

    # Save report
    validator.save_performance_report("model_performance_report.json")

    print("\nBest models:")
    for name, data in validator.best_models.items():
        perf = data["performance"]
        print(f"{name}: Accuracy={perf.accuracy:.3f}, F1={perf.f1_score:.3f}")
