"""
Enhanced Multi-Model Ensemble Framework for GPT-Trader Phase 1.

This module provides a sophisticated ensemble of machine learning models
that builds upon the existing Random Forest foundation to include:
- XGBoost for gradient boosting
- LightGBM for efficient boosting
- CatBoost for categorical handling
- Support Vector Machines for non-linear patterns
- Gaussian Process for uncertainty quantification

Integrates seamlessly with existing regime detection and strategy selection systems.
"""

from __future__ import annotations

import logging

# import pickle  # Replaced with joblib for security
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Optional advanced boosting libraries
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Performance metrics for individual models."""

    model_name: str
    mse: float
    mae: float
    r2: float
    prediction_time_ms: float
    training_time_ms: float
    memory_usage_mb: float
    uncertainty_score: float | None = None
    feature_importance: dict[str, float] | None = None


@dataclass
class EnsembleConfig(BaseConfig):
    """Configuration for the ensemble model framework."""

    # Model selection
    use_random_forest: bool = True
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True
    use_svm: bool = True
    use_gaussian_process: bool = True

    # Ensemble parameters
    ensemble_method: str = "weighted_average"  # weighted_average, stacking, voting
    meta_model: str = "linear"  # linear, ridge, xgboost
    cv_folds: int = 5

    # Model-specific parameters
    random_forest_params: dict[str, Any] = None
    xgboost_params: dict[str, Any] = None
    lightgbm_params: dict[str, Any] = None
    catboost_params: dict[str, Any] = None
    svm_params: dict[str, Any] = None
    gp_params: dict[str, Any] = None

    # Training parameters
    early_stopping_rounds: int = 50
    validation_split: float = 0.2
    random_state: int = 42
    n_jobs: int = -1

    # Performance parameters
    enable_uncertainty: bool = True
    enable_feature_importance: bool = True
    performance_tracking: bool = True

    def __post_init__(self):
        """Initialize default model parameters."""
        if self.random_forest_params is None:
            self.random_forest_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }

        if self.xgboost_params is None:
            self.xgboost_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }

        if self.lightgbm_params is None:
            self.lightgbm_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbosity": -1,
            }

        if self.catboost_params is None:
            self.catboost_params = {
                "iterations": 200,
                "depth": 6,
                "learning_rate": 0.1,
                "random_state": self.random_state,
                "verbose": False,
            }

        if self.svm_params is None:
            self.svm_params = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "epsilon": 0.1}

        if self.gp_params is None:
            self.gp_params = {
                "kernel": None,  # Will be set in model
                "alpha": 1e-6,
                "normalize_y": True,
                "random_state": self.random_state,
            }


class BaseMLModel(ABC):
    """Base class for individual ML models in the ensemble."""

    def __init__(self, config: dict[str, Any], name: str) -> None:
        self.config = config
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: list[str] | None = None

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        predictions = self.predict(X)
        # Default: no uncertainty for most models
        uncertainty = np.zeros_like(predictions)
        return predictions, uncertainty

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance if available."""
        return None


class RandomForestModel(BaseMLModel):
    """Random Forest implementation for the ensemble."""

    def _create_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(**self.config)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model = self._create_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Use tree variance for uncertainty estimation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])

        # Mean and standard deviation across trees
        predictions = np.mean(tree_predictions, axis=0)
        uncertainty = np.std(tree_predictions, axis=0)

        return predictions, uncertainty

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_, strict=False))


class XGBoostModel(BaseMLModel):
    """XGBoost implementation for the ensemble."""

    def _create_model(self) -> Any:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not available")
        return xgb.XGBRegressor(**self.config)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not available")

        self.model = self._create_model()
        X_scaled = self.scaler.fit_transform(X)

        # Use validation set for early stopping
        eval_set = kwargs.get("eval_set")
        if eval_set is not None:
            X_val_scaled = self.scaler.transform(eval_set[0])
            eval_set = [(X_val_scaled, eval_set[1])]

        self.model.fit(
            X_scaled,
            y,
            eval_set=eval_set,
            early_stopping_rounds=kwargs.get("early_stopping_rounds"),
            verbose=False,
        )
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_, strict=False))


class LightGBMModel(BaseMLModel):
    """LightGBM implementation for the ensemble."""

    def _create_model(self) -> Any:
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not available")
        return lgb.LGBMRegressor(**self.config)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not available")

        self.model = self._create_model()
        X_scaled = self.scaler.fit_transform(X)

        # Use validation set for early stopping
        eval_set = kwargs.get("eval_set")
        if eval_set is not None:
            X_val_scaled = self.scaler.transform(eval_set[0])
            eval_set = [(X_val_scaled, eval_set[1])]

        self.model.fit(
            X_scaled,
            y,
            eval_set=eval_set,
            early_stopping_rounds=kwargs.get("early_stopping_rounds"),
            verbose=False,
        )
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_, strict=False))


class CatBoostModel(BaseMLModel):
    """CatBoost implementation for the ensemble."""

    def _create_model(self) -> Any:
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not available")
        return cb.CatBoostRegressor(**self.config)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not available")

        self.model = self._create_model()
        X_scaled = self.scaler.fit_transform(X)

        # Use validation set for early stopping
        eval_set = kwargs.get("eval_set")
        if eval_set is not None:
            X_val_scaled = self.scaler.transform(eval_set[0])
            eval_set = (X_val_scaled, eval_set[1])

        self.model.fit(
            X_scaled,
            y,
            eval_set=eval_set,
            early_stopping_rounds=kwargs.get("early_stopping_rounds"),
            verbose=False,
        )
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_, strict=False))


class SVMModel(BaseMLModel):
    """Support Vector Machine implementation for the ensemble."""

    def _create_model(self) -> SVR:
        return SVR(**self.config)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model = self._create_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class GaussianProcessModel(BaseMLModel):
    """Gaussian Process implementation for the ensemble with uncertainty."""

    def _create_model(self) -> GaussianProcessRegressor:
        # Create composite kernel
        kernel = (
            1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + Matern(length_scale=1.0, nu=1.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        )

        config = self.config.copy()
        config["kernel"] = kernel
        return GaussianProcessRegressor(**config)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self.model = self._create_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictions with uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        predictions, std = self.model.predict(X_scaled, return_std=True)
        return predictions, std


class EnhancedModelEnsemble:
    """
    Enhanced ensemble framework integrating multiple ML models.

    This class provides a sophisticated ensemble that combines:
    - Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost)
    - Kernel methods (SVM, Gaussian Process)
    - Uncertainty quantification
    - Dynamic model weighting
    """

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config
        self.models: dict[str, BaseMLModel] = {}
        self.model_weights: dict[str, float] = {}
        self.performance_history: list[dict[str, ModelPerformance]] = []
        self.is_fitted = False

        # Initialize models based on configuration
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize individual models based on configuration."""
        if self.config.use_random_forest:
            self.models["random_forest"] = RandomForestModel(
                self.config.random_forest_params, "Random Forest"
            )

        if self.config.use_xgboost and HAS_XGBOOST:
            self.models["xgboost"] = XGBoostModel(self.config.xgboost_params, "XGBoost")

        if self.config.use_lightgbm and HAS_LIGHTGBM:
            self.models["lightgbm"] = LightGBMModel(self.config.lightgbm_params, "LightGBM")

        if self.config.use_catboost and HAS_CATBOOST:
            self.models["catboost"] = CatBoostModel(self.config.catboost_params, "CatBoost")

        if self.config.use_svm:
            self.models["svm"] = SVMModel(self.config.svm_params, "SVM")

        if self.config.use_gaussian_process:
            self.models["gaussian_process"] = GaussianProcessModel(
                self.config.gp_params, "Gaussian Process"
            )

        # Initialize equal weights
        n_models = len(self.models)
        if n_models > 0:
            equal_weight = 1.0 / n_models
            self.model_weights = {name: equal_weight for name in self.models.keys()}

        logger.info(f"Initialized ensemble with {n_models} models: {list(self.models.keys())}")

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Fit all models in the ensemble.

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Optional feature names for importance tracking
        """
        logger.info("Training ensemble models...")

        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Validation split for early stopping
        if self.config.validation_split > 0:
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            eval_set = (X_val, y_val)
        else:
            X_train, y_train = X, y
            eval_set = None

        # Train each model
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.feature_names = feature_names

                # Prepare training arguments
                fit_kwargs = {}
                if eval_set is not None and name in ["xgboost", "lightgbm", "catboost"]:
                    fit_kwargs["eval_set"] = eval_set
                    fit_kwargs["early_stopping_rounds"] = self.config.early_stopping_rounds

                model.fit(X_train, y_train, **fit_kwargs)
                logger.info(f"Successfully trained {name}")

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                # Remove failed model
                del self.models[name]
                if name in self.model_weights:
                    del self.model_weights[name]

        # Rebalance weights after removing failed models
        if self.models:
            n_models = len(self.models)
            equal_weight = 1.0 / n_models
            self.model_weights = {name: equal_weight for name in self.models.keys()}

        self.is_fitted = True
        logger.info(f"Ensemble training completed with {len(self.models)} models")

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions using weighted average.

        Args:
            X: Feature matrix for prediction

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        weights = []

        # Get predictions from each model
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.model_weights[name])
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                continue

        if not predictions:
            raise ValueError("No models available for prediction")

        # Weighted average ensemble
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights

        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred

    def predict_with_uncertainty(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.

        Args:
            X: Feature matrix for prediction

        Returns:
            Tuple of (predictions, uncertainty_estimates)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        all_predictions = []
        all_uncertainties = []
        weights = []

        # Get predictions and uncertainties from each model
        for name, model in self.models.items():
            try:
                pred, unc = model.predict_with_uncertainty(X)
                all_predictions.append(pred)
                all_uncertainties.append(unc)
                weights.append(self.model_weights[name])
            except Exception as e:
                logger.warning(f"Prediction with uncertainty failed for {name}: {e}")
                continue

        if not all_predictions:
            raise ValueError("No models available for prediction")

        # Weighted ensemble
        predictions = np.array(all_predictions)
        uncertainties = np.array(all_uncertainties)
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Ensemble prediction
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        # Ensemble uncertainty (combination of model uncertainty and prediction variance)
        model_variance = np.average(uncertainties**2, axis=0, weights=weights)
        prediction_variance = np.average(
            (predictions - ensemble_pred) ** 2, axis=0, weights=weights
        )
        ensemble_uncertainty = np.sqrt(model_variance + prediction_variance)

        return ensemble_pred, ensemble_uncertainty

    def evaluate_models(
        self, X_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray
    ) -> dict[str, ModelPerformance]:
        """
        Evaluate individual model performance.

        Args:
            X_test: Test feature matrix
            y_test: Test target values

        Returns:
            Dictionary of model performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before evaluation")

        # Convert to numpy arrays if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        performance = {}

        for name, model in self.models.items():
            try:
                import time

                # Time prediction
                start_time = time.time()
                predictions = model.predict(X_test)
                prediction_time = (time.time() - start_time) * 1000  # ms

                # Calculate metrics
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                # Get feature importance
                feature_importance = model.get_feature_importance()

                # Get uncertainty if available
                uncertainty_score = None
                if hasattr(model, "predict_with_uncertainty"):
                    try:
                        _, uncertainties = model.predict_with_uncertainty(X_test)
                        uncertainty_score = float(np.mean(uncertainties))
                    except (AttributeError, ValueError, TypeError):
                        pass

                performance[name] = ModelPerformance(
                    model_name=name,
                    mse=float(mse),
                    mae=float(mae),
                    r2=float(r2),
                    prediction_time_ms=float(prediction_time),
                    training_time_ms=0.0,  # Would need to track during training
                    memory_usage_mb=0.0,  # Would need memory profiling
                    uncertainty_score=uncertainty_score,
                    feature_importance=feature_importance,
                )

            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")

        return performance

    def update_weights(self, performance: dict[str, ModelPerformance], metric: str = "r2") -> None:
        """
        Update model weights based on performance.

        Args:
            performance: Performance metrics for each model
            metric: Metric to use for weighting ('r2', 'mse', 'mae')
        """
        if metric == "r2":
            # Higher is better
            scores = {name: perf.r2 for name, perf in performance.items()}
            # Convert to positive weights
            min_score = min(scores.values())
            if min_score < 0:
                scores = {name: score - min_score + 0.01 for name, score in scores.items()}
        elif metric in ["mse", "mae"]:
            # Lower is better
            scores = {name: getattr(perf, metric) for name, perf in performance.items()}
            # Convert to weights (inverse of error)
            scores = {name: 1.0 / (score + 1e-8) for name, score in scores.items()}
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Normalize weights
        total_score = sum(scores.values())
        if total_score > 0:
            self.model_weights = {name: score / total_score for name, score in scores.items()}
        else:
            # Fall back to equal weights
            n_models = len(self.models)
            self.model_weights = {name: 1.0 / n_models for name in self.models.keys()}

        logger.info(f"Updated model weights based on {metric}: {self.model_weights}")

    def get_model_summary(self) -> dict[str, Any]:
        """Get summary of ensemble configuration and status."""
        return {
            "n_models": len(self.models),
            "model_names": list(self.models.keys()),
            "model_weights": self.model_weights.copy(),
            "is_fitted": self.is_fitted,
            "config": self.config.dict(),
        }

    def save_ensemble(self, filepath: str | Path) -> None:
        """Save the entire ensemble to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        ensemble_data = {
            "config": self.config,
            "model_weights": self.model_weights,
            "models": {},
            "is_fitted": self.is_fitted,
        }

        # Save each model separately to handle different serialization needs
        for name, model in self.models.items():
            try:
                model_path = filepath.parent / f"{filepath.stem}_{name}.joblib"
                joblib.dump(model, model_path)
                ensemble_data["models"][name] = str(model_path)
            except Exception as e:
                logger.error(f"Failed to save model {name}: {e}")

        # Save ensemble metadata
        joblib.dump(ensemble_data, filepath)

        logger.info(f"Ensemble saved to {filepath}")

    def load_ensemble(self, filepath: str | Path) -> None:
        """Load ensemble from disk."""
        filepath = Path(filepath)

        # Load ensemble metadata
        ensemble_data = joblib.load(filepath)

        self.config = ensemble_data["config"]
        self.model_weights = ensemble_data["model_weights"]
        self.is_fitted = ensemble_data["is_fitted"]

        # Load individual models
        self.models = {}
        for name, model_path in ensemble_data["models"].items():
            try:
                self.models[name] = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")

        logger.info(f"Ensemble loaded from {filepath}")


def create_default_ensemble(random_state: int = 42) -> EnhancedModelEnsemble:
    """Create a default ensemble configuration for quick start."""
    config = EnsembleConfig(
        use_random_forest=True,
        use_xgboost=HAS_XGBOOST,
        use_lightgbm=HAS_LIGHTGBM,
        use_catboost=HAS_CATBOOST,
        use_svm=True,
        use_gaussian_process=True,
        random_state=random_state,
    )

    return EnhancedModelEnsemble(config)
