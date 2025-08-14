"""
Base classes for ML components in GPT-Trader
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..core.base import BaseComponent, ComponentConfig, HealthStatus


class MLModel(BaseComponent, ABC):
    """Base class for all ML models in the system"""

    def __init__(self, config: ComponentConfig, db_manager=None):
        """Initialize ML model

        Args:
            config: Component configuration
            db_manager: Database manager for persistence
        """
        super().__init__(config)
        self.db_manager = db_manager
        self.model = None
        self.model_id = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_dir = Path("models")
        self.model_path = self.model_dir / f"{self.model_id}.joblib"
        self.metrics = {}
        self.feature_importance = {}
        self.is_trained = False

    def _initialize_component(self):
        """Initialize the ML component"""
        self.model_dir.mkdir(exist_ok=True)
        self.logger.info(f"Initialized ML model: {self.model_id}")

    def _start_component(self):
        """Start the ML component"""
        self.logger.info(f"Starting ML model: {self.model_id}")

    def _stop_component(self):
        """Stop the ML component"""
        self.logger.info(f"Stopping ML model: {self.model_id}")

    def _health_check(self) -> HealthStatus:
        """Check health of ML model"""
        if self.model is None:
            return HealthStatus.UNHEALTHY
        if not self.is_trained:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, float]:
        """Train the model

        Args:
            X: Feature matrix
            y: Target variable (optional for unsupervised models)

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification models)

        Args:
            X: Feature matrix

        Returns:
            Probability array
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support probability predictions"
            )

    def save_model(self) -> str:
        """Save model to disk and register in database

        Returns:
            Model ID
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Ensure directory exists
        self.model_dir.mkdir(exist_ok=True)

        # Save model to disk
        joblib.dump(self.model, self.model_path)
        self.logger.info(f"Saved model to {self.model_path}")

        # Register in database if available
        if self.db_manager:
            self.db_manager.insert_record(
                "ml_models",
                {
                    "model_id": self.model_id,
                    "model_type": self.__class__.__name__,
                    "model_path": str(self.model_path),
                    "training_date": datetime.now(),
                    "performance_metrics": json.dumps(self.metrics),
                    "is_active": True,
                },
            )
            self.logger.info(f"Registered model in database: {self.model_id}")

        return self.model_id

    def load_model(self, model_id: str) -> None:
        """Load model from disk

        Args:
            model_id: ID of model to load
        """
        if self.db_manager:
            # Get model info from database
            model_info = self.db_manager.fetch_one(
                "SELECT * FROM ml_models WHERE model_id = ?", (model_id,)
            )

            if model_info:
                model_path = Path(model_info["model_path"])
                if model_path.exists():
                    self.model = joblib.load(model_path)
                    self.model_id = model_id
                    self.model_path = model_path
                    self.metrics = json.loads(model_info["performance_metrics"])
                    self.is_trained = True
                    self.logger.info(f"Loaded model from {model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            else:
                raise ValueError(f"Model not found in database: {model_id}")
        else:
            # Try direct file load
            model_path = self.model_dir / f"{model_id}.joblib"
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.model_id = model_id
                self.model_path = model_path
                self.is_trained = True
                self.logger.info(f"Loaded model from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores

        Returns:
            Dictionary of feature names to importance scores
        """
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names, self.model.feature_importances_, strict=False))
        elif hasattr(self.model, "coef_"):
            return dict(zip(self.feature_names, np.abs(self.model.coef_).flatten(), strict=False))
        else:
            return {}

    def validate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
        """Validate model performance

        Args:
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Validation metrics
        """
        predictions = self.predict(X_val)

        # Calculate basic metrics
        metrics = {}

        # For regression
        if hasattr(self.model, "predict"):
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            metrics["mse"] = mean_squared_error(y_val, predictions)
            metrics["mae"] = mean_absolute_error(y_val, predictions)
            metrics["r2"] = r2_score(y_val, predictions)

        # For classification
        if hasattr(self.model, "predict_proba"):
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            metrics["accuracy"] = accuracy_score(y_val, predictions)
            # Handle multi-class
            avg_method = "weighted" if len(np.unique(y_val)) > 2 else "binary"
            metrics["precision"] = precision_score(y_val, predictions, average=avg_method)
            metrics["recall"] = recall_score(y_val, predictions, average=avg_method)
            metrics["f1"] = f1_score(y_val, predictions, average=avg_method)

        return metrics


class FeatureEngineer(BaseComponent, ABC):
    """Base class for feature engineering components"""

    def __init__(self, config: ComponentConfig):
        """Initialize feature engineer

        Args:
            config: Component configuration
        """
        super().__init__(config)
        self.feature_cache = {}

    def _initialize_component(self):
        """Initialize the feature engineering component"""
        self.logger.info("Initialized feature engineer")

    def _start_component(self):
        """Start the feature engineering component"""
        self.logger.info("Starting feature engineer")

    def _stop_component(self):
        """Stop the feature engineering component"""
        self.logger.info("Stopping feature engineer")

    def _health_check(self) -> HealthStatus:
        """Check health of feature engineer"""
        return HealthStatus.HEALTHY

    @abstractmethod
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features from raw data

        Args:
            data: Raw input data

        Returns:
            DataFrame with engineered features
        """
        pass

    def get_feature_names(self) -> list[str]:
        """Get list of feature names

        Returns:
            List of feature names
        """
        if self.feature_cache:
            return list(next(iter(self.feature_cache.values())).columns)
        return []

    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        self.logger.info("Cleared feature cache")


class ModelRegistry:
    """Registry for managing ML models"""

    def __init__(self, db_manager):
        """Initialize model registry

        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def register_model(self, model: MLModel, metrics: dict[str, float]) -> str:
        """Register a trained model

        Args:
            model: Trained ML model
            metrics: Performance metrics

        Returns:
            Model ID
        """
        model_id = model.save_model()
        self.logger.info(f"Registered model: {model_id} with metrics: {metrics}")
        return model_id

    def get_best_model(self, model_type: str, metric: str = "accuracy") -> str:
        """Get the best performing model of a given type

        Args:
            model_type: Type of model (e.g., 'StrategySelector')
            metric: Metric to use for selection

        Returns:
            Model ID of best model
        """
        query = """
            SELECT model_id, performance_metrics
            FROM ml_models
            WHERE model_type = ? AND is_active = 1
            ORDER BY created_at DESC
        """

        models = self.db_manager.fetch_all(query, (model_type,))

        if not models:
            raise ValueError(f"No models found for type: {model_type}")

        # Find best model by metric
        best_model = None
        best_score = -float("inf")

        for model in models:
            metrics = json.loads(model["performance_metrics"])
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model["model_id"]

        if best_model is None:
            # Return most recent if metric not found
            return models[0]["model_id"]

        return best_model

    def deactivate_model(self, model_id: str):
        """Deactivate a model

        Args:
            model_id: ID of model to deactivate
        """
        self.db_manager.update_record(
            "ml_models", {"is_active": False}, "model_id = ?", (model_id,)
        )
        self.logger.info(f"Deactivated model: {model_id}")

    def get_active_models(self) -> list[dict]:
        """Get all active models

        Returns:
            List of active model records
        """
        return self.db_manager.fetch_all(
            "SELECT * FROM ml_models WHERE is_active = 1 ORDER BY created_at DESC"
        )

    def cleanup_old_models(self, keep_last_n: int = 5):
        """Clean up old model files, keeping only the last N models of each type

        Args:
            keep_last_n: Number of models to keep per type
        """
        # Get all model types
        types_query = "SELECT DISTINCT model_type FROM ml_models"
        model_types = self.db_manager.fetch_all(types_query)

        for model_type_row in model_types:
            model_type = model_type_row["model_type"]

            # Get all models of this type, ordered by date
            models = self.db_manager.fetch_all(
                """SELECT model_id, model_path
                   FROM ml_models
                   WHERE model_type = ?
                   ORDER BY created_at DESC""",
                (model_type,),
            )

            # Deactivate and delete old models
            for i, model in enumerate(models):
                if i >= keep_last_n:
                    # Deactivate in database
                    self.deactivate_model(model["model_id"])

                    # Delete file
                    model_path = Path(model["model_path"])
                    if model_path.exists():
                        model_path.unlink()
                        self.logger.info(f"Deleted old model file: {model_path}")
