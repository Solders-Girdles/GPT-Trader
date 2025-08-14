"""
Meta-Learning Framework for GPT-Trader Phase 2.

This module provides sophisticated meta-learning capabilities:
- Model-Agnostic Meta-Learning (MAML) for rapid adaptation
- Learning to learn trading strategies across different markets
- Few-shot learning for new instruments and conditions
- Meta-optimization for hyperparameter learning
- Task similarity measurement and transfer
- Learning rate adaptation and gradient-based meta-learning

Enables rapid adaptation to new trading scenarios with minimal data
by learning from the structure of multiple related tasks.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Optional deep learning for advanced meta-learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a meta-learning task."""

    name: str
    X_support: pd.DataFrame  # Support/training set
    y_support: pd.Series
    X_query: pd.DataFrame  # Query/test set
    y_query: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X_support)


@dataclass
class MetaLearningConfig(BaseConfig):
    """Configuration for meta-learning framework."""

    # Meta-learning algorithm
    algorithm: str = "maml"  # maml, reptile, learning2learn, gradient_based

    # MAML parameters
    inner_lr: float = 0.01  # Inner loop learning rate
    meta_lr: float = 0.001  # Meta learning rate (outer loop)
    inner_steps: int = 5  # Number of inner loop steps
    meta_batch_size: int = 16  # Number of tasks per meta-batch

    # Task generation
    n_support_samples: int = 50  # Support set size per task
    n_query_samples: int = 25  # Query set size per task
    task_variety: str = "temporal"  # temporal, cross_asset, regime_based

    # Model architecture
    model_type: str = "neural"  # neural, linear, ensemble
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1

    # Training parameters
    meta_epochs: int = 100
    early_stopping_patience: int = 20
    min_improvement: float = 0.001

    # Task similarity and selection
    task_similarity_threshold: float = 0.3
    max_tasks_per_batch: int = 20
    use_task_weighting: bool = True

    # Few-shot adaptation
    few_shot_samples: int = 10
    adaptation_steps: int = 10
    adaptation_lr: float = 0.01

    # Performance tracking
    validation_tasks: int = 5
    performance_window: int = 50

    # Advanced features
    use_gradient_checkpointing: bool = False
    use_higher_order_gradients: bool = True
    regularization_strength: float = 0.01

    # Random state
    random_state: int = 42


@dataclass
class MetaLearningResult:
    """Result from meta-learning process."""

    meta_loss: float
    adaptation_performance: dict[str, float]
    task_performances: list[float]
    convergence_epoch: int
    training_time: float
    n_tasks_trained: int


class TaskGenerator:
    """Generate meta-learning tasks from trading data."""

    def __init__(self, config: MetaLearningConfig) -> None:
        self.config = config

    def generate_temporal_tasks(
        self, data: pd.DataFrame, target_col: str = "future_return", n_tasks: int = 50
    ) -> list[Task]:
        """Generate tasks from different time periods."""
        tasks = []

        # Calculate minimum data needed
        min_samples = self.config.n_support_samples + self.config.n_query_samples
        if len(data) < min_samples * n_tasks:
            logger.warning(
                f"Insufficient data for {n_tasks} tasks. Reducing to {len(data) // min_samples}"
            )
            n_tasks = max(1, len(data) // min_samples)

        # Create features if not already present
        features_df = self._create_features(data)

        # Create target if not present
        if target_col not in data.columns:
            if "Close" in data.columns:
                target = data["Close"].shift(-1) / data["Close"] - 1
                target = target.fillna(0)
            else:
                target = pd.Series(np.random.randn(len(data)), index=data.index)
        else:
            target = data[target_col]

        # Generate tasks by sliding window
        step_size = max(1, (len(data) - min_samples) // n_tasks)

        for i in range(n_tasks):
            start_idx = i * step_size
            end_idx = start_idx + min_samples

            if end_idx >= len(data):
                break

            # Task data
            task_features = features_df.iloc[start_idx:end_idx]
            task_target = target.iloc[start_idx:end_idx]

            # Split into support and query
            support_size = self.config.n_support_samples

            X_support = task_features.iloc[:support_size]
            y_support = task_target.iloc[:support_size]
            X_query = task_features.iloc[support_size:]
            y_query = task_target.iloc[support_size:]

            task = Task(
                name=f"temporal_task_{i}",
                X_support=X_support,
                y_support=y_support,
                X_query=X_query,
                y_query=y_query,
                metadata={
                    "start_date": task_features.index[0],
                    "end_date": task_features.index[-1],
                    "task_type": "temporal",
                },
            )

            tasks.append(task)

        logger.info(f"Generated {len(tasks)} temporal tasks")
        return tasks

    def generate_cross_asset_tasks(
        self, data_dict: dict[str, pd.DataFrame], target_col: str = "future_return"
    ) -> list[Task]:
        """Generate tasks from different assets."""
        tasks = []

        for asset_name, asset_data in data_dict.items():
            if len(asset_data) < (self.config.n_support_samples + self.config.n_query_samples):
                continue

            # Create features
            features_df = self._create_features(asset_data)

            # Create target
            if target_col not in asset_data.columns:
                if "Close" in asset_data.columns:
                    target = asset_data["Close"].shift(-1) / asset_data["Close"] - 1
                    target = target.fillna(0)
                else:
                    continue
            else:
                target = asset_data[target_col]

            # Use recent data for task
            recent_features = features_df.tail(
                self.config.n_support_samples + self.config.n_query_samples
            )
            recent_target = target.tail(self.config.n_support_samples + self.config.n_query_samples)

            # Split support/query
            support_size = self.config.n_support_samples

            X_support = recent_features.iloc[:support_size]
            y_support = recent_target.iloc[:support_size]
            X_query = recent_features.iloc[support_size:]
            y_query = recent_target.iloc[support_size:]

            task = Task(
                name=f"asset_{asset_name}",
                X_support=X_support,
                y_support=y_support,
                X_query=X_query,
                y_query=y_query,
                metadata={"asset": asset_name, "task_type": "cross_asset"},
            )

            tasks.append(task)

        logger.info(f"Generated {len(tasks)} cross-asset tasks")
        return tasks

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create trading features from OHLCV data."""
        features = pd.DataFrame(index=data.index)

        if "Close" in data.columns:
            close = data["Close"]

            # Returns and volatility
            features["returns"] = close.pct_change()
            features["volatility_5"] = features["returns"].rolling(5).std()
            features["volatility_20"] = features["returns"].rolling(20).std()

            # Moving averages
            features["ma_5"] = close.rolling(5).mean()
            features["ma_20"] = close.rolling(20).mean()
            features["ma_ratio"] = features["ma_5"] / features["ma_20"] - 1

            # Momentum
            features["momentum_5"] = close / close.shift(5) - 1
            features["momentum_10"] = close / close.shift(10) - 1

            # RSI approximation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            features["rsi"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            bb_ma = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            features["bb_position"] = (close - bb_ma) / (2 * bb_std + 1e-8)

        if "Volume" in data.columns:
            volume = data["Volume"]
            features["volume_ratio"] = volume / volume.rolling(20).mean()
            features["volume_roc"] = volume.pct_change()

        if "High" in data.columns and "Low" in data.columns:
            features["high_low_ratio"] = (data["High"] - data["Low"]) / data["Close"]

        # Fill missing values
        features = features.fillna(method="ffill").fillna(0)

        return features


class BaseMetaLearner(ABC):
    """Base class for meta-learning algorithms."""

    def __init__(self, config: MetaLearningConfig) -> None:
        self.config = config
        self.is_fitted = False
        self.meta_parameters = None

    @abstractmethod
    def meta_train(self, tasks: list[Task]) -> MetaLearningResult:
        """Meta-train on a collection of tasks."""
        pass

    @abstractmethod
    def adapt(self, task: Task) -> Any:
        """Adapt to a new task with few examples."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, adapted_model: Any = None) -> np.ndarray:
        """Make predictions using adapted model."""
        pass


class MAMLLearner(BaseMetaLearner):
    """Model-Agnostic Meta-Learning implementation."""

    def __init__(self, config: MetaLearningConfig) -> None:
        super().__init__(config)
        self.base_model = None
        self.meta_optimizer = None
        self.task_losses = []

    def _create_base_model(self, input_size: int) -> Any:
        """Create base model for MAML."""
        if self.config.model_type == "neural" and HAS_TORCH:
            return self._create_neural_model(input_size)
        else:
            # Fallback to sklearn model
            return Ridge(alpha=1.0)

    def _create_neural_model(self, input_size: int) -> nn.Module:
        """Create neural network for MAML."""
        layers = []
        current_size = input_size

        for hidden_size in self.config.hidden_layers:
            layers.extend(
                [nn.Linear(current_size, hidden_size), nn.ReLU(), nn.Dropout(self.config.dropout)]
            )
            current_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, 1))

        return nn.Sequential(*layers)

    def meta_train(self, tasks: list[Task]) -> MetaLearningResult:
        """Meta-train using MAML algorithm."""
        if not tasks:
            raise ValueError("No tasks provided for meta-training")

        logger.info(f"Starting MAML meta-training on {len(tasks)} tasks...")
        start_time = time.time()

        # Get input size from first task
        input_size = tasks[0].X_support.shape[1]

        # Create base model
        self.base_model = self._create_base_model(input_size)

        if HAS_TORCH and isinstance(self.base_model, nn.Module):
            return self._neural_meta_train(tasks, start_time)
        else:
            return self._sklearn_meta_train(tasks, start_time)

    def _neural_meta_train(self, tasks: list[Task], start_time: float) -> MetaLearningResult:
        """Neural network MAML training."""
        # Meta optimizer
        self.meta_optimizer = optim.Adam(self.base_model.parameters(), lr=self.config.meta_lr)

        best_meta_loss = float("inf")
        patience_counter = 0
        convergence_epoch = 0

        for epoch in range(self.config.meta_epochs):
            meta_loss = 0.0
            task_performances = []

            # Sample meta-batch
            batch_tasks = np.random.choice(
                tasks, min(self.config.meta_batch_size, len(tasks)), replace=False
            )

            # Meta-gradient accumulation

            for task in batch_tasks:
                # Inner loop adaptation
                adapted_params = self._inner_loop_adaptation(task)

                # Evaluate on query set
                query_loss = self._evaluate_on_query(task, adapted_params)
                meta_loss += query_loss

                # Calculate task performance
                task_r2 = self._calculate_task_r2(task, adapted_params)
                task_performances.append(task_r2)

            # Average meta loss
            meta_loss /= len(batch_tasks)

            # Meta-parameter update
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            # Early stopping check
            if meta_loss < best_meta_loss - self.config.min_improvement:
                best_meta_loss = meta_loss
                patience_counter = 0
                convergence_epoch = epoch
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}, Meta Loss: {meta_loss:.6f}, Avg Task R²: {np.mean(task_performances):.4f}"
                )

        self.is_fitted = True
        training_time = time.time() - start_time

        return MetaLearningResult(
            meta_loss=float(best_meta_loss),
            adaptation_performance={"avg_r2": np.mean(task_performances[-10:])},
            task_performances=task_performances[-len(batch_tasks) :],
            convergence_epoch=convergence_epoch,
            training_time=training_time,
            n_tasks_trained=len(tasks),
        )

    def _sklearn_meta_train(self, tasks: list[Task], start_time: float) -> MetaLearningResult:
        """Sklearn-based meta-training (simplified MAML)."""

        # Collect all task adaptations
        all_performances = []
        adapted_models = []

        for task in tasks:
            # Clone base model for each task
            adapted_model = clone(self.base_model)

            # Fit on support set
            X_support = task.X_support.fillna(0)
            adapted_model.fit(X_support, task.y_support)

            # Evaluate on query set
            X_query = task.X_query.fillna(0)
            predictions = adapted_model.predict(X_query)
            task_r2 = r2_score(task.y_query, predictions)

            all_performances.append(task_r2)
            adapted_models.append(adapted_model)

        # Simple meta-learning: average model parameters (for linear models)
        if hasattr(self.base_model, "coef_"):
            # Average coefficients across adapted models
            all_coefs = [model.coef_ for model in adapted_models if hasattr(model, "coef_")]
            if all_coefs:
                self.base_model.coef_ = np.mean(all_coefs, axis=0)

            all_intercepts = [
                model.intercept_ for model in adapted_models if hasattr(model, "intercept_")
            ]
            if all_intercepts:
                self.base_model.intercept_ = np.mean(all_intercepts)

        self.is_fitted = True
        training_time = time.time() - start_time

        return MetaLearningResult(
            meta_loss=1.0 - np.mean(all_performances),  # Convert R² to loss
            adaptation_performance={"avg_r2": np.mean(all_performances)},
            task_performances=all_performances,
            convergence_epoch=0,  # No epochs in sklearn version
            training_time=training_time,
            n_tasks_trained=len(tasks),
        )

    def _inner_loop_adaptation(self, task: Task) -> dict[str, torch.Tensor]:
        """Perform inner loop adaptation for a single task."""
        if not HAS_TORCH:
            return {}

        # Clone model parameters
        adapted_params = {name: param.clone() for name, param in self.base_model.named_parameters()}

        # Prepare data
        X_support = torch.FloatTensor(task.X_support.fillna(0).values)
        y_support = torch.FloatTensor(task.y_support.values).unsqueeze(1)

        # Inner loop optimizer
        inner_optimizer = optim.SGD(
            [{"params": list(adapted_params.values()), "lr": self.config.inner_lr}]
        )

        # Inner loop steps
        for _step in range(self.config.inner_steps):
            # Forward pass with adapted parameters
            predictions = self._forward_with_params(X_support, adapted_params)
            loss = F.mse_loss(predictions, y_support)

            # Backward pass
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return adapted_params

    def _forward_with_params(
        self, X: torch.Tensor, params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using specific parameters."""
        x = X
        param_iter = iter(params.values())

        # Iterate through layers
        for _i, layer in enumerate(self.base_model):
            if isinstance(layer, nn.Linear):
                weight = next(param_iter)
                bias = next(param_iter)
                x = F.linear(x, weight, bias)
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            elif isinstance(layer, nn.Dropout):
                x = F.dropout(x, training=self.training)

        return x

    def _evaluate_on_query(
        self, task: Task, adapted_params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Evaluate adapted model on query set."""
        X_query = torch.FloatTensor(task.X_query.fillna(0).values)
        y_query = torch.FloatTensor(task.y_query.values).unsqueeze(1)

        predictions = self._forward_with_params(X_query, adapted_params)
        return F.mse_loss(predictions, y_query)

    def _calculate_task_r2(self, task: Task, adapted_params: dict[str, torch.Tensor]) -> float:
        """Calculate R² score for a task."""
        X_query = torch.FloatTensor(task.X_query.fillna(0).values)

        with torch.no_grad():
            predictions = self._forward_with_params(X_query, adapted_params)

        pred_numpy = predictions.squeeze().numpy()
        return r2_score(task.y_query.values, pred_numpy)

    def adapt(self, task: Task) -> Any:
        """Adapt to a new task."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be trained first")

        if HAS_TORCH and isinstance(self.base_model, nn.Module):
            return self._neural_adapt(task)
        else:
            return self._sklearn_adapt(task)

    def _neural_adapt(self, task: Task) -> dict[str, torch.Tensor]:
        """Neural adaptation to new task."""
        return self._inner_loop_adaptation(task)

    def _sklearn_adapt(self, task: Task) -> Any:
        """Sklearn adaptation to new task."""
        adapted_model = clone(self.base_model)
        X_support = task.X_support.fillna(0)
        adapted_model.fit(X_support, task.y_support)
        return adapted_model

    def predict(self, X: pd.DataFrame, adapted_model: Any = None) -> np.ndarray:
        """Make predictions using adapted model."""
        if adapted_model is None:
            if not self.is_fitted:
                raise ValueError("No model available for prediction")
            adapted_model = self.base_model

        X_clean = X.fillna(0)

        if HAS_TORCH and isinstance(adapted_model, dict):
            # Neural model with adapted parameters
            X_tensor = torch.FloatTensor(X_clean.values)
            with torch.no_grad():
                predictions = self._forward_with_params(X_tensor, adapted_model)
            return predictions.squeeze().numpy()
        else:
            # Sklearn model
            return adapted_model.predict(X_clean)


class LearningToLearnFramework:
    """
    Comprehensive learning-to-learn framework for trading.

    Implements various meta-learning algorithms and provides
    unified interface for rapid adaptation to new trading scenarios.
    """

    def __init__(self, config: MetaLearningConfig) -> None:
        self.config = config
        self.task_generator = TaskGenerator(config)
        self.meta_learner = None
        self.task_history = []
        self.adaptation_history = []

        # Initialize meta-learner
        self._initialize_meta_learner()

    def _initialize_meta_learner(self) -> None:
        """Initialize meta-learning algorithm."""
        if self.config.algorithm == "maml":
            self.meta_learner = MAMLLearner(self.config)
        else:
            # Default to MAML
            self.meta_learner = MAMLLearner(self.config)

        logger.info(f"Initialized {self.config.algorithm} meta-learner")

    def train_on_historical_data(
        self, data: pd.DataFrame | dict[str, pd.DataFrame], target_col: str = "future_return"
    ) -> MetaLearningResult:
        """Train meta-learner on historical data."""

        logger.info("Generating meta-learning tasks from historical data...")

        if isinstance(data, dict):
            # Multiple assets - cross-asset tasks
            tasks = self.task_generator.generate_cross_asset_tasks(data, target_col)
        else:
            # Single asset - temporal tasks
            tasks = self.task_generator.generate_temporal_tasks(data, target_col)

        if not tasks:
            raise ValueError("No tasks generated from provided data")

        # Filter tasks by quality
        quality_tasks = self._filter_quality_tasks(tasks)

        logger.info(f"Training meta-learner on {len(quality_tasks)} high-quality tasks...")

        # Meta-train
        result = self.meta_learner.meta_train(quality_tasks)

        # Store task history
        self.task_history.extend(quality_tasks)

        logger.info(f"Meta-training completed. Meta-loss: {result.meta_loss:.6f}")

        return result

    def _filter_quality_tasks(self, tasks: list[Task]) -> list[Task]:
        """Filter tasks based on quality metrics."""
        quality_tasks = []

        for task in tasks:
            # Check data quality
            if len(task.X_support) < self.config.n_support_samples // 2:
                continue

            if len(task.X_query) < self.config.n_query_samples // 2:
                continue

            # Check for valid targets (not all zeros/NaN)
            if task.y_support.std() < 1e-8 or task.y_query.std() < 1e-8:
                continue

            # Check for reasonable data range
            if task.X_support.isnull().sum().sum() > len(task.X_support) * 0.5:
                continue

            quality_tasks.append(task)

        logger.info(f"Filtered {len(tasks)} tasks to {len(quality_tasks)} quality tasks")
        return quality_tasks

    def adapt_to_new_scenario(
        self,
        support_data: pd.DataFrame,
        support_target: pd.Series,
        query_data: pd.DataFrame | None = None,
        query_target: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Rapidly adapt to new trading scenario with few examples.

        Args:
            support_data: Support set features (few examples)
            support_target: Support set targets
            query_data: Optional query set for evaluation
            query_target: Optional query targets

        Returns:
            Adaptation results and adapted model
        """

        if not self.meta_learner.is_fitted:
            raise ValueError("Meta-learner must be trained first")

        logger.info(f"Adapting to new scenario with {len(support_data)} support examples")

        # Create task
        if query_data is None or query_target is None:
            # Split support data for evaluation
            X_supp, X_query, y_supp, y_query = train_test_split(
                support_data, support_target, test_size=0.4, random_state=self.config.random_state
            )
        else:
            X_supp, y_supp = support_data, support_target
            X_query, y_query = query_data, query_target

        adaptation_task = Task(
            name=f"adaptation_{len(self.adaptation_history)}",
            X_support=X_supp,
            y_support=y_supp,
            X_query=X_query,
            y_query=y_query,
            metadata={"adaptation": True, "timestamp": time.time()},
        )

        # Adapt model
        start_time = time.time()
        adapted_model = self.meta_learner.adapt(adaptation_task)
        adaptation_time = time.time() - start_time

        # Evaluate adaptation performance
        predictions = self.meta_learner.predict(X_query, adapted_model)

        performance_metrics = {
            "r2_score": r2_score(y_query, predictions),
            "mse": mean_squared_error(y_query, predictions),
            "mae": mean_absolute_error(y_query, predictions),
        }

        # Store adaptation history
        adaptation_record = {
            "task": adaptation_task,
            "adapted_model": adapted_model,
            "performance": performance_metrics,
            "adaptation_time": adaptation_time,
            "support_size": len(X_supp),
            "query_size": len(X_query),
        }

        self.adaptation_history.append(adaptation_record)

        logger.info(
            f"Adaptation completed in {adaptation_time:.3f}s. "
            f"Performance - R²: {performance_metrics['r2_score']:.4f}, "
            f"MSE: {performance_metrics['mse']:.6f}"
        )

        return {
            "adapted_model": adapted_model,
            "performance_metrics": performance_metrics,
            "adaptation_time": adaptation_time,
            "predictions": predictions,
            "task": adaptation_task,
        }

    def predict_with_adaptation(self, X: pd.DataFrame, adapted_model: Any) -> np.ndarray:
        """Make predictions using adapted model."""
        return self.meta_learner.predict(X, adapted_model)

    def get_task_similarity(self, task1: Task, task2: Task) -> float:
        """Calculate similarity between two tasks."""

        # Feature distribution similarity
        try:
            # Compare feature statistics
            stats1 = task1.X_support.describe()
            stats2 = task2.X_support.describe()

            feature_similarity = []
            for col in stats1.columns:
                if col in stats2.columns:
                    # Compare means and stds
                    mean_diff = abs(stats1.loc["mean", col] - stats2.loc["mean", col])
                    std_diff = abs(stats1.loc["std", col] - stats2.loc["std", col])

                    # Normalized similarity (inverse of differences)
                    col_sim = 1.0 / (1.0 + mean_diff + std_diff)
                    feature_similarity.append(col_sim)

            feature_sim = np.mean(feature_similarity) if feature_similarity else 0.0

            # Target distribution similarity
            target_corr = np.corrcoef(task1.y_support, task2.y_support)[0, 1]
            target_sim = max(0, target_corr) if not np.isnan(target_corr) else 0.0

            # Combined similarity
            overall_similarity = 0.7 * feature_sim + 0.3 * target_sim

        except Exception as e:
            logger.warning(f"Error calculating task similarity: {e}")
            overall_similarity = 0.0

        return overall_similarity

    def select_similar_tasks(
        self, target_task: Task, candidate_tasks: list[Task] | None = None
    ) -> list[Task]:
        """Select tasks similar to target for focused training."""

        if candidate_tasks is None:
            candidate_tasks = self.task_history

        if not candidate_tasks:
            return []

        # Calculate similarities
        similarities = []
        for task in candidate_tasks:
            similarity = self.get_task_similarity(target_task, task)
            similarities.append((task, similarity))

        # Filter by threshold and sort
        similar_tasks = [
            task for task, sim in similarities if sim >= self.config.task_similarity_threshold
        ]

        similar_tasks.sort(key=lambda x: similarities[candidate_tasks.index(x)][1], reverse=True)

        # Limit number of tasks
        return similar_tasks[: self.config.max_tasks_per_batch]

    def get_meta_learning_summary(self) -> dict[str, Any]:
        """Get summary of meta-learning performance."""

        if not self.adaptation_history:
            return {"status": "no_adaptations_performed"}

        # Performance statistics
        performances = [adapt["performance"]["r2_score"] for adapt in self.adaptation_history]
        adaptation_times = [adapt["adaptation_time"] for adapt in self.adaptation_history]
        support_sizes = [adapt["support_size"] for adapt in self.adaptation_history]

        summary = {
            "total_tasks_trained": len(self.task_history),
            "total_adaptations": len(self.adaptation_history),
            "avg_adaptation_performance": np.mean(performances),
            "best_adaptation_performance": np.max(performances),
            "avg_adaptation_time": np.mean(adaptation_times),
            "avg_support_size": np.mean(support_sizes),
            "meta_learner_type": self.config.algorithm,
            "is_trained": self.meta_learner.is_fitted,
        }

        # Recent performance trend
        if len(performances) >= 10:
            recent_perf = performances[-10:]
            early_perf = performances[:10]
            summary["performance_improvement"] = np.mean(recent_perf) - np.mean(early_perf)

        return summary


def create_meta_learning_system(
    config: MetaLearningConfig | None = None,
) -> LearningToLearnFramework:
    """Create default meta-learning system."""

    if config is None:
        config = MetaLearningConfig(
            algorithm="maml",
            inner_lr=0.01,
            meta_lr=0.001,
            inner_steps=5,
            meta_epochs=100,
            n_support_samples=50,
            n_query_samples=25,
            task_variety="temporal",
        )

    return LearningToLearnFramework(config)


# Integration with trading strategy adaptation
class MetaLearningStrategyAdaptor:
    """
    Meta-learning adaptor for trading strategies.

    Uses meta-learning to rapidly adapt trading strategies
    to new market conditions with minimal data.
    """

    def __init__(self, framework: LearningToLearnFramework) -> None:
        self.framework = framework
        self.strategy_adaptations = {}
        self.adaptation_performance = defaultdict(list)

    def train_strategy_meta_learner(
        self, historical_strategies: dict[str, dict[str, Any]]
    ) -> MetaLearningResult:
        """
        Train meta-learner on historical strategy performance.

        Args:
            historical_strategies: Dict of {strategy_name: {data: DataFrame, performance: metrics}}
        """

        # Convert strategy data to meta-learning format
        strategy_data = {}

        for strategy_name, strategy_info in historical_strategies.items():
            if "data" in strategy_info:
                # Add strategy performance as target
                data = strategy_info["data"].copy()

                # Use returns or Sharpe ratio as target
                if "performance" in strategy_info:
                    perf_metrics = strategy_info["performance"]
                    if "sharpe_ratio" in perf_metrics:
                        # Create target based on rolling Sharpe ratio
                        returns = data["Close"].pct_change()
                        rolling_sharpe = (
                            returns.rolling(20).mean() / returns.rolling(20).std() * np.sqrt(252)
                        ).fillna(0)
                        data["strategy_performance"] = rolling_sharpe
                    else:
                        data["strategy_performance"] = 0.0
                else:
                    data["strategy_performance"] = 0.0

                strategy_data[strategy_name] = data

        # Train meta-learner
        result = self.framework.train_on_historical_data(
            strategy_data, target_col="strategy_performance"
        )

        logger.info(f"Strategy meta-learner trained on {len(strategy_data)} strategies")

        return result

    def adapt_strategy_to_new_market(
        self, new_market_data: pd.DataFrame, few_shot_examples: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """
        Adapt strategy to new market using meta-learning.

        Args:
            new_market_data: New market data to adapt to
            few_shot_examples: Optional few examples with known good performance
        """

        if not self.framework.meta_learner.is_fitted:
            raise ValueError("Strategy meta-learner must be trained first")

        # Prepare support data
        if few_shot_examples is not None:
            support_data = few_shot_examples
            # Use simple momentum as target for demonstration
            support_target = support_data["Close"].pct_change().rolling(5).mean().fillna(0)
        else:
            # Use recent market data as support
            support_size = min(len(new_market_data) // 2, self.framework.config.n_support_samples)
            support_data = new_market_data.tail(support_size * 2).head(support_size)
            support_target = support_data["Close"].pct_change().rolling(5).mean().fillna(0)

        # Query data (for evaluation)
        query_data = new_market_data.tail(min(len(new_market_data) // 4, 50))
        query_target = query_data["Close"].pct_change().rolling(5).mean().fillna(0)

        # Perform adaptation
        adaptation_result = self.framework.adapt_to_new_scenario(
            support_data, support_target, query_data, query_target
        )

        # Store adaptation
        market_id = f"market_{hash(str(new_market_data.index))}"
        self.strategy_adaptations[market_id] = adaptation_result

        # Track performance
        self.adaptation_performance[market_id].append(adaptation_result["performance_metrics"])

        return {
            "market_id": market_id,
            "adaptation_result": adaptation_result,
            "adapted_strategy_performance": adaptation_result["performance_metrics"],
            "confidence": max(0, adaptation_result["performance_metrics"]["r2_score"]),
        }

    def predict_strategy_signal(self, market_data: pd.DataFrame, market_id: str) -> dict[str, Any]:
        """Generate strategy signal using adapted meta-learner."""

        if market_id not in self.strategy_adaptations:
            return {"error": "Market not adapted. Call adapt_strategy_to_new_market first."}

        adapted_model = self.strategy_adaptations[market_id]["adapted_model"]

        # Make prediction
        prediction = self.framework.predict_with_adaptation(market_data.tail(1), adapted_model)

        # Convert to trading signal
        signal_strength = prediction[0] if len(prediction) > 0 else 0.0

        if signal_strength > 0.01:
            signal = "BUY"
            confidence = min(signal_strength * 10, 1.0)
        elif signal_strength < -0.01:
            signal = "SELL"
            confidence = min(abs(signal_strength) * 10, 1.0)
        else:
            signal = "HOLD"
            confidence = 0.5

        return {
            "signal": signal,
            "strength": signal_strength,
            "confidence": confidence,
            "market_id": market_id,
        }

    def get_adaptation_summary(self) -> dict[str, Any]:
        """Get summary of strategy adaptations."""

        if not self.strategy_adaptations:
            return {"status": "no_adaptations"}

        # Performance across adaptations
        all_r2_scores = []
        all_adaptation_times = []

        for _market_id, adaptation in self.strategy_adaptations.items():
            all_r2_scores.append(adaptation["performance_metrics"]["r2_score"])
            all_adaptation_times.append(adaptation["adaptation_time"])

        return {
            "total_market_adaptations": len(self.strategy_adaptations),
            "avg_adaptation_r2": np.mean(all_r2_scores),
            "best_adaptation_r2": np.max(all_r2_scores),
            "avg_adaptation_time": np.mean(all_adaptation_times),
            "adaptation_success_rate": sum(1 for r2 in all_r2_scores if r2 > 0.1)
            / len(all_r2_scores),
            "markets_adapted": list(self.strategy_adaptations.keys()),
        }


# Example usage function
def demonstrate_meta_learning(
    historical_data: pd.DataFrame, new_market_data: pd.DataFrame
) -> dict[str, Any]:
    """Demonstrate meta-learning on trading data."""

    try:
        # Create meta-learning system
        framework = create_meta_learning_system()

        # Train on historical data
        logger.info("Training meta-learner on historical data...")
        training_result = framework.train_on_historical_data(
            historical_data, target_col="future_return"
        )

        # Adapt to new market (few-shot learning)
        logger.info("Adapting to new market scenario...")
        adaptation_result = framework.adapt_to_new_scenario(
            new_market_data.head(20),  # Few examples
            new_market_data.head(20)["Close"].pct_change().fillna(0),  # Simple target
            new_market_data.tail(10),  # Query set
            new_market_data.tail(10)["Close"].pct_change().fillna(0),
        )

        # Get framework summary
        summary = framework.get_meta_learning_summary()

        return {
            "demo_completed": True,
            "training_result": training_result,
            "adaptation_result": adaptation_result,
            "framework_summary": summary,
            "meta_learning_success": adaptation_result["performance_metrics"]["r2_score"] > 0.1,
        }

    except Exception as e:
        logger.error(f"Meta-learning demo failed: {e}")
        return {"demo_completed": False, "error": str(e), "framework_created": True}
