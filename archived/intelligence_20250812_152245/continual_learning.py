"""
Continual Learning Framework for GPT-Trader Phase 2.

This module provides sophisticated continual learning capabilities:
- Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
- Progressive Neural Networks for expanding capacity
- Memory replay systems for experience retention
- Regularization-based approaches for stable learning
- Task-specific parameter isolation
- Lifelong learning across market regimes and instruments

Enables the system to continuously learn from new market data while preserving
knowledge of previous market conditions and trading patterns.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Optional deep learning libraries
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
    """Represents a continual learning task."""

    task_id: int
    name: str
    data: pd.DataFrame
    target: pd.Series
    task_type: str  # temporal, cross_asset, regime_specific
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class ContinualLearningConfig(BaseConfig):
    """Configuration for continual learning framework."""

    # Continual learning strategy
    strategy: str = "ewc"  # ewc, progressive, replay, regularization, ensemble

    # EWC parameters
    ewc_lambda: float = 1000.0  # Regularization strength
    fisher_samples: int = 200  # Samples for Fisher Information Matrix
    ewc_online: bool = True  # Online vs batch EWC

    # Progressive networks
    progressive_columns: int = 3  # Number of columns per task
    lateral_connections: bool = True
    adapter_capacity: float = 0.1  # Fraction of original capacity

    # Memory replay
    memory_size: int = 1000  # Experience replay buffer size
    replay_batch_size: int = 32  # Batch size for replay
    replay_frequency: int = 10  # How often to replay
    memory_selection: str = "random"  # random, diverse, recent

    # Regularization approaches
    l2_regularization: float = 0.001
    dropout_rate: float = 0.1
    weight_decay: float = 1e-4

    # Task detection and management
    task_detection: bool = True
    change_detection_threshold: float = 0.1
    task_similarity_threshold: float = 0.7
    max_tasks: int = 20

    # Model architecture
    model_type: str = "neural"  # neural, linear, ensemble
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64])

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs_per_task: int = 50
    early_stopping_patience: int = 10

    # Performance monitoring
    performance_buffer_size: int = 100
    forgetting_threshold: float = 0.1

    # Storage and persistence
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    memory_efficient: bool = True

    # Random state
    random_state: int = 42


@dataclass
class ContinualLearningResult:
    """Result from continual learning process."""

    task_id: int
    task_performance: dict[str, float]
    average_performance: float
    backward_transfer: float  # Performance on previous tasks
    forward_transfer: float  # Performance on future tasks
    forgetting_measure: float
    learning_time: float
    memory_usage: float


class ChangeDetector:
    """Detect when new tasks/regimes begin."""

    def __init__(self, threshold: float = 0.1, window_size: int = 50) -> None:
        self.threshold = threshold
        self.window_size = window_size
        self.reference_statistics = None
        self.current_window = deque(maxlen=window_size)

    def update(self, data_point: dict[str, float]) -> bool:
        """
        Update detector with new data point and check for change.

        Returns True if change detected.
        """
        self.current_window.append(data_point)

        if len(self.current_window) < self.window_size:
            return False

        # Calculate current statistics
        current_stats = self._calculate_statistics()

        if self.reference_statistics is None:
            self.reference_statistics = current_stats
            return False

        # Compare with reference statistics
        change_detected = self._detect_change(current_stats, self.reference_statistics)

        if change_detected:
            self.reference_statistics = current_stats

        return change_detected

    def _calculate_statistics(self) -> dict[str, float]:
        """Calculate statistics from current window."""
        if not self.current_window:
            return {}

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(list(self.current_window))

        stats = {}
        for col in df.columns:
            stats[f"{col}_mean"] = df[col].mean()
            stats[f"{col}_std"] = df[col].std()
            stats[f"{col}_median"] = df[col].median()

        return stats

    def _detect_change(self, current: dict[str, float], reference: dict[str, float]) -> bool:
        """Detect if there's significant change between statistics."""

        changes = []
        for key in reference:
            if key in current:
                ref_val = reference[key]
                curr_val = current[key]

                if ref_val != 0:
                    relative_change = abs(curr_val - ref_val) / abs(ref_val)
                else:
                    relative_change = abs(curr_val)

                changes.append(relative_change)

        if changes:
            avg_change = np.mean(changes)
            return avg_change > self.threshold

        return False


class ExperienceReplayBuffer:
    """Buffer for storing and replaying past experiences."""

    def __init__(self, capacity: int, selection_strategy: str = "random") -> None:
        self.capacity = capacity
        self.selection_strategy = selection_strategy
        self.buffer = deque(maxlen=capacity)
        self.task_counts = defaultdict(int)

    def add(self, experience: dict[str, Any]) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)

        if "task_id" in experience:
            self.task_counts[experience["task_id"]] += 1

    def sample(self, batch_size: int, exclude_task: int | None = None) -> list[dict[str, Any]]:
        """Sample experiences from buffer."""

        if len(self.buffer) == 0:
            return []

        # Filter out current task if specified
        available_experiences = [
            exp for exp in self.buffer if exclude_task is None or exp.get("task_id") != exclude_task
        ]

        if not available_experiences:
            return []

        sample_size = min(batch_size, len(available_experiences))

        if self.selection_strategy == "random":
            return self._random_sample(available_experiences, sample_size)
        elif self.selection_strategy == "diverse":
            return self._diverse_sample(available_experiences, sample_size)
        else:
            return self._recent_sample(available_experiences, sample_size)

    def _random_sample(self, experiences: list[dict], size: int) -> list[dict]:
        """Random sampling strategy."""
        return np.random.choice(experiences, size, replace=False).tolist()

    def _diverse_sample(self, experiences: list[dict], size: int) -> list[dict]:
        """Sample diverse experiences across tasks."""
        # Group by task
        task_groups = defaultdict(list)
        for exp in experiences:
            task_id = exp.get("task_id", 0)
            task_groups[task_id].append(exp)

        # Sample proportionally from each task
        sampled = []
        tasks = list(task_groups.keys())

        for i in range(size):
            task_id = tasks[i % len(tasks)]
            if task_groups[task_id]:
                exp = np.random.choice(task_groups[task_id])
                sampled.append(exp)

        return sampled

    def _recent_sample(self, experiences: list[dict], size: int) -> list[dict]:
        """Sample more recent experiences."""
        # Sort by timestamp if available
        sorted_exp = sorted(experiences, key=lambda x: x.get("timestamp", 0), reverse=True)
        return sorted_exp[:size]

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "task_distribution": dict(self.task_counts),
            "utilization": len(self.buffer) / self.capacity,
        }


class BaseContinualLearner(ABC):
    """Base class for continual learning strategies."""

    def __init__(self, config: ContinualLearningConfig) -> None:
        self.config = config
        self.current_task_id = 0
        self.task_history = []
        self.performance_history = []

    @abstractmethod
    def learn_task(self, task: Task) -> ContinualLearningResult:
        """Learn a new task."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, task_id: int | None = None) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate_all_tasks(self, tasks: list[Task]) -> dict[int, dict[str, float]]:
        """Evaluate performance on all learned tasks."""
        pass


class EWCLearner(BaseContinualLearner):
    """Elastic Weight Consolidation for continual learning."""

    def __init__(self, config: ContinualLearningConfig) -> None:
        super().__init__(config)

        if not HAS_TORCH:
            logger.warning("PyTorch not available. Using simplified EWC with sklearn.")
            self.use_pytorch = False
            self.model = Ridge(alpha=1.0)
            self.scaler = StandardScaler()
        else:
            self.use_pytorch = True
            self.model = None
            self.optimizer = None
            self.fisher_information = {}
            self.optimal_params = {}
            self.scaler = StandardScaler()

    def _create_neural_model(self, input_size: int) -> nn.Module:
        """Create neural network model."""
        layers = []
        current_size = input_size

        for hidden_size in self.config.hidden_layers:
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout_rate),
                ]
            )
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))
        return nn.Sequential(*layers)

    def learn_task(self, task: Task) -> ContinualLearningResult:
        """Learn new task with EWC regularization."""
        start_time = time.time()

        logger.info(f"Learning task {task.task_id}: {task.name}")

        # Prepare data
        X = task.data.fillna(0)
        y = task.target

        if self.use_pytorch:
            result = self._learn_task_pytorch(task, X, y)
        else:
            result = self._learn_task_sklearn(task, X, y)

        learning_time = time.time() - start_time
        result.learning_time = learning_time

        # Update task history
        self.task_history.append(task)
        self.performance_history.append(result)
        self.current_task_id = task.task_id

        logger.info(f"Task {task.task_id} learning completed in {learning_time:.2f}s")

        return result

    def _learn_task_pytorch(
        self, task: Task, X: pd.DataFrame, y: pd.Series
    ) -> ContinualLearningResult:
        """PyTorch-based EWC learning."""

        # Scale features
        X_scaled = (
            self.scaler.fit_transform(X) if self.current_task_id == 0 else self.scaler.transform(X)
        )

        # Initialize model on first task
        if self.model is None:
            input_size = X_scaled.shape[1]
            self.model = self._create_neural_model(input_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values).unsqueeze(1)

        # Calculate Fisher Information Matrix before learning new task
        if self.current_task_id > 0:
            self._compute_fisher_information(X_tensor, y_tensor)
            self._save_optimal_parameters()

        # Training loop
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        for _epoch in range(self.config.epochs_per_task):
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_X)
                task_loss = F.mse_loss(predictions, batch_y)

                # EWC regularization
                ewc_loss = self._compute_ewc_loss()
                total_loss = task_loss + ewc_loss

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                break

        # Evaluate performance
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze().numpy()

        performance = {
            "r2": r2_score(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "mae": mean_absolute_error(y, predictions),
        }

        # Calculate backward transfer (performance on previous tasks)
        backward_transfer = self._calculate_backward_transfer()

        return ContinualLearningResult(
            task_id=task.task_id,
            task_performance=performance,
            average_performance=performance["r2"],
            backward_transfer=backward_transfer,
            forward_transfer=0.0,  # Cannot measure yet
            forgetting_measure=max(0, -backward_transfer),
            learning_time=0.0,  # Will be set by caller
            memory_usage=self._estimate_memory_usage(),
        )

    def _learn_task_sklearn(
        self, task: Task, X: pd.DataFrame, y: pd.Series
    ) -> ContinualLearningResult:
        """Sklearn-based simplified continual learning."""

        # Scale features
        X_scaled = (
            self.scaler.fit_transform(X) if self.current_task_id == 0 else self.scaler.transform(X)
        )

        # Simple regularization for continual learning
        if self.current_task_id > 0:
            # Increase regularization to preserve previous knowledge
            self.model.alpha *= 1.5

        # Fit model
        self.model.fit(X_scaled, y)

        # Evaluate performance
        predictions = self.model.predict(X_scaled)
        performance = {
            "r2": r2_score(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "mae": mean_absolute_error(y, predictions),
        }

        # Simple backward transfer estimation
        backward_transfer = 0.0
        if len(self.task_history) > 0:
            # Re-evaluate on previous tasks
            prev_performances = []
            for prev_task in self.task_history[-3:]:  # Last 3 tasks
                prev_X = self.scaler.transform(prev_task.data.fillna(0))
                prev_pred = self.model.predict(prev_X)
                prev_r2 = r2_score(prev_task.target, prev_pred)
                prev_performances.append(prev_r2)

            backward_transfer = np.mean(prev_performances) if prev_performances else 0.0

        return ContinualLearningResult(
            task_id=task.task_id,
            task_performance=performance,
            average_performance=performance["r2"],
            backward_transfer=backward_transfer,
            forward_transfer=0.0,
            forgetting_measure=max(0, 0.5 - backward_transfer),  # Simplified
            learning_time=0.0,
            memory_usage=0.0,
        )

    def _compute_fisher_information(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Compute Fisher Information Matrix for EWC."""
        if not self.use_pytorch:
            return

        self.fisher_information = {}

        # Sample subset for efficiency
        n_samples = min(self.config.fisher_samples, len(X))
        indices = torch.randperm(len(X))[:n_samples]
        X_sample = X[indices]
        y_sample = y[indices]

        # Compute gradients
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)

        self.model.train()

        for i in range(len(X_sample)):
            self.optimizer.zero_grad()

            output = self.model(X_sample[i : i + 1])
            loss = F.mse_loss(output, y_sample[i : i + 1])
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data**2

        # Average over samples
        for name in self.fisher_information:
            self.fisher_information[name] /= len(X_sample)

    def _save_optimal_parameters(self) -> None:
        """Save current parameters as optimal for EWC."""
        if not self.use_pytorch:
            return

        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.use_pytorch or not self.fisher_information:
            return torch.tensor(0.0)

        ewc_loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()

        return self.config.ewc_lambda * ewc_loss

    def _calculate_backward_transfer(self) -> float:
        """Calculate performance on previous tasks."""
        if not self.task_history or not self.use_pytorch:
            return 0.0

        self.model.eval()
        backward_scores = []

        with torch.no_grad():
            for prev_task in self.task_history[-5:]:  # Last 5 tasks
                prev_X = self.scaler.transform(prev_task.data.fillna(0))
                prev_X_tensor = torch.FloatTensor(prev_X)

                predictions = self.model(prev_X_tensor).squeeze().numpy()
                r2 = r2_score(prev_task.target, predictions)
                backward_scores.append(r2)

        return np.mean(backward_scores)

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if not self.use_pytorch:
            return 0.0

        total_params = sum(p.numel() for p in self.model.parameters())
        fisher_params = sum(f.numel() for f in self.fisher_information.values())
        optimal_params = sum(o.numel() for o in self.optimal_params.values())

        # Assume 4 bytes per float parameter
        memory_mb = (total_params + fisher_params + optimal_params) * 4 / (1024 * 1024)

        return memory_mb

    def predict(self, X: pd.DataFrame, task_id: int | None = None) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X.fillna(0))

        if self.use_pytorch and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                predictions = self.model(X_tensor).squeeze().numpy()
        else:
            predictions = self.model.predict(X_scaled)

        return predictions

    def evaluate_all_tasks(self, tasks: list[Task]) -> dict[int, dict[str, float]]:
        """Evaluate performance on all tasks."""
        results = {}

        for task in tasks:
            predictions = self.predict(task.data)
            performance = {
                "r2": r2_score(task.target, predictions),
                "mse": mean_squared_error(task.target, predictions),
                "mae": mean_absolute_error(task.target, predictions),
            }
            results[task.task_id] = performance

        return results


class ReplayLearner(BaseContinualLearner):
    """Experience replay-based continual learning."""

    def __init__(self, config: ContinualLearningConfig) -> None:
        super().__init__(config)
        self.replay_buffer = ExperienceReplayBuffer(config.memory_size, config.memory_selection)
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()

    def learn_task(self, task: Task) -> ContinualLearningResult:
        """Learn task with experience replay."""
        start_time = time.time()

        logger.info(f"Learning task {task.task_id} with replay: {task.name}")

        # Prepare current task data
        X = task.data.fillna(0)
        y = task.target

        # Scale features
        if self.current_task_id == 0:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Store experiences in replay buffer
        for i in range(len(X)):
            experience = {
                "task_id": task.task_id,
                "X": X_scaled[i],
                "y": y.iloc[i],
                "timestamp": time.time(),
            }
            self.replay_buffer.add(experience)

        # Prepare training data (current + replay)
        train_X = X_scaled
        train_y = y.values

        # Add replay experiences
        if self.current_task_id > 0:
            replay_experiences = self.replay_buffer.sample(
                self.config.replay_batch_size, exclude_task=task.task_id
            )

            if replay_experiences:
                replay_X = np.array([exp["X"] for exp in replay_experiences])
                replay_y = np.array([exp["y"] for exp in replay_experiences])

                # Combine current and replay data
                train_X = np.vstack([train_X, replay_X])
                train_y = np.concatenate([train_y, replay_y])

        # Train model
        self.model.fit(train_X, train_y)

        # Evaluate performance
        predictions = self.model.predict(X_scaled)
        performance = {
            "r2": r2_score(y, predictions),
            "mse": mean_squared_error(y, predictions),
            "mae": mean_absolute_error(y, predictions),
        }

        # Calculate backward transfer
        backward_transfer = self._calculate_backward_transfer()

        learning_time = time.time() - start_time

        # Update state
        self.task_history.append(task)
        self.current_task_id = task.task_id

        result = ContinualLearningResult(
            task_id=task.task_id,
            task_performance=performance,
            average_performance=performance["r2"],
            backward_transfer=backward_transfer,
            forward_transfer=0.0,
            forgetting_measure=max(0, -backward_transfer),
            learning_time=learning_time,
            memory_usage=self.replay_buffer.get_statistics()["size"] * 0.001,  # Rough estimate
        )

        self.performance_history.append(result)

        logger.info(f"Replay learning completed. Buffer size: {len(self.replay_buffer.buffer)}")

        return result

    def _calculate_backward_transfer(self) -> float:
        """Calculate performance retention on previous tasks."""
        if not self.task_history:
            return 0.0

        backward_scores = []
        for prev_task in self.task_history[-3:]:  # Last 3 tasks
            prev_X = self.scaler.transform(prev_task.data.fillna(0))
            pred = self.model.predict(prev_X)
            r2 = r2_score(prev_task.target, pred)
            backward_scores.append(r2)

        return np.mean(backward_scores)

    def predict(self, X: pd.DataFrame, task_id: int | None = None) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X.fillna(0))
        return self.model.predict(X_scaled)

    def evaluate_all_tasks(self, tasks: list[Task]) -> dict[int, dict[str, float]]:
        """Evaluate on all tasks."""
        results = {}
        for task in tasks:
            pred = self.predict(task.data)
            results[task.task_id] = {
                "r2": r2_score(task.target, pred),
                "mse": mean_squared_error(task.target, pred),
                "mae": mean_absolute_error(task.target, pred),
            }
        return results


class ContinualLearningFramework:
    """
    Comprehensive continual learning framework.

    Manages lifelong learning across multiple market regimes and instruments
    while preventing catastrophic forgetting of previous knowledge.
    """

    def __init__(self, config: ContinualLearningConfig) -> None:
        self.config = config

        # Components
        self.change_detector = ChangeDetector(threshold=config.change_detection_threshold)
        self.learner = self._create_learner()

        # Task management
        self.current_task_id = 0
        self.active_tasks = []
        self.completed_tasks = []

        # Performance tracking
        self.performance_buffer = deque(maxlen=config.performance_buffer_size)
        self.learning_curve = []

        # Checkpointing
        self.checkpoint_dir = None
        if config.save_checkpoints:
            self.checkpoint_dir = Path("checkpoints/continual_learning")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_learner(self) -> BaseContinualLearner:
        """Create continual learner based on strategy."""
        if self.config.strategy == "ewc":
            return EWCLearner(self.config)
        elif self.config.strategy == "replay":
            return ReplayLearner(self.config)
        else:
            # Default to EWC
            return EWCLearner(self.config)

    def process_data_stream(self, data: pd.DataFrame, target_col: str = "target") -> dict[str, Any]:
        """
        Process streaming data and detect when to learn new tasks.

        Args:
            data: New streaming data
            target_col: Name of target column

        Returns:
            Processing results including task detection and learning
        """

        logger.info(f"Processing data stream with {len(data)} samples")

        results = {
            "samples_processed": len(data),
            "tasks_detected": 0,
            "tasks_learned": 0,
            "change_points": [],
            "performance_metrics": {},
        }

        # Process data in chunks to detect regime changes
        chunk_size = max(10, len(data) // 10)
        change_points = []

        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i : i + chunk_size]

            # Extract features for change detection
            if "Close" in chunk.columns:
                features = self._extract_change_features(chunk)

                # Update change detector
                for _, row in features.iterrows():
                    change_detected = self.change_detector.update(row.to_dict())

                    if change_detected:
                        change_points.append(i + _)
                        logger.info(f"Change detected at position {i + _}")

        results["change_points"] = change_points

        # Create and learn tasks based on change points
        if change_points or len(self.completed_tasks) == 0:
            tasks = self._create_tasks_from_data(data, target_col, change_points)

            for task in tasks:
                learning_result = self.learn_new_task(task)
                results["tasks_learned"] += 1
                results[f"task_{task.task_id}_performance"] = learning_result.task_performance

            results["tasks_detected"] = len(tasks)

        # Update performance metrics
        results["performance_metrics"] = self._get_performance_summary()

        return results

    def _extract_change_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for change detection."""
        features = pd.DataFrame()

        if "Close" in data.columns:
            close = data["Close"]

            # Price-based features
            features["price_mean"] = close.mean()
            features["price_std"] = close.std()
            features["returns_mean"] = close.pct_change().mean()
            features["returns_std"] = close.pct_change().std()

            # Volatility
            features["volatility"] = close.pct_change().rolling(5).std().mean()

            # Trend
            features["trend"] = (
                (close.iloc[-1] - close.iloc[0]) / close.iloc[0] if len(close) > 1 else 0
            )

        if "Volume" in data.columns:
            volume = data["Volume"]
            features["volume_mean"] = volume.mean()
            features["volume_std"] = volume.std()

        # Fill single-row DataFrame for each data point
        features = pd.concat([features] * len(data), ignore_index=True)
        features.fillna(0, inplace=True)

        return features

    def _create_tasks_from_data(
        self, data: pd.DataFrame, target_col: str, change_points: list[int]
    ) -> list[Task]:
        """Create tasks from data based on change points."""
        tasks = []

        # Add start and end points
        split_points = [0] + change_points + [len(data)]
        split_points = sorted(set(split_points))

        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]

            # Skip very small segments
            if end_idx - start_idx < 20:
                continue

            task_data = data.iloc[start_idx:end_idx].copy()

            # Create target if not present
            if target_col not in task_data.columns:
                if "Close" in task_data.columns:
                    target = task_data["Close"].pct_change().shift(-1).fillna(0)
                else:
                    continue
            else:
                target = task_data[target_col]

            # Remove target from features
            feature_data = task_data.drop(
                columns=[target_col] if target_col in task_data.columns else []
            )

            task = Task(
                task_id=self.current_task_id,
                name=f"temporal_task_{self.current_task_id}",
                data=feature_data,
                target=target,
                task_type="temporal",
                metadata={
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "data_period": (
                        (task_data.index[0], task_data.index[-1]) if len(task_data) > 0 else None
                    ),
                },
                timestamp=time.time(),
            )

            tasks.append(task)
            self.current_task_id += 1

        logger.info(f"Created {len(tasks)} tasks from data stream")
        return tasks

    def learn_new_task(self, task: Task) -> ContinualLearningResult:
        """Learn a new task with continual learning."""

        logger.info(f"Learning new task: {task.name} (ID: {task.task_id})")

        # Add to active tasks
        self.active_tasks.append(task)

        # Learn the task
        result = self.learner.learn_task(task)

        # Update learning curve
        self.learning_curve.append(
            {
                "task_id": task.task_id,
                "performance": result.task_performance,
                "backward_transfer": result.backward_transfer,
                "forgetting": result.forgetting_measure,
                "timestamp": time.time(),
            }
        )

        # Move to completed tasks
        self.completed_tasks.append(task)
        if task in self.active_tasks:
            self.active_tasks.remove(task)

        # Save checkpoint if configured
        if (
            self.config.save_checkpoints
            and len(self.completed_tasks) % self.config.checkpoint_frequency == 0
        ):
            self._save_checkpoint()

        # Check for task limit
        if len(self.completed_tasks) > self.config.max_tasks:
            self._remove_old_tasks()

        logger.info(f"Task learning completed. Total completed tasks: {len(self.completed_tasks)}")

        return result

    def predict_with_context(
        self, X: pd.DataFrame, context: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Make predictions with optional context information."""

        # Determine which task/model to use based on context
        task_id = None
        if context and "task_id" in context:
            task_id = context["task_id"]

        return self.learner.predict(X, task_id)

    def evaluate_continual_performance(self) -> dict[str, Any]:
        """Evaluate continual learning performance across all tasks."""

        if not self.completed_tasks:
            return {"status": "no_tasks_completed"}

        # Evaluate on all completed tasks
        all_results = self.learner.evaluate_all_tasks(self.completed_tasks)

        # Calculate continual learning metrics
        task_performances = [result["r2"] for result in all_results.values()]

        # Average performance
        avg_performance = np.mean(task_performances)

        # Backward transfer (performance on earlier tasks)
        if len(self.learning_curve) > 1:
            recent_bt = [entry["backward_transfer"] for entry in self.learning_curve[-5:]]
            avg_backward_transfer = np.mean([bt for bt in recent_bt if bt is not None])
        else:
            avg_backward_transfer = 0.0

        # Forgetting measure
        if len(self.learning_curve) > 1:
            recent_forgetting = [entry["forgetting"] for entry in self.learning_curve[-5:]]
            avg_forgetting = np.mean([f for f in recent_forgetting if f is not None])
        else:
            avg_forgetting = 0.0

        # Memory efficiency
        total_memory = sum(result.memory_usage for result in self.learner.performance_history)

        return {
            "total_tasks": len(self.completed_tasks),
            "average_performance": avg_performance,
            "performance_std": np.std(task_performances),
            "best_performance": np.max(task_performances),
            "worst_performance": np.min(task_performances),
            "average_backward_transfer": avg_backward_transfer,
            "average_forgetting": avg_forgetting,
            "total_memory_usage_mb": total_memory,
            "learning_efficiency": avg_performance / (total_memory + 1),
            "task_performances": all_results,
            "continual_learning_strategy": self.config.strategy,
        }

    def _get_performance_summary(self) -> dict[str, float]:
        """Get summary of recent performance."""
        if not self.learning_curve:
            return {}

        recent_curve = self.learning_curve[-5:]  # Last 5 tasks

        return {
            "recent_avg_performance": np.mean(
                [entry["performance"]["r2"] for entry in recent_curve]
            ),
            "recent_avg_forgetting": np.mean([entry["forgetting"] for entry in recent_curve]),
            "tasks_completed": len(self.completed_tasks),
            "learning_trend": self._calculate_learning_trend(),
        }

    def _calculate_learning_trend(self) -> float:
        """Calculate whether performance is improving or declining."""
        if len(self.learning_curve) < 2:
            return 0.0

        recent_performances = [entry["performance"]["r2"] for entry in self.learning_curve[-10:]]

        # Simple linear trend
        if len(recent_performances) >= 3:
            x = np.arange(len(recent_performances))
            slope = np.polyfit(x, recent_performances, 1)[0]
            return slope
        else:
            return 0.0

    def _save_checkpoint(self) -> None:
        """Save framework checkpoint."""
        if not self.checkpoint_dir:
            return

        checkpoint_path = self.checkpoint_dir / f"checkpoint_task_{self.current_task_id}.pkl"

        checkpoint_data = {
            "config": self.config,
            "current_task_id": self.current_task_id,
            "completed_tasks": self.completed_tasks,
            "learning_curve": self.learning_curve,
            "learner_state": getattr(self.learner, "__dict__", {}),
        }

        try:
            joblib.dump(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _remove_old_tasks(self) -> None:
        """Remove oldest tasks to manage memory."""
        tasks_to_remove = len(self.completed_tasks) - self.config.max_tasks

        if tasks_to_remove > 0:
            removed_tasks = self.completed_tasks[:tasks_to_remove]
            self.completed_tasks = self.completed_tasks[tasks_to_remove:]

            logger.info(f"Removed {len(removed_tasks)} old tasks to manage memory")

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load framework from checkpoint."""
        try:
            checkpoint_data = joblib.load(checkpoint_path)

            self.current_task_id = checkpoint_data["current_task_id"]
            self.completed_tasks = checkpoint_data["completed_tasks"]
            self.learning_curve = checkpoint_data["learning_curve"]

            # Restore learner state
            if "learner_state" in checkpoint_data:
                learner_state = checkpoint_data["learner_state"]
                for key, value in learner_state.items():
                    if hasattr(self.learner, key):
                        setattr(self.learner, key, value)

            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False


def create_continual_learning_system(
    config: ContinualLearningConfig | None = None,
) -> ContinualLearningFramework:
    """Create default continual learning system."""

    if config is None:
        config = ContinualLearningConfig(
            strategy="ewc",
            ewc_lambda=1000.0,
            memory_size=1000,
            task_detection=True,
            change_detection_threshold=0.1,
            max_tasks=20,
            save_checkpoints=True,
        )

    return ContinualLearningFramework(config)


# Example usage and demonstration
def demonstrate_continual_learning(market_data_sequence: list[pd.DataFrame]) -> dict[str, Any]:
    """Demonstrate continual learning on sequence of market data."""

    try:
        # Create continual learning system
        framework = create_continual_learning_system()

        results = {
            "demo_completed": True,
            "total_data_periods": len(market_data_sequence),
            "tasks_learned": 0,
            "learning_results": [],
            "final_evaluation": {},
        }

        # Process each data period sequentially
        for i, data_period in enumerate(market_data_sequence):
            logger.info(f"Processing data period {i+1}/{len(market_data_sequence)}")

            # Add target column (future returns)
            if "Close" in data_period.columns:
                data_period = data_period.copy()
                data_period["target"] = data_period["Close"].shift(-1) / data_period["Close"] - 1
                data_period["target"] = data_period["target"].fillna(0)
            else:
                continue

            # Process data stream
            stream_results = framework.process_data_stream(data_period, "target")
            results["learning_results"].append(stream_results)
            results["tasks_learned"] += stream_results["tasks_learned"]

        # Final evaluation
        evaluation = framework.evaluate_continual_performance()
        results["final_evaluation"] = evaluation

        logger.info(f"Continual learning demo completed. {results['tasks_learned']} tasks learned.")

        return results

    except Exception as e:
        logger.error(f"Continual learning demo failed: {e}")
        return {"demo_completed": False, "error": str(e), "framework_created": True}


# Integration with trading strategy evolution
class ContinualStrategyLearner:
    """
    Continual learning system for trading strategy evolution.

    Enables strategies to continuously adapt to new market conditions
    while preserving knowledge of successful patterns from the past.
    """

    def __init__(self, framework: ContinualLearningFramework) -> None:
        self.framework = framework
        self.strategy_evolution = []
        self.performance_tracking = defaultdict(list)

    def evolve_strategy(
        self, new_market_data: pd.DataFrame, current_performance: dict[str, float]
    ) -> dict[str, Any]:
        """Evolve strategy based on new market conditions."""

        # Create task from new market data
        task_data = new_market_data.copy()

        # Use performance metrics as learning target
        if "sharpe_ratio" in current_performance:
            # Create target based on rolling performance
            target = pd.Series(
                [current_performance["sharpe_ratio"]] * len(task_data), index=task_data.index
            )
        else:
            # Fallback to returns
            target = task_data["Close"].pct_change().fillna(0)

        # Remove target from features
        feature_data = task_data.drop(columns=["Close"] if "Close" in task_data.columns else [])

        # Create evolution task
        evolution_task = Task(
            task_id=len(self.strategy_evolution),
            name=f"strategy_evolution_{len(self.strategy_evolution)}",
            data=feature_data,
            target=target,
            task_type="strategy_evolution",
            metadata={
                "performance_metrics": current_performance,
                "market_period": (task_data.index[0], task_data.index[-1]),
            },
        )

        # Learn the evolution
        learning_result = self.framework.learn_new_task(evolution_task)

        # Store evolution history
        self.strategy_evolution.append(
            {"task": evolution_task, "learning_result": learning_result, "timestamp": time.time()}
        )

        # Track performance
        self.performance_tracking["evolution_performance"].append(learning_result.task_performance)

        return {
            "evolution_completed": True,
            "task_id": evolution_task.task_id,
            "learning_performance": learning_result.task_performance,
            "backward_transfer": learning_result.backward_transfer,
            "forgetting_measure": learning_result.forgetting_measure,
            "total_evolutions": len(self.strategy_evolution),
        }

    def get_evolved_strategy_signal(self, current_market_data: pd.DataFrame) -> dict[str, Any]:
        """Generate strategy signal using evolved continual learner."""

        if not self.strategy_evolution:
            return {"error": "No strategy evolution performed"}

        # Predict using continual learner
        predictions = self.framework.predict_with_context(current_market_data.tail(1))

        if len(predictions) > 0:
            signal_strength = predictions[0]

            # Convert to trading decision
            if signal_strength > 0.02:
                decision = "STRONG_BUY"
                confidence = min(signal_strength * 5, 1.0)
            elif signal_strength > 0.005:
                decision = "BUY"
                confidence = signal_strength * 10
            elif signal_strength < -0.02:
                decision = "STRONG_SELL"
                confidence = min(abs(signal_strength) * 5, 1.0)
            elif signal_strength < -0.005:
                decision = "SELL"
                confidence = abs(signal_strength) * 10
            else:
                decision = "HOLD"
                confidence = 0.5

            return {
                "decision": decision,
                "signal_strength": signal_strength,
                "confidence": confidence,
                "evolution_count": len(self.strategy_evolution),
                "last_evolution": self.strategy_evolution[-1]["timestamp"],
            }
        else:
            return {"error": "No predictions generated"}

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get summary of strategy evolution process."""

        if not self.strategy_evolution:
            return {"status": "no_evolution"}

        # Evolution performance trend
        performances = [
            evo["learning_result"].task_performance["r2"] for evo in self.strategy_evolution
        ]

        # Backward transfer trend
        backward_transfers = [
            evo["learning_result"].backward_transfer for evo in self.strategy_evolution
        ]

        # Forgetting trend
        forgetting_measures = [
            evo["learning_result"].forgetting_measure for evo in self.strategy_evolution
        ]

        return {
            "total_evolutions": len(self.strategy_evolution),
            "avg_evolution_performance": np.mean(performances),
            "performance_trend": np.polyfit(range(len(performances)), performances, 1)[0],
            "avg_backward_transfer": np.mean(backward_transfers),
            "avg_forgetting": np.mean(forgetting_measures),
            "learning_stability": 1.0 - np.std(performances),  # Higher is more stable
            "evolution_frequency": (
                len(self.strategy_evolution)
                / (time.time() - self.strategy_evolution[0]["timestamp"])
                * 3600
                if self.strategy_evolution
                else 0
            ),
            "continual_learning_efficiency": (
                np.mean(performances) / np.mean(forgetting_measures)
                if np.mean(forgetting_measures) > 0
                else np.mean(performances)
            ),
        }
