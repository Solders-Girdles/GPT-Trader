"""
DL-003: LSTM Training Framework
Phase 4 - Week 1

Training framework with:
- Training time < 30 minutes for 2 years of data
- Early stopping and checkpointing
- Automatic hyperparameter optimization
- GPU support with fallback to CPU
- TensorBoard integration for monitoring
"""

import json
import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# Import components
try:
    from .lstm_architecture import LSTMArchitecture, LSTMConfig, TaskType
    from .lstm_data_pipeline import LSTMDataPipeline, SequenceConfig
except ImportError:
    # Handle direct execution
    from lstm_architecture import LSTMArchitecture, TaskType
    from lstm_data_pipeline import LSTMDataPipeline

# Try to import deep learning frameworks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Supported optimizers"""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Learning rate schedulers"""

    NONE = "none"
    STEP = "step"
    COSINE = "cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


@dataclass
class TrainingConfig:
    """Configuration for LSTM training"""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Optimizer settings
    optimizer_type: OptimizerType = OptimizerType.ADAM
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: float = 0.9  # For SGD

    # Learning rate scheduling
    scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-6
    restore_best_weights: bool = True

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10  # Save every N epochs
    save_best_only: bool = True

    # Validation
    validation_split: float = 0.2
    validation_frequency: int = 1

    # Performance targets
    max_training_time_minutes: float = 30.0
    target_accuracy: float | None = None
    target_loss: float | None = None

    # Monitoring
    tensorboard_logging: bool = True
    log_frequency: int = 10  # Log every N batches

    # Hyperparameter optimization
    use_hyperopt: bool = False
    hyperopt_trials: int = 50
    hyperopt_timeout_minutes: float = 120.0

    # Hardware
    use_gpu: bool = True
    mixed_precision: bool = False
    num_workers: int = 4

    # Paths
    log_dir: str = "logs/lstm_training"
    checkpoint_dir: str = "checkpoints/lstm"

    def __post_init__(self):
        """Validate configuration"""
        if self.max_training_time_minutes <= 0:
            raise ValueError("max_training_time_minutes must be positive")

        if self.validation_split < 0 or self.validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")


@dataclass
class TrainingResults:
    """Results from training session"""

    # Training metrics
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)

    # Best metrics
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0

    # Training info
    total_epochs: int = 0
    training_time_seconds: float = 0.0
    early_stopped: bool = False

    # Hyperparameter optimization results
    best_hyperparams: dict[str, Any] | None = None
    hyperopt_trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary"""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "total_epochs": self.total_epochs,
            "training_time_seconds": self.training_time_seconds,
            "early_stopped": self.early_stopped,
            "best_hyperparams": self.best_hyperparams,
        }


class LSTMTrainingFramework:
    """
    Comprehensive training framework for LSTM models.

    Features:
    - Multiple backend support (PyTorch, TensorFlow, sklearn)
    - Early stopping and checkpointing
    - Hyperparameter optimization
    - Performance monitoring
    - GPU acceleration
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.data_pipeline = None
        self.results = TrainingResults()

        # Setup directories
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.tb_writer = None
        if config.tensorboard_logging:
            try:
                if TORCH_AVAILABLE:
                    from torch.utils.tensorboard import SummaryWriter

                    self.tb_writer = SummaryWriter(config.log_dir)
                elif TF_AVAILABLE:
                    # TensorFlow has built-in TensorBoard support
                    pass
            except ImportError:
                logger.warning("TensorBoard not available")

        logger.info("Initialized LSTM training framework")

    def train(
        self,
        model: LSTMArchitecture,
        data_pipeline: LSTMDataPipeline,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        train_lengths: np.ndarray | None = None,
        val_lengths: np.ndarray | None = None,
    ) -> TrainingResults:
        """
        Train LSTM model with comprehensive monitoring.

        Args:
            model: LSTM architecture to train
            data_pipeline: Data pipeline for preprocessing
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            train_lengths: Training sequence lengths
            val_lengths: Validation sequence lengths

        Returns:
            TrainingResults with metrics and history
        """
        start_time = time.time()
        self.model = model
        self.data_pipeline = data_pipeline

        logger.info(f"Starting LSTM training with {model.backend} backend")

        # Prepare validation data if not provided
        if X_val is None:
            split_idx = int(len(X_train) * (1 - self.config.validation_split))
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]

            if train_lengths is not None:
                val_lengths = train_lengths[split_idx:]
                train_lengths = train_lengths[:split_idx]

        # Train based on backend
        if model.backend == "torch":
            results = self._train_torch(X_train, y_train, X_val, y_val, train_lengths, val_lengths)
        elif model.backend == "tensorflow":
            results = self._train_tensorflow(
                X_train, y_train, X_val, y_val, train_lengths, val_lengths
            )
        else:
            results = self._train_sklearn(X_train, y_train, X_val, y_val)

        # Record training time
        total_time = time.time() - start_time
        results.training_time_seconds = total_time

        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Check if training time exceeded target
        if total_time > self.config.max_training_time_minutes * 60:
            logger.warning(
                f"Training time ({total_time:.1f}s) exceeded target "
                f"({self.config.max_training_time_minutes * 60:.1f}s)"
            )

        self.results = results
        return results

    def _train_torch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_lengths: np.ndarray | None,
        val_lengths: np.ndarray | None,
    ) -> TrainingResults:
        """Train PyTorch model"""

        device = self.model.device
        model = self.model.model

        # Setup optimizer
        if self.config.optimizer_type == OptimizerType.ADAM:
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == OptimizerType.ADAMW:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == OptimizerType.SGD:
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # Setup scheduler
        scheduler = None
        if self.config.scheduler_type == SchedulerType.STEP:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma
            )
        elif self.config.scheduler_type == SchedulerType.COSINE:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        elif self.config.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.config.lr_patience,
                factor=self.config.lr_factor,
                verbose=True,
            )

        # Setup loss function
        if model.config.task_type == TaskType.REGRESSION:
            criterion = nn.MSELoss()
        elif model.config.task_type == TaskType.BINARY_CLASSIFICATION:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Training loop
        results = TrainingResults()
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Create batches
            for batch_idx, (X_batch, y_batch, lengths_batch) in enumerate(
                self.data_pipeline.create_data_loader(X_train, y_train, train_lengths)
            ):
                # Move to device
                X_batch = torch.FloatTensor(X_batch).to(device)
                y_batch = torch.FloatTensor(y_batch).to(device)

                if lengths_batch is not None:
                    lengths_batch = torch.LongTensor(lengths_batch).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_batch, lengths_batch)

                # Calculate loss
                if model.config.task_type == TaskType.REGRESSION:
                    loss = criterion(outputs.squeeze(), y_batch)
                else:
                    if model.config.task_type == TaskType.BINARY_CLASSIFICATION:
                        loss = criterion(outputs.squeeze(), y_batch)
                        predicted = (outputs.squeeze() > 0.5).float()
                    else:
                        loss = criterion(outputs, y_batch.long())
                        predicted = torch.argmax(outputs, dim=1)

                    train_correct += (predicted == y_batch).sum().item()
                    train_total += y_batch.size(0)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if hasattr(model.config, "gradient_clipping"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), model.config.gradient_clipping
                    )

                optimizer.step()
                train_loss += loss.item()

                # Log batch metrics
                if self.tb_writer and batch_idx % self.config.log_frequency == 0:
                    global_step = epoch * len(X_train) // self.config.batch_size + batch_idx
                    self.tb_writer.add_scalar("Batch/Loss", loss.item(), global_step)

            # Validation phase
            val_loss, val_accuracy = self._validate_torch(
                model, criterion, X_val, y_val, val_lengths
            )

            # Record metrics
            avg_train_loss = train_loss / (len(X_train) // self.config.batch_size)
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0

            results.train_losses.append(avg_train_loss)
            results.val_losses.append(val_loss)
            results.train_accuracies.append(train_accuracy)
            results.val_accuracies.append(val_accuracy)

            # Log to TensorBoard
            if self.tb_writer:
                self.tb_writer.add_scalar("Epoch/Train_Loss", avg_train_loss, epoch)
                self.tb_writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
                self.tb_writer.add_scalar("Epoch/Train_Accuracy", train_accuracy, epoch)
                self.tb_writer.add_scalar("Epoch/Val_Accuracy", val_accuracy, epoch)
                self.tb_writer.add_scalar(
                    "Epoch/Learning_Rate", optimizer.param_groups[0]["lr"], epoch
                )

            # Update learning rate
            if scheduler:
                if self.config.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping check
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                results.best_epoch = epoch
                results.best_val_loss = val_loss
                results.best_val_accuracy = val_accuracy
                patience_counter = 0

                # Save best model
                if self.config.save_best_only:
                    self._save_checkpoint(model, optimizer, epoch, val_loss, is_best=True)
            else:
                patience_counter += 1

            # Save regular checkpoint
            if self.config.save_checkpoints and epoch % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(model, optimizer, epoch, val_loss, is_best=False)

            # Early stopping
            if self.config.early_stopping and patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                results.early_stopped = True
                break

            # Progress logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}"
                )

        results.total_epochs = epoch + 1
        return results

    def _validate_torch(
        self,
        model,
        criterion,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_lengths: np.ndarray | None,
    ) -> tuple[float, float]:
        """Validate PyTorch model"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch, lengths_batch in self.data_pipeline.create_data_loader(
                X_val, y_val, val_lengths, shuffle=False
            ):
                X_batch = torch.FloatTensor(X_batch).to(self.model.device)
                y_batch = torch.FloatTensor(y_batch).to(self.model.device)

                if lengths_batch is not None:
                    lengths_batch = torch.LongTensor(lengths_batch).to(self.model.device)

                outputs = model(X_batch, lengths_batch)

                if model.config.task_type == TaskType.REGRESSION:
                    loss = criterion(outputs.squeeze(), y_batch)
                else:
                    if model.config.task_type == TaskType.BINARY_CLASSIFICATION:
                        loss = criterion(outputs.squeeze(), y_batch)
                        predicted = (outputs.squeeze() > 0.5).float()
                    else:
                        loss = criterion(outputs, y_batch.long())
                        predicted = torch.argmax(outputs, dim=1)

                    val_correct += (predicted == y_batch).sum().item()
                    val_total += y_batch.size(0)

                val_loss += loss.item()

        avg_val_loss = val_loss / (len(X_val) // self.config.batch_size)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0

        return avg_val_loss, val_accuracy

    def _train_tensorflow(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_lengths: np.ndarray | None,
        val_lengths: np.ndarray | None,
    ) -> TrainingResults:
        """Train TensorFlow model"""

        model = self.model.model

        # Setup callbacks
        callbacks = []

        if self.config.early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                restore_best_weights=self.config.restore_best_weights,
                verbose=1,
            )
            callbacks.append(early_stopping)

        if self.config.save_checkpoints:
            checkpoint_path = Path(self.config.checkpoint_dir) / "best_model.h5"
            checkpoint = keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_loss",
                save_best_only=self.config.save_best_only,
                verbose=1,
            )
            callbacks.append(checkpoint)

        if self.config.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                verbose=1,
            )
            callbacks.append(lr_scheduler)

        if self.config.tensorboard_logging:
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=self.config.log_dir, histogram_freq=1, write_graph=True, update_freq="epoch"
            )
            callbacks.append(tensorboard)

        # Handle variable length sequences for TensorFlow
        if train_lengths is not None:
            # Mask sequences
            train_mask = np.zeros_like(X_train[:, :, 0])
            for i, length in enumerate(train_lengths):
                train_mask[i, :length] = 1
            X_train = X_train * train_mask[:, :, np.newaxis]

        if val_lengths is not None:
            val_mask = np.zeros_like(X_val[:, :, 0])
            for i, length in enumerate(val_lengths):
                val_mask[i, :length] = 1
            X_val = X_val * val_mask[:, :, np.newaxis]

        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Convert history to results
        results = TrainingResults()
        results.train_losses = history.history["loss"]
        results.val_losses = history.history["val_loss"]

        if "accuracy" in history.history:
            results.train_accuracies = history.history["accuracy"]
            results.val_accuracies = history.history["val_accuracy"]

        # Find best epoch
        best_epoch = np.argmin(results.val_losses)
        results.best_epoch = best_epoch
        results.best_val_loss = results.val_losses[best_epoch]
        if results.val_accuracies:
            results.best_val_accuracy = results.val_accuracies[best_epoch]

        results.total_epochs = len(results.train_losses)
        results.early_stopped = results.total_epochs < self.config.epochs

        return results

    def _train_sklearn(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> TrainingResults:
        """Train sklearn model (fallback)"""

        # Flatten sequences for sklearn
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        # Train model
        self.model.model.fit(X_train_flat, y_train)

        # Evaluate
        train_pred = self.model.model.predict(X_train_flat)
        val_pred = self.model.model.predict(X_val_flat)

        # Calculate metrics
        if hasattr(self.model.model, "predict_proba"):
            from sklearn.metrics import accuracy_score

            train_accuracy = accuracy_score(y_train, train_pred)
            val_accuracy = accuracy_score(y_val, val_pred)
        else:
            train_accuracy = 0.0
            val_accuracy = 0.0

        from sklearn.metrics import mean_squared_error

        train_loss = mean_squared_error(y_train, train_pred)
        val_loss = mean_squared_error(y_val, val_pred)

        results = TrainingResults()
        results.train_losses = [train_loss]
        results.val_losses = [val_loss]
        results.train_accuracies = [train_accuracy]
        results.val_accuracies = [val_accuracy]
        results.best_val_loss = val_loss
        results.best_val_accuracy = val_accuracy
        results.total_epochs = 1

        return results

    def _save_checkpoint(
        self, model, optimizer, epoch: int, val_loss: float, is_best: bool = False
    ):
        """Save model checkpoint"""
        if not TORCH_AVAILABLE:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.model.config,
        }

        if is_best:
            path = Path(self.config.checkpoint_dir) / "best_model.pth"
        else:
            path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"

        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")

    def optimize_hyperparameters(
        self,
        model_factory: Callable[[dict[str, Any]], LSTMArchitecture],
        data_pipeline: LSTMDataPipeline,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            model_factory: Function that creates model from hyperparameters
            data_pipeline: Data pipeline for preprocessing
            X_train: Training data
            y_train: Training targets
            X_val: Validation data
            y_val: Validation targets

        Returns:
            Best hyperparameters found
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, skipping hyperparameter optimization")
            return {}

        def objective(trial):
            # Define hyperparameter search space
            hyperparams = {
                "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256, 512]),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            }

            # Create model with suggested hyperparameters
            model = model_factory(hyperparams)

            # Create temporary training config
            temp_config = TrainingConfig(
                epochs=min(50, self.config.epochs),  # Reduced epochs for hyperopt
                batch_size=hyperparams["batch_size"],
                learning_rate=hyperparams["learning_rate"],
                early_stopping=True,
                patience=10,
                tensorboard_logging=False,
                save_checkpoints=False,
            )

            # Train with temporary config
            temp_trainer = LSTMTrainingFramework(temp_config)
            results = temp_trainer.train(model, data_pipeline, X_train, y_train, X_val, y_val)

            return results.best_val_loss

        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=self.config.hyperopt_trials,
            timeout=self.config.hyperopt_timeout_minutes * 60,
        )

        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")

        # Store results
        self.results.best_hyperparams = best_params
        self.results.hyperopt_trials = [
            {"params": trial.params, "value": trial.value} for trial in study.trials
        ]

        return best_params

    def save_results(self, path: str) -> None:
        """Save training results to file"""
        results_dict = self.results.to_dict()

        with open(path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Training results saved to {path}")

    def close(self) -> None:
        """Cleanup resources"""
        if self.tb_writer:
            self.tb_writer.close()


def create_lstm_training_framework(
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    early_stopping: bool = True,
    patience: int = 15,
    use_gpu: bool = True,
    tensorboard_logging: bool = True,
) -> LSTMTrainingFramework:
    """
    Factory function to create LSTM training framework.

    Args:
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        early_stopping: Enable early stopping
        patience: Early stopping patience
        use_gpu: Use GPU if available
        tensorboard_logging: Enable TensorBoard logging

    Returns:
        Configured LSTMTrainingFramework instance
    """
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        early_stopping=early_stopping,
        patience=patience,
        use_gpu=use_gpu,
        tensorboard_logging=tensorboard_logging,
    )

    return LSTMTrainingFramework(config)
