"""
Deep Learning Module for GPT-Trader Phase 1.

This module provides neural network architectures optimized for financial time series:
- LSTM for temporal pattern recognition
- CNN for price chart pattern detection
- Transformer for attention-based modeling
- Multi-scale architectures for different prediction horizons

Integrates with existing regime detection and ensemble framework.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Optional deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import callbacks, layers, models, optimizers
    from tensorflow.keras.layers import (
        GRU,
        LSTM,
        Add,
        BatchNormalization,
        Concatenate,
        Conv1D,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
        LayerNormalization,
        MaxPooling1D,
        MultiHeadAttention,
        TimeDistributed,
    )
    from tensorflow.keras.models import Model, Sequential

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow not available. Install with: pip install tensorflow")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class DeepLearningConfig(BaseConfig):
    """Configuration for deep learning models."""

    # Model architecture
    model_type: str = "lstm"  # lstm, gru, cnn, transformer, multi_scale
    sequence_length: int = 60
    prediction_horizon: int = 1

    # LSTM/GRU parameters
    lstm_units: list[int] = None
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.2

    # CNN parameters
    conv_filters: list[int] = None
    kernel_sizes: list[int] = None
    pool_sizes: list[int] = None

    # Transformer parameters
    num_heads: int = 8
    transformer_units: int = 128
    num_transformer_blocks: int = 4

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5

    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.001

    # Feature scaling
    feature_scaler: str = "standard"  # standard, minmax, robust
    target_scaler: str = "standard"

    # Advanced options
    use_batch_norm: bool = True
    use_residual_connections: bool = False
    use_attention: bool = False

    def __post_init__(self):
        """Initialize default parameters."""
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]
        if self.conv_filters is None:
            self.conv_filters = [64, 32, 16]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3]
        if self.pool_sizes is None:
            self.pool_sizes = [2, 2, 2]


class FinancialTimeSeriesDataset:
    """Dataset class for financial time series data."""

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        feature_scaler: str = "standard",
        target_scaler: str = "standard",
    ) -> None:
        self.data = data
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Initialize scalers
        if feature_scaler == "standard":
            self.feature_scaler = StandardScaler()
        elif feature_scaler == "minmax":
            self.feature_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported feature scaler: {feature_scaler}")

        if target_scaler == "standard":
            self.target_scaler = StandardScaler()
        elif target_scaler == "minmax":
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported target scaler: {target_scaler}")

        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare sequences for training."""
        # Separate features and target
        feature_columns = [col for col in self.data.columns if col != self.target_column]

        # Scale features
        features_scaled = self.feature_scaler.fit_transform(self.data[feature_columns])
        features_df = pd.DataFrame(features_scaled, columns=feature_columns, index=self.data.index)

        # Scale target
        target_scaled = self.target_scaler.fit_transform(self.data[[self.target_column]])
        target_df = pd.DataFrame(target_scaled, columns=[self.target_column], index=self.data.index)

        # Create sequences
        X, y = [], []

        for i in range(self.sequence_length, len(self.data) - self.prediction_horizon + 1):
            # Input sequence
            X.append(features_df.iloc[i - self.sequence_length : i].values)

            # Target (future value)
            if self.prediction_horizon == 1:
                y.append(target_df.iloc[i].values[0])
            else:
                y.append(target_df.iloc[i : i + self.prediction_horizon].values.flatten())

        self.X = np.array(X)
        self.y = np.array(y)
        self.n_features = len(feature_columns)

        logger.info(f"Prepared dataset: X shape {self.X.shape}, y shape {self.y.shape}")

    def get_train_test_split(self, test_size: float = 0.2) -> tuple[np.ndarray, ...]:
        """Get train/test split respecting time series order."""
        split_idx = int(len(self.X) * (1 - test_size))

        X_train, X_test = self.X[:split_idx], self.X[split_idx:]
        y_train, y_test = self.y[:split_idx], self.y[split_idx:]

        return X_train, X_test, y_train, y_test

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled target values."""
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled).flatten()


class BaseDeepModel(ABC):
    """Base class for deep learning models."""

    def __init__(self, config: DeepLearningConfig) -> None:
        self.config = config
        self.model = None
        self.history = None
        self.is_fitted = False

    @abstractmethod
    def build_model(self, input_shape: tuple[int, ...], output_shape: int = 1) -> Any:
        """Build the model architecture."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def save_model(self, filepath: str | Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not built yet")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(filepath))

    def load_model(self, filepath: str | Path) -> None:
        """Load model from disk."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")
        self.model = keras.models.load_model(str(filepath))
        self.is_fitted = True


class LSTMModel(BaseDeepModel):
    """LSTM model for financial time series prediction."""

    def build_model(self, input_shape: tuple[int, ...], output_shape: int = 1) -> Model:
        """Build LSTM architecture."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")

        model = Sequential()

        # First LSTM layer
        model.add(
            LSTM(
                self.config.lstm_units[0],
                return_sequences=len(self.config.lstm_units) > 1,
                input_shape=input_shape,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
            )
        )

        if self.config.use_batch_norm:
            model.add(BatchNormalization())

        # Additional LSTM layers
        for i, units in enumerate(self.config.lstm_units[1:], 1):
            return_sequences = i < len(self.config.lstm_units) - 1
            model.add(
                LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout,
                )
            )

            if self.config.use_batch_norm:
                model.add(BatchNormalization())

        # Dense layers
        model.add(Dense(32, activation="relu"))
        if self.config.use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(self.config.dropout_rate))

        # Output layer
        model.add(Dense(output_shape, activation="linear"))

        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the LSTM model."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        output_shape = y.shape[1] if y.ndim > 1 else 1
        self.model = self.build_model(input_shape, output_shape)

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                monitor="val_loss",
            ),
            callbacks.ReduceLROnPlateau(
                patience=self.config.reduce_lr_patience, factor=0.5, min_lr=1e-7, monitor="val_loss"
            ),
        ]

        # Train model
        self.history = self.model.fit(
            X,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=1,
        )

        self.is_fitted = True
        logger.info("LSTM model training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the LSTM model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class CNNModel(BaseDeepModel):
    """CNN model for pattern recognition in financial time series."""

    def build_model(self, input_shape: tuple[int, ...], output_shape: int = 1) -> Model:
        """Build CNN architecture."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")

        model = Sequential()

        # Convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(
                self.config.conv_filters,
                self.config.kernel_sizes,
                self.config.pool_sizes,
                strict=False,
            )
        ):
            if i == 0:
                model.add(
                    Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        activation="relu",
                        input_shape=input_shape,
                    )
                )
            else:
                model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"))

            if self.config.use_batch_norm:
                model.add(BatchNormalization())

            model.add(MaxPooling1D(pool_size=pool_size))
            model.add(Dropout(self.config.dropout_rate))

        # Global pooling
        model.add(GlobalAveragePooling1D())

        # Dense layers
        model.add(Dense(64, activation="relu"))
        if self.config.use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(self.config.dropout_rate))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(self.config.dropout_rate))

        # Output layer
        model.add(Dense(output_shape, activation="linear"))

        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the CNN model."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        output_shape = y.shape[1] if y.ndim > 1 else 1
        self.model = self.build_model(input_shape, output_shape)

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                monitor="val_loss",
            ),
            callbacks.ReduceLROnPlateau(
                patience=self.config.reduce_lr_patience, factor=0.5, min_lr=1e-7, monitor="val_loss"
            ),
        ]

        # Train model
        self.history = self.model.fit(
            X,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=1,
        )

        self.is_fitted = True
        logger.info("CNN model training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the CNN model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class TransformerModel(BaseDeepModel):
    """Transformer model with attention mechanisms for financial time series."""

    def _transformer_block(
        self, inputs, head_size: int, num_heads: int, ff_dim: int, dropout: float = 0.0
    ):
        """Build a transformer block."""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout
        )(inputs, inputs)

        attention_output = Dropout(dropout)(attention_output)
        res1 = Add()([inputs, attention_output])
        res1 = LayerNormalization(epsilon=1e-6)(res1)

        # Feed-forward network
        ffn_output = Dense(ff_dim, activation="relu")(res1)
        ffn_output = Dropout(dropout)(ffn_output)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)

        ffn_output = Dropout(dropout)(ffn_output)
        res2 = Add()([res1, ffn_output])
        return LayerNormalization(epsilon=1e-6)(res2)

    def build_model(self, input_shape: tuple[int, ...], output_shape: int = 1) -> Model:
        """Build Transformer architecture."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")

        inputs = Input(shape=input_shape)

        # Initial projection
        x = Dense(self.config.transformer_units)(inputs)

        # Transformer blocks
        for _ in range(self.config.num_transformer_blocks):
            x = self._transformer_block(
                x,
                head_size=self.config.transformer_units // self.config.num_heads,
                num_heads=self.config.num_heads,
                ff_dim=self.config.transformer_units * 2,
                dropout=self.config.dropout_rate,
            )

        # Global average pooling
        x = GlobalAveragePooling1D()(x)

        # Final dense layers
        x = Dense(64, activation="relu")(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(self.config.dropout_rate)(x)

        # Output layer
        outputs = Dense(output_shape, activation="linear")(x)

        model = Model(inputs, outputs)

        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the Transformer model."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        output_shape = y.shape[1] if y.ndim > 1 else 1
        self.model = self.build_model(input_shape, output_shape)

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                monitor="val_loss",
            ),
            callbacks.ReduceLROnPlateau(
                patience=self.config.reduce_lr_patience, factor=0.5, min_lr=1e-7, monitor="val_loss"
            ),
        ]

        # Train model
        self.history = self.model.fit(
            X,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=1,
        )

        self.is_fitted = True
        logger.info("Transformer model training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the Transformer model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class MultiScaleModel(BaseDeepModel):
    """Multi-scale model combining LSTM and CNN for different time horizons."""

    def build_model(self, input_shape: tuple[int, ...], output_shape: int = 1) -> Model:
        """Build multi-scale architecture."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")

        inputs = Input(shape=input_shape)

        # LSTM branch for long-term patterns
        lstm_branch = LSTM(64, return_sequences=True)(inputs)
        lstm_branch = LSTM(32)(lstm_branch)
        lstm_branch = Dense(32, activation="relu")(lstm_branch)

        # CNN branch for short-term patterns
        cnn_branch = Conv1D(64, 3, activation="relu")(inputs)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        cnn_branch = Conv1D(32, 3, activation="relu")(cnn_branch)
        cnn_branch = GlobalAveragePooling1D()(cnn_branch)
        cnn_branch = Dense(32, activation="relu")(cnn_branch)

        # Attention branch for feature importance
        if self.config.use_attention:
            attention_branch = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
            attention_branch = GlobalAveragePooling1D()(attention_branch)
            attention_branch = Dense(32, activation="relu")(attention_branch)

            # Combine all branches
            combined = Concatenate()([lstm_branch, cnn_branch, attention_branch])
        else:
            # Combine LSTM and CNN branches
            combined = Concatenate()([lstm_branch, cnn_branch])

        # Final processing
        x = Dense(64, activation="relu")(combined)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(self.config.dropout_rate)(x)

        # Output layer
        outputs = Dense(output_shape, activation="linear")(x)

        model = Model(inputs, outputs)

        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the multi-scale model."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available")

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        output_shape = y.shape[1] if y.ndim > 1 else 1
        self.model = self.build_model(input_shape, output_shape)

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                monitor="val_loss",
            ),
            callbacks.ReduceLROnPlateau(
                patience=self.config.reduce_lr_patience, factor=0.5, min_lr=1e-7, monitor="val_loss"
            ),
        ]

        # Train model
        self.history = self.model.fit(
            X,
            y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=1,
        )

        self.is_fitted = True
        logger.info("Multi-scale model training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the multi-scale model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class DeepLearningFramework:
    """
    Framework for managing deep learning models in GPT-Trader.

    Provides unified interface for different neural network architectures
    and integrates with existing ensemble and regime detection systems.
    """

    def __init__(self, config: DeepLearningConfig) -> None:
        self.config = config
        self.model = None
        self.dataset = None

    def create_model(self, model_type: str | None = None) -> BaseDeepModel:
        """Create a deep learning model based on configuration."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available for deep learning")

        model_type = model_type or self.config.model_type

        if model_type == "lstm":
            return LSTMModel(self.config)
        elif model_type == "cnn":
            return CNNModel(self.config)
        elif model_type == "transformer":
            return TransformerModel(self.config)
        elif model_type == "multi_scale":
            return MultiScaleModel(self.config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def prepare_data(self, data: pd.DataFrame, target_column: str) -> FinancialTimeSeriesDataset:
        """Prepare time series data for deep learning."""
        self.dataset = FinancialTimeSeriesDataset(
            data=data,
            target_column=target_column,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon,
            feature_scaler=self.config.feature_scaler,
            target_scaler=self.config.target_scaler,
        )
        return self.dataset

    def train_model(
        self, data: pd.DataFrame, target_column: str, model_type: str | None = None
    ) -> BaseDeepModel:
        """Train a deep learning model end-to-end."""
        # Prepare data
        dataset = self.prepare_data(data, target_column)
        X_train, X_test, y_train, y_test = dataset.get_train_test_split()

        # Create and train model
        self.model = self.create_model(model_type)
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        test_predictions = self.model.predict(X_test)
        test_loss = np.mean((y_test - test_predictions.flatten()) ** 2)

        logger.info(f"Model training completed. Test MSE: {test_loss:.6f}")

        return self.model

    def predict_next(self, recent_data: pd.DataFrame) -> float:
        """Predict next value given recent data."""
        if self.model is None or self.dataset is None:
            raise ValueError("Model must be trained before prediction")

        # Prepare input sequence
        feature_columns = [col for col in recent_data.columns if col != self.dataset.target_column]

        # Scale features
        features_scaled = self.dataset.feature_scaler.transform(
            recent_data[feature_columns].tail(self.config.sequence_length)
        )

        # Reshape for prediction
        X_pred = features_scaled.reshape(1, self.config.sequence_length, -1)

        # Make prediction
        y_pred_scaled = self.model.predict(X_pred)

        # Inverse transform
        y_pred = self.dataset.inverse_transform_target(y_pred_scaled)

        return float(y_pred[0])

    def get_model_summary(self) -> dict[str, Any]:
        """Get summary of the deep learning framework."""
        summary = {
            "config": self.config.dict(),
            "model_trained": self.model is not None and self.model.is_fitted,
            "dataset_prepared": self.dataset is not None,
        }

        if self.model is not None and hasattr(self.model, "model"):
            if hasattr(self.model.model, "count_params"):
                summary["model_parameters"] = int(self.model.model.count_params())

        return summary


def create_financial_lstm(sequence_length: int = 60, n_features: int = 50) -> DeepLearningFramework:
    """Create a pre-configured LSTM for financial prediction."""
    config = DeepLearningConfig(
        model_type="lstm",
        sequence_length=sequence_length,
        lstm_units=[128, 64, 32],
        dropout_rate=0.2,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
    )

    return DeepLearningFramework(config)


def create_financial_transformer(sequence_length: int = 60) -> DeepLearningFramework:
    """Create a pre-configured Transformer for financial prediction."""
    config = DeepLearningConfig(
        model_type="transformer",
        sequence_length=sequence_length,
        num_heads=8,
        transformer_units=128,
        num_transformer_blocks=4,
        dropout_rate=0.1,
        epochs=150,
        batch_size=32,
        learning_rate=0.0001,
    )

    return DeepLearningFramework(config)
