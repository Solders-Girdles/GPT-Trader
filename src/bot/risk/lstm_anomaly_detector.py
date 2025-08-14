"""
LSTM-based Time Series Anomaly Detection
Phase 3, Week 3-4: RISK-010
Deep learning approach for sequential anomaly detection
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import callbacks, layers, models
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM features will be limited.")

from sklearn.preprocessing import MinMaxScaler

from ..utils.serialization import (
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM anomaly detector"""

    # Model architecture
    sequence_length: int = 20  # Length of input sequences
    n_features: int = 1  # Number of features
    lstm_units: list[int] = None  # LSTM layer sizes
    dropout_rate: float = 0.2

    # Training parameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 5

    # Anomaly detection
    threshold_multiplier: float = 2.0  # Standard deviations for threshold
    use_bidirectional: bool = True
    use_attention: bool = False

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [64, 32]


class LSTMAnomalyDetector:
    """
    LSTM-based anomaly detector for time series data.

    Uses an autoencoder architecture to learn normal patterns
    and detect deviations as anomalies.
    """

    def __init__(self, config: LSTMConfig | None = None):
        """
        Initialize LSTM detector.

        Args:
            config: LSTM configuration
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM anomaly detection")

        self.config = config or LSTMConfig()
        self.model: keras.Model | None = None
        self.scaler = MinMaxScaler()
        self.threshold: float = 0
        self.is_fitted = False

        # Training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _build_model(self) -> keras.Model:
        """Build LSTM autoencoder model"""
        # Input layer
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.n_features))

        # Encoder
        x = inputs
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1

            if self.config.use_bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(
                        units, return_sequences=return_sequences, dropout=self.config.dropout_rate
                    )
                )(x)
            else:
                x = layers.LSTM(
                    units, return_sequences=return_sequences, dropout=self.config.dropout_rate
                )(x)

        # Bottleneck (encoding)
        encoded = x

        # Decoder
        x = layers.RepeatVector(self.config.sequence_length)(encoded)

        for i, units in enumerate(reversed(self.config.lstm_units)):
            x = layers.LSTM(units, return_sequences=True, dropout=self.config.dropout_rate)(x)

        # Output layer
        outputs = layers.TimeDistributed(layers.Dense(self.config.n_features))(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile
        model.compile(optimizer=Adam(learning_rate=self.config.learning_rate), loss="mse")

        return model

    def _prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare sequences for LSTM input.

        Args:
            data: Time series data

        Returns:
            Array of sequences
        """
        sequences = []
        for i in range(len(data) - self.config.sequence_length + 1):
            seq = data[i : i + self.config.sequence_length]
            sequences.append(seq)

        return np.array(sequences)

    def fit(self, data: np.ndarray | pd.DataFrame, verbose: int = 0):
        """
        Train LSTM autoencoder on normal data.

        Args:
            data: Training data (should be mostly normal)
            verbose: Verbosity level
        """
        # Convert to numpy if needed
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Reshape if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        self.config.n_features = data.shape[1]

        # Scale data
        data_scaled = self.scaler.fit_transform(data)

        # Prepare sequences
        X = self._prepare_sequences(data_scaled)

        # Build model
        self.model = self._build_model()

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]

        # Train model
        history = self.model.fit(
            X,
            X,  # Autoencoder: input = output
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=verbose,
        )

        # Store training history
        self.train_losses = history.history["loss"]
        self.val_losses = history.history.get("val_loss", [])

        # Calculate threshold based on training reconstruction error
        train_pred = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - train_pred), axis=(1, 2))
        self.threshold = np.mean(mse) + self.config.threshold_multiplier * np.std(mse)

        self.is_fitted = True

        logger.info(f"LSTM trained for {len(history.history['loss'])} epochs")
        logger.info(f"Anomaly threshold set to {self.threshold:.6f}")

    def predict(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict reconstruction error for sequences.

        Args:
            data: Input data

        Returns:
            Reconstruction errors
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Convert to numpy if needed
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Reshape if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Scale data
        data_scaled = self.scaler.transform(data)

        # Prepare sequences
        X = self._prepare_sequences(data_scaled)

        if len(X) == 0:
            return np.array([])

        # Predict
        predictions = self.model.predict(X, verbose=0)

        # Calculate reconstruction error
        mse = np.mean(np.square(X - predictions), axis=(1, 2))

        return mse

    def detect_anomalies(self, data: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in time series data.

        Args:
            data: Input data

        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
        """
        # Get reconstruction errors
        errors = self.predict(data)

        # Detect anomalies
        anomalies = errors > self.threshold

        # Calculate anomaly scores (normalized)
        scores = errors / self.threshold

        return anomalies, scores

    def predict_next(self, data: np.ndarray | pd.DataFrame, n_steps: int = 1) -> np.ndarray:
        """
        Predict next values in sequence.

        Args:
            data: Historical data
            n_steps: Number of steps to predict

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Convert to numpy if needed
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Reshape if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Take last sequence_length points
        if len(data) >= self.config.sequence_length:
            last_sequence = data[-self.config.sequence_length :]
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((self.config.sequence_length - len(data), data.shape[1]))
            last_sequence = np.vstack([padding, data])

        # Scale
        last_sequence_scaled = self.scaler.transform(last_sequence)

        predictions = []
        current_sequence = last_sequence_scaled.copy()

        for _ in range(n_steps):
            # Reshape for model input
            X = current_sequence.reshape(1, self.config.sequence_length, self.config.n_features)

            # Predict (reconstruction)
            pred = self.model.predict(X, verbose=0)

            # Take last timestep as prediction
            next_val = pred[0, -1, :]
            predictions.append(next_val)

            # Update sequence
            current_sequence = np.vstack([current_sequence[1:], next_val])

        # Inverse transform
        predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(predictions)

        return predictions

    def save(self, filepath: str):
        """Save model and configuration"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Save keras model
        model_path = filepath + "_model.h5"
        self.model.save(model_path)

        # Save other components
        components = {
            "config": self.config,
            "scaler": self.scaler,
            "threshold": self.threshold,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        save_json(components, filepath + "_components.json")

        logger.info(f"LSTM model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and configuration"""
        # Load keras model
        model_path = filepath + "_model.h5"
        self.model = keras.models.load_model(model_path)

        # Load other components
        components = load_json(filepath + "_components.json")

        self.config = components["config"]
        self.scaler = components["scaler"]
        self.threshold = components["threshold"]
        self.train_losses = components["train_losses"]
        self.val_losses = components["val_losses"]
        self.is_fitted = True

        logger.info(f"LSTM model loaded from {filepath}")


class LSTMEnsembleDetector:
    """
    Ensemble of LSTM models for robust anomaly detection.

    Trains multiple models with different architectures and combines predictions.
    """

    def __init__(self, n_models: int = 3):
        """
        Initialize ensemble detector.

        Args:
            n_models: Number of models in ensemble
        """
        self.n_models = n_models
        self.models: list[LSTMAnomalyDetector] = []
        self.weights: np.ndarray = np.ones(n_models) / n_models

        # Create models with different configurations
        configs = [
            LSTMConfig(lstm_units=[32, 16], sequence_length=15),
            LSTMConfig(lstm_units=[64, 32], sequence_length=20),
            LSTMConfig(lstm_units=[128, 64, 32], sequence_length=30),
        ]

        for i in range(n_models):
            config = configs[i % len(configs)]
            self.models.append(LSTMAnomalyDetector(config))

    def fit(self, data: np.ndarray | pd.DataFrame, verbose: int = 0):
        """
        Train all models in ensemble.

        Args:
            data: Training data
            verbose: Verbosity level
        """
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{self.n_models}")
            model.fit(data, verbose=verbose)

    def detect_anomalies(self, data: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using ensemble voting.

        Args:
            data: Input data

        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
        """
        all_anomalies = []
        all_scores = []

        for model in self.models:
            if model.is_fitted:
                anomalies, scores = model.detect_anomalies(data)
                all_anomalies.append(anomalies)
                all_scores.append(scores)

        if not all_anomalies:
            return np.array([]), np.array([])

        # Combine predictions (weighted average)
        anomaly_probs = np.average(
            all_anomalies, axis=0, weights=self.weights[: len(all_anomalies)]
        )
        avg_scores = np.average(all_scores, axis=0, weights=self.weights[: len(all_scores)])

        # Threshold for final decision (majority vote)
        final_anomalies = anomaly_probs > 0.5

        return final_anomalies, avg_scores


def demonstrate_lstm_anomaly_detection():
    """Demonstrate LSTM anomaly detection"""

    if not TF_AVAILABLE:
        print("TensorFlow not available. Skipping LSTM demonstration.")
        print("Install with: pip install tensorflow")
        return

    print("LSTM Anomaly Detection Demo")
    print("=" * 60)

    # Generate synthetic time series with anomalies
    np.random.seed(42)

    # Normal pattern: sine wave with noise
    t = np.linspace(0, 100, 1000)
    normal_data = np.sin(t) + 0.1 * np.random.randn(len(t))

    # Inject anomalies
    anomaly_indices = [200, 400, 600, 800]
    for idx in anomaly_indices:
        # Different types of anomalies
        if idx == 200:
            normal_data[idx : idx + 10] += 2  # Level shift
        elif idx == 400:
            normal_data[idx : idx + 5] *= 3  # Amplitude change
        elif idx == 600:
            normal_data[idx : idx + 10] = np.random.randn(10) * 2  # Random noise
        else:
            normal_data[idx : idx + 5] = 0  # Dropout

    # Split data
    train_data = normal_data[:700]
    test_data = normal_data[700:]

    print("\nTraining LSTM autoencoder...")

    # Single model
    config = LSTMConfig(
        sequence_length=20,
        lstm_units=[32, 16],
        epochs=20,  # Reduced for demo
        threshold_multiplier=2.5,
    )

    detector = LSTMAnomalyDetector(config)
    detector.fit(train_data, verbose=0)

    print(f"Training complete. Threshold: {detector.threshold:.6f}")

    # Detect anomalies
    print("\nDetecting anomalies in test data...")
    anomalies, scores = detector.detect_anomalies(test_data)

    print(f"Found {np.sum(anomalies)} anomalies in {len(anomalies)} sequences")

    if np.sum(anomalies) > 0:
        print("\nAnomaly details:")
        anomaly_positions = np.where(anomalies)[0]
        for pos in anomaly_positions[:5]:  # Show first 5
            print(f"  Position {pos}: Score = {scores[pos]:.2f}")

    # Test ensemble
    print("\n" + "=" * 40)
    print("Testing Ensemble Detector...")

    ensemble = LSTMEnsembleDetector(n_models=2)  # Reduced for demo
    ensemble.fit(train_data[:500], verbose=0)  # Smaller training set for speed

    ensemble_anomalies, ensemble_scores = ensemble.detect_anomalies(test_data)
    print(f"Ensemble found {np.sum(ensemble_anomalies)} anomalies")

    # Prediction test
    print("\n" + "=" * 40)
    print("Testing Next Value Prediction...")

    next_values = detector.predict_next(test_data[:50], n_steps=5)
    print(f"Predicted next 5 values: {next_values.flatten()}")

    print("\n✅ LSTM Anomaly Detection operational!")


if __name__ == "__main__":
    # Set TensorFlow to use less memory for demo
    if TF_AVAILABLE:
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    demonstrate_lstm_anomaly_detection()
