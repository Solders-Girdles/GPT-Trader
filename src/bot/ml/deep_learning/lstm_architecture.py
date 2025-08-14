"""
DL-001: LSTM Architecture Design
Phase 4 - Week 1

LSTM neural network architecture supporting:
- Variable sequence lengths (10-100 timesteps)
- 50+ input features
- Regression and classification modes
- Bidirectional LSTM option
- Dropout regularization
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Try multiple deep learning frameworks with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    BACKEND = "torch"
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        BACKEND = "tensorflow"
    except ImportError:
        # Fallback to sklearn-based sequence models
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        BACKEND = "sklearn"

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported task types"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


@dataclass
class LSTMConfig:
    """Configuration for LSTM architecture"""
    # Architecture
    input_size: int = 50  # Number of features
    hidden_size: int = 128  # LSTM hidden units
    num_layers: int = 2  # Number of LSTM layers
    dropout: float = 0.2  # Dropout rate
    bidirectional: bool = True  # Use bidirectional LSTM
    
    # Sequence parameters
    sequence_length: int = 30  # Default sequence length
    min_sequence_length: int = 10  # Minimum supported length
    max_sequence_length: int = 100  # Maximum supported length
    
    # Task configuration
    task_type: TaskType = TaskType.REGRESSION
    num_classes: int = 2  # For classification tasks
    output_size: int = 1  # For regression tasks
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Regularization
    batch_norm: bool = True
    layer_norm: bool = False
    gradient_clipping: float = 1.0


class LSTMArchitecture:
    """
    LSTM Architecture for financial time series prediction.
    
    Supports variable sequence lengths and multiple task types.
    Provides fallback implementations for different backends.
    """
    
    def __init__(self, config: LSTMConfig):
        self.config = config
        self.backend = BACKEND
        self.model = None
        self.device = None
        
        logger.info(f"Initializing LSTM with backend: {self.backend}")
        
        # Initialize device for PyTorch
        if self.backend == "torch":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the LSTM model based on available backend"""
        if self.backend == "torch":
            self._build_torch_model()
        elif self.backend == "tensorflow":
            self._build_tensorflow_model()
        else:
            self._build_sklearn_model()
    
    def _build_torch_model(self) -> None:
        """Build PyTorch LSTM model"""
        
        class TorchLSTMModel(nn.Module):
            def __init__(self, config: LSTMConfig):
                super().__init__()
                self.config = config
                
                # LSTM layer
                self.lstm = nn.LSTM(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_layers,
                    dropout=config.dropout if config.num_layers > 1 else 0,
                    bidirectional=config.bidirectional,
                    batch_first=True
                )
                
                # Calculate LSTM output size
                lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
                
                # Batch normalization
                if config.batch_norm:
                    self.batch_norm = nn.BatchNorm1d(lstm_output_size)
                
                # Layer normalization
                if config.layer_norm:
                    self.layer_norm = nn.LayerNorm(lstm_output_size)
                
                # Dropout
                self.dropout = nn.Dropout(config.dropout)
                
                # Output layers
                self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
                self.fc2 = nn.Linear(lstm_output_size // 2, lstm_output_size // 4)
                
                # Final output layer
                if config.task_type == TaskType.REGRESSION:
                    self.output = nn.Linear(lstm_output_size // 4, config.output_size)
                elif config.task_type == TaskType.BINARY_CLASSIFICATION:
                    self.output = nn.Linear(lstm_output_size // 4, 1)
                else:  # Multiclass
                    self.output = nn.Linear(lstm_output_size // 4, config.num_classes)
                
                # Initialize weights
                self._init_weights()
            
            def _init_weights(self) -> None:
                """Initialize model weights"""
                for name, param in self.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1
                        if 'bias_ih' in name:
                            n = param.size(0)
                            param.data[(n//4):(n//2)].fill_(1)
            
            def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
                """Forward pass with variable length support"""
                batch_size, seq_len, _ = x.shape
                
                # Handle variable length sequences
                if lengths is not None:
                    # Pack padded sequence
                    x = nn.utils.rnn.pack_padded_sequence(
                        x, lengths.cpu(), batch_first=True, enforce_sorted=False
                    )
                
                # LSTM forward pass
                lstm_out, (hidden, cell) = self.lstm(x)
                
                # Unpack if we packed
                if lengths is not None:
                    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                        lstm_out, batch_first=True
                    )
                
                # Use last output (for each sequence)
                if lengths is not None:
                    # Get last actual output for each sequence
                    idx = (lengths - 1).long().unsqueeze(-1).unsqueeze(-1)
                    idx = idx.expand(batch_size, 1, lstm_out.size(2))
                    last_output = lstm_out.gather(1, idx).squeeze(1)
                else:
                    last_output = lstm_out[:, -1, :]  # Use last timestep
                
                # Apply normalization
                if self.config.batch_norm and last_output.size(0) > 1:
                    last_output = self.batch_norm(last_output)
                
                if self.config.layer_norm:
                    last_output = self.layer_norm(last_output)
                
                # Dropout
                last_output = self.dropout(last_output)
                
                # Fully connected layers
                x = F.relu(self.fc1(last_output))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                
                # Output layer
                output = self.output(x)
                
                # Apply activation based on task type
                if self.config.task_type == TaskType.BINARY_CLASSIFICATION:
                    output = torch.sigmoid(output)
                elif self.config.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                    output = F.softmax(output, dim=1)
                
                return output
        
        self.model = TorchLSTMModel(self.config).to(self.device)
        logger.info(f"Built PyTorch LSTM model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _build_tensorflow_model(self) -> None:
        """Build TensorFlow LSTM model"""
        
        # Input layer with variable length support
        inputs = keras.Input(shape=(None, self.config.input_size), name='input_sequences')
        
        # Masking layer for variable lengths
        masked = layers.Masking(mask_value=0.0)(inputs)
        
        # LSTM layers
        x = masked
        for i in range(self.config.num_layers):
            return_sequences = i < self.config.num_layers - 1
            
            x = layers.LSTM(
                self.config.hidden_size,
                return_sequences=return_sequences,
                dropout=self.config.dropout,
                recurrent_dropout=self.config.dropout,
                bidirectional=self.config.bidirectional,
                name=f'lstm_{i}'
            )(x)
            
            if self.config.batch_norm:
                x = layers.BatchNormalization()(x)
        
        # Dense layers
        x = layers.Dense(self.config.hidden_size // 2, activation='relu')(x)
        x = layers.Dropout(self.config.dropout)(x)
        x = layers.Dense(self.config.hidden_size // 4, activation='relu')(x)
        x = layers.Dropout(self.config.dropout)(x)
        
        # Output layer
        if self.config.task_type == TaskType.REGRESSION:
            outputs = layers.Dense(self.config.output_size, name='output')(x)
        elif self.config.task_type == TaskType.BINARY_CLASSIFICATION:
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        else:  # Multiclass
            outputs = layers.Dense(self.config.num_classes, activation='softmax', name='output')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        if self.config.task_type == TaskType.REGRESSION:
            loss = 'mse'
            metrics = ['mae']
        elif self.config.task_type == TaskType.BINARY_CLASSIFICATION:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Built TensorFlow LSTM model with {self.model.count_params()} parameters")
    
    def _build_sklearn_model(self) -> None:
        """Build sklearn-based sequence model as fallback"""
        logger.warning("Using sklearn fallback - limited sequence modeling capability")
        
        if self.config.task_type == TaskType.REGRESSION:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        logger.info("Built sklearn fallback model")
    
    def predict(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions on input sequences"""
        if self.backend == "torch":
            return self._predict_torch(X, lengths)
        elif self.backend == "tensorflow":
            return self._predict_tensorflow(X, lengths)
        else:
            return self._predict_sklearn(X)
    
    def _predict_torch(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """PyTorch prediction"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            lengths_tensor = torch.LongTensor(lengths).to(self.device) if lengths is not None else None
            
            predictions = self.model(X_tensor, lengths_tensor)
            return predictions.cpu().numpy()
    
    def _predict_tensorflow(self, X: np.ndarray, lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """TensorFlow prediction"""
        if lengths is not None:
            # Mask sequences to proper lengths
            mask = np.zeros_like(X[:, :, 0])
            for i, length in enumerate(lengths):
                mask[i, :length] = 1
            X = X * mask[:, :, np.newaxis]
        
        return self.model.predict(X, verbose=0)
    
    def _predict_sklearn(self, X: np.ndarray) -> np.ndarray:
        """Sklearn prediction"""
        # Flatten sequences for sklearn
        X_flat = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_flat)
        
        # Ensure output has correct shape for regression
        if self.config.task_type == TaskType.REGRESSION:
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, self.config.output_size)
        
        return predictions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model architecture summary"""
        summary = {
            "backend": self.backend,
            "config": {
                "input_size": self.config.input_size,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "bidirectional": self.config.bidirectional,
                "task_type": self.config.task_type.value,
                "dropout": self.config.dropout
            }
        }
        
        if self.backend == "torch" and self.model:
            summary["parameters"] = sum(p.numel() for p in self.model.parameters())
            summary["trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        elif self.backend == "tensorflow" and self.model:
            summary["parameters"] = self.model.count_params()
        
        return summary
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        if self.backend == "torch":
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, path)
        elif self.backend == "tensorflow":
            self.model.save(path)
        else:
            import joblib
            joblib.dump(self.model, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk"""
        if self.backend == "torch":
            checkpoint = torch.load(path, map_location=self.device)
            self.config = checkpoint['config']
            self._build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif self.backend == "tensorflow":
            self.model = keras.models.load_model(path)
        else:
            import joblib
            self.model = joblib.load(path)
        
        logger.info(f"Model loaded from {path}")


def create_lstm_architecture(
    input_size: int = 50,
    sequence_length: int = 30,
    task_type: TaskType = TaskType.REGRESSION,
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    dropout: float = 0.2
) -> LSTMArchitecture:
    """
    Factory function to create LSTM architecture with common configurations.
    
    Args:
        input_size: Number of input features
        sequence_length: Length of input sequences
        task_type: Type of prediction task
        hidden_size: Size of LSTM hidden layers
        num_layers: Number of LSTM layers
        bidirectional: Use bidirectional LSTM
        dropout: Dropout rate for regularization
        
    Returns:
        Configured LSTMArchitecture instance
    """
    config = LSTMConfig(
        input_size=input_size,
        sequence_length=sequence_length,
        task_type=task_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout
    )
    
    return LSTMArchitecture(config)