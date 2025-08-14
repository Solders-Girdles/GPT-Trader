"""
DL-002: LSTM Data Pipeline
Phase 4 - Week 1

Data pipeline for LSTM training with:
- Overlapping sequences with proper time alignment
- Missing data handling
- Batch processing for efficiency
- Data augmentation capabilities
- Time series train/val/test splitting
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ScalingMethod(Enum):
    """Supported scaling methods"""
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


class AugmentationMethod(Enum):
    """Data augmentation methods"""
    NOISE = "noise"
    JITTER = "jitter"
    SCALING = "scaling"
    TIME_WARPING = "time_warping"
    MAGNITUDE_WARPING = "magnitude_warping"


@dataclass
class SequenceConfig:
    """Configuration for sequence generation"""
    # Sequence parameters
    sequence_length: int = 30
    overlap_ratio: float = 0.8  # Overlap between consecutive sequences
    min_sequence_length: int = 10
    max_sequence_length: int = 100
    
    # Data handling
    fill_method: str = "forward"  # forward, backward, linear, mean
    max_missing_ratio: float = 0.1  # Maximum missing data ratio per sequence
    
    # Scaling
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    feature_wise_scaling: bool = True
    temporal_scaling: bool = False  # Scale each sequence independently
    
    # Batching
    batch_size: int = 32
    shuffle_sequences: bool = True
    
    # Augmentation
    augmentation_ratio: float = 0.0  # Ratio of augmented sequences
    augmentation_methods: List[AugmentationMethod] = None
    noise_std: float = 0.01
    jitter_std: float = 0.03
    scaling_range: Tuple[float, float] = (0.9, 1.1)
    
    # Time series splitting
    test_size: float = 0.2
    validation_size: float = 0.2
    gap_size: int = 0  # Gap between train/val/test sets


class LSTMDataPipeline:
    """
    Data pipeline for LSTM training with time series considerations.
    
    Handles variable sequence lengths, missing data, and proper
    temporal splitting for financial time series.
    """
    
    def __init__(self, config: SequenceConfig):
        self.config = config
        self.scaler = None
        self.feature_scalers = {}
        self.is_fitted = False
        
        # Initialize scaler
        if config.scaling_method == ScalingMethod.STANDARD:
            self.scaler = StandardScaler()
        elif config.scaling_method == ScalingMethod.ROBUST:
            self.scaler = RobustScaler()
        elif config.scaling_method == ScalingMethod.MINMAX:
            self.scaler = MinMaxScaler()
        
        if config.augmentation_methods is None:
            self.config.augmentation_methods = [AugmentationMethod.NOISE, AugmentationMethod.JITTER]
        
        logger.info(f"Initialized LSTM data pipeline with sequence length {config.sequence_length}")
    
    def create_sequences(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        time_col: str = 'datetime'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create overlapping sequences from time series data.
        
        Args:
            data: Input DataFrame with time series data
            target_col: Name of target column
            feature_cols: List of feature column names
            time_col: Name of timestamp column
            
        Returns:
            Tuple of (X_sequences, y_sequences, sequence_lengths, timestamps)
        """
        # Sort by time
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Handle missing data
        data = self._handle_missing_data(data, feature_cols + [target_col])
        
        # Extract features and targets
        X = data[feature_cols].values
        y = data[target_col].values
        timestamps = data[time_col].values
        
        # Create sequences
        X_sequences, y_sequences, lengths, seq_timestamps = self._create_overlapping_sequences(
            X, y, timestamps
        )
        
        logger.info(f"Created {len(X_sequences)} sequences from {len(data)} data points")
        
        return X_sequences, y_sequences, lengths, seq_timestamps
    
    def _handle_missing_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle missing data in the time series"""
        data = data.copy()
        
        for col in columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            
            if missing_ratio > self.config.max_missing_ratio:
                logger.warning(f"Column {col} has {missing_ratio:.2%} missing data")
            
            # Fill missing data based on method
            if self.config.fill_method == "forward":
                data[col] = data[col].fillna(method='ffill')
            elif self.config.fill_method == "backward":
                data[col] = data[col].fillna(method='bfill')
            elif self.config.fill_method == "linear":
                data[col] = data[col].interpolate(method='linear')
            elif self.config.fill_method == "mean":
                data[col] = data[col].fillna(data[col].mean())
            
            # Forward fill any remaining missing values
            data[col] = data[col].fillna(method='ffill')
            data[col] = data[col].fillna(method='bfill')
        
        return data
    
    def _create_overlapping_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create overlapping sequences with proper time alignment"""
        n_samples, n_features = X.shape
        sequence_length = self.config.sequence_length
        
        # Calculate step size based on overlap ratio
        step_size = max(1, int(sequence_length * (1 - self.config.overlap_ratio)))
        
        sequences_X = []
        sequences_y = []
        lengths = []
        seq_timestamps = []
        
        for i in range(0, n_samples - sequence_length + 1, step_size):
            # Extract sequence
            seq_X = X[i:i + sequence_length]
            seq_y = y[i + sequence_length - 1]  # Target is the last value
            seq_ts = timestamps[i + sequence_length - 1]
            
            # Check for missing data in sequence
            missing_ratio = np.isnan(seq_X).sum() / (sequence_length * n_features)
            if missing_ratio <= self.config.max_missing_ratio:
                sequences_X.append(seq_X)
                sequences_y.append(seq_y)
                lengths.append(sequence_length)
                seq_timestamps.append(seq_ts)
        
        # Convert to numpy arrays
        X_sequences = np.array(sequences_X)
        y_sequences = np.array(sequences_y)
        lengths = np.array(lengths)
        seq_timestamps = np.array(seq_timestamps)
        
        return X_sequences, y_sequences, lengths, seq_timestamps
    
    def fit_transform(
        self,
        X_sequences: np.ndarray,
        y_sequences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scalers and transform data"""
        
        if self.config.scaling_method == ScalingMethod.NONE:
            self.is_fitted = True
            return X_sequences, y_sequences
        
        # Fit scaler on training data
        if self.config.feature_wise_scaling:
            # Reshape for feature-wise scaling
            n_samples, seq_len, n_features = X_sequences.shape
            X_reshaped = X_sequences.reshape(-1, n_features)
            
            # Fit and transform
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X_sequences = X_scaled.reshape(n_samples, seq_len, n_features)
        else:
            # Global scaling
            X_flat = X_sequences.reshape(-1, 1)
            X_scaled = self.scaler.fit_transform(X_flat)
            X_sequences = X_scaled.reshape(X_sequences.shape)
        
        self.is_fitted = True
        logger.info("Fitted data scalers")
        
        return X_sequences, y_sequences
    
    def transform(
        self,
        X_sequences: np.ndarray,
        y_sequences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted scalers"""
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        if self.config.scaling_method == ScalingMethod.NONE:
            return X_sequences, y_sequences
        
        # Transform using fitted scaler
        if self.config.feature_wise_scaling:
            n_samples, seq_len, n_features = X_sequences.shape
            X_reshaped = X_sequences.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_reshaped)
            X_sequences = X_scaled.reshape(n_samples, seq_len, n_features)
        else:
            X_flat = X_sequences.reshape(-1, 1)
            X_scaled = self.scaler.transform(X_flat)
            X_sequences = X_scaled.reshape(X_sequences.shape)
        
        return X_sequences, y_sequences
    
    def apply_augmentation(
        self,
        X_sequences: np.ndarray,
        y_sequences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to sequences"""
        
        if self.config.augmentation_ratio <= 0:
            return X_sequences, y_sequences
        
        n_samples = len(X_sequences)
        n_augmented = int(n_samples * self.config.augmentation_ratio)
        
        if n_augmented == 0:
            return X_sequences, y_sequences
        
        # Select random sequences to augment
        indices = np.random.choice(n_samples, n_augmented, replace=True)
        
        augmented_X = []
        augmented_y = []
        
        for idx in indices:
            X_seq = X_sequences[idx].copy()
            y_seq = y_sequences[idx]
            
            # Apply random augmentation method
            method = np.random.choice(self.config.augmentation_methods)
            
            if method == AugmentationMethod.NOISE:
                X_seq = self._add_noise(X_seq)
            elif method == AugmentationMethod.JITTER:
                X_seq = self._add_jitter(X_seq)
            elif method == AugmentationMethod.SCALING:
                X_seq = self._apply_scaling(X_seq)
            elif method == AugmentationMethod.MAGNITUDE_WARPING:
                X_seq = self._magnitude_warping(X_seq)
            
            augmented_X.append(X_seq)
            augmented_y.append(y_seq)
        
        # Combine original and augmented data
        X_combined = np.concatenate([X_sequences, np.array(augmented_X)], axis=0)
        y_combined = np.concatenate([y_sequences, np.array(augmented_y)], axis=0)
        
        logger.info(f"Applied augmentation: {len(X_sequences)} -> {len(X_combined)} sequences")
        
        return X_combined, y_combined
    
    def _add_noise(self, X_seq: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to sequence"""
        noise = np.random.normal(0, self.config.noise_std, X_seq.shape)
        return X_seq + noise
    
    def _add_jitter(self, X_seq: np.ndarray) -> np.ndarray:
        """Add temporal jitter to sequence"""
        seq_len, n_features = X_seq.shape
        
        # Create small random shifts
        shifts = np.random.normal(0, self.config.jitter_std, (seq_len, n_features))
        
        return X_seq + shifts
    
    def _apply_scaling(self, X_seq: np.ndarray) -> np.ndarray:
        """Apply random scaling to sequence"""
        scale_factor = np.random.uniform(*self.config.scaling_range)
        return X_seq * scale_factor
    
    def _magnitude_warping(self, X_seq: np.ndarray) -> np.ndarray:
        """Apply magnitude warping to sequence"""
        seq_len, n_features = X_seq.shape
        
        # Create smooth warping curve
        warp_steps = np.random.randint(2, 6)
        warp_points = np.random.uniform(0.8, 1.2, warp_steps)
        warp_curve = np.interp(np.linspace(0, 1, seq_len), 
                              np.linspace(0, 1, warp_steps), 
                              warp_points)
        
        return X_seq * warp_curve[:, np.newaxis]
    
    def time_series_split(
        self,
        X_sequences: np.ndarray,
        y_sequences: np.ndarray,
        timestamps: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Split time series data into train/validation/test sets.
        
        Maintains temporal order and applies gap between sets.
        """
        n_samples = len(X_sequences)
        
        # Calculate split indices
        test_size = int(n_samples * self.config.test_size)
        val_size = int(n_samples * self.config.validation_size)
        
        # Apply gaps
        test_start = n_samples - test_size
        val_end = test_start - self.config.gap_size
        val_start = val_end - val_size
        train_end = val_start - self.config.gap_size
        
        # Ensure we have enough data
        if train_end <= 0:
            raise ValueError("Not enough data for train/val/test split with specified sizes and gaps")
        
        # Create splits
        train_indices = slice(0, train_end)
        val_indices = slice(val_start, val_end)
        test_indices = slice(test_start, n_samples)
        
        train_data = (
            X_sequences[train_indices],
            y_sequences[train_indices],
            timestamps[train_indices]
        )
        
        val_data = (
            X_sequences[val_indices],
            y_sequences[val_indices],
            timestamps[val_indices]
        )
        
        test_data = (
            X_sequences[test_indices],
            y_sequences[test_indices],
            timestamps[test_indices]
        )
        
        logger.info(f"Split data: Train={len(train_data[0])}, Val={len(val_data[0])}, Test={len(test_data[0])}")
        
        return train_data, val_data, test_data
    
    def create_data_loader(
        self,
        X_sequences: np.ndarray,
        y_sequences: np.ndarray,
        lengths: Optional[np.ndarray] = None,
        shuffle: bool = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """
        Create batch iterator for training.
        
        Args:
            X_sequences: Input sequences
            y_sequences: Target sequences
            lengths: Sequence lengths (for variable length support)
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of (X_batch, y_batch, lengths_batch)
        """
        if shuffle is None:
            shuffle = self.config.shuffle_sequences
        
        n_samples = len(X_sequences)
        batch_size = self.config.batch_size
        
        # Create indices
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X_sequences[batch_indices]
            y_batch = y_sequences[batch_indices]
            lengths_batch = lengths[batch_indices] if lengths is not None else None
            
            yield X_batch, y_batch, lengths_batch
    
    def get_sequence_stats(self, X_sequences: np.ndarray) -> Dict[str, Any]:
        """Get statistics about the sequence data"""
        n_samples, seq_len, n_features = X_sequences.shape
        
        stats = {
            "n_sequences": n_samples,
            "sequence_length": seq_len,
            "n_features": n_features,
            "total_timesteps": n_samples * seq_len,
            "feature_means": np.mean(X_sequences, axis=(0, 1)),
            "feature_stds": np.std(X_sequences, axis=(0, 1)),
            "missing_ratio": np.isnan(X_sequences).sum() / X_sequences.size
        }
        
        return stats


def create_lstm_data_pipeline(
    sequence_length: int = 30,
    overlap_ratio: float = 0.8,
    scaling_method: ScalingMethod = ScalingMethod.STANDARD,
    augmentation_ratio: float = 0.1,
    batch_size: int = 32
) -> LSTMDataPipeline:
    """
    Factory function to create LSTM data pipeline with common configurations.
    
    Args:
        sequence_length: Length of sequences to create
        overlap_ratio: Overlap between consecutive sequences
        scaling_method: Method for scaling features
        augmentation_ratio: Ratio of augmented data to add
        batch_size: Batch size for training
        
    Returns:
        Configured LSTMDataPipeline instance
    """
    config = SequenceConfig(
        sequence_length=sequence_length,
        overlap_ratio=overlap_ratio,
        scaling_method=scaling_method,
        augmentation_ratio=augmentation_ratio,
        batch_size=batch_size
    )
    
    return LSTMDataPipeline(config)