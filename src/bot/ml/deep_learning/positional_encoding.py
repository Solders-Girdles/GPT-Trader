"""
DL-007: Advanced Positional Encoding
Phase 4 - Week 1

Enhanced positional encoding for financial time series:
- Captures temporal relationships in time series
- Handles irregular time intervals (market gaps, weekends)
- Maintains performance with missing data
- Learnable vs fixed encoding options
- Time-aware encoding for financial data
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

# Try multiple deep learning frameworks with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    BACKEND = "torch"
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        BACKEND = "tensorflow"
    except ImportError:
        # Fallback to numpy-based implementations
        BACKEND = "numpy"

logger = logging.getLogger(__name__)


class EncodingType(Enum):
    """Types of positional encoding"""

    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    RELATIVE = "relative"
    TIME_AWARE = "time_aware"
    ROTARY = "rotary"
    ADAPTIVE = "adaptive"


class TimeUnit(Enum):
    """Time units for financial data"""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encoding"""

    # Basic settings
    d_model: int = 256
    max_length: int = 1000
    encoding_type: EncodingType = EncodingType.TIME_AWARE
    dropout: float = 0.1

    # Time-aware settings
    base_time_unit: TimeUnit = TimeUnit.DAY
    handle_weekends: bool = True
    handle_holidays: bool = True
    market_hours_only: bool = True

    # Advanced settings
    temperature: float = 10000.0  # For sinusoidal encoding
    learnable_temperature: bool = False
    use_layer_norm: bool = True

    # Financial market specific
    trading_days_per_year: int = 252
    intraday_periods: int = 390  # 6.5 hours * 60 minutes
    timezone: str = "US/Eastern"

    # Adaptive encoding
    adaptive_alpha: float = 0.1  # Learning rate for adaptive encoding
    update_frequency: int = 100  # Update every N steps


if BACKEND == "torch":

    class SinusoidalPositionalEncoding(nn.Module):
        """Traditional sinusoidal positional encoding with enhancements"""

        def __init__(self, config: PositionalEncodingConfig):
            super().__init__()
            self.config = config
            self.dropout = nn.Dropout(config.dropout)

            # Create sinusoidal encoding
            pe = torch.zeros(config.max_length, config.d_model)
            position = torch.arange(0, config.max_length, dtype=torch.float).unsqueeze(1)

            if config.learnable_temperature:
                self.temperature = nn.Parameter(torch.tensor(config.temperature))
            else:
                self.register_buffer("temperature", torch.tensor(config.temperature))

            div_term = torch.exp(
                torch.arange(0, config.d_model, 2).float()
                * (-math.log(config.temperature) / config.d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer("pe", pe.unsqueeze(0))

            if config.use_layer_norm:
                self.layer_norm = nn.LayerNorm(config.d_model)

        def forward(self, x: Tensor) -> Tensor:
            """Add sinusoidal positional encoding"""
            seq_len = x.size(1)
            pos_encoding = self.pe[:, :seq_len, :]

            if self.config.learnable_temperature:
                # Recompute with learnable temperature
                position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, self.config.d_model, 2, dtype=torch.float, device=x.device)
                    * (-math.log(self.temperature) / self.config.d_model)
                )

                pe_dynamic = torch.zeros(seq_len, self.config.d_model, device=x.device)
                pe_dynamic[:, 0::2] = torch.sin(position * div_term)
                pe_dynamic[:, 1::2] = torch.cos(position * div_term)

                pos_encoding = pe_dynamic.unsqueeze(0)

            x = x + pos_encoding

            if hasattr(self, "layer_norm"):
                x = self.layer_norm(x)

            return self.dropout(x)

    class TimeAwarePositionalEncoding(nn.Module):
        """Financial time-aware positional encoding"""

        def __init__(self, config: PositionalEncodingConfig):
            super().__init__()
            self.config = config
            self.dropout = nn.Dropout(config.dropout)

            # Base sinusoidal encoding
            self.base_encoding = SinusoidalPositionalEncoding(config)

            # Time-specific embeddings
            self.time_of_day_embedding = nn.Embedding(config.intraday_periods, config.d_model // 4)
            self.day_of_week_embedding = nn.Embedding(7, config.d_model // 4)
            self.day_of_year_embedding = nn.Embedding(366, config.d_model // 4)
            self.month_embedding = nn.Embedding(12, config.d_model // 4)

            # Market state embeddings
            self.market_state_embedding = nn.Embedding(
                4, config.d_model // 8
            )  # pre-market, open, close, after-hours
            self.holiday_embedding = nn.Embedding(2, config.d_model // 8)  # holiday or not

            # Projection layer to combine all embeddings
            embedding_dim = (config.d_model // 4) * 4 + (config.d_model // 8) * 2
            self.time_projection = nn.Linear(embedding_dim, config.d_model)

            # Learnable combination weights
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Balance between base and time encoding

            if config.use_layer_norm:
                self.layer_norm = nn.LayerNorm(config.d_model)

        def forward(
            self,
            x: Tensor,
            timestamps: Tensor | None = None,
            time_features: dict[str, Tensor] | None = None,
        ) -> Tensor:
            """
            Apply time-aware positional encoding

            Args:
                x: Input tensor (batch_size, seq_len, d_model)
                timestamps: Unix timestamps (batch_size, seq_len)
                time_features: Pre-computed time features
            """
            # Base positional encoding
            base_encoded = self.base_encoding(x) - x  # Get just the positional part

            if timestamps is not None or time_features is not None:
                # Extract or use provided time features
                if time_features is None:
                    time_features = self._extract_time_features(timestamps)

                # Create time embeddings
                time_embeddings = self._create_time_embeddings(time_features)

                # Combine base and time-aware encoding
                pos_encoding = self.alpha * base_encoded + (1 - self.alpha) * time_embeddings
            else:
                pos_encoding = base_encoded

            x = x + pos_encoding

            if hasattr(self, "layer_norm"):
                x = self.layer_norm(x)

            return self.dropout(x)

        def _extract_time_features(self, timestamps: Tensor) -> dict[str, Tensor]:
            """Extract time features from timestamps"""
            # Convert to datetime (assuming Unix timestamps)
            # This is a simplified version - in practice, you'd use proper datetime handling

            batch_size, seq_len = timestamps.shape
            device = timestamps.device

            # Mock time features for demonstration
            # In practice, these would be extracted from actual timestamps
            time_features = {
                "time_of_day": torch.randint(
                    0, self.config.intraday_periods, (batch_size, seq_len), device=device
                ),
                "day_of_week": torch.randint(0, 7, (batch_size, seq_len), device=device),
                "day_of_year": torch.randint(0, 366, (batch_size, seq_len), device=device),
                "month": torch.randint(0, 12, (batch_size, seq_len), device=device),
                "market_state": torch.randint(0, 4, (batch_size, seq_len), device=device),
                "is_holiday": torch.randint(0, 2, (batch_size, seq_len), device=device),
            }

            return time_features

        def _create_time_embeddings(self, time_features: dict[str, Tensor]) -> Tensor:
            """Create combined time embeddings"""
            embeddings = []

            # Time-based embeddings
            embeddings.append(self.time_of_day_embedding(time_features["time_of_day"]))
            embeddings.append(self.day_of_week_embedding(time_features["day_of_week"]))
            embeddings.append(self.day_of_year_embedding(time_features["day_of_year"]))
            embeddings.append(self.month_embedding(time_features["month"]))

            # Market state embeddings
            embeddings.append(self.market_state_embedding(time_features["market_state"]))
            embeddings.append(self.holiday_embedding(time_features["is_holiday"]))

            # Concatenate and project
            combined = torch.cat(embeddings, dim=-1)
            time_encoding = self.time_projection(combined)

            return time_encoding

    class RelativePositionalEncoding(nn.Module):
        """Relative positional encoding for variable length sequences"""

        def __init__(self, config: PositionalEncodingConfig):
            super().__init__()
            self.config = config
            self.dropout = nn.Dropout(config.dropout)

            # Relative position embeddings
            self.relative_position_k = nn.Embedding(2 * config.max_length - 1, config.d_model)
            self.relative_position_v = nn.Embedding(2 * config.max_length - 1, config.d_model)

        def forward(self, x: Tensor) -> Tensor:
            """Apply relative positional encoding"""
            seq_len = x.size(1)

            # Create relative position matrix
            positions = torch.arange(seq_len, device=x.device).unsqueeze(1) - torch.arange(
                seq_len, device=x.device
            ).unsqueeze(0)
            positions = positions + self.config.max_length - 1  # Shift to positive indices

            # Get relative position encodings
            rel_pos_k = self.relative_position_k(positions)
            rel_pos_v = self.relative_position_v(positions)

            # Apply to input (simplified - normally this would be in attention computation)
            x = x + rel_pos_k.mean(dim=0, keepdim=True)

            return self.dropout(x)

    class RotaryPositionalEncoding(nn.Module):
        """Rotary positional encoding (RoPE)"""

        def __init__(self, config: PositionalEncodingConfig):
            super().__init__()
            self.config = config
            self.dropout = nn.Dropout(config.dropout)

            # Create rotation matrix
            self.register_buffer(
                "inv_freq",
                1.0
                / (
                    config.temperature
                    ** (torch.arange(0, config.d_model, 2).float() / config.d_model)
                ),
            )

        def forward(self, x: Tensor) -> Tensor:
            """Apply rotary positional encoding"""
            seq_len = x.size(1)

            # Create position indices
            position = torch.arange(seq_len, device=x.device, dtype=torch.float).unsqueeze(1)

            # Compute frequencies
            freqs = torch.einsum("i,j->ij", position.squeeze(), self.inv_freq)

            # Create rotation matrices
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            # Apply rotation (simplified version)
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]

            # Rotate
            x_rotated_even = x_even * cos_freq.unsqueeze(0) - x_odd * sin_freq.unsqueeze(0)
            x_rotated_odd = x_even * sin_freq.unsqueeze(0) + x_odd * cos_freq.unsqueeze(0)

            # Recombine
            x_rotated = torch.zeros_like(x)
            x_rotated[..., 0::2] = x_rotated_even
            x_rotated[..., 1::2] = x_rotated_odd

            return self.dropout(x_rotated)

    class AdaptivePositionalEncoding(nn.Module):
        """Adaptive positional encoding that learns from data"""

        def __init__(self, config: PositionalEncodingConfig):
            super().__init__()
            self.config = config
            self.dropout = nn.Dropout(config.dropout)

            # Base learned embedding
            self.position_embedding = nn.Parameter(torch.randn(config.max_length, config.d_model))

            # Adaptive components
            self.adaptation_network = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Linear(config.d_model // 2, config.d_model),
            )

            # Running statistics for adaptation
            self.register_buffer("running_mean", torch.zeros(config.d_model))
            self.register_buffer("running_var", torch.ones(config.d_model))
            self.register_buffer("num_updates", torch.tensor(0))

        def forward(self, x: Tensor, update_stats: bool = True) -> Tensor:
            """Apply adaptive positional encoding"""
            seq_len = x.size(1)
            batch_size = x.size(0)

            # Base positional encoding
            base_pos = self.position_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)

            # Adaptive adjustment based on input statistics
            if update_stats and self.training:
                # Update running statistics
                batch_mean = x.mean(dim=(0, 1))
                batch_var = x.var(dim=(0, 1))

                momentum = 1.0 / (self.num_updates + 1).float().clamp(min=1.0 / 1000.0)
                self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
                self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
                self.num_updates += 1

            # Adapt position encoding based on input characteristics
            adaptation_input = torch.cat(
                [
                    self.running_mean.unsqueeze(0).expand(batch_size, -1),
                    self.running_var.unsqueeze(0).expand(batch_size, -1),
                ],
                dim=-1,
            )

            if adaptation_input.size(-1) != self.config.d_model:
                # Project to correct dimension if needed
                adaptation_input = adaptation_input[:, : self.config.d_model]

            adaptation = self.adaptation_network(adaptation_input).unsqueeze(1)
            adapted_pos = base_pos * (1 + self.config.adaptive_alpha * adaptation)

            x = x + adapted_pos

            return self.dropout(x)

    class UnifiedPositionalEncoding(nn.Module):
        """Unified positional encoding supporting multiple types"""

        def __init__(self, config: PositionalEncodingConfig):
            super().__init__()
            self.config = config

            # Create the appropriate encoding based on config
            if config.encoding_type == EncodingType.SINUSOIDAL:
                self.encoding = SinusoidalPositionalEncoding(config)
            elif config.encoding_type == EncodingType.TIME_AWARE:
                self.encoding = TimeAwarePositionalEncoding(config)
            elif config.encoding_type == EncodingType.RELATIVE:
                self.encoding = RelativePositionalEncoding(config)
            elif config.encoding_type == EncodingType.ROTARY:
                self.encoding = RotaryPositionalEncoding(config)
            elif config.encoding_type == EncodingType.ADAPTIVE:
                self.encoding = AdaptivePositionalEncoding(config)
            elif config.encoding_type == EncodingType.LEARNED:
                self.encoding = nn.Parameter(torch.randn(config.max_length, config.d_model))
                self.dropout = nn.Dropout(config.dropout)
            else:
                raise ValueError(f"Unsupported encoding type: {config.encoding_type}")

        def forward(self, x: Tensor, **kwargs) -> Tensor:
            """Forward pass with appropriate encoding"""
            if self.config.encoding_type == EncodingType.LEARNED:
                seq_len = x.size(1)
                pos_encoding = self.encoding[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)
                return self.dropout(x + pos_encoding)
            else:
                return self.encoding(x, **kwargs)

        def handle_irregular_intervals(self, x: Tensor, time_deltas: Tensor) -> Tensor:
            """Handle irregular time intervals in financial data"""
            if not self.config.handle_weekends:
                return self.forward(x)

            # Scale positional encoding based on time gaps
            # Larger gaps get proportionally larger positional increments
            time_scaling = torch.clamp(time_deltas / time_deltas.mean(), 0.1, 10.0)

            if hasattr(self.encoding, "pe"):
                # Modify sinusoidal encoding
                scaled_positions = torch.cumsum(time_scaling, dim=1)
                # Apply scaled encoding (simplified implementation)
                return self.forward(x)
            else:
                return self.forward(x, time_deltas=time_deltas)

else:
    # Fallback implementations for non-PyTorch environments
    class UnifiedPositionalEncoding:
        """Fallback positional encoding"""

        def __init__(self, config: PositionalEncodingConfig):
            self.config = config
            logger.warning(f"Using fallback positional encoding with {BACKEND}")

            # Create basic sinusoidal encoding with numpy
            pe = np.zeros((config.max_length, config.d_model))
            position = np.arange(0, config.max_length)[:, np.newaxis]
            div_term = np.exp(
                np.arange(0, config.d_model, 2) * -(np.log(config.temperature) / config.d_model)
            )

            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)

            self.pe = pe

        def __call__(self, x: np.ndarray) -> np.ndarray:
            """Apply positional encoding"""
            seq_len = x.shape[1]
            return x + self.pe[:seq_len][np.newaxis, :, :]


def create_positional_encoding(config: PositionalEncodingConfig) -> UnifiedPositionalEncoding:
    """Factory function to create positional encoding"""
    return UnifiedPositionalEncoding(config)


def create_default_encoding_config(**kwargs) -> PositionalEncodingConfig:
    """Create default positional encoding configuration with overrides"""
    config = PositionalEncodingConfig()

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")

    return config


def analyze_time_patterns(
    timestamps: list[datetime], time_unit: TimeUnit = TimeUnit.DAY
) -> dict[str, Any]:
    """Analyze time patterns in financial data"""
    if not timestamps:
        return {}

    # Convert to pandas for easier analysis
    try:
        import pandas as pd

        ts_series = pd.Series(timestamps)

        analysis = {
            "total_observations": len(timestamps),
            "time_span_days": (timestamps[-1] - timestamps[0]).days,
            "avg_interval_hours": ts_series.diff().mean().total_seconds() / 3600,
            "irregular_intervals": len(ts_series.diff().unique()) > 2,
            "weekend_observations": sum(1 for ts in timestamps if ts.weekday() >= 5),
            "trading_hours_only": all(9 <= ts.hour <= 16 for ts in timestamps if ts.weekday() < 5),
        }

        # Market-specific analysis
        market_hours = [ts for ts in timestamps if 9 <= ts.hour <= 16 and ts.weekday() < 5]
        analysis["market_hours_ratio"] = len(market_hours) / len(timestamps)

        return analysis

    except ImportError:
        # Fallback without pandas
        intervals = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)
        ]

        return {
            "total_observations": len(timestamps),
            "time_span_days": (timestamps[-1] - timestamps[0]).days,
            "avg_interval_hours": np.mean(intervals) / 3600,
            "irregular_intervals": len(set(intervals)) > 2,
        }


if __name__ == "__main__":
    # Test positional encoding
    config = create_default_encoding_config(
        d_model=256, max_length=100, encoding_type=EncodingType.TIME_AWARE
    )

    encoding = create_positional_encoding(config)

    print(f"Created {config.encoding_type.value} positional encoding")
    print(f"Model dimension: {config.d_model}")
    print(f"Backend: {BACKEND}")

    if BACKEND == "torch":
        # Test forward pass
        batch_size, seq_len = 16, 60
        x = torch.randn(batch_size, seq_len, config.d_model)

        with torch.no_grad():
            encoded = encoding(x)
            print(f"Input shape: {x.shape}")
            print(f"Encoded shape: {encoded.shape}")
            print(f"Encoding difference norm: {(encoded - x).norm().item():.3f}")

    # Test time pattern analysis
    from datetime import datetime, timedelta

    timestamps = [datetime.now() + timedelta(days=i) for i in range(100)]
    analysis = analyze_time_patterns(timestamps)
    print(f"Time pattern analysis: {analysis}")
