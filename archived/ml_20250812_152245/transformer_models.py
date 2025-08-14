"""
Transformer Models for Market Prediction

State-of-the-art transformer architectures for financial time series:
- Market-aware attention mechanisms
- Multi-asset correlation modeling
- Temporal pattern recognition
- High-frequency trading signals
- Sentiment and news integration
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Transformer models will be limited.")


@dataclass
class TransformerConfig:
    """Configuration for Transformer models"""

    d_model: int = 512  # Model dimension
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 6  # Number of transformer layers
    d_ff: int = 2048  # Feed-forward dimension
    max_seq_length: int = 100  # Maximum sequence length
    dropout: float = 0.1
    learning_rate: float = 0.0001
    warmup_steps: int = 4000
    batch_size: int = 32
    epochs: int = 100

    # Market-specific configurations
    n_assets: int = 1  # Number of assets to model
    n_features: int = 10  # Features per asset
    prediction_horizon: int = 1  # Steps ahead to predict
    use_positional_encoding: bool = True
    use_temporal_encoding: bool = True  # Time-aware encoding


if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer"""

        def __init__(self, d_model: int, max_len: int = 5000) -> None:
            super().__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer("pe", pe)

        def forward(self, x):
            return x + self.pe[: x.size(0), :]

    class TemporalEncoding(nn.Module):
        """Time-aware encoding for financial data"""

        def __init__(self, d_model: int) -> None:
            super().__init__()
            # Encodings for different time scales
            self.minute_encoding = nn.Embedding(60, d_model // 4)
            self.hour_encoding = nn.Embedding(24, d_model // 4)
            self.day_encoding = nn.Embedding(7, d_model // 4)
            self.month_encoding = nn.Embedding(12, d_model // 4)

        def forward(self, x, timestamps):
            """
            Add temporal information to embeddings
            timestamps: (batch, seq_len, 4) - [minute, hour, day, month]
            """
            _batch_size, _seq_len = x.size(0), x.size(1)

            # Extract time components
            minutes = timestamps[..., 0].long()
            hours = timestamps[..., 1].long()
            days = timestamps[..., 2].long()
            months = timestamps[..., 3].long()

            # Get embeddings
            minute_emb = self.minute_encoding(minutes)
            hour_emb = self.hour_encoding(hours)
            day_emb = self.day_encoding(days)
            month_emb = self.month_encoding(months)

            # Concatenate and project
            temporal_emb = torch.cat([minute_emb, hour_emb, day_emb, month_emb], dim=-1)

            return x + temporal_emb

    class MarketAttention(nn.Module):
        """Market-aware multi-head attention"""

        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)

            # Market-specific attention bias
            self.volume_attention = nn.Linear(1, n_heads)
            self.volatility_attention = nn.Linear(1, n_heads)

        def forward(self, query, key, value, mask=None, market_features=None):
            batch_size = query.size(0)

            # Linear transformations and split into heads
            Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Add market-aware attention bias if provided
            if market_features is not None:
                volume_bias = self.volume_attention(market_features["volume"].unsqueeze(-1))
                volatility_bias = self.volatility_attention(
                    market_features["volatility"].unsqueeze(-1)
                )
                market_bias = (volume_bias + volatility_bias).transpose(1, 2).unsqueeze(-1)
                scores = scores + market_bias

            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            # Softmax and dropout
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Apply attention to values
            context = torch.matmul(attention_weights, V)

            # Concatenate heads
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

            # Final linear transformation
            output = self.W_o(context)

            return output, attention_weights

    class TransformerBlock(nn.Module):
        """Transformer encoder block with market-aware attention"""

        def __init__(self, config: TransformerConfig) -> None:
            super().__init__()

            self.attention = MarketAttention(config.d_model, config.n_heads, config.dropout)
            self.norm1 = nn.LayerNorm(config.d_model)
            self.norm2 = nn.LayerNorm(config.d_model)

            self.feed_forward = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout),
            )

        def forward(self, x, mask=None, market_features=None):
            # Self-attention with residual connection
            attn_output, attn_weights = self.attention(x, x, x, mask, market_features)
            x = self.norm1(x + attn_output)

            # Feed-forward with residual connection
            ff_output = self.feed_forward(x)
            x = self.norm2(x + ff_output)

            return x, attn_weights

    class MarketTransformer(nn.Module):
        """Transformer model for market prediction"""

        def __init__(self, config: TransformerConfig) -> None:
            super().__init__()
            self.config = config

            # Input projection
            self.input_projection = nn.Linear(config.n_features * config.n_assets, config.d_model)

            # Positional encoding
            if config.use_positional_encoding:
                self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)

            # Temporal encoding
            if config.use_temporal_encoding:
                self.temporal_encoding = TemporalEncoding(config.d_model)

            # Transformer layers
            self.transformer_blocks = nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.n_layers)]
            )

            # Output heads
            self.prediction_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, config.n_assets * 3),  # Buy/Hold/Sell per asset
            )

            self.regression_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, config.n_assets),  # Price prediction per asset
            )

            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x, timestamps=None, mask=None, market_features=None):
            """
            Forward pass through transformer

            Args:
                x: (batch, seq_len, n_features * n_assets)
                timestamps: (batch, seq_len, 4) - temporal information
                mask: (batch, seq_len) - attention mask
                market_features: dict with volume, volatility, etc.

            Returns:
                predictions: (batch, n_assets * 3) - trading signals
                price_pred: (batch, n_assets) - price predictions
                attention_weights: list of attention weight matrices
            """
            # Input projection
            x = self.input_projection(x)
            x = self.dropout(x)

            # Add positional encoding
            if self.config.use_positional_encoding:
                x = x.transpose(0, 1)  # (seq_len, batch, d_model)
                x = self.pos_encoding(x)
                x = x.transpose(0, 1)  # (batch, seq_len, d_model)

            # Add temporal encoding
            if self.config.use_temporal_encoding and timestamps is not None:
                x = self.temporal_encoding(x, timestamps)

            # Pass through transformer blocks
            attention_weights = []
            for transformer in self.transformer_blocks:
                x, attn = transformer(x, mask, market_features)
                attention_weights.append(attn)

            # Global pooling (use last token for prediction)
            x = x[:, -1, :]  # (batch, d_model)

            # Generate predictions
            predictions = self.prediction_head(x)
            price_pred = self.regression_head(x)

            return predictions, price_pred, attention_weights

    class MarketDataset(torch.utils.data.Dataset):
        """Dataset for market data with temporal information"""

        def __init__(
            self,
            data: pd.DataFrame,
            sequence_length: int = 60,
            prediction_horizon: int = 1,
            features: list[str] = None,
        ) -> None:
            self.data = data
            self.sequence_length = sequence_length
            self.prediction_horizon = prediction_horizon
            self.features = features or ["Open", "High", "Low", "Close", "Volume"]

            # Normalize data
            self.feature_data = data[self.features].values
            self.mean = self.feature_data.mean(axis=0)
            self.std = self.feature_data.std(axis=0) + 1e-8
            self.feature_data = (self.feature_data - self.mean) / self.std

            # Extract temporal features if datetime index
            self.timestamps = None
            if isinstance(data.index, pd.DatetimeIndex):
                self.timestamps = np.array(
                    [
                        data.index.minute,
                        data.index.hour,
                        data.index.dayofweek,
                        data.index.month - 1,  # 0-indexed
                    ]
                ).T

        def __len__(self) -> int:
            return len(self.data) - self.sequence_length - self.prediction_horizon + 1

        def __getitem__(self, idx):
            # Get sequence
            seq_data = self.feature_data[idx : idx + self.sequence_length]

            # Get target (next price movement)
            current_price = self.data["Close"].iloc[idx + self.sequence_length - 1]
            future_price = self.data["Close"].iloc[
                idx + self.sequence_length - 1 + self.prediction_horizon
            ]

            # Classification target
            price_change = (future_price - current_price) / current_price
            if price_change > 0.001:  # 0.1% threshold
                target_class = 2  # Buy
            elif price_change < -0.001:
                target_class = 0  # Sell
            else:
                target_class = 1  # Hold

            # Regression target (normalized price change)
            target_price = price_change

            # Get timestamps if available
            seq_timestamps = None
            if self.timestamps is not None:
                seq_timestamps = self.timestamps[idx : idx + self.sequence_length]

            return {
                "sequence": torch.FloatTensor(seq_data),
                "timestamps": (
                    torch.LongTensor(seq_timestamps)
                    if seq_timestamps is not None
                    else torch.zeros(self.sequence_length, 4)
                ),
                "target_class": torch.LongTensor([target_class]),
                "target_price": torch.FloatTensor([target_price]),
            }

else:
    # Fallback classes when PyTorch is not available
    class TransformerBlock:
        def __init__(self, config: TransformerConfig) -> None:
            warnings.warn("PyTorch not available, TransformerBlock is a stub")

    class MarketTransformer:
        def __init__(self, config: TransformerConfig) -> None:
            self.config = config
            warnings.warn("PyTorch not available, MarketTransformer is a stub")


class TransformerTrader:
    """
    High-level interface for transformer-based trading strategies.

    Features:
    - Multi-asset modeling
    - Attention visualization
    - Real-time inference
    - Risk-aware predictions
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        self.config = config or TransformerConfig()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.device = None

        if TORCH_AVAILABLE:
            # Set device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

            self.logger.info(f"Using device: {self.device}")
        else:
            self.logger.warning("PyTorch not available, transformer features limited")

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Train transformer model on market data.

        Args:
            train_data: Training market data
            val_data: Validation market data
            features: Feature columns to use

        Returns:
            Training history and metrics
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        # Create datasets
        train_dataset = MarketDataset(
            train_data, self.config.max_seq_length, self.config.prediction_horizon, features
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0
        )

        val_loader = None
        if val_data is not None:
            val_dataset = MarketDataset(
                val_data, self.config.max_seq_length, self.config.prediction_horizon, features
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
            )

        # Initialize model
        self.model = MarketTransformer(self.config).to(self.device)

        # Optimizer with warmup
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.98), eps=1e-9
        )

        # Learning rate scheduler
        def lr_lambda(step):
            if step == 0:
                return 1
            d_model = self.config.d_model
            warmup = self.config.warmup_steps
            return min(step**-0.5, step * warmup**-1.5) * d_model**0.5

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Loss functions
        class_criterion = nn.CrossEntropyLoss()
        reg_criterion = nn.MSELoss()

        # Training loop
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_loss = float("inf")

        self.logger.info("Starting transformer training...")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                # Move to device
                sequences = batch["sequence"].to(self.device)
                timestamps = batch["timestamps"].to(self.device)
                target_class = batch["target_class"].squeeze().to(self.device)
                target_price = batch["target_price"].squeeze().to(self.device)

                # Forward pass
                predictions, price_pred, _ = self.model(sequences, timestamps)

                # Reshape predictions for multi-asset
                predictions = predictions.view(-1, 3)  # (batch * n_assets, 3)

                # Calculate loss
                class_loss = class_criterion(predictions, target_class)
                reg_loss = reg_criterion(price_pred.squeeze(), target_price)
                total_loss = class_loss + 0.1 * reg_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Track metrics
                train_loss += total_loss.item()
                _, predicted = torch.max(predictions, 1)
                train_correct += (predicted == target_class).sum().item()
                train_total += target_class.size(0)

            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        sequences = batch["sequence"].to(self.device)
                        timestamps = batch["timestamps"].to(self.device)
                        target_class = batch["target_class"].squeeze().to(self.device)
                        target_price = batch["target_price"].squeeze().to(self.device)

                        predictions, price_pred, _ = self.model(sequences, timestamps)
                        predictions = predictions.view(-1, 3)

                        class_loss = class_criterion(predictions, target_class)
                        reg_loss = reg_criterion(price_pred.squeeze(), target_price)
                        total_loss = class_loss + 0.1 * reg_loss

                        val_loss += total_loss.item()
                        _, predicted = torch.max(predictions, 1)
                        val_correct += (predicted == target_class).sum().item()
                        val_total += target_class.size(0)

                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total

                history["val_loss"].append(avg_val_loss)
                history["val_acc"].append(val_acc)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Save best model
                    torch.save(self.model.state_dict(), "best_transformer.pth")

            # Log progress
            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, "
                    f"val_loss={history['val_loss'][-1] if history['val_loss'] else 0:.4f}, "
                    f"val_acc={history['val_acc'][-1] if history['val_acc'] else 0:.3f}"
                )

        training_time = time.time() - start_time

        return {
            "history": history,
            "training_time": training_time,
            "best_val_loss": best_val_loss,
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        }

    def predict(self, data: pd.DataFrame, return_attention: bool = False) -> dict[str, Any]:
        """
        Generate predictions using trained transformer.

        Args:
            data: Market data for prediction
            return_attention: Whether to return attention weights

        Returns:
            Predictions and optional attention weights
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {"error": "Model not available"}

        self.model.eval()

        dataset = MarketDataset(data, self.config.max_seq_length, self.config.prediction_horizon)

        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )

        all_predictions = []
        all_price_preds = []
        all_attention_weights = []

        with torch.no_grad():
            for batch in loader:
                sequences = batch["sequence"].to(self.device)
                timestamps = batch["timestamps"].to(self.device)

                predictions, price_pred, attention_weights = self.model(sequences, timestamps)

                # Convert to probabilities
                predictions = F.softmax(predictions.view(-1, 3), dim=1)

                all_predictions.append(predictions.cpu().numpy())
                all_price_preds.append(price_pred.cpu().numpy())

                if return_attention:
                    # Average attention across heads and layers
                    avg_attention = torch.stack(attention_weights).mean(dim=(0, 2))
                    all_attention_weights.append(avg_attention.cpu().numpy())

        results = {
            "predictions": np.concatenate(all_predictions),
            "price_predictions": np.concatenate(all_price_preds),
        }

        if return_attention:
            results["attention_weights"] = np.concatenate(all_attention_weights)

        return results

    def analyze_attention(self, data: pd.DataFrame, sample_idx: int = 0) -> dict[str, Any]:
        """
        Analyze attention patterns for interpretability.

        Args:
            data: Market data
            sample_idx: Sample index to analyze

        Returns:
            Attention analysis results
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {"error": "Model not available"}

        # Get single sample
        dataset = MarketDataset(data, self.config.max_seq_length)
        sample = dataset[sample_idx]

        # Forward pass with attention
        self.model.eval()
        with torch.no_grad():
            sequences = sample["sequence"].unsqueeze(0).to(self.device)
            timestamps = sample["timestamps"].unsqueeze(0).to(self.device)

            _, _, attention_weights = self.model(sequences, timestamps)

        # Analyze attention patterns
        analysis = {}

        for layer_idx, layer_attention in enumerate(attention_weights):
            # Average across heads
            avg_attention = layer_attention.mean(dim=1).squeeze().cpu().numpy()

            # Find most attended positions
            top_positions = np.argsort(avg_attention[-1])[-5:]  # Top 5 for last position

            analysis[f"layer_{layer_idx}"] = {
                "avg_attention": avg_attention.tolist(),
                "top_attended_positions": top_positions.tolist(),
                "attention_entropy": -np.sum(
                    avg_attention * np.log(avg_attention + 1e-9), axis=-1
                ).mean(),
            }

        return analysis


def benchmark_transformer():
    """Benchmark transformer model performance"""
    print("🚀 Transformer Model Benchmark")
    print("=" * 50)

    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available. Install with: pip install torch")
        return

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 2000

    dates = pd.date_range(start="2022-01-01", periods=n_samples, freq="H")

    # Generate correlated market data
    returns = np.random.multivariate_normal(
        [0.0001, 0.0001], [[0.01, 0.005], [0.005, 0.01]], n_samples
    )

    prices = 100 * np.cumprod(1 + returns, axis=0)

    data = pd.DataFrame(
        {
            "Open": prices[:, 0] * np.random.uniform(0.99, 1.01, n_samples),
            "High": prices[:, 0] * np.random.uniform(1.01, 1.03, n_samples),
            "Low": prices[:, 0] * np.random.uniform(0.97, 0.99, n_samples),
            "Close": prices[:, 0],
            "Volume": np.random.lognormal(15, 0.5, n_samples),
        },
        index=dates,
    )

    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"📊 Data shape: {data.shape}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")

    # Configure small transformer for testing
    config = TransformerConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_length=30,
        batch_size=16,
        epochs=5,  # Quick test
        learning_rate=0.001,
    )

    # Initialize trader
    trader = TransformerTrader(config)

    # Train model
    print("\n🤖 Training Transformer model...")
    results = trader.train(train_data, val_data)

    print("\n📊 Training Results:")
    print(f"   Training time: {results['training_time']:.2f}s")
    print(f"   Final train accuracy: {results['final_train_acc']:.3f}")
    if results.get("final_val_acc"):
        print(f"   Final validation accuracy: {results['final_val_acc']:.3f}")
    print(f"   Best validation loss: {results['best_val_loss']:.4f}")

    # Generate predictions
    print("\n🔮 Generating predictions...")
    predictions = trader.predict(val_data, return_attention=True)

    if "predictions" in predictions:
        pred_classes = np.argmax(predictions["predictions"], axis=1)
        signal_distribution = np.bincount(pred_classes, minlength=3)

        print("   Signal distribution:")
        print(
            f"      Sell: {signal_distribution[0]} ({signal_distribution[0]/len(pred_classes):.1%})"
        )
        print(
            f"      Hold: {signal_distribution[1]} ({signal_distribution[1]/len(pred_classes):.1%})"
        )
        print(
            f"      Buy: {signal_distribution[2]} ({signal_distribution[2]/len(pred_classes):.1%})"
        )

    # Analyze attention
    print("\n🔍 Analyzing attention patterns...")
    attention_analysis = trader.analyze_attention(val_data)

    if attention_analysis and "layer_0" in attention_analysis:
        layer_0 = attention_analysis["layer_0"]
        print(f"   Layer 0 attention entropy: {layer_0['attention_entropy']:.3f}")
        print(f"   Top attended positions: {layer_0['top_attended_positions']}")

    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    benchmark_transformer()
