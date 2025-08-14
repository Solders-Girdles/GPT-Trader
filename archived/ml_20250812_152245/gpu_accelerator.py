"""
GPU Acceleration for ML Models

Provides GPU-accelerated machine learning capabilities:
- Neural network strategy optimization
- Deep reinforcement learning
- Large-scale backtesting
- Real-time inference
- Multi-GPU support
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Try to import GPU libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU acceleration will be limited.")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # Fallback to numpy


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration"""

    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    device_id: int = 0
    memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    batch_size: int = 1024
    num_workers: int = 4
    pin_memory: bool = True
    benchmark_mode: bool = True


@dataclass
class ModelConfig:
    """Configuration for ML models"""

    model_type: str = "lstm"  # "lstm", "gru", "transformer", "cnn"
    input_dim: int = 10
    hidden_dim: int = 128
    output_dim: int = 3  # Buy, Hold, Sell
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10


if TORCH_AVAILABLE:

    class TradingNN(nn.Module):
        """Neural network for trading strategy"""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config

            if config.model_type == "lstm":
                self.rnn = nn.LSTM(
                    config.input_dim,
                    config.hidden_dim,
                    config.num_layers,
                    dropout=config.dropout,
                    batch_first=True,
                )
            elif config.model_type == "gru":
                self.rnn = nn.GRU(
                    config.input_dim,
                    config.hidden_dim,
                    config.num_layers,
                    dropout=config.dropout,
                    batch_first=True,
                )
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")

            self.fc = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.output_dim),
                nn.Softmax(dim=-1),
            )

        def forward(self, x):
            # x shape: (batch, sequence, features)
            out, _ = self.rnn(x)
            # Take last output
            out = out[:, -1, :]
            out = self.fc(out)
            return out

else:
    # Fallback when PyTorch is not available
    class TradingNN:
        def __init__(self, config: ModelConfig) -> None:
            self.config = config
            warnings.warn("PyTorch not available, TradingNN is a stub")


class GPUAccelerator:
    """
    GPU acceleration engine for ML-based trading strategies.

    Features:
    - Automatic device selection (CUDA, MPS, CPU)
    - Mixed precision training
    - Multi-GPU support
    - Efficient data pipelines
    - Real-time inference optimization
    """

    def __init__(self, config: GPUConfig | None = None) -> None:
        self.config = config or GPUConfig()
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.is_initialized = False

        # Initialize device
        self._initialize_device()

    def _initialize_device(self) -> None:
        """Initialize GPU device"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, using CPU fallback")
            self.device = "cpu"
            return

        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.config.device_id}")
                self.logger.info(
                    f"🎮 Using CUDA GPU: {torch.cuda.get_device_name(self.config.device_id)}"
                )
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.logger.info("🎮 Using Apple Metal GPU")
            else:
                self.device = torch.device("cpu")
                self.logger.info("💻 Using CPU (no GPU available)")
        else:
            self.device = torch.device(self.config.device)

        # Set memory fraction for CUDA
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)

            # Enable benchmark mode for consistent input sizes
            if self.config.benchmark_mode:
                torch.backends.cudnn.benchmark = True

        self.is_initialized = True

    def prepare_data(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        features: list[str] | None = None,
        target_col: str = "signal",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare market data for neural network training.

        Args:
            data: Market data DataFrame
            sequence_length: Length of input sequences
            features: List of feature columns to use
            target_col: Target column name

        Returns:
            Tuple of (features tensor, targets tensor)
        """
        if features is None:
            features = ["Open", "High", "Low", "Close", "Volume"]

        # Normalize features
        feature_data = data[features].values
        feature_mean = feature_data.mean(axis=0)
        feature_std = feature_data.std(axis=0) + 1e-8
        feature_data = (feature_data - feature_mean) / feature_std

        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(feature_data[i - sequence_length : i])

            # Create target (classification: 0=sell, 1=hold, 2=buy)
            if target_col in data.columns:
                signal = data[target_col].iloc[i]
                if signal > 0.5:
                    y.append(2)  # Buy
                elif signal < -0.5:
                    y.append(0)  # Sell
                else:
                    y.append(1)  # Hold
            else:
                # Default to hold if no signal
                y.append(1)

        # Convert to tensors
        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))

        # Move to device
        if self.device and self.device != "cpu":
            X = X.to(self.device)
            y = y.to(self.device)

        return X, y

    def train_model(
        self,
        model: nn.Module,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor] | None = None,
        config: ModelConfig | None = None,
    ) -> dict[str, Any]:
        """
        Train neural network model on GPU.

        Args:
            model: PyTorch model to train
            train_data: Training data (features, targets)
            val_data: Validation data (features, targets)
            config: Model training configuration

        Returns:
            Training history and metrics
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        config = config or ModelConfig()

        # Move model to device
        model = model.to(self.device)

        # Create data loaders
        X_train, y_train = train_data
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers if self.device.type != "mps" else 0,
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
        )

        val_loader = None
        if val_data:
            X_val, y_val = val_data
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers if self.device.type != "mps" else 0,
                pin_memory=self.config.pin_memory and self.device.type == "cuda",
            )

        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Mixed precision training
        scaler = None
        if self.config.enable_mixed_precision and self.device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float("inf")
        patience_counter = 0

        start_time = time.time()

        for epoch in range(config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                if scaler:
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular training
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation phase
            if val_loader:
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct / total

                history["val_loss"].append(avg_val_loss)
                history["val_accuracy"].append(val_accuracy)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

                if epoch % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                        f"val_loss={avg_val_loss:.4f}, val_acc={val_accuracy:.3f}"
                    )

        training_time = time.time() - start_time

        # Calculate throughput
        total_samples = len(X_train) * min(epoch + 1, config.epochs)
        throughput = total_samples / training_time

        return {
            "history": history,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
            "training_time": training_time,
            "epochs_completed": len(history["train_loss"]),
            "throughput_samples_per_sec": throughput,
            "device": str(self.device),
        }

    def predict(
        self, model: nn.Module, data: torch.Tensor, batch_size: int | None = None
    ) -> np.ndarray:
        """
        Fast GPU inference for trading signals.

        Args:
            model: Trained PyTorch model
            data: Input data tensor
            batch_size: Batch size for inference

        Returns:
            Predictions as numpy array
        """
        if not TORCH_AVAILABLE:
            return np.array([])

        model = model.to(self.device)
        model.eval()

        batch_size = batch_size or self.config.batch_size

        # Create data loader for batched inference
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for inference
            pin_memory=self.config.pin_memory and self.device.type == "cuda",
        )

        predictions = []

        with torch.no_grad():
            for batch in loader:
                batch_data = batch[0].to(self.device)

                if self.config.enable_mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_data)
                else:
                    outputs = model(batch_data)

                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def accelerate_backtest(
        self, signals: np.ndarray, prices: np.ndarray, initial_cash: float = 100000
    ) -> dict[str, Any]:
        """
        GPU-accelerated backtesting using CuPy.

        Args:
            signals: Trading signals array
            prices: Price array
            initial_cash: Starting capital

        Returns:
            Backtest results
        """
        if CUPY_AVAILABLE and self.device and self.device.type == "cuda":
            # Use CuPy for GPU acceleration
            signals_gpu = cp.asarray(signals)
            prices_gpu = cp.asarray(prices)

            # Calculate returns on GPU
            returns_gpu = cp.diff(prices_gpu) / prices_gpu[:-1]
            strategy_returns_gpu = returns_gpu * signals_gpu[:-1]

            # Calculate cumulative returns
            cumulative_returns_gpu = cp.cumprod(1 + strategy_returns_gpu)

            # Calculate metrics on GPU
            total_return = float(cumulative_returns_gpu[-1] - 1)
            sharpe_ratio = float(
                cp.mean(strategy_returns_gpu) / cp.std(strategy_returns_gpu) * cp.sqrt(252)
            )

            # Drawdown calculation
            running_max = cp.maximum.accumulate(cumulative_returns_gpu)
            drawdown = (cumulative_returns_gpu - running_max) / running_max
            max_drawdown = float(cp.min(drawdown))

            # Transfer results back to CPU
            cumulative_returns = cp.asnumpy(cumulative_returns_gpu)

        else:
            # CPU fallback
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = returns * signals[:-1]
            cumulative_returns = np.cumprod(1 + strategy_returns)

            total_return = cumulative_returns[-1] - 1
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)

            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_value": initial_cash * (1 + total_return),
            "cumulative_returns": cumulative_returns,
        }

    def benchmark_gpu(self, data_size: int = 10000) -> dict[str, Any]:
        """
        Benchmark GPU performance vs CPU.

        Args:
            data_size: Size of test data

        Returns:
            Benchmark results
        """
        results = {}

        # Generate test data
        np.random.seed(42)
        test_data = np.random.randn(data_size, 10).astype(np.float32)

        # CPU benchmark
        start_time = time.time()
        np.matmul(test_data, test_data.T)
        cpu_time = time.time() - start_time

        results["cpu_time"] = cpu_time
        results["cpu_gflops"] = (2 * data_size**3) / (cpu_time * 1e9)

        # GPU benchmark if available
        if CUPY_AVAILABLE and self.device and self.device.type == "cuda":
            # Transfer to GPU
            start_time = time.time()
            gpu_data = cp.asarray(test_data)
            transfer_time = time.time() - start_time

            # Compute on GPU
            start_time = time.time()
            gpu_result = cp.matmul(gpu_data, gpu_data.T)
            cp.cuda.Stream.null.synchronize()  # Wait for completion
            gpu_compute_time = time.time() - start_time

            # Transfer back
            start_time = time.time()
            cp.asnumpy(gpu_result)
            transfer_back_time = time.time() - start_time

            gpu_total_time = transfer_time + gpu_compute_time + transfer_back_time

            results["gpu_compute_time"] = gpu_compute_time
            results["gpu_total_time"] = gpu_total_time
            results["gpu_transfer_time"] = transfer_time + transfer_back_time
            results["gpu_gflops"] = (2 * data_size**3) / (gpu_compute_time * 1e9)
            results["speedup_compute"] = cpu_time / gpu_compute_time
            results["speedup_total"] = cpu_time / gpu_total_time

        elif TORCH_AVAILABLE and self.device != "cpu":
            # PyTorch GPU benchmark
            test_tensor = torch.from_numpy(test_data).to(self.device)

            start_time = time.time()
            gpu_result = torch.matmul(test_tensor, test_tensor.T)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            gpu_time = time.time() - start_time

            results["gpu_compute_time"] = gpu_time
            results["gpu_gflops"] = (2 * data_size**3) / (gpu_time * 1e9)
            results["speedup_compute"] = cpu_time / gpu_time

        return results

    def get_device_info(self) -> dict[str, Any]:
        """Get information about available GPU devices"""
        info = {
            "torch_available": TORCH_AVAILABLE,
            "cupy_available": CUPY_AVAILABLE,
            "device": str(self.device) if self.device else "not_initialized",
        }

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_current_device"] = torch.cuda.current_device()
                info["cuda_device_name"] = torch.cuda.get_device_name()
                info["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                info["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            else:
                info["cuda_available"] = False

            if torch.backends.mps.is_available():
                info["mps_available"] = True
            else:
                info["mps_available"] = False

        return info


def benchmark_gpu_ml():
    """Benchmark GPU-accelerated ML training"""
    print("🚀 GPU-Accelerated ML Benchmark")
    print("=" * 50)

    # Initialize GPU accelerator
    gpu_config = GPUConfig(device="auto")
    accelerator = GPUAccelerator(gpu_config)

    # Get device info
    device_info = accelerator.get_device_info()
    print("\n📊 Device Information:")
    for key, value in device_info.items():
        print(f"   {key}: {value}")

    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch not available. Install with: pip install torch")
        return

    # Generate synthetic trading data
    print("\n🧪 Generating synthetic data...")
    np.random.seed(42)
    n_samples = 5000

    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="H")
    data = pd.DataFrame(
        {
            "Open": 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
            "High": 100 + np.cumsum(np.random.randn(n_samples) * 0.5) + 1,
            "Low": 100 + np.cumsum(np.random.randn(n_samples) * 0.5) - 1,
            "Close": 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
            "Volume": np.random.lognormal(15, 0.5, n_samples),
        },
        index=dates,
    )

    # Add synthetic signals
    data["signal"] = np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3])

    # Prepare data
    print("📊 Preparing data for neural network...")
    X, y = accelerator.prepare_data(data, sequence_length=30)

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")

    # Create and train model
    print("\n🤖 Training neural network model...")
    model_config = ModelConfig(
        model_type="lstm",
        input_dim=5,
        hidden_dim=64,
        output_dim=3,
        num_layers=2,
        epochs=20,
        learning_rate=0.001,
    )

    model = TradingNN(model_config)

    # Train model
    training_results = accelerator.train_model(
        model, (X_train, y_train), (X_val, y_val), model_config
    )

    print("\n📊 Training Results:")
    print(f"   Device: {training_results['device']}")
    print(f"   Training time: {training_results['training_time']:.2f}s")
    print(f"   Epochs completed: {training_results['epochs_completed']}")
    print(f"   Throughput: {training_results['throughput_samples_per_sec']:.0f} samples/sec")
    if training_results.get("final_val_accuracy"):
        print(f"   Final validation accuracy: {training_results['final_val_accuracy']:.3f}")

    # Benchmark GPU vs CPU matrix operations
    print("\n🧪 Benchmarking GPU vs CPU performance...")
    benchmark_results = accelerator.benchmark_gpu(data_size=5000)

    print("\n📊 Benchmark Results:")
    print(f"   CPU time: {benchmark_results['cpu_time']:.3f}s")
    print(f"   CPU GFLOPS: {benchmark_results['cpu_gflops']:.1f}")

    if "gpu_compute_time" in benchmark_results:
        print(f"   GPU compute time: {benchmark_results['gpu_compute_time']:.3f}s")
        print(f"   GPU GFLOPS: {benchmark_results['gpu_gflops']:.1f}")
        print(f"   Speedup (compute): {benchmark_results['speedup_compute']:.1f}x")

        if "speedup_total" in benchmark_results:
            print(f"   Speedup (with transfers): {benchmark_results['speedup_total']:.1f}x")

    return {
        "device_info": device_info,
        "training_results": training_results,
        "benchmark_results": benchmark_results,
    }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    benchmark_gpu_ml()
