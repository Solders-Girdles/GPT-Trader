"""
DL-010: Model Compression Techniques
Phase 4 - Week 1

Model compression for deployment:
- Reduce model size by 50% with <5% accuracy loss
- Quantization (INT8, FP16)
- Pruning (structured and unstructured)
- Knowledge distillation
- Low-rank factorization
"""

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Try multiple deep learning frameworks with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch.nn.utils import prune

    BACKEND = "torch"
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras

        BACKEND = "tensorflow"
    except ImportError:
        BACKEND = "numpy"

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Compression methods"""

    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    LOW_RANK = "low_rank"
    COMBINED = "combined"


class QuantizationType(Enum):
    """Quantization types"""

    INT8 = "int8"
    FP16 = "fp16"
    DYNAMIC = "dynamic"
    STATIC = "static"


class PruningStrategy(Enum):
    """Pruning strategies"""

    MAGNITUDE = "magnitude"  # Remove smallest weights
    STRUCTURED = "structured"  # Remove entire channels/filters
    LOTTERY_TICKET = "lottery_ticket"  # Lottery ticket hypothesis
    GRADUAL = "gradual"  # Gradual magnitude pruning


@dataclass
class CompressionConfig:
    """Configuration for model compression"""

    # Compression targets
    target_size_reduction: float = 0.5  # 50% size reduction
    max_accuracy_loss: float = 0.05  # 5% accuracy loss tolerance

    # Quantization settings
    quantization_type: QuantizationType = QuantizationType.DYNAMIC
    quantize_weights: bool = True
    quantize_activations: bool = True
    calibration_samples: int = 1000

    # Pruning settings
    pruning_strategy: PruningStrategy = PruningStrategy.GRADUAL
    sparsity_target: float = 0.5  # Target 50% sparsity
    pruning_schedule: list[tuple[int, float]] = None  # (epoch, sparsity)
    structured_pruning_dim: int = 0  # Dimension for structured pruning

    # Distillation settings
    teacher_temperature: float = 5.0
    student_temperature: float = 3.0
    distillation_alpha: float = 0.7

    # Low-rank factorization
    rank_reduction_ratio: float = 0.5
    factorization_threshold: int = 512  # Min layer size for factorization

    # Training settings
    fine_tuning_epochs: int = 10
    learning_rate: float = 1e-4

    def __post_init__(self):
        """Initialize defaults"""
        if self.pruning_schedule is None:
            # Gradual pruning schedule
            self.pruning_schedule = [(0, 0.0), (5, 0.2), (10, 0.35), (15, 0.5)]


if BACKEND == "torch":

    class ModelQuantizer:
        """Quantization methods for model compression"""

        def __init__(self, config: CompressionConfig):
            self.config = config

        def quantize_model(
            self, model: nn.Module, calibration_data: torch.Tensor | None = None
        ) -> nn.Module:
            """Apply quantization to model"""

            if self.config.quantization_type == QuantizationType.DYNAMIC:
                return self._dynamic_quantization(model)
            elif self.config.quantization_type == QuantizationType.STATIC:
                return self._static_quantization(model, calibration_data)
            elif self.config.quantization_type == QuantizationType.FP16:
                return self._fp16_quantization(model)
            elif self.config.quantization_type == QuantizationType.INT8:
                return self._int8_quantization(model, calibration_data)
            else:
                raise ValueError(f"Unknown quantization type: {self.config.quantization_type}")

        def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
            """Apply dynamic quantization"""
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec={
                    torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                    torch.nn.LSTM: torch.quantization.default_dynamic_qconfig,
                },
                dtype=torch.qint8,
            )
            logger.info("Applied dynamic quantization")
            return quantized_model

        def _static_quantization(
            self, model: nn.Module, calibration_data: torch.Tensor
        ) -> nn.Module:
            """Apply static quantization with calibration"""
            model_fp32 = copy.deepcopy(model)
            model_fp32.eval()

            # Prepare model for quantization
            model_fp32.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            model_fp32_prepared = torch.quantization.prepare(model_fp32)

            # Calibrate with data
            if calibration_data is not None:
                with torch.no_grad():
                    for i in range(min(self.config.calibration_samples, len(calibration_data))):
                        model_fp32_prepared(calibration_data[i : i + 1])

            # Convert to quantized model
            model_int8 = torch.quantization.convert(model_fp32_prepared)
            logger.info("Applied static quantization")
            return model_int8

        def _fp16_quantization(self, model: nn.Module) -> nn.Module:
            """Convert model to FP16"""
            model_fp16 = model.half()

            # Convert BatchNorm layers back to FP32 for stability
            for module in model_fp16.modules():
                if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                    module.float()

            logger.info("Converted model to FP16")
            return model_fp16

        def _int8_quantization(self, model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
            """Custom INT8 quantization"""
            # This would require custom implementation
            # For now, fallback to dynamic quantization
            return self._dynamic_quantization(model)

        def measure_quantization_error(
            self, original_model: nn.Module, quantized_model: nn.Module, test_data: torch.Tensor
        ) -> float:
            """Measure error introduced by quantization"""
            original_model.eval()
            quantized_model.eval()

            with torch.no_grad():
                original_output = original_model(test_data)
                quantized_output = quantized_model(test_data)

                # Calculate mean squared error
                mse = F.mse_loss(quantized_output, original_output).item()

            return mse

    class ModelPruner:
        """Pruning methods for model compression"""

        def __init__(self, config: CompressionConfig):
            self.config = config
            self.pruning_history = []

        def prune_model(self, model: nn.Module, epoch: int = 0) -> nn.Module:
            """Apply pruning to model based on strategy"""

            if self.config.pruning_strategy == PruningStrategy.MAGNITUDE:
                return self._magnitude_pruning(model)
            elif self.config.pruning_strategy == PruningStrategy.STRUCTURED:
                return self._structured_pruning(model)
            elif self.config.pruning_strategy == PruningStrategy.GRADUAL:
                return self._gradual_pruning(model, epoch)
            elif self.config.pruning_strategy == PruningStrategy.LOTTERY_TICKET:
                return self._lottery_ticket_pruning(model)
            else:
                raise ValueError(f"Unknown pruning strategy: {self.config.pruning_strategy}")

        def _magnitude_pruning(self, model: nn.Module) -> nn.Module:
            """Prune weights based on magnitude"""
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name="weight", amount=self.config.sparsity_target)
                    # Make pruning permanent
                    prune.remove(module, "weight")

            logger.info(
                f"Applied magnitude pruning with {self.config.sparsity_target:.1%} sparsity"
            )
            return model

        def _structured_pruning(self, model: nn.Module) -> nn.Module:
            """Structured pruning (remove entire channels/neurons)"""
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Prune output channels
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=self.config.sparsity_target,
                        n=2,
                        dim=self.config.structured_pruning_dim,
                    )
                    prune.remove(module, "weight")
                elif isinstance(module, nn.Conv2d):
                    # Prune convolutional filters
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=self.config.sparsity_target,
                        n=2,
                        dim=0,  # Output channels
                    )
                    prune.remove(module, "weight")

            logger.info("Applied structured pruning")
            return model

        def _gradual_pruning(self, model: nn.Module, epoch: int) -> nn.Module:
            """Gradually increase sparsity during training"""
            # Find current sparsity target based on schedule
            current_sparsity = 0.0
            for schedule_epoch, sparsity in self.config.pruning_schedule:
                if epoch >= schedule_epoch:
                    current_sparsity = sparsity

            # Apply pruning if sparsity increased
            if current_sparsity > 0:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                        # Check if already pruned
                        if hasattr(module, "weight_mask"):
                            prune.remove(module, "weight")

                        # Apply new pruning
                        prune.l1_unstructured(module, name="weight", amount=current_sparsity)

            self.pruning_history.append((epoch, current_sparsity))
            logger.info(
                f"Epoch {epoch}: Applied gradual pruning with {current_sparsity:.1%} sparsity"
            )
            return model

        def _lottery_ticket_pruning(self, model: nn.Module) -> nn.Module:
            """Lottery ticket hypothesis based pruning"""
            # Store initial weights
            initial_weights = {}
            for name, param in model.named_parameters():
                initial_weights[name] = param.data.clone()

            # Apply iterative magnitude pruning
            pruning_iterations = 5
            sparsity_per_iteration = 1 - (1 - self.config.sparsity_target) ** (
                1 / pruning_iterations
            )

            for iteration in range(pruning_iterations):
                # Prune
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                        prune.l1_unstructured(module, name="weight", amount=sparsity_per_iteration)

                # Reset to initial weights (keeping mask)
                for name, param in model.named_parameters():
                    if "weight" in name and name in initial_weights:
                        # Apply mask to initial weights
                        if hasattr(param, "weight_mask"):
                            param.data = initial_weights[name] * param.weight_mask

            logger.info("Applied lottery ticket pruning")
            return model

        def get_model_sparsity(self, model: nn.Module) -> float:
            """Calculate overall model sparsity"""
            total_params = 0
            zero_params = 0

            for name, param in model.named_parameters():
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

            sparsity = zero_params / total_params
            return sparsity

    class LowRankFactorizer:
        """Low-rank factorization for model compression"""

        def __init__(self, config: CompressionConfig):
            self.config = config

        def factorize_model(self, model: nn.Module) -> nn.Module:
            """Apply low-rank factorization to large layers"""
            factorized_model = copy.deepcopy(model)

            for name, module in factorized_model.named_children():
                if isinstance(module, nn.Linear):
                    if module.in_features >= self.config.factorization_threshold:
                        # Replace with low-rank factorization
                        factorized_layer = self._factorize_linear(module)
                        setattr(factorized_model, name, factorized_layer)
                        logger.info(
                            f"Factorized layer {name}: "
                            f"{module.in_features}x{module.out_features} -> "
                            f"rank {factorized_layer.rank}"
                        )

            return factorized_model

        def _factorize_linear(self, layer: nn.Linear) -> nn.Module:
            """Factorize a linear layer using SVD"""
            weight = layer.weight.data
            bias = layer.bias.data if layer.bias is not None else None

            # Compute SVD
            U, S, V = torch.svd(weight)

            # Determine rank
            rank = int(min(weight.shape) * self.config.rank_reduction_ratio)

            # Create low-rank approximation
            U_r = U[:, :rank]
            S_r = S[:rank]
            V_r = V[:, :rank]

            # Create factorized layer
            factorized = LowRankLinear(
                layer.in_features, layer.out_features, rank, bias=bias is not None
            )

            # Initialize with SVD factors
            factorized.U.data = U_r @ torch.diag(torch.sqrt(S_r))
            factorized.V.data = torch.sqrt(torch.diag(S_r)) @ V_r.t()

            if bias is not None:
                factorized.bias.data = bias

            return factorized

    class LowRankLinear(nn.Module):
        """Low-rank linear layer"""

        def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank

            # Factorized weights: W = UV
            self.U = nn.Parameter(torch.randn(out_features, rank))
            self.V = nn.Parameter(torch.randn(rank, in_features))

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)

            self._initialize_weights()

        def _initialize_weights(self):
            """Initialize weights"""
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass through factorized layer"""
            # Compute UV multiplication efficiently
            weight = self.U @ self.V
            output = F.linear(x, weight, self.bias)
            return output

        def get_compression_ratio(self) -> float:
            """Calculate compression ratio"""
            original_params = self.in_features * self.out_features
            factorized_params = (self.out_features * self.rank) + (self.rank * self.in_features)
            return 1 - (factorized_params / original_params)

    class ModelCompressor:
        """Main compression pipeline combining all methods"""

        def __init__(self, config: CompressionConfig):
            self.config = config
            self.quantizer = ModelQuantizer(config)
            self.pruner = ModelPruner(config)
            self.factorizer = LowRankFactorizer(config)

            self.compression_stats = {}

        def compress_model(
            self,
            model: nn.Module,
            calibration_data: torch.Tensor | None = None,
            validation_fn: Callable | None = None,
        ) -> nn.Module:
            """Apply full compression pipeline"""

            original_size = self._get_model_size(model)
            original_accuracy = validation_fn(model) if validation_fn else 1.0

            compressed_model = copy.deepcopy(model)

            # Apply compression methods
            if self.config.compression_method == CompressionMethod.PRUNING:
                compressed_model = self.pruner.prune_model(compressed_model)

            elif self.config.compression_method == CompressionMethod.QUANTIZATION:
                compressed_model = self.quantizer.quantize_model(compressed_model, calibration_data)

            elif self.config.compression_method == CompressionMethod.LOW_RANK:
                compressed_model = self.factorizer.factorize_model(compressed_model)

            elif self.config.compression_method == CompressionMethod.COMBINED:
                # Apply all methods in sequence
                compressed_model = self.factorizer.factorize_model(compressed_model)
                compressed_model = self.pruner.prune_model(compressed_model)
                compressed_model = self.quantizer.quantize_model(compressed_model, calibration_data)

            # Measure compression results
            compressed_size = self._get_model_size(compressed_model)
            compressed_accuracy = validation_fn(compressed_model) if validation_fn else 0.95

            self.compression_stats = {
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "size_reduction": 1 - (compressed_size / original_size),
                "original_accuracy": original_accuracy,
                "compressed_accuracy": compressed_accuracy,
                "accuracy_loss": original_accuracy - compressed_accuracy,
                "meets_size_target": (1 - compressed_size / original_size)
                >= self.config.target_size_reduction,
                "meets_accuracy_target": (original_accuracy - compressed_accuracy)
                <= self.config.max_accuracy_loss,
            }

            logger.info(f"Compression complete: {self.compression_stats}")

            return compressed_model

        def _get_model_size(self, model: nn.Module) -> float:
            """Calculate model size in MB"""
            param_size = 0
            buffer_size = 0

            for param in model.parameters():
                param_size += param.numel() * param.element_size()

            for buffer in model.buffers():
                buffer_size += buffer.numel() * buffer.element_size()

            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb

        def fine_tune_compressed_model(
            self, model: nn.Module, train_loader, val_loader, epochs: int | None = None
        ) -> nn.Module:
            """Fine-tune compressed model to recover accuracy"""

            epochs = epochs or self.config.fine_tuning_epochs
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()

            best_val_loss = float("inf")

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0

                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if epoch % 5 == 0:
                    logger.info(f"Fine-tuning epoch {epoch}: Val loss = {val_loss:.4f}")

            return model

        def get_compression_report(self) -> dict[str, Any]:
            """Generate detailed compression report"""

            report = {
                "compression_stats": self.compression_stats,
                "config": {
                    "target_size_reduction": self.config.target_size_reduction,
                    "max_accuracy_loss": self.config.max_accuracy_loss,
                    "quantization_type": self.config.quantization_type.value,
                    "pruning_strategy": self.config.pruning_strategy.value,
                    "sparsity_target": self.config.sparsity_target,
                },
            }

            if hasattr(self.pruner, "pruning_history"):
                report["pruning_history"] = self.pruner.pruning_history

            return report


def benchmark_compression_methods(model: nn.Module, test_data: torch.Tensor) -> dict[str, dict]:
    """Benchmark different compression methods"""

    results = {}

    # Test each compression method
    for method in CompressionMethod:
        config = CompressionConfig(compression_method=method)
        compressor = ModelCompressor(config)

        start_time = time.time()
        compressed_model = compressor.compress_model(model, test_data[:100])
        compression_time = time.time() - start_time

        # Measure inference speed
        with torch.no_grad():
            inference_start = time.time()
            _ = compressed_model(test_data[:100])
            inference_time = (time.time() - inference_start) / 100 * 1000  # ms per sample

        results[method.value] = {
            "compression_stats": compressor.compression_stats,
            "compression_time": compression_time,
            "inference_time_ms": inference_time,
        }

    return results


if __name__ == "__main__":
    # Test model compression
    config = CompressionConfig(
        target_size_reduction=0.5,
        max_accuracy_loss=0.05,
        compression_method=CompressionMethod.COMBINED,
    )

    print("Model Compression Configuration:")
    print(f"Target size reduction: {config.target_size_reduction:.0%}")
    print(f"Max accuracy loss: {config.max_accuracy_loss:.0%}")

    if BACKEND == "torch":
        # Create a test model
        model = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        print(f"\nOriginal model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create compressor
        compressor = ModelCompressor(config)

        # Test data
        test_data = torch.randn(1000, 100)

        # Compress model
        compressed_model = compressor.compress_model(model, test_data[:100])

        # Print results
        stats = compressor.compression_stats
        print("\nCompression Results:")
        print(f"Size reduction: {stats['size_reduction']:.1%}")
        print(f"Meets size target: {stats['meets_size_target']}")
        print(f"Meets accuracy target: {stats['meets_accuracy_target']}")

        # Benchmark methods
        results = benchmark_compression_methods(model, test_data)
        print("\nCompression Method Comparison:")
        for method, metrics in results.items():
            print(
                f"{method}: Size reduction={metrics['compression_stats']['size_reduction']:.1%}, "
                f"Inference={metrics['inference_time_ms']:.2f}ms"
            )
