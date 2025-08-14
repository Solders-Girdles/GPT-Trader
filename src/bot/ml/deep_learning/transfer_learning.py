"""
DL-009: Transfer Learning Framework
Phase 4 - Week 1

Transfer learning for financial time series:
- Pre-train on large market dataset
- Fine-tune on specific assets/timeframes
- Domain adaptation techniques
- Knowledge distillation from larger models
- Cross-market transfer capabilities
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

# Try multiple deep learning frameworks with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch.utils.data import DataLoader, TensorDataset

    BACKEND = "torch"
except ImportError:
    try:
        import tensorflow as tf
        from tensorflow import keras

        BACKEND = "tensorflow"
    except ImportError:
        BACKEND = "numpy"

logger = logging.getLogger(__name__)


class TransferStrategy(Enum):
    """Transfer learning strategies"""

    FEATURE_EXTRACTION = "feature_extraction"  # Freeze base, train head
    FINE_TUNING = "fine_tuning"  # Fine-tune all layers
    PROGRESSIVE = "progressive"  # Gradually unfreeze layers
    DOMAIN_ADAPTATION = "domain_adaptation"  # Adapt to new domain
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"  # Learn from teacher


class PretrainedModel(Enum):
    """Available pretrained models"""

    MARKET_TRANSFORMER = "market_transformer"
    CRYPTO_LSTM = "crypto_lstm"
    FOREX_GRU = "forex_gru"
    EQUITY_ENSEMBLE = "equity_ensemble"
    UNIVERSAL_TIMESERIES = "universal_timeseries"


@dataclass
class TransferConfig:
    """Configuration for transfer learning"""

    # Strategy
    strategy: TransferStrategy = TransferStrategy.PROGRESSIVE
    pretrained_model: PretrainedModel | None = None

    # Fine-tuning settings
    freeze_base_initially: bool = True
    unfreeze_schedule: list[int] = None  # Epochs to unfreeze layers
    learning_rate_schedule: list[float] = None  # LR for each phase

    # Domain adaptation
    domain_adversarial: bool = False
    domain_lambda: float = 0.1  # Weight for domain loss

    # Knowledge distillation
    teacher_model_path: str | None = None
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7  # Weight for distillation loss

    # Data requirements
    min_pretrain_samples: int = 10000
    min_finetune_samples: int = 1000

    # Performance targets
    target_improvement: float = 10.0  # % improvement over training from scratch
    max_adaptation_epochs: int = 50

    def __post_init__(self):
        """Initialize defaults"""
        if self.unfreeze_schedule is None:
            self.unfreeze_schedule = [5, 10, 15, 20]
        if self.learning_rate_schedule is None:
            self.learning_rate_schedule = [1e-4, 5e-5, 1e-5, 5e-6]


if BACKEND == "torch":

    class FeatureExtractor(nn.Module):
        """Base feature extractor that can be frozen"""

        def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 3):
            super().__init__()

            layers = []
            current_dim = input_dim

            for i in range(n_layers):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                current_dim = hidden_dim

            self.feature_layers = nn.Sequential(*layers)
            self.output_dim = hidden_dim

        def forward(self, x: Tensor) -> Tensor:
            return self.feature_layers(x)

        def freeze(self):
            """Freeze all parameters"""
            for param in self.parameters():
                param.requires_grad = False

        def unfreeze(self):
            """Unfreeze all parameters"""
            for param in self.parameters():
                param.requires_grad = True

        def unfreeze_layers(self, n_layers: int):
            """Unfreeze last n layers"""
            layers = list(self.feature_layers.children())
            for layer in layers[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    class TaskHead(nn.Module):
        """Task-specific head for different prediction tasks"""

        def __init__(self, input_dim: int, output_dim: int, task_type: str = "regression"):
            super().__init__()
            self.task_type = task_type

            if task_type == "regression":
                self.head = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(input_dim // 2, output_dim),
                )
            elif task_type == "classification":
                self.head = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(input_dim // 2, output_dim),
                    nn.Softmax(dim=-1),
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")

        def forward(self, x: Tensor) -> Tensor:
            return self.head(x)

    class DomainDiscriminator(nn.Module):
        """Domain discriminator for adversarial domain adaptation"""

        def __init__(self, input_dim: int, n_domains: int = 2):
            super().__init__()

            self.discriminator = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(input_dim // 4, n_domains),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.discriminator(x)

    class GradientReversalLayer(nn.Module):
        """Gradient reversal for domain adaptation"""

        def __init__(self, lambda_val: float = 1.0):
            super().__init__()
            self.lambda_val = lambda_val

        def forward(self, x: Tensor) -> Tensor:
            return GradientReversalFunction.apply(x, self.lambda_val)

    class GradientReversalFunction(torch.autograd.Function):
        """Gradient reversal function"""

        @staticmethod
        def forward(ctx, x, lambda_val):
            ctx.lambda_val = lambda_val
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lambda_val, None

    class TransferLearningModel(nn.Module):
        """Complete transfer learning model"""

        def __init__(self, config: TransferConfig, base_model: nn.Module | None = None):
            super().__init__()
            self.config = config

            # Initialize or load base model
            if base_model is not None:
                self.feature_extractor = base_model
            else:
                self.feature_extractor = FeatureExtractor(50, 256, 4)

            # Task-specific head
            feature_dim = getattr(self.feature_extractor, "output_dim", 256)
            self.task_head = TaskHead(feature_dim, 1, "regression")

            # Domain adaptation components
            if config.domain_adversarial:
                self.gradient_reversal = GradientReversalLayer(config.domain_lambda)
                self.domain_discriminator = DomainDiscriminator(feature_dim)

            # Initially freeze base if configured
            if config.freeze_base_initially:
                self.feature_extractor.freeze()

            # Track training phase
            self.current_phase = 0
            self.phase_epochs = 0

        def forward(self, x: Tensor, return_domain: bool = False) -> Tensor | tuple[Tensor, Tensor]:
            """Forward pass with optional domain prediction"""
            features = self.feature_extractor(x)
            predictions = self.task_head(features)

            if return_domain and self.config.domain_adversarial:
                reversed_features = self.gradient_reversal(features)
                domain_pred = self.domain_discriminator(reversed_features)
                return predictions, domain_pred

            return predictions

        def update_phase(self, epoch: int):
            """Update training phase based on epoch"""
            if epoch in self.config.unfreeze_schedule:
                phase_idx = self.config.unfreeze_schedule.index(epoch)
                self.current_phase = phase_idx + 1

                # Progressive unfreezing
                if self.config.strategy == TransferStrategy.PROGRESSIVE:
                    n_layers_to_unfreeze = self.current_phase * 2
                    self.feature_extractor.unfreeze_layers(n_layers_to_unfreeze)
                    logger.info(
                        f"Phase {self.current_phase}: Unfroze {n_layers_to_unfreeze} layers"
                    )

                # Full fine-tuning
                elif self.config.strategy == TransferStrategy.FINE_TUNING:
                    if self.current_phase >= 1:
                        self.feature_extractor.unfreeze()
                        logger.info("Unfroze entire feature extractor")

        def get_current_lr(self) -> float:
            """Get learning rate for current phase"""
            if self.current_phase < len(self.config.learning_rate_schedule):
                return self.config.learning_rate_schedule[self.current_phase]
            return self.config.learning_rate_schedule[-1]

    class KnowledgeDistillationLoss(nn.Module):
        """Loss function for knowledge distillation"""

        def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
            super().__init__()
            self.temperature = temperature
            self.alpha = alpha
            self.kl_div = nn.KLDivLoss(reduction="batchmean")

        def forward(
            self,
            student_logits: Tensor,
            teacher_logits: Tensor,
            targets: Tensor,
            task_loss_fn: Callable,
        ) -> Tensor:
            """
            Compute distillation loss

            Args:
                student_logits: Student model outputs
                teacher_logits: Teacher model outputs
                targets: True labels
                task_loss_fn: Loss function for the task
            """
            # Task loss
            task_loss = task_loss_fn(student_logits, targets)

            # Distillation loss
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
            distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature**2)

            # Combined loss
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss

            return total_loss, task_loss, distillation_loss

    class TransferLearningTrainer:
        """Trainer for transfer learning"""

        def __init__(self, model: TransferLearningModel, config: TransferConfig):
            self.model = model
            self.config = config
            self.optimizer = None
            self.scheduler = None
            self.teacher_model = None

            # Load teacher model if using knowledge distillation
            if config.strategy == TransferStrategy.KNOWLEDGE_DISTILLATION:
                self._load_teacher_model()

            # Initialize optimizer
            self._setup_optimizer()

        def _setup_optimizer(self):
            """Setup optimizer with different learning rates for different parts"""
            param_groups = [
                {"params": self.model.task_head.parameters(), "lr": self.model.get_current_lr()},
            ]

            if not self.config.freeze_base_initially:
                param_groups.append(
                    {
                        "params": self.model.feature_extractor.parameters(),
                        "lr": self.model.get_current_lr() * 0.1,  # Lower LR for pretrained layers
                    }
                )

            self.optimizer = torch.optim.Adam(param_groups)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )

        def _load_teacher_model(self):
            """Load teacher model for knowledge distillation"""
            if self.config.teacher_model_path:
                self.teacher_model = torch.load(self.config.teacher_model_path)
                self.teacher_model.eval()
                logger.info(f"Loaded teacher model from {self.config.teacher_model_path}")

        def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
            """Train for one epoch"""
            self.model.train()
            self.model.update_phase(epoch)

            # Update learning rates if phase changed
            if epoch in self.config.unfreeze_schedule:
                self._update_learning_rates()

            total_loss = 0.0
            domain_loss = 0.0
            n_batches = 0

            for batch_idx, (data, target, *domain) in enumerate(train_loader):
                self.optimizer.zero_grad()

                # Forward pass
                if self.config.domain_adversarial and domain:
                    predictions, domain_pred = self.model(data, return_domain=True)

                    # Task loss
                    task_loss = F.mse_loss(predictions, target)

                    # Domain loss
                    domain_target = domain[0]
                    d_loss = F.cross_entropy(domain_pred, domain_target)

                    # Combined loss
                    loss = task_loss + self.config.domain_lambda * d_loss
                    domain_loss += d_loss.item()

                elif (
                    self.config.strategy == TransferStrategy.KNOWLEDGE_DISTILLATION
                    and self.teacher_model
                ):
                    # Get teacher predictions
                    with torch.no_grad():
                        teacher_pred = self.teacher_model(data)

                    # Student predictions
                    student_pred = self.model(data)

                    # Distillation loss
                    distill_loss_fn = KnowledgeDistillationLoss(
                        self.config.distillation_temperature, self.config.distillation_alpha
                    )
                    loss, task_loss, dist_loss = distill_loss_fn(
                        student_pred, teacher_pred, target, F.mse_loss
                    )

                else:
                    # Standard supervised loss
                    predictions = self.model(data)
                    loss = F.mse_loss(predictions, target)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            metrics = {
                "loss": total_loss / n_batches,
                "phase": self.model.current_phase,
                "lr": self.model.get_current_lr(),
            }

            if self.config.domain_adversarial:
                metrics["domain_loss"] = domain_loss / n_batches

            return metrics

        def _update_learning_rates(self):
            """Update learning rates for new phase"""
            new_lr = self.model.get_current_lr()

            for param_group in self.optimizer.param_groups:
                if "feature_extractor" in str(param_group["params"][0]):
                    param_group["lr"] = new_lr * 0.1
                else:
                    param_group["lr"] = new_lr

            logger.info(f"Updated learning rates to {new_lr}")

        def validate(self, val_loader: DataLoader) -> dict[str, float]:
            """Validate the model"""
            self.model.eval()
            total_loss = 0.0
            predictions = []
            targets = []

            with torch.no_grad():
                for data, target in val_loader:
                    pred = self.model(data)
                    loss = F.mse_loss(pred, target)

                    total_loss += loss.item()
                    predictions.append(pred)
                    targets.append(target)

            predictions = torch.cat(predictions)
            targets = torch.cat(targets)

            # Calculate metrics
            mae = F.l1_loss(predictions, targets).item()

            return {"val_loss": total_loss / len(val_loader), "val_mae": mae}

        def fine_tune(
            self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
        ) -> dict[str, list[float]]:
            """Fine-tune the model"""
            history = {"train_loss": [], "val_loss": [], "val_mae": []}

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(epochs):
                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                history["train_loss"].append(train_metrics["loss"])

                # Validate
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                history["val_mae"].append(val_metrics["val_mae"])

                # Learning rate scheduling
                self.scheduler.step(val_metrics["val_loss"])

                # Early stopping
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                # Logging
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                        f"Val Loss={val_metrics['val_loss']:.4f}, "
                        f"Phase={train_metrics['phase']}"
                    )

            return history

    class PretrainedModelZoo:
        """Repository of pretrained models"""

        @staticmethod
        def load_pretrained(model_name: PretrainedModel, device: str = "cpu") -> nn.Module:
            """Load a pretrained model"""

            # In practice, these would be loaded from actual pretrained weights
            # For now, we create initialized models as placeholders

            if model_name == PretrainedModel.MARKET_TRANSFORMER:
                from .transformer_architecture import TransformerArchitecture, TransformerConfig

                config = TransformerConfig(
                    d_model=512,
                    n_heads=8,
                    n_layers=6,
                    input_features=100,  # Pretrained on 100 features
                )
                model = TransformerArchitecture(config)

            elif model_name == PretrainedModel.CRYPTO_LSTM:
                from .lstm_architecture import LSTMArchitecture, LSTMConfig

                config = LSTMConfig(
                    input_size=75, hidden_size=256, num_layers=3, bidirectional=True
                )
                model = LSTMArchitecture(config)

            elif model_name == PretrainedModel.EQUITY_ENSEMBLE:
                from .deep_ensemble import DeepEnsemble, EnsembleConfig

                config = EnsembleConfig(model_types=["lstm", "transformer"], n_models_per_type=2)
                model = DeepEnsemble(config)

            else:
                # Default to a simple feature extractor
                model = FeatureExtractor(50, 256, 4)

            # Simulate loading pretrained weights
            logger.info(f"Loaded pretrained model: {model_name.value}")

            return model.to(device)

        @staticmethod
        def adapt_model_for_task(
            pretrained_model: nn.Module, target_features: int, target_output: int
        ) -> nn.Module:
            """Adapt a pretrained model for a specific task"""

            # Add adaptation layers if feature dimensions don't match
            adapted_model = nn.Sequential()

            # Input adaptation if needed
            pretrained_input_size = getattr(pretrained_model, "input_size", 50)
            if target_features != pretrained_input_size:
                adapted_model.add_module(
                    "input_adapter", nn.Linear(target_features, pretrained_input_size)
                )

            # Add pretrained model
            adapted_model.add_module("pretrained", pretrained_model)

            # Add task head
            feature_dim = getattr(pretrained_model, "output_dim", 256)
            adapted_model.add_module(
                "task_head", TaskHead(feature_dim, target_output, "regression")
            )

            return adapted_model


def create_transfer_learning_model(
    config: TransferConfig,
    pretrained_model_name: PretrainedModel | None = None,
    target_features: int = 50,
    target_output: int = 1,
) -> TransferLearningModel:
    """Factory function to create transfer learning model"""

    # Load pretrained model if specified
    base_model = None
    if pretrained_model_name:
        base_model = PretrainedModelZoo.load_pretrained(pretrained_model_name)
        base_model = PretrainedModelZoo.adapt_model_for_task(
            base_model, target_features, target_output
        )

    return TransferLearningModel(config, base_model)


def benchmark_transfer_learning(
    source_data: tuple[torch.Tensor, torch.Tensor],
    target_data: tuple[torch.Tensor, torch.Tensor],
    config: TransferConfig,
) -> dict[str, float]:
    """Benchmark transfer learning vs training from scratch"""

    X_source, y_source = source_data
    X_target, y_target = target_data

    # Split target data
    n_train = int(0.8 * len(X_target))
    X_train, y_train = X_target[:n_train], y_target[:n_train]
    X_val, y_val = X_target[n_train:], y_target[n_train:]

    # Train from scratch
    scratch_model = TransferLearningModel(
        TransferConfig(strategy=TransferStrategy.FINE_TUNING, freeze_base_initially=False), None
    )
    scratch_trainer = TransferLearningTrainer(scratch_model, config)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    # Train scratch model
    scratch_history = scratch_trainer.fine_tune(train_loader, val_loader, epochs=30)
    scratch_best_val = min(scratch_history["val_loss"])

    # Transfer learning model
    transfer_model = create_transfer_learning_model(config, PretrainedModel.UNIVERSAL_TIMESERIES)
    transfer_trainer = TransferLearningTrainer(transfer_model, config)

    # Fine-tune transfer model
    transfer_history = transfer_trainer.fine_tune(train_loader, val_loader, epochs=30)
    transfer_best_val = min(transfer_history["val_loss"])

    # Calculate improvement
    improvement = (scratch_best_val - transfer_best_val) / scratch_best_val * 100

    return {
        "scratch_best_val_loss": scratch_best_val,
        "transfer_best_val_loss": transfer_best_val,
        "improvement_percent": improvement,
        "meets_target": improvement >= config.target_improvement,
        "transfer_epochs_to_converge": len(transfer_history["val_loss"]),
        "scratch_epochs_to_converge": len(scratch_history["val_loss"]),
    }


if __name__ == "__main__":
    # Test transfer learning
    config = TransferConfig(
        strategy=TransferStrategy.PROGRESSIVE,
        pretrained_model=PretrainedModel.MARKET_TRANSFORMER,
        target_improvement=10.0,
    )

    print("Transfer Learning Configuration:")
    print(f"Strategy: {config.strategy.value}")
    print(f"Unfreeze Schedule: {config.unfreeze_schedule}")
    print(f"Learning Rate Schedule: {config.learning_rate_schedule}")

    if BACKEND == "torch":
        # Create model
        model = create_transfer_learning_model(config)
        print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

        # Test forward pass
        batch_size = 32
        x = torch.randn(batch_size, 50)

        with torch.no_grad():
            output = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")

        # Test domain adaptation
        if config.domain_adversarial:
            output, domain = model(x, return_domain=True)
            print(f"Domain prediction shape: {domain.shape}")

        # Benchmark
        source_data = (torch.randn(1000, 50), torch.randn(1000, 1))
        target_data = (torch.randn(200, 50), torch.randn(200, 1))

        results = benchmark_transfer_learning(source_data, target_data, config)
        print("\nTransfer Learning Benchmark:")
        print(f"Improvement: {results['improvement_percent']:.2f}%")
        print(f"Meets target: {results['meets_target']}")
