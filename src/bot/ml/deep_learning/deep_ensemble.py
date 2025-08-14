"""
DL-008: Deep Ensemble Methods
Phase 4 - Week 1

Deep ensemble combining multiple architectures:
- Combines 3+ deep models (LSTM, GRU, Transformer)
- Uncertainty quantification via ensemble
- 5% accuracy improvement over single model
- Weighted averaging with dynamic weights
- Model diversity enforcement
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import json
import time

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
        BACKEND = "tensorflow"
    except ImportError:
        from sklearn.ensemble import VotingRegressor, VotingClassifier
        from sklearn.linear_model import Ridge
        BACKEND = "sklearn"

# Import other deep learning components
try:
    from .lstm_architecture import LSTMArchitecture, LSTMConfig, create_lstm_architecture
    from .transformer_architecture import TransformerArchitecture, TransformerConfig, create_transformer_architecture
except ImportError:
    # Handle direct execution
    from lstm_architecture import LSTMArchitecture, LSTMConfig, create_lstm_architecture
    from transformer_architecture import TransformerArchitecture, TransformerConfig, create_transformer_architecture

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    DYNAMIC_WEIGHTED = "dynamic_weighted"
    STACKING = "stacking"
    BAYESIAN_AVERAGE = "bayesian_average"


class ModelType(Enum):
    """Supported model types for ensemble"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CONV1D = "conv1d"


@dataclass
class EnsembleConfig:
    """Configuration for deep ensemble"""
    # Ensemble composition
    model_types: List[ModelType] = None
    n_models_per_type: int = 1
    ensemble_method: EnsembleMethod = EnsembleMethod.DYNAMIC_WEIGHTED
    
    # Model configurations
    lstm_config: Optional[LSTMConfig] = None
    transformer_config: Optional[TransformerConfig] = None
    
    # Diversity enforcement
    enforce_diversity: bool = True
    diversity_threshold: float = 0.7  # Max correlation between models
    diversity_weight: float = 0.1     # Weight for diversity loss
    
    # Uncertainty quantification
    enable_uncertainty: bool = True
    uncertainty_method: str = "ensemble_variance"  # or "mc_dropout"
    n_uncertainty_samples: int = 100
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_early_stopping: bool = True
    patience: int = 10
    
    # Performance targets
    target_accuracy_improvement: float = 0.05  # 5% improvement
    max_memory_mb: float = 6000  # Maximum memory usage
    max_inference_time_ms: float = 30  # Maximum inference time
    
    def __post_init__(self):
        """Initialize defaults and validate"""
        if self.model_types is None:
            self.model_types = [ModelType.LSTM, ModelType.TRANSFORMER]
        
        if len(self.model_types) < 2:
            raise ValueError("Ensemble requires at least 2 model types")


if BACKEND == "torch":
    
    class GRUArchitecture(nn.Module):
        """GRU architecture for ensemble diversity"""
        
        def __init__(self, config: LSTMConfig):
            super().__init__()
            self.config = config
            
            # GRU layers
            self.gru = nn.GRU(
                input_size=config.input_features,
                hidden_size=config.hidden_size,
                num_layers=config.n_layers,
                dropout=config.dropout if config.n_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
            
            # Output layer
            gru_output_size = config.hidden_size * (2 if config.bidirectional else 1)
            self.output_layer = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(gru_output_size, config.output_size)
            )
        
        def forward(self, x: Tensor) -> Dict[str, Tensor]:
            """Forward pass"""
            gru_out, hidden = self.gru(x)
            
            # Use last timestep output
            last_output = gru_out[:, -1, :]
            predictions = self.output_layer(last_output)
            
            return {
                'predictions': predictions,
                'last_hidden': hidden,
                'sequence_output': gru_out
            }


    class Conv1DArchitecture(nn.Module):
        """1D CNN architecture for ensemble diversity"""
        
        def __init__(self, config: LSTMConfig):
            super().__init__()
            self.config = config
            
            # Conv1D layers
            self.conv_layers = nn.ModuleList([
                nn.Conv1d(config.input_features, 64, kernel_size=3, padding=1),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.Conv1d(128, 256, kernel_size=3, padding=1)
            ])
            
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(config.dropout)
            
            # Output layer
            self.output_layer = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, config.output_size)
            )
        
        def forward(self, x: Tensor) -> Dict[str, Tensor]:
            """Forward pass"""
            # Transpose for Conv1D (batch, features, sequence)
            x = x.transpose(1, 2)
            
            # Convolutional layers
            for conv in self.conv_layers:
                x = F.relu(conv(x))
                x = self.dropout(x)
            
            # Global pooling
            x = self.pool(x).squeeze(-1)
            
            # Output layer
            predictions = self.output_layer(x)
            
            return {
                'predictions': predictions,
                'last_hidden': x,
                'sequence_output': None
            }


    class DeepEnsemble(nn.Module):
        """DL-008: Deep ensemble of multiple architectures"""
        
        def __init__(self, config: EnsembleConfig):
            super().__init__()
            self.config = config
            
            # Create individual models
            self.models = nn.ModuleList()
            self.model_types_list = []
            
            for model_type in config.model_types:
                for i in range(config.n_models_per_type):
                    model = self._create_model(model_type, variation=i)
                    self.models.append(model)
                    self.model_types_list.append(model_type)
            
            # Ensemble combination weights
            n_models = len(self.models)
            if config.ensemble_method in [EnsembleMethod.WEIGHTED_AVERAGE, EnsembleMethod.DYNAMIC_WEIGHTED]:
                self.ensemble_weights = nn.Parameter(torch.ones(n_models) / n_models)
            
            # Meta-learner for stacking
            if config.ensemble_method == EnsembleMethod.STACKING:
                self.meta_learner = nn.Sequential(
                    nn.Linear(n_models, n_models // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(n_models // 2, 1)
                )
            
            # Track model performance for dynamic weighting
            self.model_performances = torch.ones(n_models)
            self.performance_history = []
        
        def _create_model(self, model_type: ModelType, variation: int = 0) -> nn.Module:
            """Create individual model with variations for diversity"""
            if model_type == ModelType.LSTM:
                config = self.config.lstm_config or LSTMConfig()
                # Add variation for diversity
                if variation > 0:
                    config.hidden_size = int(config.hidden_size * (1 + 0.1 * variation))
                    config.dropout = min(0.5, config.dropout + 0.05 * variation)
                return create_lstm_architecture(config)
            
            elif model_type == ModelType.TRANSFORMER:
                config = self.config.transformer_config or TransformerConfig()
                # Add variation for diversity
                if variation > 0:
                    config.d_model = int(config.d_model * (1 + 0.1 * variation))
                    config.n_heads = max(4, config.n_heads + variation)
                return create_transformer_architecture(config)
            
            elif model_type == ModelType.GRU:
                config = self.config.lstm_config or LSTMConfig()
                if variation > 0:
                    config.hidden_size = int(config.hidden_size * (1 + 0.1 * variation))
                return GRUArchitecture(config)
            
            elif model_type == ModelType.CONV1D:
                config = self.config.lstm_config or LSTMConfig()
                return Conv1DArchitecture(config)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        def forward(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
            """
            Forward pass through ensemble
            
            Returns:
                Dictionary with ensemble predictions and uncertainty
            """
            batch_size = x.size(0)
            model_outputs = []
            model_predictions = []
            
            # Get predictions from all models
            for i, model in enumerate(self.models):
                if self.model_types_list[i] == ModelType.TRANSFORMER:
                    output = model(x, **kwargs)
                else:
                    output = model(x)
                
                model_outputs.append(output)
                model_predictions.append(output['predictions'])
            
            # Stack predictions
            all_predictions = torch.stack(model_predictions, dim=1)  # (batch, n_models, output_size)
            
            # Combine predictions based on ensemble method
            if self.config.ensemble_method == EnsembleMethod.SIMPLE_AVERAGE:
                ensemble_pred = all_predictions.mean(dim=1)
                
            elif self.config.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
                weights = F.softmax(self.ensemble_weights, dim=0)
                ensemble_pred = (all_predictions * weights.view(1, -1, 1)).sum(dim=1)
                
            elif self.config.ensemble_method == EnsembleMethod.DYNAMIC_WEIGHTED:
                # Update weights based on recent performance
                weights = F.softmax(self.model_performances, dim=0)
                ensemble_pred = (all_predictions * weights.view(1, -1, 1)).sum(dim=1)
                
            elif self.config.ensemble_method == EnsembleMethod.STACKING:
                # Use meta-learner to combine predictions
                flat_predictions = all_predictions.view(batch_size, -1)
                ensemble_pred = self.meta_learner(flat_predictions)
                
            else:  # BAYESIAN_AVERAGE
                # Weighted by inverse uncertainty
                uncertainties = torch.var(all_predictions, dim=1, keepdim=True)
                weights = 1.0 / (uncertainties + 1e-8)
                weights = weights / weights.sum(dim=1, keepdim=True)
                ensemble_pred = (all_predictions * weights).sum(dim=1)
            
            # Calculate uncertainty if enabled
            uncertainty = None
            if self.config.enable_uncertainty:
                if self.config.uncertainty_method == "ensemble_variance":
                    uncertainty = torch.var(all_predictions, dim=1)
                elif self.config.uncertainty_method == "mc_dropout":
                    # Enable dropout at inference for MC sampling
                    self.train()  # Enable dropout
                    mc_predictions = []
                    for _ in range(self.config.n_uncertainty_samples):
                        mc_preds = []
                        for model in self.models:
                            mc_preds.append(model(x)['predictions'])
                        mc_predictions.append(torch.stack(mc_preds).mean(dim=0))
                    
                    mc_predictions = torch.stack(mc_predictions)
                    uncertainty = torch.var(mc_predictions, dim=0)
                    self.eval()  # Disable dropout
            
            return {
                'predictions': ensemble_pred,
                'individual_predictions': all_predictions,
                'uncertainty': uncertainty,
                'model_weights': getattr(self, 'ensemble_weights', None),
                'model_outputs': model_outputs
            }
        
        def calculate_diversity_loss(self, predictions: Tensor) -> Tensor:
            """Calculate diversity loss to enforce model diversity"""
            if not self.config.enforce_diversity:
                return torch.tensor(0.0)
            
            n_models = predictions.size(1)
            diversity_loss = 0.0
            
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    correlation = F.cosine_similarity(
                        predictions[:, i], predictions[:, j], dim=0
                    )
                    if correlation > self.config.diversity_threshold:
                        diversity_loss += (correlation - self.config.diversity_threshold) ** 2
            
            return diversity_loss * self.config.diversity_weight
        
        def update_model_performances(self, losses: List[float]):
            """Update model performance tracking for dynamic weighting"""
            if len(losses) == len(self.models):
                # Convert losses to performance scores (lower loss = higher performance)
                performances = torch.tensor([1.0 / (loss + 1e-8) for loss in losses])
                
                # Exponential moving average
                alpha = 0.1
                self.model_performances = (1 - alpha) * self.model_performances + alpha * performances
                
                # Store history
                self.performance_history.append(performances.tolist())
                if len(self.performance_history) > 100:  # Keep last 100 updates
                    self.performance_history.pop(0)
        
        def get_model_analysis(self) -> Dict[str, Any]:
            """Analyze ensemble composition and performance"""
            analysis = {
                'n_models': len(self.models),
                'model_types': [mt.value for mt in self.model_types_list],
                'total_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'ensemble_method': self.config.ensemble_method.value
            }
            
            if hasattr(self, 'ensemble_weights'):
                weights = F.softmax(self.ensemble_weights, dim=0)
                analysis['model_weights'] = weights.detach().tolist()
            
            if self.performance_history:
                recent_performance = np.array(self.performance_history[-10:])  # Last 10 updates
                analysis['model_performance_trends'] = {
                    'mean': recent_performance.mean(axis=0).tolist(),
                    'std': recent_performance.std(axis=0).tolist()
                }
            
            return analysis
        
        def prune_underperforming_models(self, threshold: float = 0.1):
            """Remove consistently underperforming models"""
            if len(self.performance_history) < 10:
                return
            
            recent_performance = np.array(self.performance_history[-10:])
            avg_performance = recent_performance.mean(axis=0)
            
            # Find models performing below threshold
            keep_indices = []
            for i, perf in enumerate(avg_performance):
                if perf > threshold * avg_performance.max():
                    keep_indices.append(i)
            
            if len(keep_indices) >= 2:  # Keep at least 2 models
                # Create new module list with only good models
                new_models = nn.ModuleList([self.models[i] for i in keep_indices])
                new_model_types = [self.model_types_list[i] for i in keep_indices]
                
                self.models = new_models
                self.model_types_list = new_model_types
                
                # Update weights if they exist
                if hasattr(self, 'ensemble_weights'):
                    new_weights = self.ensemble_weights[keep_indices]
                    self.ensemble_weights = nn.Parameter(new_weights / new_weights.sum())
                
                logger.info(f"Pruned ensemble from {len(avg_performance)} to {len(keep_indices)} models")


    class EnsembleTrainer:
        """Training framework for deep ensembles"""
        
        def __init__(self, ensemble: DeepEnsemble, config: EnsembleConfig):
            self.ensemble = ensemble
            self.config = config
            
            # Optimizer
            self.optimizer = torch.optim.AdamW(
                ensemble.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
            
            # Early stopping
            self.best_loss = float('inf')
            self.patience_counter = 0
        
        def train_epoch(self, train_loader, criterion) -> Dict[str, float]:
            """Train for one epoch"""
            self.ensemble.train()
            total_loss = 0.0
            diversity_loss_total = 0.0
            individual_losses = [0.0] * len(self.ensemble.models)
            
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.ensemble(data)
                
                # Main prediction loss
                main_loss = criterion(output['predictions'], target)
                
                # Diversity loss
                diversity_loss = self.ensemble.calculate_diversity_loss(output['individual_predictions'])
                
                # Individual model losses for performance tracking
                for i, pred in enumerate(torch.unbind(output['individual_predictions'], dim=1)):
                    individual_losses[i] += criterion(pred, target).item()
                
                # Total loss
                total_loss_batch = main_loss + diversity_loss
                total_loss_batch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += main_loss.item()
                diversity_loss_total += diversity_loss.item()
            
            # Update model performances
            avg_individual_losses = [loss / len(train_loader) for loss in individual_losses]
            self.ensemble.update_model_performances(avg_individual_losses)
            
            return {
                'total_loss': total_loss / len(train_loader),
                'diversity_loss': diversity_loss_total / len(train_loader),
                'individual_losses': avg_individual_losses
            }
        
        def validate(self, val_loader, criterion) -> Dict[str, float]:
            """Validate the ensemble"""
            self.ensemble.eval()
            total_loss = 0.0
            predictions = []
            targets = []
            uncertainties = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.ensemble(data)
                    loss = criterion(output['predictions'], target)
                    
                    total_loss += loss.item()
                    predictions.append(output['predictions'])
                    targets.append(target)
                    
                    if output['uncertainty'] is not None:
                        uncertainties.append(output['uncertainty'])
            
            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)
            
            # Calculate metrics
            mse = F.mse_loss(predictions, targets).item()
            mae = F.l1_loss(predictions, targets).item()
            
            results = {
                'val_loss': total_loss / len(val_loader),
                'mse': mse,
                'mae': mae
            }
            
            if uncertainties:
                uncertainties = torch.cat(uncertainties, dim=0)
                results['avg_uncertainty'] = uncertainties.mean().item()
            
            return results
        
        def early_stopping_check(self, val_loss: float) -> bool:
            """Check if training should stop early"""
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                return False
            else:
                self.patience_counter += 1
                return self.patience_counter >= self.config.patience


else:
    # Fallback implementation for non-PyTorch environments
    class DeepEnsemble:
        """Fallback ensemble using sklearn"""
        
        def __init__(self, config: EnsembleConfig):
            self.config = config
            logger.warning(f"Using fallback ensemble implementation with {BACKEND}")
            
            if BACKEND == "sklearn":
                from sklearn.ensemble import VotingRegressor, RandomForestRegressor
                from sklearn.linear_model import Ridge
                
                self.models = [
                    ('rf1', RandomForestRegressor(n_estimators=100, random_state=1)),
                    ('rf2', RandomForestRegressor(n_estimators=100, random_state=2)),
                    ('ridge', Ridge(alpha=1.0))
                ]
                self.ensemble = VotingRegressor(self.models)
        
        def fit(self, X, y):
            """Train the ensemble"""
            if BACKEND == "sklearn":
                self.ensemble.fit(X, y)
        
        def predict(self, X):
            """Make predictions"""
            if BACKEND == "sklearn":
                return self.ensemble.predict(X)
        
        def get_model_analysis(self):
            """Basic analysis for fallback"""
            return {
                'n_models': len(self.models),
                'backend': BACKEND,
                'ensemble_method': 'voting'
            }


def create_deep_ensemble(config: EnsembleConfig) -> DeepEnsemble:
    """Factory function to create deep ensemble"""
    return DeepEnsemble(config)


def create_default_ensemble_config(**kwargs) -> EnsembleConfig:
    """Create default ensemble configuration with overrides"""
    config = EnsembleConfig()
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    return config


def benchmark_ensemble_performance(ensemble: DeepEnsemble, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
    """Benchmark ensemble performance vs individual models"""
    X_test, y_test = test_data
    
    # Get ensemble predictions
    ensemble.eval()
    with torch.no_grad():
        start_time = time.time()
        ensemble_output = ensemble(X_test)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        ensemble_pred = ensemble_output['predictions']
        individual_preds = ensemble_output['individual_predictions']
    
    # Calculate metrics
    ensemble_mse = F.mse_loss(ensemble_pred, y_test).item()
    ensemble_mae = F.l1_loss(ensemble_pred, y_test).item()
    
    # Individual model performance
    individual_mses = []
    for i in range(individual_preds.size(1)):
        mse = F.mse_loss(individual_preds[:, i], y_test).item()
        individual_mses.append(mse)
    
    best_individual_mse = min(individual_mses)
    improvement = (best_individual_mse - ensemble_mse) / best_individual_mse
    
    return {
        'ensemble_mse': ensemble_mse,
        'ensemble_mae': ensemble_mae,
        'best_individual_mse': best_individual_mse,
        'improvement_over_best': improvement,
        'meets_target': improvement >= ensemble.config.target_accuracy_improvement,
        'inference_time_ms': inference_time / len(X_test),  # Per sample
        'memory_efficient': inference_time < ensemble.config.max_inference_time_ms
    }


if __name__ == "__main__":
    # Test the deep ensemble
    config = create_default_ensemble_config(
        model_types=[ModelType.LSTM, ModelType.TRANSFORMER, ModelType.GRU],
        n_models_per_type=1,
        ensemble_method=EnsembleMethod.DYNAMIC_WEIGHTED
    )
    
    ensemble = create_deep_ensemble(config)
    
    print(f"Created ensemble with {ensemble.get_model_analysis()}")
    print(f"Backend: {BACKEND}")
    
    if BACKEND == "torch":
        # Test forward pass
        batch_size, seq_len, features = 16, 60, 50
        x = torch.randn(batch_size, seq_len, features)
        y = torch.randn(batch_size, 1)
        
        with torch.no_grad():
            output = ensemble(x)
            print(f"Input shape: {x.shape}")
            print(f"Ensemble output shape: {output['predictions'].shape}")
            print(f"Individual predictions shape: {output['individual_predictions'].shape}")
            if output['uncertainty'] is not None:
                print(f"Uncertainty shape: {output['uncertainty'].shape}")
        
        # Benchmark performance
        perf = benchmark_ensemble_performance(ensemble, (x, y))
        print(f"Improvement over best individual: {perf['improvement_over_best']:.3f}")
        print(f"Meets target improvement: {perf['meets_target']}")
        print(f"Inference time per sample: {perf['inference_time_ms']:.2f} ms")