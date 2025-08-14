"""
DL-004: Attention Mechanisms
Phase 4 - Week 1

Attention mechanisms with:
- Attention weights visualizable
- Improves prediction accuracy by >3%
- Identifies important time periods
- Self-attention and cross-attention support
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try multiple deep learning frameworks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """Types of attention mechanisms"""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD = "multi_head"
    ADDITIVE = "additive"
    DOT_PRODUCT = "dot_product"
    SCALED_DOT_PRODUCT = "scaled_dot_product"


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    # Attention type
    attention_type: AttentionType = AttentionType.SCALED_DOT_PRODUCT
    
    # Dimensions
    d_model: int = 128  # Model dimension
    d_k: int = 64  # Key/Query dimension
    d_v: int = 64  # Value dimension
    
    # Multi-head attention
    num_heads: int = 8
    
    # Regularization
    dropout: float = 0.1
    
    # Positional encoding
    use_positional_encoding: bool = True
    max_sequence_length: int = 1000
    
    # Temperature for scaled attention
    temperature: float = 1.0
    
    # Additive attention specific
    hidden_size: int = 128  # For additive attention
    
    # Visualization
    save_attention_weights: bool = True
    attention_weight_threshold: float = 0.1  # Minimum weight to consider significant


class AttentionMechanism:
    """
    Comprehensive attention mechanism implementation.
    
    Supports multiple attention types with visualization capabilities
    and performance improvements for time series prediction.
    """
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.attention_weights = None
        self.backend = "torch" if TORCH_AVAILABLE else ("tensorflow" if TF_AVAILABLE else "numpy")
        
        logger.info(f"Initialized attention mechanism with backend: {self.backend}")
    
    def build_attention_layer(self, input_shape: Tuple[int, ...]):
        """Build attention layer based on backend and configuration"""
        
        if self.backend == "torch":
            return self._build_torch_attention(input_shape)
        elif self.backend == "tensorflow":
            return self._build_tensorflow_attention(input_shape)
        else:
            return self._build_numpy_attention(input_shape)
    
    def _build_torch_attention(self, input_shape: Tuple[int, ...]):
        """Build PyTorch attention layers"""
        
        if self.config.attention_type == AttentionType.MULTI_HEAD:
            return TorchMultiHeadAttention(self.config)
        elif self.config.attention_type == AttentionType.SELF_ATTENTION:
            return TorchSelfAttention(self.config)
        elif self.config.attention_type == AttentionType.ADDITIVE:
            return TorchAdditiveAttention(self.config)
        else:
            return TorchScaledDotProductAttention(self.config)
    
    def _build_tensorflow_attention(self, input_shape: Tuple[int, ...]):
        """Build TensorFlow attention layers"""
        
        if self.config.attention_type == AttentionType.MULTI_HEAD:
            return TensorFlowMultiHeadAttention(self.config)
        else:
            return TensorFlowSelfAttention(self.config)
    
    def _build_numpy_attention(self, input_shape: Tuple[int, ...]):
        """Build NumPy-based attention (fallback)"""
        return NumpyAttention(self.config)
    
    def visualize_attention_weights(
        self,
        attention_weights: np.ndarray,
        sequence_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "Attention Weights"
    ) -> None:
        """
        Visualize attention weights as heatmap.
        
        Args:
            attention_weights: Attention weight matrix [seq_len, seq_len] or [heads, seq_len, seq_len]
            sequence_labels: Labels for sequence positions
            save_path: Path to save the visualization
            title: Title for the plot
        """
        
        # Handle multi-head attention
        if attention_weights.ndim == 3:
            # Average across heads or show multiple subplots
            if attention_weights.shape[0] <= 4:
                # Show multiple heads
                fig, axes = plt.subplots(1, attention_weights.shape[0], figsize=(5*attention_weights.shape[0], 4))
                if attention_weights.shape[0] == 1:
                    axes = [axes]
                
                for i, ax in enumerate(axes):
                    sns.heatmap(attention_weights[i], ax=ax, cmap='Blues', 
                               xticklabels=sequence_labels, yticklabels=sequence_labels)
                    ax.set_title(f"Head {i+1}")
                
                plt.suptitle(title)
            else:
                # Average across heads
                attention_weights = np.mean(attention_weights, axis=0)
                plt.figure(figsize=(10, 8))
                sns.heatmap(attention_weights, cmap='Blues', 
                           xticklabels=sequence_labels, yticklabels=sequence_labels)
                plt.title(f"{title} (Average across {attention_weights.shape[0]} heads)")
        else:
            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_weights, cmap='Blues', 
                       xticklabels=sequence_labels, yticklabels=sequence_labels)
            plt.title(title)
        
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
        
        plt.show()
    
    def analyze_attention_patterns(
        self,
        attention_weights: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns to identify important time periods.
        
        Args:
            attention_weights: Attention weight matrix
            timestamps: Timestamps for sequence positions
            
        Returns:
            Dictionary with attention analysis results
        """
        
        # Handle multi-head attention
        if attention_weights.ndim == 3:
            attention_weights = np.mean(attention_weights, axis=0)
        
        seq_len = attention_weights.shape[0]
        
        # Calculate attention statistics
        attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=1)
        attention_concentration = np.max(attention_weights, axis=1)
        attention_spread = np.std(attention_weights, axis=1)
        
        # Identify important positions
        important_positions = []
        for i in range(seq_len):
            max_attention_idx = np.argmax(attention_weights[i])
            max_attention_value = attention_weights[i, max_attention_idx]
            
            if max_attention_value > self.config.attention_weight_threshold:
                important_positions.append({
                    'query_pos': i,
                    'key_pos': max_attention_idx,
                    'attention_weight': max_attention_value,
                    'timestamp': timestamps[i] if timestamps is not None else i
                })
        
        # Calculate global attention patterns
        global_attention = np.mean(attention_weights, axis=0)
        most_attended_positions = np.argsort(global_attention)[-5:][::-1]  # Top 5
        
        # Temporal attention patterns
        temporal_patterns = {}
        if timestamps is not None:
            # Group by time periods (e.g., days, hours)
            # This would need more sophisticated time analysis
            pass
        
        analysis = {
            'attention_entropy': attention_entropy.tolist(),
            'attention_concentration': attention_concentration.tolist(),
            'attention_spread': attention_spread.tolist(),
            'important_positions': important_positions,
            'most_attended_positions': most_attended_positions.tolist(),
            'global_attention': global_attention.tolist(),
            'avg_entropy': np.mean(attention_entropy),
            'avg_concentration': np.mean(attention_concentration),
            'temporal_patterns': temporal_patterns
        }
        
        return analysis
    
    def compute_attention_metrics(
        self,
        predictions_with_attention: np.ndarray,
        predictions_without_attention: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute metrics to evaluate attention mechanism effectiveness.
        
        Args:
            predictions_with_attention: Predictions from model with attention
            predictions_without_attention: Predictions from model without attention
            targets: True target values
            
        Returns:
            Dictionary with attention effectiveness metrics
        """
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
        
        # Regression metrics
        mse_with = mean_squared_error(targets, predictions_with_attention)
        mse_without = mean_squared_error(targets, predictions_without_attention)
        
        mae_with = mean_absolute_error(targets, predictions_with_attention)
        mae_without = mean_absolute_error(targets, predictions_without_attention)
        
        # Calculate improvements
        mse_improvement = (mse_without - mse_with) / mse_without * 100
        mae_improvement = (mae_without - mae_with) / mae_without * 100
        
        metrics = {
            'mse_with_attention': mse_with,
            'mse_without_attention': mse_without,
            'mse_improvement_percent': mse_improvement,
            'mae_with_attention': mae_with,
            'mae_without_attention': mae_without,
            'mae_improvement_percent': mae_improvement,
            'meets_3_percent_improvement': mse_improvement >= 3.0 or mae_improvement >= 3.0
        }
        
        # Classification metrics if applicable
        if len(np.unique(targets)) <= 10:  # Likely classification
            try:
                acc_with = accuracy_score(targets, np.round(predictions_with_attention))
                acc_without = accuracy_score(targets, np.round(predictions_without_attention))
                acc_improvement = (acc_with - acc_without) * 100
                
                metrics.update({
                    'accuracy_with_attention': acc_with,
                    'accuracy_without_attention': acc_without,
                    'accuracy_improvement_percent': acc_improvement
                })
            except:
                pass
        
        return metrics


# PyTorch Attention Implementations
if TORCH_AVAILABLE:
    
    class TorchMultiHeadAttention(nn.Module):
        """Multi-head attention implementation in PyTorch"""
        
        def __init__(self, config: AttentionConfig):
            super().__init__()
            self.config = config
            self.d_model = config.d_model
            self.num_heads = config.num_heads
            self.d_k = config.d_k
            self.d_v = config.d_v
            
            assert config.d_model % config.num_heads == 0
            
            self.W_q = nn.Linear(config.d_model, config.num_heads * config.d_k, bias=False)
            self.W_k = nn.Linear(config.d_model, config.num_heads * config.d_k, bias=False)
            self.W_v = nn.Linear(config.d_model, config.num_heads * config.d_v, bias=False)
            self.W_o = nn.Linear(config.num_heads * config.d_v, config.d_model)
            
            self.dropout = nn.Dropout(config.dropout)
            self.scale = config.d_k ** -0.5
            
        def forward(self, query, key, value, mask=None):
            batch_size, seq_len, _ = query.shape
            
            # Linear transformations
            Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
            
            # Scaled dot-product attention
            attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
            
            # Concatenate heads
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.num_heads * self.d_v
            )
            
            # Final linear transformation
            output = self.W_o(attention_output)
            
            return output, attention_weights
        
        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            output = torch.matmul(attention_weights, V)
            
            return output, attention_weights
    
    
    class TorchSelfAttention(nn.Module):
        """Self-attention implementation in PyTorch"""
        
        def __init__(self, config: AttentionConfig):
            super().__init__()
            self.config = config
            
            self.W_q = nn.Linear(config.d_model, config.d_k, bias=False)
            self.W_k = nn.Linear(config.d_model, config.d_k, bias=False)
            self.W_v = nn.Linear(config.d_model, config.d_v, bias=False)
            
            self.dropout = nn.Dropout(config.dropout)
            self.scale = config.d_k ** -0.5
            
        def forward(self, x, mask=None):
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            output = torch.matmul(attention_weights, V)
            
            return output, attention_weights
    
    
    class TorchAdditiveAttention(nn.Module):
        """Additive (Bahdanau) attention implementation in PyTorch"""
        
        def __init__(self, config: AttentionConfig):
            super().__init__()
            self.config = config
            
            self.W_q = nn.Linear(config.d_model, config.hidden_size, bias=False)
            self.W_k = nn.Linear(config.d_model, config.hidden_size, bias=False)
            self.v = nn.Linear(config.hidden_size, 1, bias=False)
            
            self.dropout = nn.Dropout(config.dropout)
            
        def forward(self, query, key, value, mask=None):
            # Additive attention mechanism
            energy = torch.tanh(self.W_q(query).unsqueeze(2) + self.W_k(key).unsqueeze(1))
            scores = self.v(energy).squeeze(-1)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            output = torch.matmul(attention_weights.unsqueeze(1), value).squeeze(1)
            
            return output, attention_weights
    
    
    class TorchScaledDotProductAttention(nn.Module):
        """Scaled dot-product attention implementation in PyTorch"""
        
        def __init__(self, config: AttentionConfig):
            super().__init__()
            self.config = config
            self.scale = config.d_k ** -0.5
            self.dropout = nn.Dropout(config.dropout)
            
        def forward(self, query, key, value, mask=None):
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            output = torch.matmul(attention_weights, value)
            
            return output, attention_weights


# TensorFlow Attention Implementations
if TF_AVAILABLE:
    
    class TensorFlowMultiHeadAttention(layers.Layer):
        """Multi-head attention implementation in TensorFlow"""
        
        def __init__(self, config: AttentionConfig, **kwargs):
            super().__init__(**kwargs)
            self.config = config
            self.num_heads = config.num_heads
            self.d_model = config.d_model
            self.d_k = config.d_k
            self.d_v = config.d_v
            
            self.W_q = layers.Dense(config.num_heads * config.d_k, use_bias=False)
            self.W_k = layers.Dense(config.num_heads * config.d_k, use_bias=False)
            self.W_v = layers.Dense(config.num_heads * config.d_v, use_bias=False)
            self.W_o = layers.Dense(config.d_model)
            
            self.dropout = layers.Dropout(config.dropout)
            self.scale = config.d_k ** -0.5
            
        def call(self, query, key, value, mask=None, training=None):
            batch_size = tf.shape(query)[0]
            seq_len = tf.shape(query)[1]
            
            # Linear transformations and reshape
            Q = self.W_q(query)
            K = self.W_k(key)
            V = self.W_v(value)
            
            Q = tf.reshape(Q, (batch_size, seq_len, self.num_heads, self.d_k))
            K = tf.reshape(K, (batch_size, seq_len, self.num_heads, self.d_k))
            V = tf.reshape(V, (batch_size, seq_len, self.num_heads, self.d_v))
            
            Q = tf.transpose(Q, [0, 2, 1, 3])
            K = tf.transpose(K, [0, 2, 1, 3])
            V = tf.transpose(V, [0, 2, 1, 3])
            
            # Scaled dot-product attention
            attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
            
            # Concatenate heads
            attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
            attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.num_heads * self.d_v))
            
            # Final linear transformation
            output = self.W_o(attention_output)
            
            return output, attention_weights
        
        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            scores = tf.matmul(Q, K, transpose_b=True) * self.scale
            
            if mask is not None:
                scores += (mask * -1e9)
            
            attention_weights = tf.nn.softmax(scores, axis=-1)
            attention_weights = self.dropout(attention_weights)
            
            output = tf.matmul(attention_weights, V)
            
            return output, attention_weights
    
    
    class TensorFlowSelfAttention(layers.Layer):
        """Self-attention implementation in TensorFlow"""
        
        def __init__(self, config: AttentionConfig, **kwargs):
            super().__init__(**kwargs)
            self.config = config
            
            self.W_q = layers.Dense(config.d_k, use_bias=False)
            self.W_k = layers.Dense(config.d_k, use_bias=False)
            self.W_v = layers.Dense(config.d_v, use_bias=False)
            
            self.dropout = layers.Dropout(config.dropout)
            self.scale = config.d_k ** -0.5
            
        def call(self, x, mask=None, training=None):
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)
            
            scores = tf.matmul(Q, K, transpose_b=True) * self.scale
            
            if mask is not None:
                scores += (mask * -1e9)
            
            attention_weights = tf.nn.softmax(scores, axis=-1)
            attention_weights = self.dropout(attention_weights, training=training)
            
            output = tf.matmul(attention_weights, V)
            
            return output, attention_weights


# NumPy Attention Implementation (Fallback)
class NumpyAttention:
    """NumPy-based attention mechanism (fallback implementation)"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        
        # Initialize weight matrices
        self.W_q = np.random.normal(0, 0.02, (config.d_model, config.d_k))
        self.W_k = np.random.normal(0, 0.02, (config.d_model, config.d_k))
        self.W_v = np.random.normal(0, 0.02, (config.d_model, config.d_v))
        
        self.scale = config.d_k ** -0.5
    
    def forward(self, query, key, value, mask=None):
        """Forward pass through attention mechanism"""
        
        # Linear transformations
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        # Compute attention scores
        scores = np.dot(Q, K.T) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Apply dropout (simplified)
        if self.config.dropout > 0:
            dropout_mask = np.random.binomial(1, 1-self.config.dropout, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1-self.config.dropout)
        
        # Compute output
        output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
    def _softmax(self, x):
        """Compute softmax values for array x"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def create_attention_mechanism(
    attention_type: AttentionType = AttentionType.SCALED_DOT_PRODUCT,
    d_model: int = 128,
    num_heads: int = 8,
    dropout: float = 0.1,
    save_attention_weights: bool = True
) -> AttentionMechanism:
    """
    Factory function to create attention mechanism.
    
    Args:
        attention_type: Type of attention mechanism
        d_model: Model dimension
        num_heads: Number of attention heads (for multi-head)
        dropout: Dropout rate
        save_attention_weights: Whether to save attention weights for visualization
        
    Returns:
        Configured AttentionMechanism instance
    """
    config = AttentionConfig(
        attention_type=attention_type,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        save_attention_weights=save_attention_weights
    )
    
    return AttentionMechanism(config)