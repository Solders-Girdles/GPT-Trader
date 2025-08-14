"""
Machine Learning Module

Advanced ML capabilities for trading:
- Transformer models for market prediction
- Deep reinforcement learning agents
- GPU-accelerated training and inference
- Real-time prediction pipelines
- Multi-GPU and distributed training
"""

from .gpu_accelerator import GPUAccelerator, GPUConfig, ModelConfig, TradingNN, benchmark_gpu_ml
from .reinforcement_learning import (
    DQNAgent,
    PPOAgent,
    RLConfig,
    TradingEnvironment,
    benchmark_rl,
    train_rl_agent,
)
from .transformer_models import TransformerConfig, TransformerTrader, benchmark_transformer

__all__ = [
    # GPU Acceleration
    "GPUAccelerator",
    "GPUConfig",
    "ModelConfig",
    "TradingNN",
    "benchmark_gpu_ml",
    # Transformer Models
    "TransformerConfig",
    "TransformerTrader",
    "benchmark_transformer",
    # Reinforcement Learning
    "RLConfig",
    "TradingEnvironment",
    "DQNAgent",
    "PPOAgent",
    "train_rl_agent",
    "benchmark_rl",
]
