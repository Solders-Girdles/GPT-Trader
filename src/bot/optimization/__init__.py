"""
Optimization module for GPT-Trader.
"""

from .config import OptimizationConfig, ParameterDefinition, ParameterSpace, StrategyConfig
from .deployment_pipeline import DeploymentConfig, DeploymentPipeline, run_deployment_pipeline
from .engine import OptimizationEngine
from .parallel_optimizer import OptimizationConfig as ParallelOptimizationConfig
from .parallel_optimizer import OptimizationResult, ParallelOptimizer, benchmark_multiprocessing
from .walk_forward_validator import (
    WalkForwardConfig,
    WalkForwardValidator,
    run_walk_forward_validation,
)

__all__ = [
    "OptimizationConfig",
    "ParameterSpace",
    "StrategyConfig",
    "ParameterDefinition",
    "OptimizationEngine",
    "ParallelOptimizer",
    "ParallelOptimizationConfig",
    "OptimizationResult",
    "benchmark_multiprocessing",
    "DeploymentConfig",
    "DeploymentPipeline",
    "run_deployment_pipeline",
    "WalkForwardConfig",
    "WalkForwardValidator",
    "run_walk_forward_validation",
]
