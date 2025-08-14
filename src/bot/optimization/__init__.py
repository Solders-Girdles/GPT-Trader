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


def run_optimization(
    symbol: str,
    start_date: str,
    end_date: str,
    strategy: str,
    parameters: list | None = None,
    metric: str = "sharpe",
    method: str = "grid",
) -> dict:
    """Stub function for CLI compatibility.

    This is a temporary implementation to make the CLI work.
    TODO: Implement proper optimization functionality.
    """
    return {"best_score": 0.0, "best_params": {}, "message": "Optimization not yet implemented"}


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
    "run_optimization",
]
