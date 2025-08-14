"""
Distributed Computing Module

High-performance distributed computing capabilities:
- Ray-based distributed optimization
- GPU acceleration support
- Horizontal scaling across multiple machines
- Real-time streaming data processing
- Distributed caching with Redis
"""

from .ray_engine import (
    DistributedConfig,
    DistributedResult,
    DistributedWorker,
    RayDistributedEngine,
    benchmark_ray_distributed,
)

__all__ = [
    "RayDistributedEngine",
    "DistributedConfig",
    "DistributedResult",
    "DistributedWorker",
    "benchmark_ray_distributed",
]
