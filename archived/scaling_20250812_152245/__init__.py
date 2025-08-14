"""
Auto-scaling Module

Dynamic resource management based on market conditions:
- Market volatility-based scaling
- Predictive resource allocation
- Cost optimization
- Multi-resource coordination
- Event-driven scaling
"""

from .auto_scaler import (
    AutoScaler,
    MarketAnalyzer,
    MarketCondition,
    MarketMetrics,
    ResourceConfig,
    ResourceManager,
    ResourceType,
    ScalingAction,
    ScalingPolicy,
)

__all__ = [
    "MarketCondition",
    "ResourceType",
    "ScalingAction",
    "ResourceConfig",
    "MarketMetrics",
    "ScalingPolicy",
    "MarketAnalyzer",
    "ResourceManager",
    "AutoScaler",
]
