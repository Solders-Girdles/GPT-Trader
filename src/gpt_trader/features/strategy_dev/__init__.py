"""
Strategy Development Toolkit.

A comprehensive suite for evolving and improving trading strategies:

- **Strategy Lab**: Experiment tracking, A/B testing, parameter optimization
- **Performance Monitor**: Real-time metrics, alerts, regime-specific analytics
- **Config Manager**: YAML profiles, strategy registry, hot-reload configuration

Usage:
    from gpt_trader.features.strategy_dev import (
        # Lab
        Experiment,
        ExperimentTracker,
        ParameterGrid,
        # Monitor
        PerformanceMonitor,
        MetricsAggregator,
        AlertManager,
        # Config
        ConfigManager,
        StrategyRegistry,
        StrategyProfile,
    )
"""

from gpt_trader.features.strategy_dev.config import (
    ConfigManager,
    StrategyProfile,
    StrategyRegistry,
)
from gpt_trader.features.strategy_dev.lab import (
    Experiment,
    ExperimentResult,
    ExperimentTracker,
    ParameterGrid,
)
from gpt_trader.features.strategy_dev.monitor import (
    AlertManager,
    AlertRule,
    MetricsAggregator,
    PerformanceMonitor,
    PerformanceSnapshot,
)

__all__ = [
    # Lab
    "Experiment",
    "ExperimentResult",
    "ExperimentTracker",
    "ParameterGrid",
    # Monitor
    "AlertManager",
    "AlertRule",
    "MetricsAggregator",
    "PerformanceMonitor",
    "PerformanceSnapshot",
    # Config
    "ConfigManager",
    "StrategyProfile",
    "StrategyRegistry",
]
