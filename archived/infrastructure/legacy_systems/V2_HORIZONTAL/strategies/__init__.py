"""
Strategy module for GPT-Trader V2.

Provides the strategy base classes and concrete strategy implementations.
All strategies are automatically registered with the global registry.
"""

from .base import StrategyBase, StrategyConfig, StrategyMetrics
from .strategy import SimpleMAStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .volatility import VolatilityStrategy
from .breakout import BreakoutStrategy
from .factory import (
    StrategyFactory,
    StrategyRegistry,
    StrategyInfo,
    get_strategy_factory,
    get_strategy_registry,
    register_strategy,
    create_strategy,
    list_available_strategies,
    strategy_parameter_info
)

# Auto-register all available strategies
register_strategy(SimpleMAStrategy)
register_strategy(MomentumStrategy)
register_strategy(MeanReversionStrategy)
register_strategy(VolatilityStrategy)
register_strategy(BreakoutStrategy)

__all__ = [
    # Base classes
    'StrategyBase',
    'StrategyConfig', 
    'StrategyMetrics',
    
    # Factory system
    'StrategyFactory',
    'StrategyRegistry',
    'StrategyInfo',
    'get_strategy_factory',
    'get_strategy_registry',
    'register_strategy',
    'create_strategy',
    'list_available_strategies',
    'strategy_parameter_info',
    
    # Concrete strategies
    'SimpleMAStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'VolatilityStrategy',
    'BreakoutStrategy'
]