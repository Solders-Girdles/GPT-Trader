from .stateful import StatefulBaselineStrategy, StatefulPerpsStrategy
from .strategy import (
    Action,
    BaselinePerpsStrategy,
    BaseStrategyConfig,
    Decision,
    IndicatorState,
    PerpsStrategy,
    PerpsStrategyConfig,
    SpotStrategy,
    SpotStrategyConfig,
    StrategyConfig,
)

__all__ = [
    # Stateless strategy classes
    "BaselinePerpsStrategy",  # Base technical strategy
    "SpotStrategy",  # Spot trading (no shorts)
    "PerpsStrategy",  # Perpetuals trading (full functionality)
    # Stateful strategy classes (O(1) incremental updates)
    "StatefulBaselineStrategy",  # Stateful version with serialization
    "StatefulPerpsStrategy",  # Alias for StatefulBaselineStrategy
    # Types
    "Decision",
    "Action",
    "IndicatorState",
    # Config classes
    "BaseStrategyConfig",
    "SpotStrategyConfig",
    "PerpsStrategyConfig",
    "StrategyConfig",  # Backward compat alias
]
