from __future__ import annotations

from typing import Dict, Type

from ..features.strategies.interfaces import IStrategy
from ..features.strategies.scalp import ScalpStrategy
from ..features.strategies.momentum import MomentumStrategy
from ..features.strategies.mean_reversion import MeanReversionStrategy
from ..features.strategies.breakout import BreakoutStrategy
from ..features.strategies.ma_crossover import MAStrategy
from ..features.strategies.volatility import VolatilityStrategy


class StrategyRegistry:
    _map: Dict[str, Type[IStrategy]] = {
        "scalp": ScalpStrategy,
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy,
        "ma_crossover": MAStrategy,
        "volatility": VolatilityStrategy,
    }

    @classmethod
    def create(cls, name: str, **params) -> IStrategy:
        key = (name or "").lower()
        if key not in cls._map:
            raise ValueError(f"Unknown strategy: {name}")
        return cls._map[key](**params)  # type: ignore[misc]

    @classmethod
    def list_strategies(cls) -> Dict[str, str]:
        return {k: v.__name__ for k, v in cls._map.items()}

# Backwards-compatible stub to avoid breaking existing imports
class SliceRegistry:
    """Placeholder SliceRegistry to satisfy existing imports.

    Note: Full implementation of dynamic slice discovery is out of scope here.
    """

    @classmethod
    def list_slices(cls) -> Dict[str, str]:
        return {}
