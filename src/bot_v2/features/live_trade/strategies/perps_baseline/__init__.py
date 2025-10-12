"""Modular baseline strategy for perpetuals trading."""

from bot_v2.features.live_trade.strategies.decisions import Action, Decision

from .config import StrategyConfig
from .signals import StrategySignal, build_signal
from .state import StrategyState
from .strategy import BaselinePerpsStrategy

__all__ = [
    "Action",
    "Decision",
    "BaselinePerpsStrategy",
    "StrategyConfig",
    "StrategySignal",
    "StrategyState",
    "build_signal",
]
