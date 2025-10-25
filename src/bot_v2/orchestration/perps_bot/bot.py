"""Main Coinbase trading bot implementation."""

from __future__ import annotations

from .configuration import PerpsBotConfigurationMixin
from .coordinator_setup import PerpsBotCoordinatorMixin
from .initialization import PerpsBotInitializationMixin
from .lifecycle import PerpsBotLifecycleMixin
from .runtime_accessors import PerpsBotRuntimeAccessMixin
from .symbol_processing import PerpsBotSymbolProcessingMixin, _CallableSymbolProcessor


class PerpsBot(
    PerpsBotConfigurationMixin,
    PerpsBotSymbolProcessingMixin,
    PerpsBotLifecycleMixin,
    PerpsBotRuntimeAccessMixin,
    PerpsBotCoordinatorMixin,
    PerpsBotInitializationMixin,
):
    """Core trading bot orchestrating strategies, execution, and monitoring."""


# Backwards-compatibility alias for the spot-first runtime name.
CoinbaseTrader = PerpsBot

__all__ = ["PerpsBot", "CoinbaseTrader", "_CallableSymbolProcessor"]
