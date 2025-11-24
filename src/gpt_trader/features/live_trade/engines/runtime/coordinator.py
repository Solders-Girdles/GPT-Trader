"""
Simplified Runtime Engine.
"""
from gpt_trader.features.live_trade.engines.base import BaseEngine, CoordinatorContext

class RuntimeEngine(BaseEngine):
    @property
    def name(self) -> str:
        return "runtime"

    def initialize(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        return context or self.context

__all__ = ["RuntimeEngine"]
