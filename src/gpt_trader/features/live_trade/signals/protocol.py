"""
Protocols for the Signal Ensemble architecture.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gpt_trader.core import Product
from gpt_trader.features.live_trade.signals.types import SignalOutput

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.strategies.base import MarketDataContext


@dataclass
class StrategyContext:
    """Context object containing all market data needed for decision making.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTC-USD")
        current_mark: Current mark/spot price
        position_state: Current position info, or None if no position
        recent_marks: Historical prices (oldest first)
        equity: Account equity for position sizing
        product: Product specification from exchange
        candles: Historical OHLCV candles (optional)
        market_data: Enhanced market microstructure data (optional)
    """

    symbol: str
    current_mark: Decimal
    position_state: dict[str, Any] | None
    recent_marks: Sequence[Decimal]
    equity: Decimal
    product: Product | None
    candles: Sequence[Any] | None = None
    market_data: "MarketDataContext | None" = None


@runtime_checkable
class SignalGenerator(Protocol):
    """Interface for a component that generates a trading signal."""

    def generate(self, context: StrategyContext) -> SignalOutput:
        """Generate a trading signal based on the provided context."""
        ...


@runtime_checkable
class SignalCombiner(Protocol):
    """Interface for a component that combines multiple signals."""

    def combine(self, signals: list[SignalOutput], context: StrategyContext) -> SignalOutput:
        """Combine multiple signals into a single net signal."""
        ...
