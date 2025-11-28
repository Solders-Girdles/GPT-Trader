"""Base strategy protocol and abstract types.

This module defines the interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gpt_trader.features.brokerages.core.interfaces import Product

if TYPE_CHECKING:
    from .perps_baseline.strategy import Decision


@runtime_checkable
class StrategyProtocol(Protocol):
    """Protocol defining the interface for all trading strategies.

    Any class implementing this protocol can be used as a trading strategy
    in the system. The protocol ensures strategies provide the core `decide`
    method that generates trading decisions.
    """

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
    ) -> "Decision":
        """Generate a trading decision based on market data.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            current_mark: Current mark/spot price
            position_state: Current position info, or None if no position
            recent_marks: Historical prices (oldest first)
            equity: Account equity for position sizing
            product: Product specification from exchange

        Returns:
            Decision with action, reason, confidence, and indicator state
        """
        ...


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Provides a template for implementing trading strategies with
    common structure. Subclasses must implement the `decide` method.
    """

    @abstractmethod
    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
    ) -> "Decision":
        """Generate a trading decision based on market data."""
        ...


__all__ = ["BaseStrategy", "StrategyProtocol"]
