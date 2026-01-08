"""Datamodels for strategy orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.core import Balance, Position, Product

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.market_data_features import DepthSnapshot


@dataclass(slots=True)
class SymbolProcessingContext:
    """Context for strategy decision-making on a single symbol.

    Contains all market data, account state, and position information
    needed for a strategy to make trading decisions.
    """

    symbol: str
    balances: Sequence[Balance]
    equity: Decimal
    positions: dict[str, Position]
    position_state: dict[str, Any] | None
    position_quantity: Decimal
    marks: list[Decimal]
    product: Product | None

    # Order book data (optional, for advanced strategies)
    orderbook_snapshot: DepthSnapshot | None = field(default=None)
    """Latest order book snapshot with bid/ask depth."""

    # Trade flow statistics (optional, for volume analysis)
    trade_volume_stats: dict[str, Any] | None = field(default=None)
    """Rolling trade statistics: vwap, avg_size, aggressor_ratio, etc."""

    # Spread in basis points (optional, derived from orderbook)
    spread_bps: Decimal | None = field(default=None)
    """Current bid-ask spread in basis points."""


__all__ = ["SymbolProcessingContext"]
