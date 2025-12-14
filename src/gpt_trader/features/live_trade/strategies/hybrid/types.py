"""Types and dataclasses for hybrid trading strategies.

This module defines the core types used by hybrid strategies that can trade
across both spot and CFM futures markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class TradingMode(Enum):
    """Trading mode for strategy decisions.

    Determines which market venue the decision should execute on.
    """

    SPOT_ONLY = "spot_only"
    CFM_ONLY = "cfm_only"
    HYBRID = "hybrid"  # Can trade both markets


class Action(Enum):
    """Trading action to take."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class HybridDecision:
    """A trading decision from a hybrid strategy.

    Extends the standard Decision with market mode and leverage information
    for CFM futures trading.
    """

    action: Action
    symbol: str
    mode: TradingMode
    quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    leverage: int = 1
    reason: str = ""
    confidence: float = 0.0
    indicators: dict[str, Any] = field(default_factory=dict)

    def is_actionable(self) -> bool:
        """Check if this decision requires execution."""
        return self.action not in (Action.HOLD,)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "mode": self.mode.value,
            "quantity": str(self.quantity),
            "leverage": self.leverage,
            "reason": self.reason,
            "confidence": self.confidence,
            "indicators": self.indicators,
        }


@dataclass
class HybridMarketData:
    """Market data for hybrid strategy decision-making.

    Contains prices from both spot and futures markets to enable
    arbitrage and basis trading strategies.
    """

    symbol: str
    spot_price: Decimal
    futures_price: Decimal | None = None
    spot_bid: Decimal | None = None
    spot_ask: Decimal | None = None
    futures_bid: Decimal | None = None
    futures_ask: Decimal | None = None
    funding_rate: Decimal | None = None
    mark_price: Decimal | None = None

    @property
    def basis(self) -> Decimal | None:
        """Calculate the futures/spot basis (premium/discount).

        Returns:
            Absolute basis (futures - spot) or None if futures unavailable.
        """
        if self.futures_price is None:
            return None
        return self.futures_price - self.spot_price

    @property
    def basis_percentage(self) -> Decimal | None:
        """Calculate the basis as a percentage of spot.

        Returns:
            Basis percentage ((futures - spot) / spot * 100) or None.
        """
        if self.futures_price is None or self.spot_price == 0:
            return None
        basis = self.futures_price - self.spot_price
        return (basis / self.spot_price) * 100


@dataclass
class HybridStrategyConfig:
    """Configuration for hybrid trading strategies.

    Allows independent configuration of spot and CFM components
    within a hybrid strategy.
    """

    # General settings
    enabled: bool = True
    base_symbol: str = "BTC"
    quote_currency: str = "USD"

    # Spot trading settings
    enable_spot: bool = True
    spot_position_size_pct: float = 0.25  # % of equity
    spot_symbol: str = ""  # Auto-generated if empty

    # CFM futures settings
    enable_cfm: bool = True
    cfm_position_size_pct: float = 0.25  # % of equity
    cfm_max_leverage: int = 5
    cfm_default_leverage: int = 1
    cfm_symbol: str = ""  # Auto-generated if empty

    # Basis trading settings (for strategies that exploit basis)
    basis_entry_threshold_pct: float = 0.5  # Enter when basis > 0.5%
    basis_exit_threshold_pct: float = 0.1  # Exit when basis < 0.1%

    # Risk settings
    max_total_exposure_pct: float = 0.8
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10

    def __post_init__(self) -> None:
        """Generate symbol names if not provided."""
        if not self.spot_symbol:
            self.spot_symbol = f"{self.base_symbol}-{self.quote_currency}"


@dataclass
class HybridPositionState:
    """State of positions across spot and CFM markets."""

    # Spot position
    spot_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    spot_entry_price: Decimal | None = None
    spot_side: str = "flat"  # "long", "short", "flat"

    # CFM position
    cfm_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    cfm_entry_price: Decimal | None = None
    cfm_side: str = "flat"  # "long", "short", "flat"
    cfm_leverage: int = 1

    @property
    def has_spot_position(self) -> bool:
        """Check if there's an open spot position."""
        return self.spot_quantity != 0

    @property
    def has_cfm_position(self) -> bool:
        """Check if there's an open CFM position."""
        return self.cfm_quantity != 0

    @property
    def is_basis_position(self) -> bool:
        """Check if this represents a basis trade (long spot, short futures)."""
        return (
            self.spot_side == "long"
            and self.cfm_side == "short"
            and self.has_spot_position
            and self.has_cfm_position
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "spot_quantity": str(self.spot_quantity),
            "spot_entry_price": str(self.spot_entry_price) if self.spot_entry_price else None,
            "spot_side": self.spot_side,
            "cfm_quantity": str(self.cfm_quantity),
            "cfm_entry_price": str(self.cfm_entry_price) if self.cfm_entry_price else None,
            "cfm_side": self.cfm_side,
            "cfm_leverage": self.cfm_leverage,
        }
