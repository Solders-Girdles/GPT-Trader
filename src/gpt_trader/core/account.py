"""Core account types used across all slices."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from gpt_trader.core.trading import OrderSide, OrderStatus, OrderType, TimeInForce

# Product type for distinguishing spot from derivatives
ProductType = Literal["SPOT", "FUTURE"]


@dataclass
class Order:
    """Trading order representation."""

    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")
    price: Decimal | None = None
    stop_price: Decimal | None = None
    tif: TimeInForce = TimeInForce.GTC
    client_id: str | None = None
    avg_fill_price: Decimal | None = None
    submitted_at: datetime | None = None
    updated_at: datetime | None = None
    created_at: datetime | None = None


@dataclass
class Position:
    """Trading position representation.

    Supports both spot and derivatives (CFM futures, INTX perpetuals).
    """

    symbol: str
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    side: str  # "long" or "short"
    leverage: int | None = None
    # CFM/derivatives fields
    liquidation_price: Decimal | None = None
    product_type: ProductType = "SPOT"
    contract_expiry: datetime | None = None

    @property
    def is_futures(self) -> bool:
        """Check if this is a futures position."""
        return self.product_type == "FUTURE"

    @property
    def liquidation_buffer_pct(self) -> float | None:
        """Calculate distance to liquidation as percentage.

        Returns None if liquidation price not available.
        """
        if self.liquidation_price is None or self.mark_price is None:
            return None
        if self.mark_price == 0:
            return None

        if self.side == "long":
            # Long: liquidation when price drops
            buffer = (self.mark_price - self.liquidation_price) / self.mark_price
        else:
            # Short: liquidation when price rises
            buffer = (self.liquidation_price - self.mark_price) / self.mark_price

        return float(buffer * 100)


@dataclass
class Balance:
    """Account balance for a single asset."""

    asset: str
    total: Decimal
    available: Decimal
    hold: Decimal = Decimal("0")


@dataclass
class CFMBalance:
    """CFM (Coinbase Financial Markets) futures balance summary.

    Tracks margin, buying power, and liquidation metrics for US-regulated futures.
    """

    futures_buying_power: Decimal
    total_usd_balance: Decimal
    available_margin: Decimal
    initial_margin: Decimal
    unrealized_pnl: Decimal
    daily_realized_pnl: Decimal
    liquidation_threshold: Decimal
    liquidation_buffer_amount: Decimal
    liquidation_buffer_percentage: float  # As percentage (e.g., 261.09)

    @property
    def is_at_risk(self) -> bool:
        """Check if account is approaching liquidation (buffer < 50%)."""
        return self.liquidation_buffer_percentage < 50.0

    @property
    def margin_utilization_pct(self) -> float:
        """Calculate margin utilization as percentage."""
        if self.total_usd_balance == 0:
            return 0.0
        return float(self.initial_margin / self.total_usd_balance * 100)


@dataclass
class UnifiedBalance:
    """Combined balance across spot and CFM."""

    spot_balance: Decimal
    cfm_balance: Decimal
    cfm_available_margin: Decimal
    cfm_buying_power: Decimal
    total_equity: Decimal

    @property
    def has_cfm(self) -> bool:
        """Check if CFM balance is present."""
        return self.cfm_balance > 0
