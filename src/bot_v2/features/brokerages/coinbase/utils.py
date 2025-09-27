from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Any, Protocol

from ..core.interfaces import MarketType, Product
from .errors import InvalidRequestError, NotFoundError
from .models import to_product

logger = logging.getLogger(__name__)

class ProductClient(Protocol):
    def get_products(self) -> dict[str, Any]:
        ...

    def get_product(self, symbol: str) -> dict[str, Any]:
        ...

def quantize_to_increment(value: Decimal, increment: Decimal) -> Decimal:
    if increment is None or increment == 0:
        return value
    # quantize requires exponent like Decimal('0.01'); handle arbitrary increments by division
    # Compute floor to nearest multiple of increment
    q = (value / increment).to_integral_value(rounding=ROUND_DOWN)
    return (q * increment).quantize(increment)
def enforce_perp_rules(product: Product, qty: Decimal, price: Decimal | None = None) -> tuple[Decimal, Decimal | None]:
    """Enforce perpetual product rules for quantity and price.

    Args:
        product: Product with rules to enforce
        qty: Desired quantity (will be quantized to step_size)
        price: Optional price (will be quantized to price_increment)

    Returns:
        Tuple of (adjusted_qty, adjusted_price)

    Raises:
        InvalidRequestError: If values violate minimum requirements
    """
    # Quantize quantity to step_size
    adjusted_qty = quantize_to_increment(qty, product.step_size)

    # Check minimum size
    if adjusted_qty < product.min_size:
        raise InvalidRequestError(
            f"Quantity {adjusted_qty} below minimum size {product.min_size} for {product.symbol}"
        )

    # Handle price if provided
    adjusted_price = None
    if price is not None:
        # Quantize price to price_increment
        adjusted_price = quantize_to_increment(price, product.price_increment)

        # Check min_notional if applicable
        if product.min_notional is not None:
            notional = adjusted_qty * adjusted_price
            if notional < product.min_notional:
                raise InvalidRequestError(
                    f"Notional {notional} below minimum {product.min_notional} for {product.symbol}"
                )

    return adjusted_qty, adjusted_price

@dataclass
class MarkCache:
    """Cache for mark prices with TTL.

    Maintains latest mark price per symbol with staleness tracking.
    """
    ttl_seconds: int = 30  # Default 30s TTL

    def __post_init__(self) -> None:
        self._marks: dict[str, tuple[Decimal, datetime]] = {}
        self._warned_symbols: set[str] = set()  # Track symbols we've warned about

    def set_mark(self, symbol: str, mark: Decimal) -> None:
        """Set the mark price for a symbol."""
        self._marks[symbol] = (mark, datetime.utcnow())

    def get_mark(self, symbol: str) -> Decimal | None:
        """Get the mark price if not stale.

        Returns:
            Mark price or None if missing/stale
        """
        if symbol not in self._marks:
            return None

        mark, timestamp = self._marks[symbol]
        age = datetime.utcnow() - timestamp

        if age > timedelta(seconds=self.ttl_seconds):
            if symbol not in self._warned_symbols:
                logger.warning("Mark price for %s is stale (age: %s)", symbol, age)
                self._warned_symbols.add(symbol)
            return None

        self._warned_symbols.discard(symbol)
        return mark

@dataclass
class FundingCalculator:
    """Calculate funding accrual for perpetual positions.

    Implements discrete funding at scheduled times.
    Convention: Positive funding rate means longs pay shorts.
    """

    def __init__(self) -> None:
        self._last_funding_times: dict[str, datetime] = {}
        self._warned_symbols: set[str] = set()

    def accrue_if_due(
        self,
        symbol: str,
        position_size: Decimal,
        position_side: str,  # "long" or "short"
        mark_price: Decimal,
        funding_rate: Decimal | None,
        next_funding_time: datetime | None,
        now: datetime | None = None
    ) -> Decimal:
        """Calculate funding delta if due.

        Args:
            symbol: Trading symbol
            position_size: Absolute position size (positive)
            position_side: "long" or "short"
            mark_price: Current mark price
            funding_rate: Funding rate per interval (e.g., 0.0001 for 0.01%)
            next_funding_time: Next scheduled funding time
            now: Current time (default: utcnow)

        Returns:
            Funding delta (negative = payment, positive = receipt)
        """
        if now is None:
            now = datetime.utcnow()

        # Skip if no funding data
        if funding_rate is None or next_funding_time is None:
            if symbol not in self._warned_symbols:
                logger.warning("No funding data available for %s", symbol)
                self._warned_symbols.add(symbol)
            return Decimal("0")

        # Check if funding is due
        last_funding = self._last_funding_times.get(symbol)

        # If we haven't tracked this symbol yet, check if we're past funding time
        if last_funding is None:
            if now >= next_funding_time:
                # Funding is due - record it
                self._last_funding_times[symbol] = next_funding_time
                # But don't accrue on first observation (avoid double-counting)
                return Decimal("0")
            else:
                # Not due yet
                return Decimal("0")

        # Check if new funding period
        if next_funding_time > last_funding and now >= next_funding_time:
            # Funding is due
            self._last_funding_times[symbol] = next_funding_time

            # Calculate funding payment
            # notional = position_size * mark_price
            # funding_payment = notional * funding_rate
            notional = position_size * mark_price
            funding_payment = notional * funding_rate

            # Apply side logic:
            # Positive funding rate: longs pay (negative delta), shorts receive (positive delta)
            # Negative funding rate: shorts pay (negative delta), longs receive (positive delta)
            if position_side.lower() == "long":
                return -funding_payment  # Long pays when rate positive
            else:  # short
                return funding_payment   # Short receives when rate positive

        return Decimal("0")  # No funding due

@dataclass
class PositionState:
    """Track state of a perpetual position for PnL calculation."""
    symbol: str
    side: str  # "long" or "short"
    qty: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    last_funding_ts: datetime | None = None

    def update_from_fill(self, fill_qty: Decimal, fill_price: Decimal, fill_side: str) -> Decimal:
        """Update position from a fill.

        Args:
            fill_qty: Absolute fill quantity
            fill_price: Fill execution price
            fill_side: "buy" or "sell"

        Returns:
            Realized PnL from this fill
        """
        realized_delta = Decimal("0")

        # Determine if this is increasing or reducing
        is_buy = fill_side.lower() == "buy"
        is_long = self.side.lower() == "long"

        if (is_long and is_buy) or (not is_long and not is_buy):
            # Increasing position - update weighted average entry
            if self.qty > 0:
                # Weighted average
                total_value = (self.qty * self.entry_price) + (fill_qty * fill_price)
                self.qty += fill_qty
                self.entry_price = total_value / self.qty
            else:
                # New position
                self.qty = fill_qty
                self.entry_price = fill_price
        else:
            # Reducing position - realize PnL
            close_qty = min(fill_qty, self.qty)

            if is_long:
                # Long position: profit when sell price > entry
                realized_delta = (fill_price - self.entry_price) * close_qty
            else:
                # Short position: profit when buy price < entry
                realized_delta = (self.entry_price - fill_price) * close_qty

            self.realized_pnl += realized_delta
            self.qty -= close_qty

            # If position flipped, reset entry
            if fill_qty > close_qty:
                # Flipped to opposite side
                self.qty = fill_qty - close_qty
                self.entry_price = fill_price
                self.side = "long" if is_buy else "short"

        return realized_delta

    def get_unrealized_pnl(self, mark_price: Decimal | None) -> Decimal:
        """Calculate unrealized PnL at current mark.

        Returns:
            Unrealized PnL or 0 if no mark price
        """
        if mark_price is None or self.qty == 0:
            return Decimal("0")

        if self.side.lower() == "long":
            return (mark_price - self.entry_price) * self.qty
        else:  # short
            return (self.entry_price - mark_price) * self.qty

@dataclass
class ProductCatalog:
    ttl_seconds: int = 900

    def __post_init__(self) -> None:
        self._cache: dict[str, Product] = {}
        self._last_refresh: datetime | None = None

    def _is_expired(self) -> bool:
        if not self._last_refresh:
            return True
        return datetime.utcnow() - self._last_refresh > timedelta(seconds=self.ttl_seconds)

    def refresh(self, client: ProductClient) -> None:
        data = client.get_products() or {}
        items = data.get("products") or data.get("data") or []
        cache: dict[str, Product] = {}
        for p in items:
            prod = to_product(p)
            cache[prod.symbol] = prod
        if cache:
            self._cache = cache
            self._last_refresh = datetime.utcnow()

    def get(self, client: ProductClient, symbol: str) -> Product:
        if self._is_expired():
            self.refresh(client)
        prod = self._cache.get(symbol)
        if not prod:
            # Try refresh once more in case of new listing
            self.refresh(client)
            prod = self._cache.get(symbol)
        if not prod:
            raise NotFoundError(f"Product not found: {symbol}")
        return prod

    def get_funding(self, client: ProductClient, symbol: str) -> tuple[Decimal | None, datetime | None]:
        """Get funding rate and next funding time for a perpetual.

        Returns:
            Tuple of (funding_rate, next_funding_time) or (None, None) if not a perpetual
        """
        prod = self.get(client, symbol)
        if prod.market_type != MarketType.PERPETUAL:
            return None, None
        return prod.funding_rate, prod.next_funding_time
