"""Utility helpers for Coinbase brokerage integrations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Any, Protocol, cast

from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError, NotFoundError
from bot_v2.features.brokerages.coinbase.models import to_product
from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.utilities import utc_now
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_utilities")

__all__ = [
    "ProductClient",
    "quantize_to_increment",
    "enforce_perp_rules",
    "MarkCache",
    "FundingEvent",
    "FundingCalculator",
    "PositionState",
    "ProductCatalog",
    "LiquidationCalculator",
]


class ProductClient(Protocol):
    def get_products(self) -> dict[str, Any]: ...

    def get_product(self, symbol: str) -> dict[str, Any]: ...


def quantize_to_increment(
    value: Decimal, increment: Decimal, *, rounding: str = ROUND_DOWN
) -> Decimal:
    if increment is None or increment == 0:
        return value
    # quantize requires exponent like Decimal('0.01'); handle arbitrary increments by division
    factor = (value / increment).to_integral_value(rounding=rounding)
    return (factor * increment).quantize(increment)


def enforce_perp_rules(
    product: Product,
    quantity: Decimal | None = None,
    price: Decimal | None = None,
) -> tuple[Decimal, Decimal | None]:
    """Enforce perpetual product rules for quantity and price.

    Args:
        product: Product with rules to enforce
        quantity: Desired quantity (will be quantized to step_size)
        price: Optional price (will be quantized to price_increment)

    Returns:
        Tuple of (adjusted_quantity, adjusted_price)

    Raises:
        InvalidRequestError: If values violate minimum requirements
    """
    if quantity is None:
        raise TypeError("enforce_perp_rules() missing required argument: 'quantity'")

    # Quantize quantity to step_size
    adjusted_quantity = quantize_to_increment(quantity, product.step_size)

    # Check minimum size
    if adjusted_quantity < product.min_size:
        raise InvalidRequestError(
            f"Quantity {adjusted_quantity} below minimum size {product.min_size} for {product.symbol}"
        )

    # Handle price if provided
    adjusted_price = None
    if price is not None:
        # Quantize price to price_increment
        adjusted_price = quantize_to_increment(price, product.price_increment)

        # Check min_notional if applicable
        if product.min_notional is not None:
            notional = adjusted_quantity * adjusted_price
            if notional < product.min_notional:
                raise InvalidRequestError(
                    f"Notional {notional} below minimum {product.min_notional} for {product.symbol}"
                )

    return adjusted_quantity, adjusted_price


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
        self._marks[symbol] = (mark, utc_now())

    def get_mark(self, symbol: str) -> Decimal | None:
        """Get the mark price if not stale.

        Returns:
            Mark price or None if missing/stale
        """
        if symbol not in self._marks:
            return None

        mark, timestamp = self._marks[symbol]
        age = utc_now() - timestamp

        if age > timedelta(seconds=self.ttl_seconds):
            if symbol not in self._warned_symbols:
                logger.warning("Mark price for %s is stale (age: %s)", symbol, age)
                self._warned_symbols.add(symbol)
            return None

        self._warned_symbols.discard(symbol)
        return mark


@dataclass
class FundingEvent:
    """Represents a single funding accrual event."""

    timestamp: datetime
    symbol: str
    rate: Decimal
    position_size: Decimal
    position_side: str
    mark_price: Decimal
    amount: Decimal  # Negative = payment, positive = receipt


@dataclass
class FundingCalculator:
    """Calculate funding accrual for perpetual positions.

    Implements discrete funding at scheduled times with full history tracking.

    Coinbase Perpetuals Funding (as of October 2025):
    ---------------------------------------------------
    US Perpetuals (CFM):
    - Accrual: Hourly (every hour on the hour)
    - Settlement: Twice daily (typically 00:00 UTC and 12:00 UTC)
    - Rate Convention: Positive funding rate means longs pay shorts

    International Perpetuals (INTX):
    - Accrual: Every 8 hours (00:00, 08:00, 16:00 UTC)
    - Settlement: At each funding time
    - Rate Convention: Positive funding rate means longs pay shorts

    Implementation Notes:
    ---------------------
    This calculator tracks accrual events (when funding is calculated).
    Actual settlement (transfer of funds) may occur at different times
    depending on the venue. The accrual happens when `now >= next_funding_time`,
    and the calculator uses the API-provided `next_funding_time` to determine
    when to apply funding charges.

    For accurate PnL tracking:
    - Call `accrue_if_due()` on each market data update
    - Ensure `next_funding_time` is refreshed from the API
    - Use `get_funding_history()` for detailed funding attribution

    Convention: Positive funding rate means longs pay shorts.
    """

    def __init__(self, *, track_history: bool = True, max_history_per_symbol: int = 1000) -> None:
        """Initialize funding calculator.

        Args:
            track_history: Whether to track individual funding events
            max_history_per_symbol: Maximum number of events to retain per symbol
        """
        self._last_funding_times: dict[str, datetime] = {}
        self._warned_symbols: set[str] = set()
        self._track_history = track_history
        self._max_history = max_history_per_symbol
        self._funding_history: dict[str, list[FundingEvent]] = {}

    def accrue_if_due(
        self,
        symbol: str,
        position_size: Decimal,
        position_side: str,  # "long" or "short"
        mark_price: Decimal,
        funding_rate: Decimal | None,
        next_funding_time: datetime | None,
        now: datetime | None = None,
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
            now = utc_now()

        # Skip if no funding data
        if funding_rate is None or next_funding_time is None:
            if symbol not in self._warned_symbols:
                logger.warning("No funding data available for %s", symbol)
                self._warned_symbols.add(symbol)
            return Decimal("0")

        def _normalize(dt: datetime | None) -> datetime | None:
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt
            return dt.replace(tzinfo=None)

        now_cmp = cast(datetime, _normalize(now))
        next_cmp = _normalize(next_funding_time)
        if next_cmp is None:
            return Decimal("0")
        next_cmp = cast(datetime, next_cmp)

        # Check if funding is due
        last_funding = self._last_funding_times.get(symbol)

        # If we haven't tracked this symbol yet, check if we're past funding time
        if last_funding is None:
            if now_cmp >= next_cmp:
                # Funding is due - record it
                self._last_funding_times[symbol] = next_funding_time
                # But don't accrue on first observation (avoid double-counting)
                return Decimal("0")
            else:
                # Not due yet
                return Decimal("0")

        # Check if new funding period
        if next_funding_time > last_funding and now_cmp >= next_cmp:
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
                funding_delta = -funding_payment  # Long pays when rate positive
            else:  # short
                funding_delta = funding_payment  # Short receives when rate positive

            # Track funding event in history
            if self._track_history:
                self._record_funding_event(
                    timestamp=next_funding_time,
                    symbol=symbol,
                    rate=funding_rate,
                    position_size=position_size,
                    position_side=position_side,
                    mark_price=mark_price,
                    amount=funding_delta,
                )

            return funding_delta

        return Decimal("0")  # No funding due

    def _record_funding_event(
        self,
        timestamp: datetime,
        symbol: str,
        rate: Decimal,
        position_size: Decimal,
        position_side: str,
        mark_price: Decimal,
        amount: Decimal,
    ) -> None:
        """Record a funding event in history."""
        event = FundingEvent(
            timestamp=timestamp,
            symbol=symbol,
            rate=rate,
            position_size=position_size,
            position_side=position_side,
            mark_price=mark_price,
            amount=amount,
        )

        if symbol not in self._funding_history:
            self._funding_history[symbol] = []

        self._funding_history[symbol].append(event)

        # Trim history if exceeds max
        if len(self._funding_history[symbol]) > self._max_history:
            self._funding_history[symbol] = self._funding_history[symbol][-self._max_history :]

    def get_funding_history(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[FundingEvent]:
        """Get funding event history for a symbol.

        Args:
            symbol: Trading symbol
            since: Only return events after this timestamp
            limit: Maximum number of events to return (most recent first)

        Returns:
            List of funding events, newest first
        """
        if not self._track_history:
            logger.warning("Funding history tracking is disabled")
            return []

        events = self._funding_history.get(symbol, [])

        # Filter by timestamp if requested
        if since is not None:
            events = [e for e in events if e.timestamp >= since]

        # Sort newest first
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        if limit is not None:
            events = events[:limit]

        return events

    def get_total_funding(self, symbol: str) -> Decimal:
        """Get total funding paid/received for a symbol.

        Returns:
            Total funding amount (negative = paid, positive = received)
        """
        if not self._track_history:
            logger.warning("Funding history tracking is disabled")
            return Decimal("0")

        events = self._funding_history.get(symbol, [])
        return sum((e.amount for e in events), Decimal("0"))

    def clear_history(self, symbol: str | None = None) -> None:
        """Clear funding history.

        Args:
            symbol: Symbol to clear, or None to clear all
        """
        if symbol is None:
            self._funding_history.clear()
        else:
            self._funding_history.pop(symbol, None)


@dataclass
class PositionState:
    """Track state of a perpetual position for PnL calculation."""

    symbol: str
    side: str  # "long" or "short"
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    last_funding_ts: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))

    def update_from_fill(
        self, fill_quantity: Decimal, fill_price: Decimal, fill_side: str
    ) -> Decimal:
        """Update position from a fill.

        Args:
            fill_quantity: Absolute fill quantity
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
            if self.quantity > 0:
                # Weighted average
                total_value = (self.quantity * self.entry_price) + (fill_quantity * fill_price)
                self.quantity += fill_quantity
                self.entry_price = total_value / self.quantity
            else:
                # New position
                self.quantity = fill_quantity
                self.entry_price = fill_price
        else:
            # Reducing position - realize PnL
            close_quantity = min(fill_quantity, self.quantity)

            if is_long:
                # Long position: profit when sell price > entry
                realized_delta = (fill_price - self.entry_price) * close_quantity
            else:
                # Short position: profit when buy price < entry
                realized_delta = (self.entry_price - fill_price) * close_quantity

            self.realized_pnl += realized_delta
            self.quantity -= close_quantity

            # If position flipped, reset entry
            if fill_quantity > close_quantity:
                # Flipped to opposite side
                self.quantity = fill_quantity - close_quantity
                self.entry_price = fill_price
                self.side = "long" if is_buy else "short"

        return realized_delta

    def get_unrealized_pnl(self, mark_price: Decimal | None) -> Decimal:
        """Calculate unrealized PnL at current mark.

        Returns:
            Unrealized PnL or 0 if no mark price
        """
        if mark_price is None or self.quantity == 0:
            return Decimal("0")

        if self.side.lower() == "long":
            return (mark_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - mark_price) * self.quantity


@dataclass
class ProductCatalog:
    ttl_seconds: int = 900

    def __post_init__(self) -> None:
        self._cache: dict[str, Product] = {}
        self._last_refresh: datetime | None = None

    def _is_expired(self) -> bool:
        if not self._last_refresh:
            return True
        now = utc_now()
        last_refresh = self._last_refresh
        if last_refresh.tzinfo is None and now.tzinfo is not None:
            last_refresh = last_refresh.replace(tzinfo=timezone.utc)
        return bool(now - last_refresh > timedelta(seconds=self.ttl_seconds))

    def refresh(self, client: ProductClient) -> None:
        data = client.get_products() or {}
        items = data.get("products") or data.get("data") or []
        cache: dict[str, Product] = {}
        for p in items:
            prod = to_product(p)
            cache[prod.symbol] = prod
        if cache:
            self._cache = cache
            self._last_refresh = utc_now()

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

    def get_funding(
        self, client: ProductClient, symbol: str
    ) -> tuple[Decimal | None, datetime | None]:
        """Get funding rate and next funding time for a perpetual.

        Returns:
            Tuple of (funding_rate, next_funding_time) or (None, None) if not a perpetual
        """
        prod = self.get(client, symbol)
        if prod.market_type != MarketType.PERPETUAL:
            return None, None
        return prod.funding_rate, prod.next_funding_time


@dataclass
class LiquidationCalculator:
    """Calculate liquidation prices for leveraged perpetual positions.

    Coinbase Margin Model (October 2025):
    --------------------------------------
    - Initial Margin: 1 / leverage (e.g., 10x leverage = 10% initial margin)
    - Maintenance Margin: Typically 50% of initial margin
    - Liquidation: Occurs when equity falls below maintenance margin requirement

    Formula:
    --------
    For Long Positions:
        liquidation_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)

    For Short Positions:
        liquidation_price = entry_price * (1 + (1/leverage) - maintenance_margin_rate)

    Where:
        maintenance_margin_rate = maintenance_margin / initial_margin
        Typically: 0.5 * (1/leverage) for Coinbase

    Example:
    --------
    Long BTC-PERP at $50,000 with 10x leverage:
        initial_margin = 1/10 = 0.1 (10%)
        maintenance_margin = 0.05 (5%)
        liquidation_price = 50000 * (1 - 0.1 + 0.05) = $47,500

    Important Notes:
    ----------------
    - Actual liquidation prices may vary based on:
      * Accumulated funding payments
      * Position size (larger positions may have higher margin requirements)
      * Market volatility adjustments
      * Exchange-specific risk parameters
    - This calculator provides estimates; always verify with exchange API
    """

    # Default maintenance margin as fraction of initial margin
    DEFAULT_MAINTENANCE_MARGIN_RATIO: Decimal = Decimal("0.5")

    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        leverage: int,
        side: str,
        *,
        maintenance_margin_ratio: Decimal | None = None,
    ) -> Decimal:
        """Calculate liquidation price for a leveraged position.

        Args:
            entry_price: Average entry price
            leverage: Position leverage (e.g., 10 for 10x)
            side: Position side ("long" or "short")
            maintenance_margin_ratio: Maintenance margin as fraction of initial margin
                                     (default: 0.5 for 50% of initial margin)

        Returns:
            Estimated liquidation price

        Raises:
            ValueError: If leverage <= 0 or entry_price <= 0
        """
        if leverage <= 0:
            raise ValueError(f"Leverage must be positive, got {leverage}")
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")

        # Use default maintenance margin ratio if not provided
        mm_ratio = maintenance_margin_ratio or self.DEFAULT_MAINTENANCE_MARGIN_RATIO

        # Initial margin requirement as decimal (e.g., 0.1 for 10x leverage)
        initial_margin_rate = Decimal("1") / Decimal(str(leverage))

        # Maintenance margin requirement
        maintenance_margin_rate = initial_margin_rate * mm_ratio

        # Calculate liquidation price based on position side
        if side.lower() == "long":
            # Long: liquidation when price drops
            # liquidation_price = entry * (1 - initial_margin + maintenance_margin)
            liq_multiplier = Decimal("1") - initial_margin_rate + maintenance_margin_rate
            liquidation_price = entry_price * liq_multiplier
        else:  # short
            # Short: liquidation when price rises
            # liquidation_price = entry * (1 + initial_margin - maintenance_margin)
            liq_multiplier = Decimal("1") + initial_margin_rate - maintenance_margin_rate
            liquidation_price = entry_price * liq_multiplier

        return liquidation_price

    def calculate_liquidation_distance(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        leverage: int,
        side: str,
        *,
        maintenance_margin_ratio: Decimal | None = None,
    ) -> dict[str, Decimal]:
        """Calculate distance to liquidation in both price and percentage terms.

        Args:
            entry_price: Average entry price
            current_price: Current mark price
            leverage: Position leverage
            side: Position side ("long" or "short")
            maintenance_margin_ratio: Maintenance margin as fraction of initial margin

        Returns:
            Dictionary with:
                - liquidation_price: Estimated liquidation price
                - price_distance: Absolute distance to liquidation
                - percentage_distance: Percentage distance to liquidation
                - buffer_percent: Buffer as percentage of current price
        """
        liq_price = self.calculate_liquidation_price(
            entry_price, leverage, side, maintenance_margin_ratio=maintenance_margin_ratio
        )

        # Calculate distances
        if side.lower() == "long":
            # For longs, liquidation is below current price
            price_distance = current_price - liq_price
            percentage_distance = (price_distance / current_price) * Decimal("100")
        else:  # short
            # For shorts, liquidation is above current price
            price_distance = liq_price - current_price
            percentage_distance = (price_distance / current_price) * Decimal("100")

        buffer_percent = (price_distance / current_price) * Decimal("100")

        return {
            "liquidation_price": liq_price,
            "price_distance": price_distance,
            "percentage_distance": percentage_distance,
            "buffer_percent": buffer_percent,
        }

    def is_at_risk(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        leverage: int,
        side: str,
        *,
        risk_threshold_percent: Decimal = Decimal("20"),
        maintenance_margin_ratio: Decimal | None = None,
    ) -> bool:
        """Check if position is at risk of liquidation.

        Args:
            entry_price: Average entry price
            current_price: Current mark price
            leverage: Position leverage
            side: Position side ("long" or "short")
            risk_threshold_percent: Alert threshold as percentage distance (default: 20%)
            maintenance_margin_ratio: Maintenance margin as fraction of initial margin

        Returns:
            True if distance to liquidation is below risk threshold
        """
        distances = self.calculate_liquidation_distance(
            entry_price,
            current_price,
            leverage,
            side,
            maintenance_margin_ratio=maintenance_margin_ratio,
        )

        return distances["percentage_distance"] < risk_threshold_percent
