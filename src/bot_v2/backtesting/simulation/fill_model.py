"""Order fill model for realistic backtesting simulation."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from bot_v2.features.brokerages.core.interfaces import Candle, Order, OrderSide, OrderType


@dataclass
class FillResult:
    """Result of an order fill attempt."""

    filled: bool
    fill_price: Decimal | None = None
    fill_quantity: Decimal | None = None
    fill_time: datetime | None = None
    is_maker: bool = False  # True if maker (added liquidity), False if taker
    slippage_bps: Decimal | None = None  # Slippage in basis points
    reason: str | None = None  # Why order wasn't filled (if applicable)


class OrderFillModel:
    """
    Realistic order fill simulation with slippage, spread, and volume modeling.

    This model attempts to accurately simulate how orders would fill on
    Coinbase Advanced Trade, accounting for:
    - Market microstructure (bid/ask spread)
    - Slippage based on order size and liquidity
    - Volume requirements for limit orders
    - Queue priority for limit orders (optional)
    """

    def __init__(
        self,
        slippage_bps: dict[str, Decimal] | None = None,
        spread_impact_pct: Decimal = Decimal("0.5"),  # Apply 50% of spread
        limit_volume_threshold: Decimal = Decimal("2.0"),  # 2x order size
        enable_queue_priority: bool = False,
    ):
        """
        Initialize fill model.

        Args:
            slippage_bps: Per-symbol slippage in basis points (default: 2 bps BTC/ETH, 5 bps others)
            spread_impact_pct: Fraction of spread to apply (0-1, default: 0.5 = 50%)
            limit_volume_threshold: Minimum bar volume / order size ratio for limit fills
            enable_queue_priority: Model partial fills based on queue position
        """
        self.slippage_bps = slippage_bps or {}
        self.spread_impact_pct = spread_impact_pct
        self.limit_volume_threshold = limit_volume_threshold
        self.enable_queue_priority = enable_queue_priority

        # Default slippage for common pairs
        self._default_slippage = {
            "BTC": Decimal("2"),  # 2 bps for BTC pairs
            "ETH": Decimal("2"),  # 2 bps for ETH pairs
        }

    def fill_market_order(
        self,
        order: Order,
        current_bar: Candle,
        best_bid: Decimal,
        best_ask: Decimal,
        next_bar: Candle | None = None,
    ) -> FillResult:
        """
        Simulate market order fill.

        Market orders are assumed to fill immediately at the next bar's open price,
        plus slippage and spread impact.

        Args:
            order: Market order to fill
            current_bar: Current candle (order placed during this bar)
            best_bid: Current best bid price
            best_ask: Current best ask price
            next_bar: Next candle (order fills at open of next bar)

        Returns:
            FillResult with fill details
        """
        # If no next bar, use current bar close
        fill_price_base = next_bar.open if next_bar else current_bar.close

        # Get slippage for this symbol
        slippage = self._get_slippage(order.symbol)

        # Calculate spread
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / Decimal("2")

        # Apply slippage and spread impact
        if order.side == OrderSide.BUY:
            # Buying: pay spread impact + slippage
            spread_cost = spread * self.spread_impact_pct / Decimal("2")
            slippage_cost = fill_price_base * slippage / Decimal("10000")
            fill_price = fill_price_base + spread_cost + slippage_cost
        else:  # SELL
            # Selling: lose spread impact + slippage
            spread_cost = spread * self.spread_impact_pct / Decimal("2")
            slippage_cost = fill_price_base * slippage / Decimal("10000")
            fill_price = fill_price_base - spread_cost - slippage_cost

        # Calculate actual slippage in bps
        actual_slippage_bps = abs(fill_price - mid_price) / mid_price * Decimal("10000")

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            fill_time=next_bar.ts if next_bar else current_bar.ts,
            is_maker=False,  # Market orders are always taker
            slippage_bps=actual_slippage_bps,
        )

    def try_fill_limit_order(
        self,
        order: Order,
        current_bar: Candle,
        best_bid: Decimal,
        best_ask: Decimal,
    ) -> FillResult:
        """
        Attempt to fill a limit order based on bar price action and volume.

        Limit orders fill if:
        1. Price is touched (limit price within bar's high/low range)
        2. Sufficient volume (bar volume > threshold * order size)

        Args:
            order: Limit order to attempt filling
            current_bar: Current candle
            best_bid: Current best bid
            best_ask: Current best ask

        Returns:
            FillResult indicating if order filled
        """
        assert order.price is not None, "Limit order must have price"

        limit_price = order.price
        is_buy = order.side == OrderSide.BUY

        # Check if price was touched
        touched = self._is_price_touched(
            limit_price=limit_price,
            is_buy=is_buy,
            bar_high=current_bar.high,
            bar_low=current_bar.low,
        )

        if not touched:
            return FillResult(
                filled=False,
                reason=f"Price not touched (limit: {limit_price}, high: {current_bar.high}, low: {current_bar.low})",
            )

        # Check volume requirement
        if not self._has_sufficient_volume(order.quantity, current_bar.volume):
            return FillResult(
                filled=False,
                reason=f"Insufficient volume (bar: {current_bar.volume}, required: {order.quantity * self.limit_volume_threshold})",
            )

        # Determine fill price
        # Conservative: assume fill at limit price (best case for trader)
        # In reality, could get price improvement, but we don't model that
        fill_price = limit_price

        # Calculate queue priority fill percentage if enabled
        fill_pct = Decimal("1.0")
        if self.enable_queue_priority:
            fill_pct = self._estimate_queue_fill_percentage(
                order=order,
                bar=current_bar,
                best_bid=best_bid,
                best_ask=best_ask,
            )

        fill_quantity = order.quantity * fill_pct

        # If fill percentage is very low, consider it unfilled
        if fill_pct < Decimal("0.1"):  # Less than 10% filled
            return FillResult(
                filled=False,
                reason=f"Partial fill too small (queue fill: {fill_pct * 100}%)",
            )

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            fill_time=current_bar.ts,
            is_maker=True,  # Limit orders are maker when filled
            slippage_bps=Decimal("0"),  # No slippage for limit orders at limit price
        )

    def try_fill_stop_order(
        self,
        order: Order,
        current_bar: Candle,
        best_bid: Decimal,
        best_ask: Decimal,
        next_bar: Candle | None = None,
    ) -> FillResult:
        """
        Attempt to fill a stop order.

        Stop orders trigger when the stop price is breached, then fill as market order.

        Args:
            order: Stop order to attempt filling
            current_bar: Current candle
            best_bid: Current best bid
            best_ask: Current best ask
            next_bar: Next candle (for market fill after trigger)

        Returns:
            FillResult indicating if order triggered and filled
        """
        assert order.stop_price is not None, "Stop order must have stop_price"

        stop_price = order.stop_price
        is_buy_stop = order.side == OrderSide.BUY  # Buy stop triggers on upward move

        # Check if stop was triggered
        triggered = self._is_stop_triggered(
            stop_price=stop_price,
            is_buy_stop=is_buy_stop,
            bar_high=current_bar.high,
            bar_low=current_bar.low,
        )

        if not triggered:
            return FillResult(
                filled=False,
                reason=f"Stop not triggered (stop: {stop_price}, high: {current_bar.high}, low: {current_bar.low})",
            )

        # Stop triggered, fill as market order
        # Create a temporary market order for fill simulation
        market_order = Order(
            id=order.id,
            client_id=order.client_id,
            symbol=order.symbol,
            side=order.side,
            type=OrderType.MARKET,
            quantity=order.quantity,
            tif=order.tif,
            status=order.status,
            submitted_at=order.submitted_at,
            updated_at=order.updated_at,
        )

        return self.fill_market_order(
            order=market_order,
            current_bar=current_bar,
            best_bid=best_bid,
            best_ask=best_ask,
            next_bar=next_bar,
        )

    def _get_slippage(self, symbol: str) -> Decimal:
        """Get slippage in basis points for a symbol."""
        if symbol in self.slippage_bps:
            return self.slippage_bps[symbol]

        # Use default based on base asset
        for base in self._default_slippage:
            if symbol.startswith(base):
                return self._default_slippage[base]

        # Default for unknown symbols
        return Decimal("5")  # 5 bps

    def _is_price_touched(
        self,
        limit_price: Decimal,
        is_buy: bool,
        bar_high: Decimal,
        bar_low: Decimal,
    ) -> bool:
        """Check if limit price was touched during the bar."""
        if is_buy:
            # Buy limit: triggers when price drops to limit or below
            return bar_low <= limit_price
        else:
            # Sell limit: triggers when price rises to limit or above
            return bar_high >= limit_price

    def _has_sufficient_volume(
        self,
        order_size: Decimal,
        bar_volume: Decimal,
    ) -> bool:
        """Check if bar has sufficient volume to fill order."""
        required_volume = order_size * self.limit_volume_threshold
        return bar_volume >= required_volume

    def _is_stop_triggered(
        self,
        stop_price: Decimal,
        is_buy_stop: bool,
        bar_high: Decimal,
        bar_low: Decimal,
    ) -> bool:
        """Check if stop price was triggered during the bar."""
        if is_buy_stop:
            # Buy stop: triggers when price rises to stop or above
            return bar_high >= stop_price
        else:
            # Sell stop: triggers when price drops to stop or below
            return bar_low <= stop_price

    def _estimate_queue_fill_percentage(
        self,
        order: Order,
        bar: Candle,
        best_bid: Decimal,
        best_ask: Decimal,
    ) -> Decimal:
        """
        Estimate what percentage of limit order would fill based on queue priority.

        This is a simplified model. In reality, queue priority depends on:
        - Time priority (earlier orders fill first)
        - Price priority (better prices fill first)
        - Order book depth at the limit price

        For simulation, we use volume as a proxy:
        - If bar volume >> order size, assume high fill rate
        - If bar volume ~ order size, assume partial fill

        Args:
            order: Limit order
            bar: Current candle
            best_bid: Current best bid
            best_ask: Current best ask

        Returns:
            Fill percentage (0-1)
        """
        # Estimate what fraction of the bar's volume was traded at our limit price
        # This is a rough heuristic
        volume_ratio = bar.volume / order.quantity

        if volume_ratio >= Decimal("10"):
            return Decimal("1.0")  # Full fill likely
        elif volume_ratio >= Decimal("5"):
            return Decimal("0.8")  # 80% fill
        elif volume_ratio >= Decimal("2"):
            return Decimal("0.5")  # 50% fill
        else:
            return Decimal("0.2")  # 20% fill
