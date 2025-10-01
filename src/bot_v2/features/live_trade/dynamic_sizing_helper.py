"""Dynamic position sizing and impact calculation helpers."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderType,
    Product,
    Quote,
)
from bot_v2.features.live_trade.advanced_execution_models.models import (
    OrderConfig,
    SizingMode,
)
from bot_v2.features.live_trade.risk import (
    LiveRiskManager,
    PositionSizingAdvice,
    PositionSizingContext,
)
from bot_v2.utilities.quantities import quantity_from

__all__ = ["DynamicSizingHelper"]

logger = logging.getLogger(__name__)


class DynamicSizingHelper:
    """Handles dynamic position sizing and impact calculations."""

    def __init__(
        self,
        broker: Any,
        risk_manager: LiveRiskManager | None = None,
        config: OrderConfig | None = None,
    ) -> None:
        """
        Initialize dynamic sizing helper.

        Args:
            broker: Broker adapter instance
            risk_manager: Risk manager for sizing decisions
            config: Order configuration
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.config = config or OrderConfig()

        # Last sizing advice for diagnostics/tests
        self._last_sizing_advice: PositionSizingAdvice | None = None

        # Optional per-symbol slippage multipliers for impact-aware sizing
        self.slippage_multipliers: dict[str, Decimal] = {}
        try:
            import os

            env_val = os.getenv("SLIPPAGE_MULTIPLIERS", "")
            if env_val:
                for pair in env_val.split(","):
                    if ":" in pair:
                        sym, mult = pair.split(":", 1)
                        self.slippage_multipliers[sym.strip()] = Decimal(str(mult.strip()))
        except Exception:
            pass

    def maybe_apply_position_sizing(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        limit_price: Decimal | None,
        product: Product | None,
        quote: Quote | None,
        leverage: int | None,
    ) -> PositionSizingAdvice | None:
        """
        Apply dynamic position sizing if enabled.

        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            order_quantity: Requested quantity
            limit_price: Limit price
            product: Product info
            quote: Quote data
            leverage: Leverage multiplier

        Returns:
            Sizing advice or None
        """
        if self.risk_manager is None:
            return None

        config = getattr(self.risk_manager, "config", None)
        if not config or not getattr(config, "enable_dynamic_position_sizing", False):
            return None

        reference_price = self.determine_reference_price(
            symbol=symbol,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            quote=quote,
            product=product,
        )

        if reference_price <= 0:
            reference_price = Decimal(str(limit_price)) if limit_price is not None else Decimal("0")

        equity = self.estimate_equity()
        if equity <= 0 and reference_price > 0:
            equity = reference_price * order_quantity

        current_position_quantity = self._extract_position_quantity(symbol)

        target_leverage = (
            Decimal(str(leverage))
            if leverage is not None and leverage > 0
            else Decimal(str(getattr(config, "max_leverage", 1)))
        )

        context = PositionSizingContext(
            symbol=symbol,
            side=side.value,
            equity=equity,
            current_price=reference_price if reference_price > 0 else Decimal("0"),
            strategy_name="AdvancedExecutionEngine",
            method=getattr(config, "position_sizing_method", "intelligent"),
            current_position_quantity=current_position_quantity,
            target_leverage=target_leverage,
            strategy_multiplier=float(getattr(config, "position_sizing_multiplier", 1.0)),
            product=product,
        )

        try:
            advice = self.risk_manager.size_position(context)
        except Exception:
            logger.exception("Dynamic sizing failed for %s", symbol)
            return None

        self._last_sizing_advice = advice
        return advice

    def determine_reference_price(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        limit_price: Decimal | None,
        quote: Quote | None,
        product: Product | None,
    ) -> Decimal:
        """
        Determine reference price for sizing calculations.

        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            limit_price: Limit price
            quote: Quote data
            product: Product info

        Returns:
            Reference price
        """
        if limit_price is not None:
            try:
                return Decimal(str(limit_price))
            except Exception:
                pass

        if order_type == OrderType.MARKET and quote is not None:
            price = quote.ask if side == OrderSide.BUY else quote.bid
            if price is not None:
                return Decimal(str(price))

        if quote is not None and quote.last is not None:
            return Decimal(str(quote.last))

        if quote is None:
            broker = getattr(self, "broker", None)
            if broker is not None and hasattr(broker, "get_quote"):
                try:
                    fallback_quote = broker.get_quote(symbol)
                except Exception:
                    fallback_quote = None
                if fallback_quote is not None:
                    price = fallback_quote.ask if side == OrderSide.BUY else fallback_quote.bid
                    if price is not None:
                        try:
                            return Decimal(str(price))
                        except Exception:
                            pass
                    elif getattr(fallback_quote, "last", None) is not None:
                        try:
                            return Decimal(str(fallback_quote.last))
                        except Exception:
                            pass

        if product is not None:
            mark = getattr(product, "mark_price", None)
            if mark is not None:
                try:
                    return Decimal(str(mark))
                except Exception:
                    pass
        return Decimal("0")

    def estimate_equity(self) -> Decimal:
        """
        Estimate current equity.

        Returns:
            Estimated equity value
        """
        if self.risk_manager is not None:
            start_equity = getattr(self.risk_manager, "start_of_day_equity", None)
            if start_equity is not None:
                try:
                    value = Decimal(str(start_equity))
                    if value > 0:
                        return value
                except Exception:
                    pass

        broker = getattr(self, "broker", None)
        if broker is not None and hasattr(broker, "list_balances"):
            try:
                balances = broker.list_balances() or []
                total = Decimal("0")
                for bal in balances:
                    amount = getattr(bal, "total", None)
                    if amount is None:
                        amount = getattr(bal, "available", None)
                    if amount is not None:
                        total += Decimal(str(amount))
                if total > 0:
                    return total
            except Exception:
                pass

        return Decimal("0")

    def _extract_position_quantity(self, symbol: str) -> Decimal:
        """Extract current position quantity for symbol."""
        positions = getattr(self.risk_manager, "positions", {}) if self.risk_manager else {}
        pos = positions.get(symbol)
        if not pos:
            return Decimal("0")
        try:
            value = quantity_from(pos, default=Decimal("0"))
            return value or Decimal("0")
        except Exception:
            return Decimal("0")

    def calculate_impact_aware_size(
        self,
        symbol: str | None,
        target_notional: Decimal,
        market_snapshot: dict[str, Any],
        max_impact_bps: Decimal | None = None,
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate position size that respects slippage constraints.

        Args:
            symbol: Trading symbol
            target_notional: Target position size in USD
            market_snapshot: Market depth and liquidity data
            max_impact_bps: Maximum acceptable impact (overrides config)

        Returns:
            (adjusted_notional, expected_impact_bps)
        """
        max_impact = max_impact_bps or self.config.max_impact_bps

        l1_depth = Decimal(str(market_snapshot.get("depth_l1", 0)))
        l10_depth = Decimal(str(market_snapshot.get("depth_l10", 0)))

        if not l1_depth or not l10_depth:
            logger.warning("Insufficient depth data for impact calculation")
            return Decimal("0"), Decimal("0")

        # Binary search for max size within impact limit
        low, high = Decimal("0"), min(target_notional, l10_depth)
        best_size = Decimal("0")
        best_impact = Decimal("0")
        # Add per-symbol slippage multiplier as extra expected impact (bps)
        extra_bps = Decimal("0")
        if symbol and symbol in self.slippage_multipliers:
            try:
                extra_bps = Decimal("10000") * Decimal(str(self.slippage_multipliers[symbol]))
            except Exception:
                extra_bps = Decimal("0")

        while high - low > Decimal("1"):  # $1 precision
            mid = (low + high) / 2
            impact = self.estimate_impact(mid, l1_depth, l10_depth) + extra_bps

            if impact <= max_impact:
                best_size = mid
                best_impact = impact
                low = mid
            else:
                high = mid

        # Apply sizing mode
        if self.config.sizing_mode == SizingMode.STRICT and best_size < target_notional:
            logger.warning(
                f"Strict mode: Cannot fit {target_notional} within {max_impact} bps impact"
            )
            return Decimal("0"), Decimal("0")
        elif self.config.sizing_mode == SizingMode.AGGRESSIVE:
            # Allow up to 2x the impact limit in aggressive mode
            if target_notional <= l10_depth:
                return (
                    target_notional,
                    self.estimate_impact(target_notional, l1_depth, l10_depth) + extra_bps,
                )

        # Conservative mode (default): use best size found
        if best_size < target_notional:
            logger.info(
                f"SIZED_DOWN: Original=${target_notional:.0f} → Adjusted=${best_size:.0f} "
                f"(Impact: {best_impact:.1f}bps, Limit: {max_impact}bps)"
            )

        return best_size, best_impact

    def estimate_impact(
        self, order_size: Decimal, l1_depth: Decimal, l10_depth: Decimal
    ) -> Decimal:
        """
        Estimate market impact in basis points.

        Uses square root model for realistic large order impact.

        Args:
            order_size: Size of order
            l1_depth: Level 1 depth
            l10_depth: Level 10 depth

        Returns:
            Impact in basis points
        """
        if order_size <= l1_depth:
            # Linear impact within L1
            return (order_size / l1_depth) * Decimal("5")
        elif order_size <= l10_depth:
            # Square root impact beyond L1
            l1_impact = Decimal("5")
            excess = order_size - l1_depth
            excess_depth = l10_depth - l1_depth if l10_depth > l1_depth else l1_depth
            excess_ratio = min(excess / excess_depth, Decimal("1"))
            additional_impact = excess_ratio ** Decimal("0.5") * Decimal("20")
            return l1_impact + additional_impact
        else:
            # Order exceeds L10 - very high impact
            return Decimal("100")  # 100 bps = 1%

    @property
    def last_sizing_advice(self) -> PositionSizingAdvice | None:
        """Get last sizing advice for diagnostics."""
        return self._last_sizing_advice
