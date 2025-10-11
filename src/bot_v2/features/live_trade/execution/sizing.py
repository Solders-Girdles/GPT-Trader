"""Position sizing utilities for live execution."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, Product, Quote
from bot_v2.features.live_trade.risk import (
    LiveRiskManager,
    PositionSizingAdvice,
    PositionSizingContext,
)
from bot_v2.utilities.quantities import quantity_from

logger = logging.getLogger(__name__)


class PositionSizer:
    """Encapsulates reference-price resolution and dynamic sizing."""

    def __init__(self, broker: Any, risk_manager: LiveRiskManager | None) -> None:
        self._broker = broker
        self._risk_manager = risk_manager
        self._last_advice: PositionSizingAdvice | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        """Run optional dynamic sizing if enabled."""
        risk_manager = self._risk_manager
        if risk_manager is None:
            return None

        config = getattr(risk_manager, "config", None)
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
            strategy_name=self.__class__.__name__,
            method=getattr(config, "position_sizing_method", "intelligent"),
            current_position_quantity=current_position_quantity,
            target_leverage=target_leverage,
            strategy_multiplier=float(getattr(config, "position_sizing_multiplier", 1.0)),
            product=product,
        )

        try:
            advice = risk_manager.size_position(context)
        except Exception:
            logger.exception("Dynamic sizing failed for %s", symbol)
            return None

        self._last_advice = advice
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
        """Best-effort resolution of reference price for risk sizing."""
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
            broker = self._broker
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
        """Estimate account equity from risk manager or broker balances."""
        risk_manager = self._risk_manager
        if risk_manager is not None:
            start_equity = getattr(risk_manager, "start_of_day_equity", None)
            if start_equity is not None:
                try:
                    value = Decimal(str(start_equity))
                    if value > 0:
                        return value
                except Exception:
                    pass

        broker = self._broker
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

    def current_positions(self) -> Mapping[str, Any]:
        """Expose positions mapping if the risk manager tracks one."""
        risk_manager = self._risk_manager
        if risk_manager is None:
            return {}
        positions = getattr(risk_manager, "positions", {})
        if not isinstance(positions, Mapping):
            return {}
        return positions

    @property
    def last_advice(self) -> PositionSizingAdvice | None:
        return self._last_advice

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_position_quantity(self, symbol: str) -> Decimal:
        positions = self.current_positions()
        pos = positions.get(symbol)
        if not pos:
            return Decimal("0")
        try:
            value = quantity_from(pos, default=Decimal("0"))
            return value or Decimal("0")
        except Exception:
            return Decimal("0")
