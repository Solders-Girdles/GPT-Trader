"""Limit and exposure checks for pre-trade validation."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk_calculations import (
    effective_symbol_leverage_cap,
)

from .exceptions import ValidationError
from .utils import coalesce_quantity, logger, to_decimal


class LimitChecksMixin:
    """Provide leverage, liquidation, and exposure validations."""

    config: Any
    event_store: Any
    _risk_info_provider: Any
    _now_provider: Any

    def validate_position_size_limit(self, symbol: str, quantity: Decimal) -> None:
        """Ensure order quantity does not exceed configured position size limits."""
        max_size = getattr(self.config, "max_position_size", None)
        if max_size is None:
            return
        try:
            limit_decimal = Decimal(str(max_size))
        except Exception:
            return
        if limit_decimal <= 0:
            return
        if abs(quantity) > limit_decimal:
            raise ValidationError(
                f"Position size {abs(quantity)} exceeds limit {limit_decimal} for {symbol}"
            )

    def validate_leverage(
        self,
        symbol: str,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """Validate that order doesn't exceed leverage limits."""
        if price is None or product is None or equity is None:
            raise TypeError("price, product, and equity are required")

        order_qty = coalesce_quantity(qty, quantity)

        if product.market_type != MarketType.PERPETUAL:
            return

        notional = order_qty * price

        if equity <= 0:
            target_leverage = Decimal("Infinity")
        else:
            target_leverage = notional / equity

        symbol_cap = effective_symbol_leverage_cap(
            symbol,
            self.config,
            now=self._now_provider(),
            risk_info_provider=self._risk_info_provider,
            logger=logger,
        )

        symbol_cap_decimal = Decimal(str(symbol_cap))

        if target_leverage > symbol_cap_decimal:
            raise ValidationError(
                f"Leverage {float(target_leverage):.1f}x exceeds {symbol} cap of {symbol_cap_decimal}x "
                f"(notional: {notional}, equity: {equity})"
            )

        max_leverage_cap = Decimal(str(self.config.max_leverage))
        if target_leverage > max_leverage_cap:
            raise ValidationError(
                f"Leverage {float(target_leverage):.1f}x exceeds global cap of {max_leverage_cap}x"
            )

    def validate_liquidation_buffer(
        self,
        symbol: str,
        qty: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """Ensure adequate buffer from liquidation after trade."""
        if price is None or product is None or equity is None:
            raise TypeError("price, product, and equity are required")

        order_qty = coalesce_quantity(qty, quantity)

        if product.market_type != MarketType.PERPETUAL:
            return

        notional = order_qty * price

        max_leverage = effective_symbol_leverage_cap(
            symbol,
            self.config,
            now=self._now_provider(),
            risk_info_provider=self._risk_info_provider,
            logger=logger,
        )
        margin_required = notional / max_leverage if max_leverage > 0 else notional

        remaining_equity = equity - margin_required
        buffer_pct = remaining_equity / equity if equity > 0 else Decimal("0")
        buffer_threshold = Decimal(str(self.config.min_liquidation_buffer_pct))

        if buffer_pct < buffer_threshold:
            raise ValidationError(
                f"Insufficient liquidation buffer for position size: {float(buffer_pct):.1%} < "
                f"{float(buffer_threshold):.1%} required "
                f"(margin needed: {margin_required}, equity: {equity})"
            )

    def validate_exposure_limits(
        self,
        symbol: str,
        notional: Decimal,
        equity: Decimal,
        current_positions: dict[str, Any] | None = None,
    ) -> None:
        """Validate per-symbol and total exposure limits."""
        max_notional_cap = getattr(self.config, "max_notional_per_symbol", {}).get(symbol)
        if max_notional_cap is not None and notional > max_notional_cap:
            raise ValidationError(
                f"Symbol notional {notional} exceeds cap {max_notional_cap} for {symbol}"
            )

        symbol_exposure_pct = notional / equity if equity > 0 else Decimal("Infinity")
        symbol_exposure_cap = Decimal(str(self.config.max_position_pct_per_symbol))

        if symbol_exposure_pct > symbol_exposure_cap:
            raise ValidationError(
                f"Symbol exposure {float(symbol_exposure_pct):.1%} exceeds cap of "
                f"{float(symbol_exposure_cap):.1%} for {symbol}"
            )

        total_exposure = notional
        if current_positions:
            for pos_symbol, pos_data in current_positions.items():
                if pos_symbol != symbol and isinstance(pos_data, dict):
                    if "notional" in pos_data:
                        pos_notional = abs(to_decimal(pos_data.get("notional")))
                    else:
                        qty_value = to_decimal(pos_data.get("quantity", pos_data.get("qty")))
                        price_value = to_decimal(pos_data.get("mark", pos_data.get("price")))
                        pos_notional = abs(qty_value * price_value)
                    total_exposure += pos_notional

        total_exposure_pct = total_exposure / equity if equity > 0 else Decimal("Infinity")
        total_exposure_cap = Decimal(str(self.config.max_exposure_pct))

        if total_exposure_pct > total_exposure_cap:
            raise ValidationError(
                f"Total exposure {float(total_exposure_pct):.1%} would exceed cap of "
                f"{float(total_exposure_cap):.1%} (new notional: {notional})"
            )


__all__ = ["LimitChecksMixin"]
