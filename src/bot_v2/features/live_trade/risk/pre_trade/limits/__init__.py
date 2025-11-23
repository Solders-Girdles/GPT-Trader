"""Aggregates limit validation helpers for pre-trade checks."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable, Mapping

from bot_v2.features.brokerages.core.interfaces import Product

from . import exposure as exposure_mod
from . import leverage as leverage_mod
from . import liquidation as liquidation_mod


class LimitChecksMixin:
    """Provide leverage, liquidation, and exposure validations."""

    config: Any
    event_store: Any
    _risk_info_provider: Callable[[str], Mapping[str, Any]] | None
    _now_provider: Callable[[], Any]

    def validate_position_size_limit(self, symbol: str, quantity: Decimal) -> None:
        """Ensure order quantity does not exceed configured position size limits."""
        exposure_mod.validate_position_size_limit(self.config, symbol, quantity)

    def validate_leverage(
        self,
        symbol: str,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
    ) -> None:
        leverage_mod.validate_leverage(
            config=self.config,
            symbol=symbol,
            quantity=quantity,
            quantity_override=None,
            price=price,
            product=product,
            equity=equity,
            now=self._now_provider(),
            risk_info_provider=self._risk_info_provider,
        )

    def validate_liquidation_buffer(
        self,
        symbol: str,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
    ) -> None:
        liquidation_mod.validate_liquidation_buffer(
            config=self.config,
            symbol=symbol,
            quantity=quantity,
            quantity_override=None,
            price=price,
            product=product,
            equity=equity,
            now=self._now_provider(),
            risk_info_provider=self._risk_info_provider,
        )

    def validate_exposure_limits(
        self,
        symbol: str,
        notional: Decimal,
        equity: Decimal,
        current_positions: dict[str, Any] | None = None,
    ) -> None:
        exposure_mod.validate_symbol_and_total_exposure(
            config=self.config,
            symbol=symbol,
            notional=notional,
            equity=equity,
            current_positions=current_positions or {},
        )

    def validate_correlation_risk(
        self,
        symbol: str,
        *,
        notional: Decimal,
        current_positions: dict[str, Any] | None = None,
    ) -> None:
        exposure_mod.validate_correlation_risk(
            config=self.config,
            symbol=symbol,
            notional=notional,
            current_positions=current_positions or {},
        )


__all__ = ["LimitChecksMixin"]
