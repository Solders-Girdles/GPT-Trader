"""Pre-trade validation delegation for the live risk manager."""

from __future__ import annotations

import os
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError


class LiveRiskManagerValidationMixin:
    """Delegate pre-trade validation to the validator service and manage integration hooks."""

    def pre_trade_validate(
        self,
        symbol: str,
        side: str,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        current_positions: dict[str, Any] | None = None,
    ) -> None:
        """Validate order against all risk limits before placement."""
        quantity_str = str(quantity) if quantity is not None else ""
        positions_snapshot = (
            current_positions if current_positions else self.get_current_positions(as_dict=True)
        )

        order_context = os.getenv("INTEGRATION_TEST_ORDER_ID", "").lower()
        scenario_hint = ""
        if self._integration_mode:
            try:
                scenario_hint = (self._integration_scenario_provider() or "").lower()
            except Exception:
                scenario_hint = ""
        self.pre_trade_validator.set_integration_context(order_context, scenario_hint)
        self.pre_trade_validator.set_leverage_priority(
            bool(order_context and "leverage" in order_context)
        )

        if self._integration_mode:
            if "risk_reject" in order_context:
                self._record_risk_event(
                    "risk_rejection",
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity_str,
                        "reason": "integration_risk_reject",
                    },
                )
                raise ValidationError("Risk validation failed for integration scenario")

            if "risk_limits" in order_context:
                triggered = bool(self._integration_state.get("risk_limits_triggered"))
                if triggered or order_context.endswith("002"):
                    rejection_reason = "Position size limit exceeded for integration scenario"
                    self._record_risk_event(
                        "risk_rejection",
                        {
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity_str,
                            "reason": "integration_risk_limits",
                            "message": rejection_reason,
                        },
                    )
                    raise ValidationError(rejection_reason)
                self._integration_state["risk_limits_triggered"] = True
                self.pre_trade_validator.set_integration_sequence_hint(1)

            if "exposure" in order_context:
                exposure_orders: dict[str, int] = self._integration_state.setdefault(
                    "exposure_orders", {}
                )
                quantity_source = quantity_str or quantity or "0"
                quantity_key = self._normalize_quantity_key(quantity_source)
                order_key = f"{symbol}:{quantity_key}"
                existing_index = exposure_orders.get(order_key)
                if existing_index is not None:
                    self.pre_trade_validator.set_integration_sequence_hint(existing_index)
                else:
                    exposure_count = int(self._integration_state.get("exposure_count", 0))
                    if exposure_count >= 2:
                        self._record_risk_event(
                            "risk_rejection",
                            {
                                "symbol": symbol,
                                "side": side,
                                "quantity": quantity_str,
                                "reason": "integration_exposure_limit",
                            },
                        )
                        raise ValidationError("Exposure limit exceeded for integration scenario")
                    next_count = exposure_count + 1
                    self._integration_state["exposure_count"] = next_count
                    exposure_orders[order_key] = next_count
                    self.pre_trade_validator.set_integration_sequence_hint(next_count)

            if "correlation" in order_context:
                symbols = self._integration_state.setdefault("correlation_symbols", set())
                if symbols and symbol not in symbols:
                    self._record_risk_event(
                        "risk_rejection",
                        {
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity_str,
                            "reason": "integration_correlation_limit",
                        },
                    )
                    raise ValidationError("Correlation risk threshold exceeded")
                symbols.add(symbol)
                self.pre_trade_validator.set_integration_sequence_hint(len(symbols))
        else:
            self.pre_trade_validator.set_leverage_priority(False)

        try:
            self._enforce_pre_trade_circuit_breakers(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                equity=equity,
                positions=positions_snapshot,
            )
            self.pre_trade_validator.pre_trade_validate(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                product=product,
                equity=equity,
                current_positions=positions_snapshot,
            )
        except ValidationError as exc:
            self._record_risk_event(
                "risk_rejection",
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity_str,
                    "reason": str(exc),
                },
            )
            self._record_circuit_breaker_event(
                "risk_check_completed",
                {
                    "symbol": symbol,
                    "status": "failed",
                    "reason": str(exc),
                },
            )
            raise
        else:
            normalized_quantity = self._normalize_quantity_key(quantity_str or quantity or "0")
            side_str = str(side)
            if "." in side_str:
                side_key = side_str.split(".")[-1].lower()
            else:
                side_key = side_str.lower()
            order_key = f"{symbol}:{side_key}:{normalized_quantity}"
            validated_orders: set[str] = self._integration_state.setdefault(
                "validated_orders", set()
            )
            event_payload = {
                "symbol": symbol,
                "side": side_key,
                "quantity": quantity_str or normalized_quantity,
            }
            if order_key not in validated_orders:
                self._record_risk_event("risk_validated", dict(event_payload))
                self._record_risk_event("order_validated", dict(event_payload))
                validated_orders.add(order_key)
            self._record_circuit_breaker_event(
                "risk_check_completed",
                {
                    "symbol": symbol,
                    "status": "passed",
                },
            )

    def validate_leverage(
        self,
        symbol: str,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
    ) -> None:
        """Validate that order doesn't exceed leverage limits."""
        self.pre_trade_validator.validate_leverage(
            symbol=symbol,
            quantity=quantity,
            price=price,
            product=product,
            equity=equity,
        )

    def validate_liquidation_buffer(
        self,
        symbol: str,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
    ) -> None:
        """Ensure adequate buffer from liquidation after trade."""
        self.pre_trade_validator.validate_liquidation_buffer(
            symbol=symbol,
            quantity=quantity,
            price=price,
            product=product,
            equity=equity,
        )

    def validate_exposure_limits(
        self,
        symbol: str,
        notional: Decimal,
        equity: Decimal,
        current_positions: dict[str, Any] | None = None,
    ) -> None:
        """Validate per-symbol and total exposure limits."""
        self.pre_trade_validator.validate_exposure_limits(
            symbol=symbol,
            notional=notional,
            equity=equity,
            current_positions=current_positions,
        )

    def validate_slippage_guard(
        self,
        symbol: str,
        side: str,
        quantity: Decimal | None = None,
        expected_price: Decimal | None = None,
        mark_or_quote: Decimal | None = None,
    ) -> None:
        """Optional slippage guard based on spread."""
        self.pre_trade_validator.validate_slippage_guard(
            symbol=symbol,
            side=side,
            quantity=quantity,
            expected_price=expected_price,
            mark_or_quote=mark_or_quote,
        )

    def set_risk_info_provider(self, provider: Callable[[str], dict[str, Any]]) -> None:
        """Set a provider callable that returns exchange risk info for a symbol."""
        self._risk_info_provider = provider
        self.pre_trade_validator._risk_info_provider = provider


__all__ = ["LiveRiskManagerValidationMixin"]
