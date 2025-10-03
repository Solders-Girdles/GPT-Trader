"""Order validation pipeline for advanced execution engine.

Coordinates dynamic sizing, post-only checks, quantization/spec validation,
risk validation, and stop requirements with clear success/failure signalling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.coinbase.specs import (
    validate_order as spec_validate_order,
)
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from bot_v2.features.live_trade.advanced_execution_models.models import (
    NormalizedOrderRequest,
    OrderConfig,
)
from bot_v2.features.live_trade.dynamic_sizing_helper import DynamicSizingHelper
from bot_v2.features.live_trade.risk import LiveRiskManager, PositionSizingAdvice
from bot_v2.features.live_trade.stop_trigger_manager import StopTriggerManager
from bot_v2.utilities.quantization import quantize_price_side_aware

logger = logging.getLogger(__name__)

__all__ = ["ValidationResult", "OrderValidationPipeline"]


@dataclass
class ValidationResult:
    """Result of order validation pipeline."""

    ok: bool
    rejection_reason: str | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    quantity: Decimal | None = None
    reduce_only: bool | None = None
    post_only_rejection: bool = False

    @property
    def failed(self) -> bool:
        return not self.ok

    @classmethod
    def success(
        cls,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        quantity: Decimal | None = None,
        reduce_only: bool | None = None,
    ) -> ValidationResult:
        return cls(
            ok=True,
            limit_price=limit_price,
            stop_price=stop_price,
            quantity=quantity,
            reduce_only=reduce_only,
        )

    @classmethod
    def failure(
        cls,
        reason: str,
        *,
        post_only_rejection: bool = False,
    ) -> ValidationResult:
        return cls(
            ok=False,
            rejection_reason=reason,
            post_only_rejection=post_only_rejection,
        )


class OrderValidationPipeline:
    """Coordinates all order validation steps for advanced execution."""

    def __init__(
        self,
        *,
        config: OrderConfig,
        sizing_helper: DynamicSizingHelper,
        stop_trigger_manager: StopTriggerManager,
        risk_manager: LiveRiskManager | None,
    ) -> None:
        self.config = config
        self.sizing_helper = sizing_helper
        self.stop_trigger_manager = stop_trigger_manager
        self.risk_manager = risk_manager

    def validate(self, request: NormalizedOrderRequest) -> ValidationResult:
        """Run full validation pipeline on a normalized request."""
        sizing_result = self._validate_sizing(request)
        if sizing_result.failed:
            return sizing_result
        self._apply_adjustments(request, sizing_result)

        post_only_result = self._validate_post_only(request)
        if post_only_result.failed:
            return post_only_result

        quantization_result = self._validate_quantization(request)
        if quantization_result.failed:
            return quantization_result
        self._apply_adjustments(request, quantization_result)

        risk_result = self._validate_risk(request)
        if risk_result.failed:
            return risk_result

        stop_valid, stop_reason = self.stop_trigger_manager.validate_stop_order_requirements(
            symbol=request.symbol,
            order_type=request.order_type,
            stop_price=request.stop_price,
        )
        if not stop_valid:
            return ValidationResult.failure(stop_reason or "stop_validation")

        return ValidationResult.success(
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            quantity=request.quantity,
            reduce_only=request.reduce_only,
        )

    def _apply_adjustments(
        self,
        request: NormalizedOrderRequest,
        result: ValidationResult,
    ) -> None:
        if result.limit_price is not None:
            request.limit_price = result.limit_price
        if result.stop_price is not None:
            request.stop_price = result.stop_price
        if result.quantity is not None:
            request.quantity = result.quantity
        if result.reduce_only is not None:
            request.reduce_only = result.reduce_only

    def _validate_sizing(self, request: NormalizedOrderRequest) -> ValidationResult:
        """Validate dynamic sizing (delegates to sizing helper)."""
        advice: PositionSizingAdvice | None = self.sizing_helper.maybe_apply_position_sizing(
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            order_quantity=request.quantity,
            limit_price=request.limit_price,
            product=request.product,
            quote=request.quote,
            leverage=request.leverage,
        )

        if advice is None:
            return ValidationResult.success()

        order_quantity = advice.target_quantity
        if order_quantity <= 0:
            logger.info(
                "Dynamic sizing prevented new order for %s (%s)",
                request.symbol,
                advice.reason or "no reason provided",
            )
            return ValidationResult.failure("position_sizing")

        if advice.reason:
            logger.debug(
                "Dynamic sizing adjusted %s quantity to %s (%s)",
                request.symbol,
                order_quantity,
                advice.reason,
            )

        reduce_only = request.reduce_only or advice.reduce_only

        return ValidationResult.success(quantity=order_quantity, reduce_only=reduce_only)

    def _validate_post_only(self, request: NormalizedOrderRequest) -> ValidationResult:
        """Validate post-only constraints."""
        if not (request.order_type == OrderType.LIMIT and request.post_only):
            return ValidationResult.success()

        if not self.config.reject_on_cross:
            return ValidationResult.success()

        limit_price = request.limit_price
        quote = request.quote
        if limit_price is None or quote is None:
            return ValidationResult.success()

        if request.side == OrderSide.BUY and limit_price >= quote.ask:
            logger.warning(
                "Post-only buy would cross at %s >= %s",
                limit_price,
                quote.ask,
            )
            return ValidationResult.failure("post_only_cross", post_only_rejection=True)

        if request.side == OrderSide.SELL and limit_price <= quote.bid:
            logger.warning(
                "Post-only sell would cross at %s <= %s",
                limit_price,
                quote.bid,
            )
            return ValidationResult.failure("post_only_cross", post_only_rejection=True)

        return ValidationResult.success()

    def _validate_quantization(self, request: NormalizedOrderRequest) -> ValidationResult:
        """Validate quantization and product specs."""
        product = request.product
        limit_price = request.limit_price
        stop_price = request.stop_price
        quantity = request.quantity

        if product is not None and product.price_increment:
            increment = product.price_increment
            if request.order_type == OrderType.LIMIT and limit_price is not None:
                limit_price = quantize_price_side_aware(limit_price, increment, request.side.value)
            if request.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
                if stop_price is not None:
                    stop_price = quantize_price_side_aware(
                        stop_price,
                        increment,
                        request.side.value,
                    )
            if request.order_type == OrderType.STOP_LIMIT and limit_price is not None:
                limit_price = quantize_price_side_aware(limit_price, increment, request.side.value)

        if product is None:
            return ValidationResult.success(
                limit_price=limit_price,
                stop_price=stop_price,
                quantity=quantity,
            )

        validation = spec_validate_order(
            product=product,
            side=request.side.value,
            quantity=quantity,
            order_type=request.order_type.value.lower(),
            price=(
                limit_price
                if request.order_type == OrderType.LIMIT and limit_price is not None
                else None
            ),
        )

        if not validation.ok:
            reason = validation.reason or "spec_violation"
            logger.warning("Spec validation failed for %s: %s", request.symbol, reason)
            return ValidationResult.failure(reason)

        if validation.adjusted_quantity is not None:
            quantity = (
                validation.adjusted_quantity
                if isinstance(validation.adjusted_quantity, Decimal)
                else Decimal(str(validation.adjusted_quantity))
            )

        if validation.adjusted_price is not None and request.order_type == OrderType.LIMIT:
            limit_price = validation.adjusted_price

        return ValidationResult.success(
            limit_price=limit_price,
            stop_price=stop_price,
            quantity=quantity,
        )

    def _validate_risk(self, request: NormalizedOrderRequest) -> ValidationResult:
        """Validate risk limits via risk manager if available."""
        if self.risk_manager is None:
            return ValidationResult.success()

        try:
            validation_price = self.sizing_helper.determine_reference_price(
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                limit_price=request.limit_price,
                quote=request.quote,
                product=request.product,
            )
            equity = self.sizing_helper.estimate_equity()
            current_positions = getattr(self.risk_manager, "positions", {})
            self.risk_manager.pre_trade_validate(
                symbol=request.symbol,
                side=request.side.value,
                quantity=request.quantity,
                price=validation_price,
                product=request.product,
                equity=equity,
                current_positions=current_positions,
            )
            return ValidationResult.success()
        except ValidationError as exc:
            logger.warning("Risk validation failed for %s: %s", request.symbol, exc)
            return ValidationResult.failure("risk")
