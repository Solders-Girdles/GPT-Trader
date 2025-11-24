"""
Order validation logic for live trading execution.

This module handles pre-trade validation including exchange rules,
mark price staleness checks, slippage guards, and order previews.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Protocol, cast, runtime_checkable

from gpt_trader.features.brokerages.coinbase.specs import validate_order as spec_validate_order
from gpt_trader.features.brokerages.core.interfaces import (
    IBrokerage,
    OrderSide,
    OrderType,
    Product,
    TimeInForce,
)
from gpt_trader.features.live_trade.risk import LiveRiskManager, ValidationError
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.quantization import quantize_price_side_aware

logger = get_logger(__name__, component="order_validation")


class OrderValidator:
    """Validates orders before submission to broker."""

    def __init__(
        self,
        broker: IBrokerage,
        risk_manager: LiveRiskManager,
        enable_order_preview: bool,
        record_preview_callback: Any,
        record_rejection_callback: Any,
    ) -> None:
        """
        Initialize order validator.

        Args:
            broker: Brokerage adapter
            risk_manager: Risk manager instance
            enable_order_preview: Whether to preview orders before submission
            record_preview_callback: Function to record preview results
            record_rejection_callback: Function to record rejections
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self.enable_order_preview = enable_order_preview
        self._record_preview = record_preview_callback
        self._record_rejection = record_rejection_callback

    def validate_exchange_rules(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        product: Product,
    ) -> tuple[Decimal, Decimal | None]:
        """
        Validate order against exchange rules and quantization.

        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            order_quantity: Order quantity
            price: Order price (None for market orders)
            effective_price: Effective price for validation
            product: Product specifications

        Returns:
            Tuple of (adjusted_quantity, adjusted_price)

        Raises:
            ValidationError: If validation fails
        """
        validator_price: Decimal | None
        if order_type == OrderType.MARKET:
            validator_price = None
        else:
            candidate = price if price is not None else effective_price
            validator_price = Decimal(str(candidate)) if candidate is not None else None

        vr = spec_validate_order(
            product=product,
            side=side.value,
            quantity=order_quantity,
            order_type=order_type.value.lower(),
            price=validator_price,
        )
        if not vr.ok:
            reason_code = vr.reason or "spec_violation"
            self._record_rejection(
                symbol, side.value, order_quantity, price or effective_price, reason_code
            )
            raise ValidationError(f"Spec validation failed: {reason_code}")

        if order_type == OrderType.LIMIT and price is not None:
            price = quantize_price_side_aware(
                Decimal(str(price)), product.price_increment, side.value
            )
        if vr.adjusted_price is not None:
            price = vr.adjusted_price
        if vr.adjusted_quantity is not None:
            order_quantity = vr.adjusted_quantity
        return order_quantity, price

    def ensure_mark_is_fresh(self, symbol: str) -> None:
        """
        Ensure mark price is not stale.

        Args:
            symbol: Trading symbol

        Raises:
            ValidationError: If mark price is stale
        """
        try:
            if self.risk_manager.check_mark_staleness(symbol):
                raise ValidationError(f"Mark price is stale for {symbol}; halting order placement")
        except ValidationError:
            raise
        except Exception:
            pass

    def enforce_slippage_guard(
        self,
        symbol: str,
        side: OrderSide,
        order_quantity: Decimal,
        effective_price: Decimal,
    ) -> None:
        """
        Enforce slippage guard to prevent excessive market impact.

        Args:
            symbol: Trading symbol
            side: Order side
            order_quantity: Order quantity
            effective_price: Effective price

        Raises:
            ValidationError: If expected slippage exceeds limit
        """
        try:
            snapshot = None
            if hasattr(self.broker, "get_market_snapshot"):
                snapshot = self.broker.get_market_snapshot(symbol)  # type: ignore[attr-defined]
            if snapshot:
                spread_bps = Decimal(str(snapshot.get("spread_bps", 0)))
                depth_l1 = Decimal(str(snapshot.get("depth_l1", 0)))
                notional = order_quantity * Decimal(str(effective_price))
                depth = depth_l1 if depth_l1 and depth_l1 > 0 else Decimal("1")
                impact_bps = Decimal("10000") * (notional / depth) * Decimal("0.5")
                expected_bps = spread_bps + impact_bps
                guard_limit = Decimal(str(self.risk_manager.config.slippage_guard_bps))
                if expected_bps > guard_limit:
                    raise ValidationError(
                        f"Expected slippage {expected_bps:.0f} bps exceeds guard {guard_limit}"
                    )
        except ValidationError:
            raise
        except Exception:
            pass

    def run_pre_trade_validation(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_quantity: Decimal,
        effective_price: Decimal,
        product: Product,
        equity: Decimal,
        current_positions: dict[str, dict[str, Any]],
    ) -> None:
        """
        Run comprehensive pre-trade validation checks.

        Args:
            symbol: Trading symbol
            side: Order side
            order_quantity: Order quantity
            effective_price: Effective price
            product: Product specifications
            equity: Current account equity
            current_positions: Current position map

        Raises:
            ValidationError: If validation fails
        """
        self.risk_manager.pre_trade_validate(
            symbol=symbol,
            side=side.value,
            quantity=order_quantity,
            price=effective_price,
            product=product,
            equity=equity,
            current_positions=current_positions,
        )

    def maybe_preview_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        effective_price: Decimal,
        stop_price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
    ) -> None:
        """
        Preview order if enabled.

        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            order_quantity: Order quantity
            effective_price: Effective price
            stop_price: Stop price for stop orders
            tif: Time in force
            reduce_only: Whether order is reduce-only
            leverage: Leverage multiplier
        """
        if not self.enable_order_preview:
            logger.debug(
                "Order preview disabled",
                operation="order_preview",
                stage="disabled",
            )
            return
        broker = self.broker
        if not isinstance(broker, _PreviewBroker):
            logger.debug(
                "Broker does not support order preview",
                operation="order_preview",
                stage="unsupported",
            )
            return
        preview_broker = cast(_PreviewBroker, broker)
        try:
            tif_value = tif if isinstance(tif, TimeInForce) else (tif or TimeInForce.GTC)
            preview_data = preview_broker.preview_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=order_quantity,
                price=effective_price,
                stop_price=stop_price,
                tif=tif_value,
                reduce_only=reduce_only,
                leverage=leverage,
            )
            self._record_preview(
                symbol, side, order_type, order_quantity, effective_price, preview_data
            )
        except ValidationError:
            raise
        except Exception as exc:
            logger.debug(
                "Preview call failed",
                error=str(exc),
                operation="order_preview",
                stage="error",
            )

    def finalize_reduce_only_flag(self, reduce_only: bool, symbol: str) -> bool:
        """
        Finalize reduce-only flag based on risk manager state.

        Args:
            reduce_only: User-requested reduce-only flag
            symbol: Trading symbol

        Returns:
            Final reduce-only flag value
        """
        if self.risk_manager.is_reduce_only_mode():
            logger.info(
                "Reduce-only mode active - forcing reduce_only=True",
                symbol=symbol,
                operation="reduce_only",
                stage="enforce",
            )
            return True
        return reduce_only


@runtime_checkable
class _PreviewBroker(Protocol):
    def preview_order(self, **kwargs: Any) -> Any: ...

    def edit_order_preview(self, order_id: str, **kwargs: Any) -> Any: ...
