"""
Pre-trade validation logic for risk management.

All synchronous checks performed before order placement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, MutableMapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk_calculations import (
    effective_mmr,
    effective_symbol_leverage_cap,
)
from bot_v2.persistence.event_store import EventStore
from bot_v2.utilities.telemetry import emit_metric

logger = logging.getLogger(__name__)


def _coalesce_quantity(*values: Decimal | None) -> Decimal:
    """Pick the first non-``None`` quantity value."""
    for value in values:
        if value is not None:
            return value
    raise TypeError("quantity must be provided")


class ValidationError(Exception):
    """Risk validation failure with clear message."""


class PreTradeValidator:
    """Performs all pre-trade validation checks."""

    def __init__(
        self,
        config: RiskConfig,
        event_store: EventStore,
        risk_info_provider: Callable[[str], dict[str, Any]] | None = None,
        impact_estimator: Any | None = None,
        is_reduce_only_mode: Callable[[], bool] | None = None,
        now_provider: Callable[[], datetime] | None = None,
        last_mark_update: MutableMapping[str, datetime | None] | None = None,
    ):
        """
        Initialize pre-trade validator.

        Args:
            config: Risk configuration
            event_store: Event store for validation metrics
            risk_info_provider: Provider for symbol-specific risk info
            impact_estimator: Optional market impact estimator
            is_reduce_only_mode: Callable to check if reduce-only mode is active
            now_provider: Time provider for testability
            last_mark_update: Shared dict tracking last mark updates
        """
        self.config = config
        self.event_store = event_store
        self._risk_info_provider = risk_info_provider
        self._impact_estimator = impact_estimator
        self._is_reduce_only_mode = is_reduce_only_mode or (lambda: False)
        self._now_provider = now_provider or (lambda: datetime.utcnow())
        self.last_mark_update = last_mark_update if last_mark_update is not None else {}

    def pre_trade_validate(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        qty: Decimal | None = None,
        price: Decimal | None = None,
        product: Product | None = None,
        equity: Decimal | None = None,
        current_positions: dict[str, Any] | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """
        Validate order against all risk limits before placement.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            qty: Order quantity
            price: Expected execution price
            product: Product metadata
            equity: Current account equity
            current_positions: Current positions for exposure calc

        Raises:
            ValidationError: If any risk check fails
        """
        if price is None or product is None or equity is None:
            raise TypeError("price, product, and equity are required")

        order_qty = _coalesce_quantity(qty, quantity)

        # Kill switch check
        if self.config.kill_switch_enabled:
            emit_metric(
                self.event_store,
                "risk_engine",
                {
                    "event_type": "kill_switch",
                    "message": "Kill switch enabled - trading halted",
                    "component": "risk_manager",
                },
                logger=logger,
            )
            raise ValidationError("Kill switch enabled - all trading halted")

        # Reduce-only mode check
        if self._is_reduce_only_mode():
            if not self._is_reducing_position(symbol, side, current_positions):
                emit_metric(
                    self.event_store,
                    "risk_engine",
                    {
                        "event_type": "reduce_only_block",
                        "symbol": symbol,
                        "message": f"Blocked increase for {symbol} (reduce-only)",
                        "component": "risk_manager",
                    },
                    logger=logger,
                )
                raise ValidationError(
                    f"Reduce-only mode active - cannot increase position for {symbol}"
                )

        if getattr(self.config, "enable_market_impact_guard", False) and self._impact_estimator:
            self._apply_market_impact_guard(
                symbol=symbol,
                side=side,
                quantity=order_qty,
                price=price,
            )

        # Run all validation checks
        self.validate_leverage(symbol, order_qty, price, product, equity)
        self.validate_liquidation_buffer(symbol, order_qty, price, product, equity)
        self.validate_exposure_limits(symbol, order_qty * price, equity, current_positions)

        # Project post-trade liquidation buffer using MMR/liq data if enabled
        if (
            self.config.enable_pre_trade_liq_projection
            and product.market_type == MarketType.PERPETUAL
        ):
            projected = self._project_liquidation_distance(
                symbol=symbol,
                side=side,
                qty=order_qty,
                price=price,
                equity=equity,
                current_positions=current_positions or {},
            )
            if projected < Decimal(str(self.config.min_liquidation_buffer_pct)):
                raise ValidationError(
                    f"Projected liquidation buffer {projected:.2%} < "
                    f"{self.config.min_liquidation_buffer_pct:.2%} for {symbol}"
                )

        # Optional slippage guard if mark price available
        if symbol in self.last_mark_update:
            self.validate_slippage_guard(symbol, side, order_qty, price, price)

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
        """
        Validate that order doesn't exceed leverage limits.

        Raises:
            ValidationError: If leverage exceeds caps
        """
        if price is None or product is None or equity is None:
            raise TypeError("price, product, and equity are required")

        order_qty = _coalesce_quantity(qty, quantity)

        if product.market_type != MarketType.PERPETUAL:
            return  # Only check leverage for perpetuals

        # Calculate notional value
        notional = order_qty * price

        # Calculate implied leverage
        target_leverage = notional / equity if equity > 0 else float("inf")

        # Check effective per-symbol cap
        symbol_cap = effective_symbol_leverage_cap(
            symbol,
            self.config,
            now=self._now_provider(),
            risk_info_provider=self._risk_info_provider,
            logger=logger,
        )

        if target_leverage > symbol_cap:
            raise ValidationError(
                f"Leverage {target_leverage:.1f}x exceeds {symbol} cap of {symbol_cap}x "
                f"(notional: {notional}, equity: {equity})"
            )

        # Global cap as a final safeguard
        if target_leverage > self.config.max_leverage:
            raise ValidationError(
                f"Leverage {target_leverage:.1f}x exceeds global cap of {self.config.max_leverage}x"
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
        """
        Ensure adequate buffer from liquidation after trade.

        Raises:
            ValidationError: If buffer would be too low
        """
        if price is None or product is None or equity is None:
            raise TypeError("price, product, and equity are required")

        order_qty = _coalesce_quantity(qty, quantity)

        if product.market_type != MarketType.PERPETUAL:
            return

        notional = order_qty * price

        # Estimate margin required
        max_leverage = effective_symbol_leverage_cap(
            symbol,
            self.config,
            now=self._now_provider(),
            risk_info_provider=self._risk_info_provider,
            logger=logger,
        )
        margin_required = notional / max_leverage if max_leverage > 0 else notional

        # Check if we'd have enough buffer after this trade
        remaining_equity = equity - margin_required
        buffer_pct = remaining_equity / equity if equity > 0 else 0

        if buffer_pct < self.config.min_liquidation_buffer_pct:
            raise ValidationError(
                f"Insufficient liquidation buffer: {buffer_pct:.1%} < "
                f"{self.config.min_liquidation_buffer_pct:.1%} required "
                f"(margin needed: {margin_required}, equity: {equity})"
            )

    def validate_exposure_limits(
        self,
        symbol: str,
        notional: Decimal,
        equity: Decimal,
        current_positions: dict[str, Any] | None = None,
    ) -> None:
        """
        Validate per-symbol and total exposure limits.

        Raises:
            ValidationError: If exposure exceeds caps
        """
        # Per-symbol exposure check
        max_notional_cap = getattr(self.config, "max_notional_per_symbol", {}).get(symbol)
        if max_notional_cap is not None and notional > max_notional_cap:
            raise ValidationError(
                f"Symbol notional {notional} exceeds cap {max_notional_cap} for {symbol}"
            )

        symbol_exposure_pct = notional / equity if equity > 0 else float("inf")

        if symbol_exposure_pct > self.config.max_position_pct_per_symbol:
            raise ValidationError(
                f"Symbol exposure {symbol_exposure_pct:.1%} exceeds cap of "
                f"{self.config.max_position_pct_per_symbol:.1%} for {symbol}"
            )

        # Total portfolio exposure check
        total_exposure = notional
        if current_positions:
            for pos_symbol, pos_data in current_positions.items():
                if pos_symbol != symbol and isinstance(pos_data, dict):
                    if "notional" in pos_data:
                        try:
                            pos_notional = abs(Decimal(str(pos_data.get("notional", 0))))
                        except Exception:
                            pos_notional = Decimal("0")
                    else:
                        qty_value = pos_data.get("quantity", pos_data.get("qty", 0))
                        price_value = pos_data.get("mark", pos_data.get("price", 0))
                        pos_notional = abs(Decimal(str(qty_value)) * Decimal(str(price_value)))
                    total_exposure += pos_notional

        total_exposure_pct = total_exposure / equity if equity > 0 else float("inf")

        if total_exposure_pct > self.config.max_exposure_pct:
            raise ValidationError(
                f"Total exposure {total_exposure_pct:.1%} would exceed cap of "
                f"{self.config.max_exposure_pct:.1%} (new notional: {notional})"
            )

    def _apply_market_impact_guard(
        self, *, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> None:
        """Evaluate market impact guard and raise if the order breaches limits."""
        from bot_v2.features.live_trade.risk.position_sizing import ImpactRequest

        threshold_raw = getattr(self.config, "max_market_impact_bps", None)
        if threshold_raw is None:
            return

        try:
            threshold = Decimal(str(threshold_raw))
        except Exception as exc:
            logger.debug("Invalid market impact threshold %s", threshold_raw, exc_info=True)
            raise ValidationError("Invalid market impact guard configuration") from exc

        if threshold <= 0:
            return

        if self._impact_estimator is None:
            return

        try:
            assessment = self._impact_estimator(
                ImpactRequest(symbol=symbol, side=side, quantity=quantity, price=price)
            )
        except Exception as exc:
            logger.exception("Impact estimator failed for %s", symbol)
            raise ValidationError(f"Impact estimator failure for {symbol}: {exc}") from exc

        if assessment is None:
            raise ValidationError("Impact estimator returned no assessment")

        status = "allowed"
        reason: str | None = None

        if assessment.estimated_impact_bps > threshold:
            status = "blocked"
            reason = "impact_exceeds_threshold"
        elif assessment.liquidity_sufficient is False:
            status = "blocked"
            reason = "insufficient_liquidity"

        metrics = {
            "event_type": "market_impact_guard",
            "symbol": symbol,
            "side": side,
            "impact_bps": float(assessment.estimated_impact_bps),
            "threshold_bps": float(threshold),
            "quantity": float(quantity),
            "price": float(price),
            "slippage_cost": float(assessment.slippage_cost),
            "liquidity_sufficient": bool(assessment.liquidity_sufficient),
            "status": status,
        }

        if assessment.recommended_slicing is not None:
            metrics["recommended_slicing"] = bool(assessment.recommended_slicing)
        if assessment.max_slice_size is not None:
            try:
                metrics["max_slice_size"] = float(assessment.max_slice_size)
            except Exception:
                metrics["max_slice_size"] = None

        if reason:
            metrics["reason"] = reason

        emit_metric(
            self.event_store,
            "risk_engine",
            metrics,
            logger=logger,
        )

        if status == "blocked":
            if reason == "impact_exceeds_threshold":
                raise ValidationError(
                    f"Market impact {assessment.estimated_impact_bps} bps exceeds cap {threshold} bps"
                )
            raise ValidationError("Market impact guard blocked trade due to liquidity")

    def validate_slippage_guard(
        self,
        symbol: str,
        side: str,
        qty: Decimal | None = None,
        expected_price: Decimal | None = None,
        mark_or_quote: Decimal | None = None,
        *,
        quantity: Decimal | None = None,
    ) -> None:
        """
        Optional slippage guard based on spread.

        Raises:
            ValidationError: If expected slippage exceeds threshold
        """
        if expected_price is None or mark_or_quote is None:
            raise TypeError("expected_price and mark_or_quote are required")

        _ = _coalesce_quantity(qty, quantity)

        if self.config.slippage_guard_bps <= 0:
            return  # Disabled

        # Calculate expected slippage
        if side.lower() == "buy":
            slippage = (expected_price - mark_or_quote) / mark_or_quote if mark_or_quote > 0 else 0
        else:
            slippage = (mark_or_quote - expected_price) / mark_or_quote if mark_or_quote > 0 else 0

        slippage_bps = slippage * 10000

        if slippage_bps > self.config.slippage_guard_bps:
            raise ValidationError(
                f"Expected slippage {slippage_bps:.0f} bps exceeds guard of "
                f"{self.config.slippage_guard_bps} bps for {symbol} "
                f"(price: {expected_price}, mark: {mark_or_quote})"
            )

    def _project_liquidation_distance(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        price: Decimal,
        equity: Decimal,
        current_positions: dict[str, Any],
    ) -> Decimal:
        """Project post-trade liquidation buffer fraction (best-effort)."""
        from bot_v2.features.live_trade.guard_errors import RiskGuardComputationError

        guard_name = "liquidation_projection"
        try:
            # Determine post-trade absolute quantity and side
            pos = current_positions.get(symbol) or {}
            cur_qty = abs(Decimal(str(pos.get("quantity", pos.get("qty", 0)))))
            cur_side = str(pos.get("side", "")).lower()
            order_side = side.lower()

            if cur_qty == 0:
                new_qty = abs(qty)
            else:
                if (cur_side == "long" and order_side == "buy") or (
                    cur_side == "short" and order_side == "sell"
                ):
                    new_qty = cur_qty + abs(qty)
                else:
                    # Reducing or flipping
                    reduce_qty = min(cur_qty, abs(qty))
                    residual = cur_qty - reduce_qty
                    new_qty = residual if residual > 0 else (abs(qty) - reduce_qty)

            if new_qty <= 0:
                return Decimal("1")  # Reducing-only → safe

            notional = Decimal(str(new_qty)) * Decimal(str(price))
            if equity <= 0 or notional <= 0:
                return Decimal("0")

            # Determine maintenance margin rate
            mmr = effective_mmr(
                symbol,
                self.config,
                now=self._now_provider(),
                risk_info_provider=self._risk_info_provider,
                logger=logger,
            )

            # Equity buffer after accounting for maintenance requirement
            buffer = (Decimal(str(equity)) - (mmr * notional)) / Decimal(str(equity))
            if buffer < Decimal("0"):
                buffer = Decimal("0")
            if buffer > Decimal("1"):
                buffer = Decimal("1")
            return buffer
        except Exception as exc:
            raise RiskGuardComputationError(
                guard=guard_name,
                message="Failed to project liquidation distance",
                details={"symbol": symbol},
                original=exc,
            ) from exc

    def _is_reducing_position(
        self, symbol: str, side: str, current_positions: dict[str, Any] | None = None
    ) -> bool:
        """Check if order would reduce existing position."""
        if not current_positions or symbol not in current_positions:
            return False

        pos_data = current_positions[symbol]
        pos_side = pos_data.get("side", "").lower()

        # Reducing if opposite sides
        if pos_side == "long" and side.lower() == "sell":
            return True
        if pos_side == "short" and side.lower() == "buy":
            return True

        return False
