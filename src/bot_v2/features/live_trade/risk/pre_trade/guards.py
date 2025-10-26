"""Guard helpers for pre-trade validation."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.guard_errors import RiskGuardComputationError
from bot_v2.features.live_trade.risk_calculations import effective_mmr
from bot_v2.utilities.telemetry import emit_metric

from .exceptions import ValidationError
from .utils import coalesce_quantity, logger, to_decimal


class GuardChecksMixin:
    """Implement guard-related helper methods."""

    config: Any
    event_store: Any
    _impact_estimator: Any
    _risk_info_provider: Any
    _now_provider: Any
    last_mark_update: dict[str, Any]

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

        def _safe_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        metrics = {
            "event_type": "market_impact_guard",
            "symbol": symbol,
            "side": side,
            "impact_bps": _safe_float(assessment.estimated_impact_bps),
            "threshold_bps": _safe_float(threshold),
            "quantity": _safe_float(quantity),
            "price": _safe_float(price),
            "slippage_cost": _safe_float(getattr(assessment, "slippage_cost", None)),
            "liquidity_sufficient": bool(getattr(assessment, "liquidity_sufficient", False)),
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
                    f"Trade exceeds maximum market impact: {assessment.estimated_impact_bps} bps > {threshold} bps"
                )
            raise ValidationError("Market impact guard blocked trade due to insufficient liquidity")

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
        """Optional slippage guard based on spread."""
        if expected_price is None or mark_or_quote is None:
            raise TypeError("expected_price and mark_or_quote are required")

        _ = coalesce_quantity(qty, quantity)

        if self.config.slippage_guard_bps <= 0:
            return

        decimal_zero = Decimal("0")
        if side.lower() == "buy":
            slippage = (
                (expected_price - mark_or_quote) / mark_or_quote
                if mark_or_quote > 0
                else decimal_zero
            )
        else:
            slippage = (
                (mark_or_quote - expected_price) / mark_or_quote
                if mark_or_quote > 0
                else decimal_zero
            )

        slippage_bps = slippage * Decimal(10000)
        guard_threshold = Decimal(str(self.config.slippage_guard_bps))

        if slippage_bps > guard_threshold:
            raise ValidationError(
                f"Expected slippage {float(slippage_bps):.0f} bps exceeds guard of "
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
        guard_name = "liquidation_projection"
        try:
            pos = current_positions.get(symbol) or {}
            cur_qty = abs(to_decimal(pos.get("quantity", pos.get("qty"))))
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
                    reduce_qty = min(cur_qty, abs(qty))
                    residual = cur_qty - reduce_qty
                    new_qty = residual if residual > 0 else (abs(qty) - reduce_qty)

            if new_qty <= 0:
                return Decimal("1")

            notional = new_qty * price
            if equity <= 0 or notional <= 0:
                return Decimal("0")

            mmr_value = effective_mmr(
                symbol,
                self.config,
                now=self._now_provider(),
                risk_info_provider=self._risk_info_provider,
                logger=logger,
            )

            equity_decimal = Decimal(str(equity))
            buffer: Decimal = (equity_decimal - (mmr_value * notional)) / equity_decimal
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

        if pos_side == "long" and side.lower() == "sell":
            return True
        if pos_side == "short" and side.lower() == "buy":
            return True

        return False


__all__ = ["GuardChecksMixin"]
