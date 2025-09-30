"""
Risk management for perpetuals live trading.

Phase 5: Leverage-aware sizing and runtime guards.
Complete isolation - no strategy logic, pure risk controls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.guard_errors import (
    RiskGuardComputationError,
    RiskGuardDataCorrupt,
    RiskGuardTelemetryError,
)
from bot_v2.features.live_trade.risk_calculations import (
    effective_mmr,
    effective_symbol_leverage_cap,
)
from bot_v2.features.live_trade.risk_runtime import (
    CircuitBreakerOutcome,
    CircuitBreakerState,
)
from bot_v2.features.live_trade.risk_runtime import (
    append_risk_metrics as runtime_append_risk_metrics,
)
from bot_v2.features.live_trade.risk_runtime import (
    check_correlation_risk as runtime_check_correlation_risk,
)
from bot_v2.features.live_trade.risk_runtime import (
    check_mark_staleness as runtime_check_mark_staleness,
)
from bot_v2.features.live_trade.risk_runtime import (
    check_volatility_circuit_breaker as runtime_check_volatility_circuit_breaker,
)
from bot_v2.persistence.event_store import EventStore

logger = logging.getLogger(__name__)


def _coalesce_quantity(*values: Decimal | None) -> Decimal:
    """Pick the first non-``None`` quantity value."""

    for value in values:
        if value is not None:
            return value
    raise TypeError("quantity must be provided")


class ValidationError(Exception):
    """Risk validation failure with clear message."""

    pass


@dataclass
class PositionSizingContext:
    """Context for position sizing requests."""

    symbol: str
    side: str  # "buy" or "sell"
    equity: Decimal
    current_price: Decimal
    strategy_name: str
    method: str
    target_leverage: Decimal
    product: Product | None = None
    current_position_quantity: Decimal = Decimal("0")
    strategy_multiplier: float = 1.0


@dataclass
class PositionSizingAdvice:
    """Advice from position sizing calculation."""

    symbol: str
    side: str
    target_notional: Decimal
    target_quantity: Decimal
    used_dynamic: bool = False
    reduce_only: bool = False
    reason: str | None = None
    fallback_used: bool = False


@dataclass
class ImpactRequest:
    """Request for market impact assessment."""

    symbol: str
    side: str
    quantity: Decimal
    price: Decimal | None = None


@dataclass
class ImpactAssessment:
    """Assessment of market impact for a trade."""

    symbol: str
    side: str
    quantity: Decimal
    estimated_impact_bps: Decimal
    slippage_cost: Decimal
    liquidity_sufficient: bool = True
    reason: str | None = None
    recommended_slicing: bool | None = None
    max_slice_size: Decimal | None = None


@dataclass
class RiskRuntimeState:
    reduce_only_mode: bool = False
    last_reduce_only_reason: str | None = None
    last_reduce_only_at: datetime | None = None


class LiveRiskManager:
    """Risk management for perpetuals live trading.

    Enforces leverage limits, liquidation buffers, exposure caps,
    and runtime guards for safe trading.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        event_store: EventStore | None = None,
        risk_info_provider: Callable[[str], dict[str, Any]] | None = None,
        position_size_estimator: (
            Callable[[PositionSizingContext], PositionSizingAdvice] | None
        ) = None,
        impact_estimator: Callable[[ImpactRequest], ImpactAssessment] | None = None,
    ):
        """
        Initialize risk manager with configuration.

        Args:
            config: Risk configuration (defaults loaded from env)
            event_store: Event store for risk metrics/events
            risk_info_provider: Provider for symbol-specific risk info
            position_size_estimator: Optional dynamic position sizing calculator
            impact_estimator: Optional callable returning market impact assessments
        """
        self.config = config or RiskConfig.from_env()
        self.event_store = event_store or EventStore()
        self._risk_info_provider = risk_info_provider
        self._position_size_estimator = position_size_estimator
        self._impact_estimator = impact_estimator

        # Runtime state
        self.daily_pnl = Decimal("0")
        self.start_of_day_equity = Decimal("0")
        self.last_mark_update: dict[str, datetime] = {}
        self.positions: dict[str, dict[str, Any]] = {}  # Track for exposure
        # Circuit breaker state
        self._cb_last_trigger: dict[str, datetime] = {}
        self.circuit_breaker_state = CircuitBreakerState()
        # Time provider for testability
        self._now_provider = lambda: datetime.utcnow()
        self._state = RiskRuntimeState(reduce_only_mode=bool(self.config.reduce_only_mode))
        self._state_listener: Callable[[RiskRuntimeState], None] | None = None

        # Log configuration
        if hasattr(self.config, "to_dict"):
            logger.info(f"LiveRiskManager initialized with config: {self.config.to_dict()}")
        else:
            logger.info("LiveRiskManager initialized with a non-standard config object")

    # ========== Pre-trade Checks (sync) ==========

    def set_state_listener(self, listener: Callable[[RiskRuntimeState], None] | None) -> None:
        self._state_listener = listener

    def set_impact_estimator(
        self, estimator: Callable[[ImpactRequest], ImpactAssessment] | None
    ) -> None:
        """Install or clear the market-impact estimator hook."""

        self._impact_estimator = estimator

    def is_reduce_only_mode(self) -> bool:
        return self._state.reduce_only_mode or bool(getattr(self.config, "reduce_only_mode", False))

    def set_reduce_only_mode(self, enabled: bool, reason: str = "") -> None:
        """Toggle reduce-only mode without mutating the shared config object."""
        if (
            self._state.reduce_only_mode == enabled
            and bool(getattr(self.config, "reduce_only_mode", False)) == enabled
        ):
            return

        self._state.reduce_only_mode = enabled
        if enabled:
            self._state.last_reduce_only_reason = reason or "unspecified"
            self._state.last_reduce_only_at = self._now()
        else:
            self._state.last_reduce_only_reason = None
            self._state.last_reduce_only_at = None

        # Mirror change onto config so legacy access patterns remain valid
        try:
            self.config.reduce_only_mode = enabled
        except Exception as exc:
            logger.debug("Failed to mirror reduce-only state onto config: %s", exc, exc_info=True)

        try:
            self.event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "event_type": "reduce_only_mode_changed",
                    "enabled": enabled,
                    "reason": reason or "unspecified",
                    "timestamp": (
                        self._state.last_reduce_only_at.isoformat()
                        if self._state.last_reduce_only_at
                        else None
                    ),
                },
            )
        except Exception as exc:
            logger.warning("Failed to persist reduce-only mode change: %s", exc)

        if self._state_listener:
            try:
                self._state_listener(self._state)
            except Exception:
                logger.exception("Reduce-only state listener failed")

    def size_position(self, context: PositionSizingContext) -> PositionSizingAdvice:
        """
        Calculate position size using dynamic estimator or fallback logic.

        Args:
            context: Position sizing context with symbol, equity, price, etc.

        Returns:
            Position sizing advice with target notional and quantity
        """
        # If reduce-only mode, return zero sizing
        if self.is_reduce_only_mode():
            advice = PositionSizingAdvice(
                symbol=context.symbol,
                side=context.side,
                target_notional=Decimal("0"),
                target_quantity=Decimal("0"),
                reduce_only=True,
                reason="reduce_only_mode",
            )
            self._record_sizing_metric(context, advice)
            return advice

        # Try dynamic estimator if available
        if self._position_size_estimator is not None:
            try:
                advice = self._position_size_estimator(context)
                self._record_sizing_metric(context, advice)
                return advice
            except Exception as exc:
                logger.exception("Position size estimator failed for %s", context.symbol)
                try:
                    self.event_store.append_metric(
                        bot_id="risk_engine",
                        metrics={
                            "event_type": "position_sizing_error",
                            "symbol": context.symbol,
                            "error": str(exc),
                        },
                    )
                except Exception:
                    pass

        # Fallback: simple target_leverage-based sizing
        target_notional = (
            context.equity * context.target_leverage * Decimal(str(context.strategy_multiplier))
        )
        target_quantity = (
            target_notional / context.current_price if context.current_price > 0 else Decimal("0")
        )

        advice = PositionSizingAdvice(
            symbol=context.symbol,
            side=context.side,
            target_notional=target_notional,
            target_quantity=target_quantity,
            fallback_used=True,
            reason="fallback",
        )
        self._record_sizing_metric(context, advice)
        return advice

    def _record_sizing_metric(
        self, context: PositionSizingContext, advice: PositionSizingAdvice
    ) -> None:
        """Record position sizing metrics to event store."""
        try:
            self.event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "event_type": "position_sizing_advice",
                    "symbol": context.symbol,
                    "side": context.side,
                    "target_notional": float(advice.target_notional),
                    "target_quantity": float(advice.target_quantity),
                    "used_dynamic": advice.used_dynamic,
                    "reduce_only": advice.reduce_only,
                    "fallback_used": advice.fallback_used,
                    "reason": advice.reason,
                },
            )
        except Exception:
            logger.debug("Failed to record position sizing metric", exc_info=True)

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
            try:
                # Persist event via EventStore instead of cross-slice logger
                self.event_store.append_metric(
                    bot_id="risk_engine",
                    metrics={
                        "event_type": "kill_switch",
                        "message": "Kill switch enabled - trading halted",
                        "component": "risk_manager",
                    },
                )
            except Exception:
                logger.exception("Failed to record kill switch metric")
            raise ValidationError("Kill switch enabled - all trading halted")

        # Reduce-only mode check
        if self.is_reduce_only_mode():
            if not self._is_reducing_position(symbol, side, current_positions):
                try:
                    self.event_store.append_metric(
                        bot_id="risk_engine",
                        metrics={
                            "event_type": "reduce_only_block",
                            "symbol": symbol,
                            "message": f"Blocked increase for {symbol} (reduce-only)",
                            "component": "risk_manager",
                        },
                    )
                except Exception:
                    logger.exception("Failed to record reduce-only block metric for %s", symbol)
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
            self.validate_slippage_guard(
                symbol, side, order_qty, price, price
            )  # Use price as mark for now

    def _now(self) -> datetime:
        return self._now_provider()

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

        # Check effective per-symbol cap (day/night schedule, provider, global)
        symbol_cap = effective_symbol_leverage_cap(
            symbol,
            self.config,
            now=self._now(),
            risk_info_provider=self._risk_info_provider,
            logger=logger,
        )

        # Always report the effective symbol-specific cap when exceeded. Tests
        # expect messaging like "exceeds ETH-PERP cap of 10x" even when the
        # per-symbol cap falls back to the global value.
        if target_leverage > symbol_cap:
            raise ValidationError(
                f"Leverage {target_leverage:.1f}x exceeds {symbol} cap of {symbol_cap}x "
                f"(notional: {notional}, equity: {equity})"
            )

        # Global cap as a final safeguard when a per-symbol cap is looser
        # than the global limit.
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

        # Simplified liquidation buffer check
        # In reality, would need liquidation price from exchange
        notional = order_qty * price

        # Estimate margin required (inverse of max leverage)
        max_leverage = effective_symbol_leverage_cap(
            symbol,
            self.config,
            now=self._now(),
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
        # Enforce explicit per-symbol notional caps if configured
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

        threshold_raw = getattr(self.config, "max_market_impact_bps", None)
        if threshold_raw is None:
            return

        try:
            threshold = Decimal(str(threshold_raw))
        except Exception as exc:  # pragma: no cover - config validation elsewhere
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

        try:
            self.event_store.append_metric(bot_id="risk_engine", metrics=metrics)
        except Exception:
            logger.debug("Failed to record market impact guard metric", exc_info=True)

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
            # Buying above mark is slippage
            slippage = (expected_price - mark_or_quote) / mark_or_quote if mark_or_quote > 0 else 0
        else:
            # Selling below mark is slippage
            slippage = (mark_or_quote - expected_price) / mark_or_quote if mark_or_quote > 0 else 0

        slippage_bps = slippage * 10000  # Convert to basis points

        if slippage_bps > self.config.slippage_guard_bps:
            raise ValidationError(
                f"Expected slippage {slippage_bps:.0f} bps exceeds guard of "
                f"{self.config.slippage_guard_bps} bps for {symbol} "
                f"(price: {expected_price}, mark: {mark_or_quote})"
            )

    # ========== Runtime Guards (async/lightweight) ==========

    def track_daily_pnl(
        self, current_equity: Decimal, positions_pnl: dict[str, dict[str, Decimal]]
    ) -> bool:
        """
        Track daily PnL and trigger reduce-only if breaching limit.

        Args:
            current_equity: Current account equity
            positions_pnl: PnL data from Phase 4 states

        Returns:
            True if reduce-only mode was triggered
        """
        # Initialize start of day if needed
        if self.start_of_day_equity == 0:
            self.start_of_day_equity = current_equity
            return False

        # Calculate total PnL
        total_realized = Decimal("0")
        total_unrealized = Decimal("0")

        for symbol, pnl_data in positions_pnl.items():
            total_realized += pnl_data.get("realized_pnl", Decimal("0"))
            total_unrealized += pnl_data.get("unrealized_pnl", Decimal("0"))

        self.daily_pnl = total_realized + total_unrealized

        # Check daily loss limit
        daily_loss_abs = -self.daily_pnl  # Negative PnL = loss

        if daily_loss_abs > self.config.daily_loss_limit:
            # Trip reduce-only mode
            self.set_reduce_only_mode(True, reason="daily_loss_limit")

            # Log risk event
            self._log_risk_event(
                "daily_loss_breach",
                {
                    "daily_pnl": str(self.daily_pnl),
                    "daily_loss_abs": str(daily_loss_abs),
                    "limit": str(self.config.daily_loss_limit),
                    "action": "reduce_only_mode_enabled",
                },
                guard="daily_loss",
            )

            logger.warning(
                f"Daily loss limit breached: ${daily_loss_abs} > ${self.config.daily_loss_limit} - Enabling reduce-only mode"
            )

            return True

        return False

    def check_liquidation_buffer(
        self, symbol: str, position_data: dict[str, Any], equity: Decimal
    ) -> bool:
        """
        Monitor liquidation buffer for position.

        Returns:
            True if reduce-only was set for this symbol
        """
        guard_name = "liquidation_buffer"

        try:
            qty = Decimal(str(position_data.get("quantity", position_data.get("qty", 0))))
            mark = Decimal(str(position_data.get("mark", 0)))
        except Exception as exc:
            raise RiskGuardDataCorrupt(
                guard=guard_name,
                message="Invalid position data for liquidation buffer",
                details={"symbol": symbol, "position_data": position_data},
                original=exc,
            ) from exc
        # Optional real liquidation price from exchange (preferred when available)
        liq_raw = position_data.get("liquidation_price") or position_data.get("liq_price")
        liquidation_price: Decimal | None = None
        try:
            if liq_raw is not None:
                liquidation_price = Decimal(str(liq_raw))
        except Exception as exc:
            raise RiskGuardDataCorrupt(
                guard=guard_name,
                message="Invalid liquidation price",
                details={"symbol": symbol, "liquidation_price": liq_raw},
                original=exc,
            ) from exc

        if qty == 0 or mark == 0:
            return False

        notional = abs(qty * mark)

        # If we have a true liquidation price, compute distance-to-liquidation as buffer
        if liquidation_price is not None and mark > 0:
            try:
                # Distance from current mark to liquidation, normalized by mark
                buffer_pct = abs(mark - liquidation_price) / mark
            except Exception as exc:
                raise RiskGuardComputationError(
                    guard=guard_name,
                    message="Failed to compute buffer using liquidation price",
                    details={"symbol": symbol},
                    original=exc,
                ) from exc
        else:
            # Fallback: estimate via leverage (conservative)
            max_leverage = self.config.leverage_max_per_symbol.get(symbol, self.config.max_leverage)
            try:
                margin_used = notional / max_leverage if max_leverage > 0 else notional
                buffer_pct = (equity - margin_used) / equity if equity > 0 else 0
            except Exception as exc:
                raise RiskGuardComputationError(
                    guard=guard_name,
                    message="Failed to compute buffer using leverage fallback",
                    details={"symbol": symbol},
                    original=exc,
                ) from exc

        if buffer_pct < self.config.min_liquidation_buffer_pct:
            # Set reduce-only for this symbol
            if symbol not in self.positions:
                self.positions[symbol] = {}
            self.positions[symbol]["reduce_only"] = True

            # Log risk event
            self._log_risk_event(
                "liquidation_buffer_breach",
                {
                    "symbol": symbol,
                    "buffer_pct": str(buffer_pct),
                    "limit": str(self.config.min_liquidation_buffer_pct),
                    "action": f"reduce_only_enabled_for_{symbol}",
                },
                guard=guard_name,
            )

            logger.warning(
                f"Liquidation buffer breach for {symbol}: {buffer_pct:.2%} < "
                f"{self.config.min_liquidation_buffer_pct:.2%} - Setting reduce-only"
            )

            return True

        return False

    # ========== Projection Helpers ==========
    def set_risk_info_provider(self, provider: Callable[[str], dict[str, Any]]) -> None:
        """Set a provider callable that returns exchange risk info for a symbol.

        The provider should return a dict with best-effort keys like
        'maintenance_margin_rate' or 'liquidation_price' for the CURRENT position.
        """
        self._risk_info_provider = provider

    def _project_liquidation_distance(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        price: Decimal,
        equity: Decimal,
        current_positions: dict[str, Any],
    ) -> Decimal:
        """Project post-trade liquidation buffer fraction (best-effort).

        This uses maintenance margin rate when available (preferred) and falls back
        to leverage-based estimation. Returns a fractional buffer where 0.15 means 15%.
        """
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

            # Determine maintenance margin rate (provider preferred; else day/night schedule; else default)
            mmr = effective_mmr(
                symbol,
                self.config,
                now=self._now(),
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

    def check_mark_staleness(self, symbol: str, mark_timestamp: datetime | None = None) -> bool:
        """Check whether mark data for ``symbol`` is stale."""
        if mark_timestamp is not None:
            self.last_mark_update[symbol] = mark_timestamp
        return runtime_check_mark_staleness(
            symbol=symbol,
            last_mark_update=self.last_mark_update,
            now=self._now,
            max_staleness_seconds=self.config.max_mark_staleness_seconds,
            log_event=self._log_risk_event,
            logger=logger,
        )

    def append_risk_metrics(self, equity: Decimal, positions: dict[str, Any]) -> None:
        """Append periodic risk metrics snapshot to the event store."""
        runtime_append_risk_metrics(
            event_store=self.event_store,
            now=self._now,
            equity=equity,
            positions=positions,
            daily_pnl=self.daily_pnl,
            start_of_day_equity=self.start_of_day_equity,
            reduce_only=self.is_reduce_only_mode(),
            kill_switch_enabled=self.config.kill_switch_enabled,
            logger=logger,
        )

    # ========== Correlation / Concentration ==========
    def check_correlation_risk(self, positions: dict[str, Any]) -> bool:
        """Check portfolio correlation and concentration risk."""
        return runtime_check_correlation_risk(
            positions,
            log_event=self._log_risk_event,
            logger=logger,
        )

    def check_volatility_circuit_breaker(
        self, symbol: str, recent_marks: list[Decimal]
    ) -> CircuitBreakerOutcome:
        """Check rolling volatility and trigger progressive circuit breakers."""

        outcome = runtime_check_volatility_circuit_breaker(
            symbol=symbol,
            recent_marks=recent_marks,
            config=self.config,
            state=self.circuit_breaker_state,
            now=self._now,
            last_trigger=self._cb_last_trigger,
            set_reduce_only=lambda enabled, reason: self.set_reduce_only_mode(
                enabled, reason=reason
            ),
            log_event=self._log_risk_event,
            logger=logger,
        )

        if outcome.triggered:
            try:
                self.circuit_breaker_state.record(
                    "volatility_circuit_breaker",
                    symbol,
                    outcome.action,
                    self._now(),
                )
            except Exception:
                logger.debug("Failed to record circuit breaker snapshot", exc_info=True)

        return outcome

    def _is_reducing_position(
        self, symbol: str, side: str, current_positions: dict[str, Any] | None = None
    ) -> bool:
        """Check if order would reduce existing position."""
        if not current_positions or symbol not in current_positions:
            return False  # No position to reduce

        pos_data = current_positions[symbol]
        pos_side = pos_data.get("side", "").lower()

        # Reducing if opposite sides
        if pos_side == "long" and side.lower() == "sell":
            return True
        if pos_side == "short" and side.lower() == "buy":
            return True

        return False

    def _log_risk_event(
        self, event_type: str, details: dict[str, Any], *, guard: str | None = None
    ) -> None:
        """Log risk event to EventStore, surfacing telemetry failures."""

        try:
            self.event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "event_type": event_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    **details,
                },
            )
        except Exception as exc:
            raise RiskGuardTelemetryError(
                guard=guard or event_type,
                message=f"Failed to persist risk event '{event_type}'",
                details=details,
                original=exc,
            ) from exc

    def reset_daily_tracking(self, current_equity: Decimal) -> None:
        """Reset daily tracking at start of new day."""
        self.daily_pnl = Decimal("0")
        self.start_of_day_equity = current_equity
        logger.info(f"Reset daily tracking - Start of day equity: {current_equity}")
