"""Order placement pipeline for the live TradingEngine.

The canonical pre-trade path: run the OrderValidator guard stack (sizing,
preview, slippage) and validate-and-place an order through degradation gating,
sizing, security validation, and the submitter. Extracted from strategy.py
following the engine's collaborator-function pattern; the engine keeps thin
delegating methods so submit_order / the main loop / tests are unchanged.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.decision_trace import OrderDecisionTrace
from gpt_trader.features.live_trade.execution.submission_result import (
    OrderSubmissionResult,
    OrderSubmissionStatus,
)
from gpt_trader.features.live_trade.risk.manager import ValidationError
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.profiling import profile_span
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.engines.strategy import TradingEngine

logger = get_logger(__name__, component="trading_engine")


async def run_order_validator_guards(
    engine: TradingEngine,
    *,
    symbol: str,
    side: OrderSide,
    price: Decimal,
    equity: Decimal,
    quantity: Decimal,
    reduce_only_flag: bool,
    trace: OrderDecisionTrace,
) -> tuple[Decimal, Decimal, bool, OrderSubmissionResult | None]:
    effective_price = price
    if engine._order_validator is None:
        trace.record_outcome("order_validation", "skipped")
        return quantity, effective_price, reduce_only_flag, None

    try:
        with profile_span("pre_trade_validation", {"symbol": symbol}) as _val_span:
            product = engine._state_collector.require_product(symbol, product=None)
            effective_price = engine._state_collector.resolve_effective_price(
                symbol, side.value.lower(), price, product
            )

            try:
                quantity, _ = engine._order_validator.validate_exchange_rules(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    order_quantity=quantity,
                    price=None,
                    effective_price=effective_price,
                    product=product,
                )
                trace.quantity = quantity
                trace.record_outcome("exchange_rules", "passed")
            except ValidationError as exc:
                trace.record_outcome("exchange_rules", "blocked", detail=str(exc))
                raise

            current_positions_dict = engine._state_collector.build_positions_dict(
                list(engine._current_positions.values())
            )
            try:
                engine._order_validator.run_pre_trade_validation(
                    symbol=symbol,
                    side=side,
                    order_quantity=quantity,
                    effective_price=effective_price,
                    product=product,
                    equity=equity,
                    current_positions=current_positions_dict,
                )
                trace.record_outcome("pre_trade_validation", "passed")
            except ValidationError as exc:
                trace.record_outcome("pre_trade_validation", "blocked", detail=str(exc))
                raise

            try:
                engine._order_validator.enforce_slippage_guard(
                    symbol, side, quantity, effective_price
                )
                trace.record_outcome("slippage_guard", "passed")
                engine._degradation.reset_slippage_failures(symbol)
            except ValidationError as slippage_exc:
                trace.record_outcome(
                    "slippage_guard",
                    "blocked",
                    detail=str(slippage_exc),
                )
                config = engine.context.risk_manager.config if engine.context.risk_manager else None
                if config is not None:
                    engine._degradation.record_slippage_failure(symbol, config)
                raise slippage_exc

            # Use container's tracker (validated at init, asserted non-None here)
            assert engine.context.container is not None
            failure_tracker = engine.context.container.validation_failure_tracker
            config = engine.context.risk_manager.config if engine.context.risk_manager else None
            preview_disable_threshold = config.preview_failure_disable_after if config else 5

            if (
                engine._order_validator.enable_order_preview
                and failure_tracker.get_failure_count("order_preview") >= preview_disable_threshold
            ):
                logger.warning(
                    "Auto-disabling order preview due to repeated failures",
                    consecutive_failures=failure_tracker.get_failure_count("order_preview"),
                    threshold=preview_disable_threshold,
                    operation="degradation",
                    stage="preview_disable",
                )
                engine._order_validator.enable_order_preview = False

            if engine._order_validator.enable_order_preview:
                try:
                    await engine._order_validator.maybe_preview_order_async(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        order_quantity=quantity,
                        effective_price=effective_price,
                        stop_price=None,
                        tif=engine.context.config.time_in_force,
                        reduce_only=reduce_only_flag,
                        leverage=None,
                    )
                    trace.record_outcome("order_preview", "passed")
                except ValidationError as exc:
                    trace.record_outcome(
                        "order_preview",
                        "blocked",
                        detail=str(exc),
                    )
                    raise
            else:
                trace.record_outcome("order_preview", "skipped")

            reduce_only_flag = engine._order_validator.finalize_reduce_only_flag(
                reduce_only_flag, symbol
            )
            trace.reduce_only_final = reduce_only_flag
    except ValidationError as exc:
        logger.warning(f"Pre-trade guard rejected order: {exc}")
        blocked_stage = None
        for stage, outcome in trace.outcomes.items():
            if outcome.get("status") == "blocked":
                blocked_stage = stage
                break
        reason_code = blocked_stage or "order_validation"
        engine._emit_trade_gate_blocked(
            gate=reason_code,
            symbol=symbol,
            side=side,
            reason=str(exc),
            params={
                "blocked_stage": reason_code,
                "reduce_only": reduce_only_flag,
            },
            decision_id=trace.decision_id,
        )
        engine._order_submitter.record_rejection(
            symbol, side.value, quantity, effective_price, reason_code
        )
        await engine._notify(
            title="Order Blocked - Guard Rejection",
            message=f"Cannot place order for {symbol}: {exc}",
            severity=AlertSeverity.WARNING,
            context={"symbol": symbol, "side": side.value, "reason": str(exc)},
        )
        trace.record_outcome("order_validation", "blocked", detail=str(exc))
        return (
            quantity,
            effective_price,
            reduce_only_flag,
            engine._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.BLOCKED,
                reason=str(exc),
            ),
        )
    except Exception as exc:
        logger.error(f"Guard check error: {exc}")
        engine._order_submitter.record_rejection(symbol, side.value, quantity, price, "guard_error")
        await engine._notify(
            title="Order Blocked - Guard Error",
            message=f"Cannot place order for {symbol}: guard check failed",
            severity=AlertSeverity.ERROR,
            context={"symbol": symbol, "side": side.value, "error": str(exc)},
        )
        trace.record_outcome("order_validation", "error", detail=str(exc))
        return (
            quantity,
            effective_price,
            reduce_only_flag,
            engine._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.FAILED,
                error=str(exc),
            ),
        )

    return quantity, effective_price, reduce_only_flag, None


async def validate_and_place_order(
    engine: TradingEngine,
    symbol: str,
    decision: Decision,
    price: Decimal,
    equity: Decimal,
    quantity_override: Decimal | None = None,
    reduce_only_requested: bool = False,
) -> OrderSubmissionResult:
    """Validate and submit an order through the guard stack.

    Returns:
        OrderSubmissionResult describing success/blocked/failed.
    """
    side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL

    # Early check: is this order actually reduce-only? (needed for degradation check)
    current_pos = engine._current_positions.get(symbol)
    is_reducing = engine._is_reduce_only_order(current_pos, side)
    reduce_only_flag = is_reducing

    decision_id = engine._order_submitter.generate_client_order_id(None)
    trace = OrderDecisionTrace(
        symbol=symbol,
        side=side.value,
        price=price,
        equity=equity,
        quantity=None,
        reduce_only=reduce_only_requested,
        reduce_only_final=reduce_only_flag,
        reason=decision.reason,
        decision_id=decision_id,
        bot_id=str(engine.context.bot_id) if engine.context.bot_id is not None else None,
    )

    config = getattr(engine.context.risk_manager, "config", None)
    kill_switch_enabled = getattr(config, "kill_switch_enabled", False) is True
    if kill_switch_enabled:
        trace.record_outcome(
            "kill_switch",
            "blocked",
            detail="kill_switch_enabled",
        )
        engine._order_submitter.record_rejection(
            symbol,
            side.value,
            Decimal("0"),
            price,
            "kill_switch",
            client_order_id=decision_id,
        )
        return engine._finalize_decision_trace(
            trace,
            status=OrderSubmissionStatus.BLOCKED,
            reason="kill_switch",
        )

    result = await engine._check_degradation_gate(
        symbol=symbol,
        side=side,
        price=price,
        trace=trace,
        reduce_only_flag=reduce_only_flag,
    )
    if result is not None:
        return result

    quantity, result = engine._calculate_quantity_and_record(
        symbol=symbol,
        side=side,
        price=price,
        equity=equity,
        quantity_override=quantity_override,
        trace=trace,
    )
    if result is not None:
        return result

    result = await engine._check_reduce_only_request(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        reduce_only_requested=reduce_only_requested,
        is_reducing=is_reducing,
        trace=trace,
    )
    if result is not None:
        return result

    result = await engine._run_security_validation(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        equity=equity,
        trace=trace,
    )
    if result is not None:
        return result

    quantity, result = await engine._apply_reduce_only_mode(
        symbol=symbol,
        side=side,
        price=price,
        quantity=quantity,
        reduce_only_flag=reduce_only_flag,
        is_reducing=is_reducing,
        current_pos=current_pos,
        trace=trace,
    )
    if result is not None:
        return result

    result = await engine._check_mark_staleness(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        reduce_only_flag=reduce_only_flag,
        trace=trace,
    )
    if result is not None:
        return result

    (
        quantity,
        effective_price,
        reduce_only_flag,
        result,
    ) = await engine._run_order_validator_guards(
        symbol=symbol,
        side=side,
        price=price,
        equity=equity,
        quantity=quantity,
        reduce_only_flag=reduce_only_flag,
        trace=trace,
    )
    if result is not None:
        return result

    # Place order via OrderSubmitter for proper ID tracking and telemetry
    submission_outcome = await engine._broker_calls(
        engine._order_submitter.submit_order_with_result,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        order_quantity=quantity,
        price=None,  # Market order
        effective_price=effective_price,
        stop_price=None,
        tif=engine.context.config.time_in_force,
        reduce_only=reduce_only_flag,
        leverage=None,
        client_order_id=decision_id,
    )

    # Notify on successful order placement
    if submission_outcome.success:
        order_id = submission_outcome.order_id
        trace.order_id = order_id
        await engine._notify(
            title="Order Executed",
            message=f"{side.value} {quantity} {symbol} at ~{price}",
            severity=AlertSeverity.INFO,
            context={
                "symbol": symbol,
                "side": side.value,
                "quantity": str(quantity),
                "price": str(price),
                "order_id": order_id,
            },
        )

        # Record trade in status reporter
        engine._status_reporter.add_trade(
            {
                "symbol": symbol,
                "side": side.value,
                "quantity": str(quantity),
                "price": str(price),
                "order_id": order_id,
            }
        )
        trace.record_outcome("submit_order", "success", order_id=order_id)
        return engine._finalize_decision_trace(
            trace,
            status=OrderSubmissionStatus.SUCCESS,
            order_id=order_id,
        )
    reason = submission_outcome.reason or "broker_rejected"
    logger.warning(
        "Order submission failed",
        symbol=symbol,
        side=side.value,
        reason=reason,
        reason_detail=submission_outcome.reason_detail,
        operation="order_submit",
        stage="failed",
    )
    trace.record_outcome(
        "submit_order",
        "failed",
        detail=reason,
        reason_detail=submission_outcome.reason_detail,
        error=submission_outcome.error,
    )
    return engine._finalize_decision_trace(
        trace,
        status=OrderSubmissionStatus.FAILED,
        reason=reason,
        error=submission_outcome.error_message,
    )
