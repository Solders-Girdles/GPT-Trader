"""Trading and portfolio event logging mixin."""

from __future__ import annotations

from typing import Any

from .levels import LogLevel


class TradingLoggingMixin:
    """Provide trading-related logging helpers."""

    def log_trade(
        self,
        action: str,
        symbol: str,
        quantity: float,
        price: float,
        strategy: str,
        success: bool = True,
        execution_time_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        level = LogLevel.INFO if success else LogLevel.ERROR
        entry = self._create_log_entry(
            level=level,
            event_type="trade_execution",
            message=f"{action.upper()} {quantity} {symbol} @ {price}",
            trade_action=action,
            symbol=symbol,
            quantity=quantity,
            price=price,
            strategy=strategy,
            success=success,
            **kwargs,
        )
        if execution_time_ms is not None:
            entry["execution_time_ms"] = execution_time_ms
        self._emit_log(entry)

    def log_order_submission(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="order_submission",
            message=f"submit {side} {quantity} {symbol} @{price if price is not None else 'mkt'}",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            **kwargs,
        )
        self._emit_log(entry)

    def log_order_status_change(
        self,
        order_id: str,
        client_order_id: str | None,
        from_status: str | None,
        to_status: str,
        exchange_error_code: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        level = (
            LogLevel.WARNING if to_status in ("REJECTED", "CANCELLED", "EXPIRED") else LogLevel.INFO
        )
        entry = self._create_log_entry(
            level=level,
            event_type="order_status_change",
            message=f"order {order_id} → {to_status}",
            order_id=order_id,
            client_order_id=client_order_id,
            from_status=from_status,
            to_status=to_status,
            exchange_error_code=exchange_error_code,
            reason=reason,
            **kwargs,
        )
        self._emit_log(entry)

    def log_position_change(
        self,
        symbol: str,
        side: str,
        size: float,
        avg_entry_price: float | None = None,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="position_change",
            message=f"position {symbol} {side} size={size}",
            symbol=symbol,
            side=side,
            size=size,
            avg_entry_price=avg_entry_price,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            **kwargs,
        )
        self._emit_log(entry)

    def log_balance_update(
        self,
        currency: str,
        available: float,
        total: float,
        change: float | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="balance_update",
            message=f"balance {currency} total={total}",
            currency=currency,
            available=available,
            total=total,
            change=change,
            reason=reason,
            **kwargs,
        )
        self._emit_log(entry)

    def log_pnl(
        self,
        symbol: str,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        fees: float | None = None,
        funding: float | None = None,
        position_size: float | None = None,
        transition: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="pnl_update",
            message=f"PnL update {symbol}",
            symbol=symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            fees=fees,
            funding=funding,
            position_size=position_size,
            transition=transition,
            **kwargs,
        )
        self._emit_log(entry)

    def log_funding(
        self,
        symbol: str,
        funding_rate: float,
        payment: float,
        period_start: str | None = None,
        period_end: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="funding_applied",
            message=f"Funding applied {symbol} rate={funding_rate}",
            symbol=symbol,
            funding_rate=funding_rate,
            payment=payment,
            period_start=period_start,
            period_end=period_end,
            **kwargs,
        )
        self._emit_log(entry)

    def log_order_round_trip(
        self,
        order_id: str,
        client_order_id: str | None,
        round_trip_ms: float,
        submitted_ts: str | None = None,
        filled_ts: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="order_round_trip",
            message=f"order {order_id} rtt={round_trip_ms:.2f}ms",
            order_id=order_id,
            client_order_id=client_order_id,
            round_trip_ms=round_trip_ms,
            submitted_ts=submitted_ts,
            filled_ts=filled_ts,
            **kwargs,
        )
        self._emit_log(entry)


__all__ = ["TradingLoggingMixin"]
