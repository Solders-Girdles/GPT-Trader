"""Metric calculations for daily reports."""

from __future__ import annotations

from typing import Any

from .models import SymbolPerformance


def calculate_pnl_metrics(
    events: list[dict[str, Any]],
    current_metrics: dict[str, Any],
) -> dict[str, float]:
    equity = float(current_metrics.get("account", {}).get("equity", 0))
    realized = 0.0
    unrealized = 0.0
    funding = 0.0
    fees = 0.0

    for event in events:
        event_type = event.get("type", "")
        if event_type == "pnl_update":
            realized += float(event.get("realized_pnl", 0))
            unrealized = float(event.get("unrealized_pnl", 0))
        elif event_type == "funding_payment":
            funding += float(event.get("amount", 0))
        elif event_type == "fill":
            fees += float(event.get("fee", 0))

    total_pnl = realized + unrealized
    prev_equity = equity - total_pnl
    equity_change = total_pnl
    equity_change_pct = (equity_change / prev_equity * 100) if prev_equity > 0 else 0

    return {
        "equity": equity,
        "equity_change": equity_change,
        "equity_change_pct": equity_change_pct,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "funding_pnl": funding,
        "total_pnl": total_pnl,
        "fees_paid": fees,
    }


def calculate_trade_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    fills = [e for e in events if e.get("type") == "fill"]
    wins = []
    losses = []

    for fill in fills:
        pnl = fill.get("pnl", 0)
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(abs(pnl))

    total_trades = len(wins) + len(losses)
    winning_trades = len(wins)
    losing_trades = len(losses)

    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    largest_win = max(wins) if wins else 0
    largest_loss = max(losses) if losses else 0

    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    sharpe_ratio = 0.0
    if total_trades > 0:
        returns = [w for w in wins] + [-loss_value for loss_value in losses]
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance**0.5
            sharpe_ratio = (mean_return / std_dev) if std_dev > 0 else 0

    max_drawdown = 0.0
    running_equity = 0.0
    peak = 0.0
    for event in sorted(events, key=lambda e: e.get("timestamp", "")):
        if event.get("type") == "fill":
            running_equity += event.get("pnl", 0)
            peak = max(peak, running_equity)
            drawdown = peak - running_equity
            max_drawdown = max(max_drawdown, drawdown)

    max_drawdown_pct = (max_drawdown / peak * 100) if peak > 0 else 0

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
    }


def calculate_symbol_metrics(events: list[dict[str, Any]]) -> list[SymbolPerformance]:
    symbol_data: dict[str, dict[str, Any]] = {}

    for event in events:
        symbol = event.get("symbol")
        if not symbol:
            continue

        if symbol not in symbol_data:
            symbol_data[symbol] = {
                "realized": 0.0,
                "unrealized": 0.0,
                "funding": 0.0,
                "wins": [],
                "losses": [],
                "trades": 0,
                "regime": None,
                "exposure": 0.0,
            }

        event_type = event.get("type", "")
        if event_type == "fill":
            pnl = event.get("pnl", 0)
            if pnl > 0:
                symbol_data[symbol]["wins"].append(pnl)
            elif pnl < 0:
                symbol_data[symbol]["losses"].append(abs(pnl))
            symbol_data[symbol]["trades"] += 1
            symbol_data[symbol]["realized"] += event.get("pnl", 0)

        elif event_type == "pnl_update":
            symbol_data[symbol]["unrealized"] = event.get("unrealized_pnl", 0)

        elif event_type == "funding_payment":
            symbol_data[symbol]["funding"] += event.get("amount", 0)

        if "regime" in event:
            symbol_data[symbol]["regime"] = event["regime"]

        if event_type == "position_update":
            qty = event.get("quantity", 0)
            price = event.get("price", 0)
            symbol_data[symbol]["exposure"] = abs(qty * price)

    performances: list[SymbolPerformance] = []
    for symbol, data in symbol_data.items():
        wins = data["wins"]
        losses = data["losses"]
        total_trades = len(wins) + len(losses)

        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        performances.append(
            SymbolPerformance(
                symbol=symbol,
                regime=data["regime"],
                realized_pnl=data["realized"],
                unrealized_pnl=data["unrealized"],
                funding_pnl=data["funding"],
                total_pnl=data["realized"] + data["unrealized"] + data["funding"],
                trades=data["trades"],
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                exposure_usd=data["exposure"],
            )
        )

    return performances


def calculate_risk_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    guard_triggers: dict[str, int] = {}
    circuit_breaker_state: dict[str, Any] = {}

    for event in events:
        event_type = event.get("type", "")
        if event_type == "guard_triggered":
            guard = event.get("guard", "unknown")
            guard_triggers[guard] = guard_triggers.get(guard, 0) + 1
        elif event_type == "circuit_breaker_triggered":
            circuit_breaker_state = {
                "triggered": True,
                "rule": event.get("rule"),
                "action": event.get("action"),
                "timestamp": event.get("timestamp"),
            }

    return {
        "guard_triggers": guard_triggers,
        "circuit_breaker_state": circuit_breaker_state,
    }


def calculate_health_metrics(events: list[dict[str, Any]]) -> dict[str, int]:
    stale_marks = 0
    ws_reconnects = 0
    unfilled = 0
    api_errors = 0

    for event in events:
        event_type = event.get("type", "")
        if event_type == "stale_mark_detected":
            stale_marks += 1
        elif event_type == "websocket_reconnect":
            ws_reconnects += 1
        elif event_type == "unfilled_order_alert":
            unfilled += 1
        elif event_type == "api_error":
            api_errors += 1

    return {
        "stale_marks_count": stale_marks,
        "ws_reconnects": ws_reconnects,
        "unfilled_orders": unfilled,
        "api_errors": api_errors,
    }


__all__ = [
    "calculate_pnl_metrics",
    "calculate_trade_metrics",
    "calculate_symbol_metrics",
    "calculate_risk_metrics",
    "calculate_health_metrics",
]
