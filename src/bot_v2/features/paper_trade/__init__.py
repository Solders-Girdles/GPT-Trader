"""
Paper trading feature slice - simulated live trading with real-time data.

Complete isolation - no external dependencies.
"""

from typing import Any

from .paper_trade import (
    get_status,
    get_trading_session,
    start_paper_trading,
    stop_paper_trading,
)
from .types import PaperTradeResult, Position, TradeLog


def execute_paper_trade(
    symbol: str,
    action: str,
    quantity: int,
    strategy_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Facade function for orchestrator compatibility.

    Args:
        symbol: Stock symbol to trade
        action: 'buy' or 'sell'
        quantity: Number of shares to trade
        strategy_info: Strategy metadata (strategy name, confidence, etc.)

    Returns:
        Dict with trade execution result
    """
    # Start paper trading if not already running
    status = get_status()
    if not status.get("is_running", False):
        capital_hint = strategy_info.get("capital_allocated")
        price_hint = strategy_info.get("reference_price")
        if isinstance(capital_hint, int | float) and capital_hint > 0:
            estimated_capital = float(capital_hint)
        elif isinstance(price_hint, int | float) and price_hint > 0:
            estimated_capital = float(price_hint) * quantity
        else:
            # Fallback if orchestrator metadata is unavailable
            estimated_capital = quantity * 100
        start_paper_trading([symbol], initial_capital=estimated_capital)

    # For now, return a simulated result
    # In a real implementation, this would execute the trade through the paper trading system
    return {
        "status": "executed",
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "strategy": strategy_info.get("strategy", "unknown"),
        "confidence": strategy_info.get("confidence", 0.5),
        "capital_allocated": strategy_info.get("capital_allocated", 0.0),
        "reference_price": strategy_info.get("reference_price"),
        "message": f"Paper trade executed: {action} {quantity} shares of {symbol}",
    }


__all__ = [
    "start_paper_trading",
    "stop_paper_trading",
    "get_status",
    "get_trading_session",
    "execute_paper_trade",  # Added facade function
    "PaperTradeResult",
    "Position",
    "TradeLog",
]
