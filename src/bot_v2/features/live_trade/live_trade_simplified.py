"""
Simplified legacy live trading interface.

This module provides a cleaned-up version of the legacy live trading interface
that uses the new standardized utilities for better maintainability and consistency.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from bot_v2.errors import NetworkError
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.broker_connection import (
    connect_broker as _connect_broker,
)
from bot_v2.features.live_trade.broker_connection import (
    disconnect as _disconnect,
)
from bot_v2.features.live_trade.broker_connection import (
    get_broker_client,
    get_connection,
    get_risk_manager,
)
from bot_v2.features.live_trade.types import (
    AccountInfo,
    BrokerConnection,
    MarketHours,
    position_to_trading_position,
)
from bot_v2.types.trading import AccountSnapshot, TradingPosition
from bot_v2.utilities import create_position_manager, create_trading_operations
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__)

# Global trading operations instance
_trading_ops: Any = None
_position_manager: Any = None


def _ensure_trading_operations() -> Any:
    """Ensure trading operations are initialized."""
    global _trading_ops, _position_manager

    if _trading_ops is None:
        broker = get_broker_client()
        risk_manager = get_risk_manager()

        if broker is None or risk_manager is None:
            raise RuntimeError("Broker connection not initialized")

        _trading_ops = create_trading_operations(broker, risk_manager)
        _position_manager = create_position_manager(_trading_ops)

    return _trading_ops


def connect_broker(
    broker_name: str = "simulated",
    api_key: str = "",
    api_secret: str = "",
    is_paper: bool = True,
    base_url: str | None = None,
) -> BrokerConnection:
    """
    Connect to the simulated broker stub used by legacy demos.

    Args:
        broker_name: Name of the broker (defaults to simulated)
        api_key: API key (not used for simulated broker)
        api_secret: API secret (not used for simulated broker)
        is_paper: Whether to use paper trading mode
        base_url: Base URL for broker API

    Returns:
        Broker connection object
    """
    return _connect_broker(
        broker_name=broker_name,
        api_key=api_key,
        api_secret=api_secret,
        is_paper=is_paper,
        base_url=base_url,
    )


def place_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal | int,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Decimal | float | None = None,
    stop_price: Decimal | float | None = None,
    time_in_force: TimeInForce = TimeInForce.GTC,
) -> Order | None:
    """
    Place an order using simplified trading operations.

    Args:
        symbol: Trading symbol (e.g., AAPL, BTC-USD)
        side: Order side
        quantity: Units to trade
        order_type: Order type
        limit_price: Limit price (for limit orders)
        stop_price: Stop price (for stop orders)
        time_in_force: Time in force

    Returns:
        Order object or None if failed
    """
    connection = get_connection()
    if not connection or not connection.is_connected:
        raise NetworkError("Not connected to broker")

    trading_ops = _ensure_trading_operations()

    order = trading_ops.place_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force,
    )

    if order:
        logger.info(f"✅ Order placed: {order.id}")
        logger.info(f"   {side.name} {quantity} {symbol} @ {order_type.name}")
    else:
        logger.error("❌ Failed to place order")

    return order


def cancel_order(order_id: str) -> bool:
    """
    Cancel an order using simplified trading operations.

    Args:
        order_id: Order ID to cancel

    Returns:
        True if cancelled successfully
    """
    trading_ops = _ensure_trading_operations()

    success = trading_ops.cancel_order(order_id)

    if success:
        logger.info(f"✅ Order {order_id} cancelled")
    else:
        logger.error(f"❌ Failed to cancel order {order_id}")

    return success


def get_positions() -> list[Position]:
    """
    Get current positions using simplified trading operations.

    Returns:
        List of Position objects
    """
    trading_ops = _ensure_trading_operations()

    positions = trading_ops.get_positions()

    if positions:
        logger.info(f"📊 Retrieved {len(positions)} positions")
        for pos in positions:
            entry_price = float(pos.entry_price)
            mark_price = float(pos.mark_price)
            quantity = float(pos.quantity)
            cost_basis = abs(float(pos.entry_price * pos.quantity))

            logger.info(f"   {pos.symbol}: {quantity:.4f} units @ ${entry_price:.2f}")
            logger.info(f"      Mark: ${mark_price:.2f}")

            pnl_value = float(pos.unrealized_pnl)
            pnl_sign = "+" if pnl_value >= 0 else "-"
            pnl_pct = (pnl_value / cost_basis) * 100 if cost_basis else 0.0
            logger.info(
                f"      P&L: {pnl_sign}${abs(pnl_value):.2f} ({pnl_sign}{abs(pnl_pct):.2f}%)"
            )
    else:
        logger.info("📊 No open positions")

    return positions


def get_positions_trading() -> list[TradingPosition]:
    """
    Return current positions using the shared trading type schema.

    Returns:
        List of TradingPosition objects
    """
    return [position_to_trading_position(pos) for pos in get_positions()]


def get_account() -> AccountInfo | None:
    """
    Get account information using simplified trading operations.

    Returns:
        AccountInfo object or None
    """
    trading_ops = _ensure_trading_operations()

    account = trading_ops.get_account()

    if account:
        logger.info("💰 Account Summary:")
        logger.info(f"   Equity: ${account.equity:,.2f}")
        logger.info(f"   Cash: ${account.cash:,.2f}")
        logger.info(f"   Buying Power: ${account.buying_power:,.2f}")
        logger.info(f"   Positions Value: ${account.positions_value:,.2f}")
        if account.pattern_day_trader:
            logger.info(f"   Day Trades Remaining: {account.day_trades_remaining}")
    else:
        logger.warning("Account information not available")

    return account


def get_account_snapshot() -> AccountSnapshot | None:
    """
    Return the active account as a shared account snapshot.

    Returns:
        AccountSnapshot object or None
    """
    account = get_account()
    return account.to_account_snapshot() if account else None


def get_orders(status: str = "open") -> list[Order]:
    """
    Get orders for the active broker session.

    Args:
        status: Order status filter

    Returns:
        List of Order objects
    """
    broker = get_broker_client()
    if broker is None:
        return []

    return broker.get_orders(status)


def get_quote(symbol: str) -> Quote | None:
    """
    Get real-time quote for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Quote object or None
    """
    broker = get_broker_client()
    if broker is None:
        return None

    return broker.get_quote(symbol)


def get_market_hours() -> MarketHours:
    """
    Return market hours information from the active broker.

    Returns:
        MarketHours object
    """
    broker = get_broker_client()
    if broker is None:
        return MarketHours(
            is_open=False, open_time=None, close_time=None, extended_hours_open=False
        )

    return broker.get_market_hours()


def close_all_positions() -> bool:
    """
    Close all open positions using simplified trading operations.

    Returns:
        True if all positions closed successfully
    """
    position_manager = _ensure_trading_operations()
    global _position_manager

    if _position_manager is None:
        _position_manager = create_position_manager(position_manager)

    success = _position_manager.close_all_positions()

    if success:
        logger.info("✅ All positions closed successfully")
    else:
        logger.error("❌ Failed to close some positions")

    return success


def disconnect() -> None:
    """Disconnect from broker and cleanup resources."""
    global _trading_ops, _position_manager

    try:
        _disconnect()
        logger.info("Disconnected from broker")
    except Exception as exc:
        logger.warning("Error during disconnect: %s", exc, exc_info=True)
    finally:
        _trading_ops = None
        _position_manager = None


# Legacy strategy function - simplified but kept for compatibility
def run_strategy(
    symbols: list[str],
    strategy_name: str = "baseline_perps",
    iterations: int = 3,
    mark_cache: dict[str, Decimal] | None = None,
    mark_windows: dict[str, list[Decimal]] | None = None,
    *,
    strategy_override: Any | None = None,
) -> dict[str, Any]:
    """
    Run a basic trading strategy for demonstration purposes.

    This is a simplified version maintained for backward compatibility.
    For production use, use the new orchestration system.
    """
    logger.info("Running legacy strategy in simplified mode")
    logger.info("Note: Consider migrating to the new orchestration system")

    # This is a minimal implementation for compatibility
    # In practice, users should migrate to the new system
    decisions = {}

    for symbol in symbols:
        try:
            # Get current quote
            quote = get_quote(symbol)
            if quote:
                logger.info(f"Strategy processing {symbol}: price ${quote.last}")
                decisions[symbol] = {"action": "HOLD", "reason": "Simplified legacy mode"}
            else:
                logger.warning(f"No quote available for {symbol}")
                decisions[symbol] = {"action": "HOLD", "reason": "No market data"}
        except Exception as exc:
            logger.error(f"Error processing {symbol}: {exc}")
            decisions[symbol] = {"action": "HOLD", "reason": f"Error: {exc}"}

    return decisions


__all__ = [
    "connect_broker",
    "disconnect",
    "place_order",
    "cancel_order",
    "get_orders",
    "close_all_positions",
    "get_positions",
    "get_positions_trading",
    "get_account",
    "get_account_snapshot",
    "get_quote",
    "get_market_hours",
    "run_strategy",
]
