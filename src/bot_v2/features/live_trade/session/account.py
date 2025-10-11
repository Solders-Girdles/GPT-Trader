"""Account and portfolio helpers for the legacy live-trade facade."""

from __future__ import annotations

import logging
from typing import cast

from bot_v2.errors import NetworkError, log_error
from bot_v2.errors.handler import RecoveryStrategy, get_error_handler
from bot_v2.features.brokerages.core.interfaces import Order, Position, Quote
from bot_v2.features.live_trade.types import AccountInfo, MarketHours, position_to_trading_position
from bot_v2.types.trading import AccountSnapshot, TradingPosition

from .registry import get_broker_client

logger = logging.getLogger(__name__)


def get_positions() -> list[Position]:
    """
    Get current positions.

    Returns:
        List of Position objects

    Raises:
        NetworkError: If broker connection issues
    """
    try:
        broker = get_broker_client()
        if broker is None:
            raise NetworkError("Broker client not initialized")

        error_handler = get_error_handler()

        def _get_positions_from_broker() -> list[Position]:
            return broker.get_positions()

        positions = cast(
            list[Position],
            error_handler.with_retry(
                _get_positions_from_broker, recovery_strategy=RecoveryStrategy.RETRY
            ),
        )

        if positions:
            logger.info("Retrieved %s positions", len(positions))
            print("📊 Current Positions:")
            for pos in positions:
                entry_price = float(pos.entry_price)
                mark_price = float(pos.mark_price)
                quantity = float(pos.quantity)
                cost_basis = abs(float(pos.entry_price * pos.quantity))
                print(f"   {pos.symbol}: {quantity:.4f} units @ ${entry_price:.2f}")
                print(f"      Mark: ${mark_price:.2f}")

                pnl_value = float(pos.unrealized_pnl)
                pnl_sign = "+" if pnl_value >= 0 else "-"
                pnl_pct = (pnl_value / cost_basis) * 100 if cost_basis else 0.0
                print(f"      P&L: {pnl_sign}${abs(pnl_value):.2f} ({pnl_sign}{abs(pnl_pct):.2f}%)")
        else:
            logger.info("No open positions")
            print("📊 No open positions")

        return positions

    except NetworkError as exc:
        log_error(exc)
        logger.error("Failed to get positions: %s", exc.message)
        print(f"❌ Failed to get positions: {exc.message}")
        return []
    except Exception as exc:
        network_error = NetworkError(
            "Unexpected error retrieving positions", context={"original_error": str(exc)}
        )
        log_error(network_error)
        logger.error("Unexpected positions error: %s", network_error.message)
        print(f"❌ Failed to get positions: {network_error.message}")
        return []


def get_positions_trading() -> list[TradingPosition]:
    """Return current positions using the shared trading type schema."""

    return [position_to_trading_position(pos) for pos in get_positions()]


def get_account() -> AccountInfo | None:
    """
    Get account information.

    Returns:
        AccountInfo object or None

    Raises:
        NetworkError: If broker connection issues
    """
    try:
        broker = get_broker_client()
        if broker is None:
            raise NetworkError("Broker client not initialized")

        error_handler = get_error_handler()

        def _get_account_from_broker() -> AccountInfo:
            return broker.get_account()

        account = cast(
            AccountInfo,
            error_handler.with_retry(
                _get_account_from_broker, recovery_strategy=RecoveryStrategy.RETRY
            ),
        )

        if account:
            logger.info("Retrieved account info for %s", account.account_id)
            print("💰 Account Summary:")
            print(f"   Equity: ${account.equity:,.2f}")
            print(f"   Cash: ${account.cash:,.2f}")
            print(f"   Buying Power: ${account.buying_power:,.2f}")
            print(f"   Positions Value: ${account.positions_value:,.2f}")
            if account.pattern_day_trader:
                print(f"   Day Trades Remaining: {account.day_trades_remaining}")
        else:
            logger.warning("Account information not available")

        return account

    except NetworkError as exc:
        log_error(exc)
        logger.error("Failed to get account: %s", exc.message)
        print(f"❌ Failed to get account: {exc.message}")
        return None
    except Exception as exc:
        network_error = NetworkError(
            "Unexpected error retrieving account", context={"original_error": str(exc)}
        )
        log_error(network_error)
        logger.error("Unexpected account error: %s", network_error.message)
        print(f"❌ Failed to get account: {network_error.message}")
        return None


def get_account_snapshot() -> AccountSnapshot | None:
    """Return the active account as a shared account snapshot."""

    account = get_account()
    return account.to_account_snapshot() if account else None


def get_orders(status: str = "open") -> list[Order]:
    """Get orders for the active broker session."""

    broker = get_broker_client()
    if broker is None:
        return []

    return broker.get_orders(status)


def get_quote(symbol: str) -> Quote | None:
    """Get real-time quote for ``symbol``."""

    broker = get_broker_client()
    if broker is None:
        return None

    return broker.get_quote(symbol)


def get_market_hours() -> MarketHours:
    """Return market hours information from the active broker."""

    broker = get_broker_client()
    if broker is None:
        return MarketHours(
            is_open=False, open_time=None, close_time=None, extended_hours_open=False
        )

    return broker.get_market_hours()


__all__ = [
    "get_positions",
    "get_positions_trading",
    "get_account",
    "get_account_snapshot",
    "get_orders",
    "get_quote",
    "get_market_hours",
]
