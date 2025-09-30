"""
Legacy live trading orchestration retained for compatibility demos.

Production trading is handled by the Coinbase-specific orchestration
(`bot_v2.orchestration`).  This module now proxies exclusively to the
``SimulatedBroker`` stub so that historical examples and tests continue to
function without maintaining third-party integrations.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from bot_v2.errors import ExecutionError, NetworkError, ValidationError, log_error
from bot_v2.errors.handler import RecoveryStrategy, get_error_handler
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
    get_execution_engine,
    get_risk_manager,
)
from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
)
from bot_v2.features.live_trade.types import (
    AccountInfo,
    BrokerConnection,
    MarketHours,
    position_to_trading_position,
)
from bot_v2.types.trading import AccountSnapshot, TradingPosition
from bot_v2.utilities.quantities import quantity_from
from bot_v2.validation import PositiveNumberValidator, SymbolValidator

logger = logging.getLogger(__name__)


def _quantity_from_position(position: Any | None) -> Decimal:
    """Resolve quantity regardless of legacy naming."""

    resolved = quantity_from(position, default=Decimal("0"))
    return resolved if resolved is not None else Decimal("0")


@dataclass
class AccountContext:
    equity: Decimal
    position_map: dict[str, Position]


@dataclass
class SymbolContext:
    symbol: str
    current_mark: Decimal
    equity: Decimal
    position_state: dict[str, Any] | None
    recent_marks: list[Decimal] | None
    product: Any


def connect_broker(
    broker_name: str = "simulated",
    api_key: str = "",
    api_secret: str = "",
    is_paper: bool = True,
    base_url: str | None = None,
) -> BrokerConnection:
    """Connect to the simulated broker stub used by legacy demos."""

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
    Place an order (template interface).

    Args:
        symbol: Trading symbol (e.g., AAPL, BTC-USD)
        side: 'buy' or 'sell'
        quantity: Units to trade
        order_type: 'market', 'limit', 'stop', 'stop_limit'
        limit_price: Limit price (for limit orders)
        stop_price: Stop price (for stop orders)
        time_in_force: 'day', 'gtc', 'ioc', 'fok'

    Returns:
        Order object or None if failed

    Raises:
        ValidationError: If inputs are invalid
        ExecutionError: If order placement fails
        NetworkError: If broker connection issues
    """
    connection = get_connection()
    if not connection or not connection.is_connected:
        raise NetworkError("Not connected to broker")

    execution_engine = get_execution_engine()
    if not execution_engine:
        raise ExecutionError("Execution engine not initialized")

    try:
        # Validate inputs
        symbol_validator = SymbolValidator()
        symbol = symbol_validator.validate(symbol, "symbol")

        quantity_validator = PositiveNumberValidator(allow_zero=False)
        quantity_validator.validate(quantity, "quantity")

        # Get account info for validation
        account = get_account()
        if not account:
            raise ExecutionError("Unable to retrieve account information")

        # Place order through execution engine (includes risk validation)
        order = execution_engine.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )

        if order:
            logger.info(f"Order placed successfully: {order.id}")
            print(f"✅ Order placed: {order.id}")
            print(f"   {side.name} {quantity} {symbol} @ {order_type.name}")
        else:
            raise ExecutionError("Order placement returned None")

        return order

    except ValidationError as e:
        log_error(e)
        logger.error(f"Order placement failed: {e.message}")
        print(f"❌ Failed to place order: {e.message}")
        raise
    except (ExecutionError, NetworkError) as e:
        log_error(e)
        logger.error(f"Order placement failed: {e.message}")
        print(f"❌ Failed to place order: {e.message}")
        return None
    except Exception as e:
        execution_error = ExecutionError(
            "Unexpected error during order placement",
            context={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "original_error": str(e),
            },
        )
        log_error(execution_error)
        logger.error(f"Unexpected order placement error: {execution_error.message}")
        print(f"❌ Failed to place order: {execution_error.message}")
        return None


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
            logger.info(f"Retrieved {len(positions)} positions")
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

    except NetworkError as e:
        log_error(e)
        logger.error(f"Failed to get positions: {e.message}")
        print(f"❌ Failed to get positions: {e.message}")
        return []
    except Exception as e:
        network_error = NetworkError(
            "Unexpected error retrieving positions", context={"original_error": str(e)}
        )
        log_error(network_error)
        logger.error(f"Unexpected positions error: {network_error.message}")
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
            logger.info(f"Retrieved account info for {account.account_id}")
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

    except NetworkError as e:
        log_error(e)
        logger.error(f"Failed to get account: {e.message}")
        print(f"❌ Failed to get account: {e.message}")
        return None
    except Exception as e:
        network_error = NetworkError(
            "Unexpected error retrieving account", context={"original_error": str(e)}
        )
        log_error(network_error)
        logger.error(f"Unexpected account error: {network_error.message}")
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


def cancel_order(order_id: str) -> bool:
    """
    Cancel an order.

    Args:
        order_id: Order ID to cancel

    Returns:
        True if cancelled successfully

    Raises:
        ValidationError: If order ID is invalid
        NetworkError: If broker connection issues
    """
    try:
        # Validate order ID
        if not order_id or not isinstance(order_id, str):
            raise ValidationError(
                "Order ID must be a non-empty string", field="order_id", value=order_id
            )

        broker = get_broker_client()
        if broker is None:
            raise NetworkError("Broker client not initialized")

        error_handler = get_error_handler()

        def _cancel_order_with_broker() -> bool:
            return broker.cancel_order(order_id)

        success = cast(
            bool,
            error_handler.with_retry(
                _cancel_order_with_broker, recovery_strategy=RecoveryStrategy.RETRY
            ),
        )

        if success:
            logger.info(f"Order cancelled successfully: {order_id}")
            print(f"✅ Order {order_id} cancelled")
        else:
            logger.warning(f"Failed to cancel order: {order_id}")
            print(f"❌ Failed to cancel order {order_id}")

        return success

    except (ValidationError, NetworkError) as e:
        log_error(e)
        logger.error(f"Order cancellation failed: {e.message}")
        print(f"❌ Failed to cancel order: {e.message}")
        return False
    except Exception as e:
        execution_error = ExecutionError(
            "Unexpected error during order cancellation",
            order_id=order_id,
            context={"original_error": str(e)},
        )
        log_error(execution_error)
        logger.error(f"Unexpected cancellation error: {execution_error.message}")
        print(f"❌ Failed to cancel order: {execution_error.message}")
        return False


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


def close_all_positions() -> bool:
    """
    Close all open positions.

    Returns:
        True if all positions closed successfully

    Raises:
        ExecutionError: If position closure fails
    """
    try:
        positions = get_positions()

        if not positions:
            logger.info("No positions to close")
            print("No positions to close")
            return True

        logger.info(f"Closing {len(positions)} positions")
        success = True
        failed_positions = []

        for position in positions:
            try:
                # Determine side for closing
                close_side = OrderSide.SELL if position.side == "long" else OrderSide.BUY

                # Place market order to close
                order = place_order(
                    symbol=position.symbol,
                    side=close_side,
                    quantity=abs(position.quantity),
                    order_type=OrderType.MARKET,
                )

                if not order:
                    success = False
                    failed_positions.append(position.symbol)
                    logger.error(f"Failed to close position: {position.symbol}")
                    print(f"❌ Failed to close {position.symbol}")
                else:
                    logger.info(f"Close order placed for {position.symbol}: {order.id}")

            except Exception as e:
                success = False
                failed_positions.append(position.symbol)
                logger.error(f"Error closing {position.symbol}: {e}")
                print(f"❌ Error closing {position.symbol}: {e}")

        if failed_positions:
            execution_error = ExecutionError(
                f"Failed to close {len(failed_positions)} positions",
                context={"failed_positions": failed_positions},
            )
            log_error(execution_error)

        return success

    except Exception as e:
        execution_error = ExecutionError(
            "Unexpected error closing all positions", context={"original_error": str(e)}
        )
        log_error(execution_error)
        logger.error(f"Failed to close positions: {execution_error.message}")
        print(f"❌ Failed to close positions: {execution_error.message}")
        return False


def run_strategy(
    symbols: list[str],
    strategy_name: str = "baseline_perps",
    iterations: int = 3,
    mark_cache: dict[str, Decimal] | None = None,
    mark_windows: dict[str, list[Decimal]] | None = None,
    *,
    strategy_override: Any | None = None,
) -> None:
    """Run a basic trading strategy for demonstration purposes."""

    broker = get_broker_client()
    connection = get_connection()
    risk_manager = get_risk_manager()
    if not broker or not connection or not connection.is_connected or not risk_manager:
        raise RuntimeError("Broker connection not initialized")

    strategy = strategy_override or BaselinePerpsStrategy(
        environment="simulated", risk_manager=risk_manager
    )

    decisions_accumulator: dict[str, Decision] = {}

    for _ in range(iterations):
        account_info = get_account()
        if not account_info:
            break

        account_ctx = AccountContext(
            equity=Decimal(str(account_info.equity)),
            position_map={pos.symbol: pos for pos in broker.get_positions()},
        )

        for symbol in symbols:
            try:
                symbol_ctx = _prepare_symbol_context(
                    symbol=symbol,
                    account_context=account_ctx,
                    mark_cache=mark_cache,
                    mark_windows=mark_windows,
                )

                if symbol_ctx is None:
                    continue

                decision = _generate_strategy_decision(strategy, symbol_ctx)
                decisions_accumulator[symbol] = decision
                _execute_decision_if_actionable(decision, symbol_ctx)
            except Exception as exc:
                logger.error("Error processing %s: %s", symbol, exc)

    return decisions_accumulator


def _validate_broker_connection() -> bool:
    connection = get_connection()
    broker = get_broker_client()
    if connection and connection.is_connected and broker is not None:
        return True
    logger.error("Broker not connected")
    return False


def _prepare_account_context() -> AccountContext | None:
    account = get_account()
    if not account:
        logger.error("Unable to get account info")
        return None

    equity = Decimal(str(account.equity))
    positions = get_positions()
    position_map = {pos.symbol: pos for pos in positions}
    return AccountContext(equity=equity, position_map=position_map)


def _prepare_symbol_context(
    *,
    symbol: str,
    account_context: AccountContext,
    mark_cache: dict[str, Decimal] | None,
    mark_windows: dict[str, list[Decimal]] | None,
) -> SymbolContext | None:
    current_mark = None
    if mark_cache and symbol in mark_cache:
        current_mark = mark_cache[symbol]
    else:
        quote = get_quote(symbol)
        if quote:
            current_mark = Decimal(str(quote.last))

    if not current_mark:
        logger.warning(f"No mark price for {symbol}")
        return None

    position_state = None
    if symbol in account_context.position_map:
        position = account_context.position_map[symbol]
        resolved_quantity = _quantity_from_position(position)
        position_state = {
            "quantity": Decimal(str(resolved_quantity)),
            "side": position.side,
            "entry": Decimal(str(position.entry_price)),
        }

    recent_marks = None
    if mark_windows and symbol in mark_windows:
        recent_marks = mark_windows[symbol]

    product = _build_template_product(symbol)

    return SymbolContext(
        symbol=symbol,
        current_mark=current_mark,
        equity=account_context.equity,
        position_state=position_state,
        recent_marks=recent_marks,
        product=product,
    )


def _generate_strategy_decision(
    strategy: BaselinePerpsStrategy, context: SymbolContext
) -> Decision:
    decision = strategy.decide(
        symbol=context.symbol,
        current_mark=context.current_mark,
        position_state=context.position_state,
        recent_marks=context.recent_marks,
        equity=context.equity,
        product=context.product,
    )
    logger.info(f"{context.symbol} decision: {decision.action.value} - {decision.reason}")
    return decision


def _execute_decision_if_actionable(decision: Decision, context: SymbolContext) -> None:
    symbol = context.symbol
    current_mark = context.current_mark
    product = context.product

    if decision.action in (Action.BUY, Action.SELL):
        side = "buy" if decision.action == Action.BUY else "sell"
        quantity = decision.quantity
        if quantity is None and decision.target_notional:
            quantity = decision.target_notional / current_mark

        if quantity is not None:
            from bot_v2.features.brokerages.coinbase.utilities import enforce_perp_rules

            quantized_quantity, _ = enforce_perp_rules(
                product=product,
                quantity=quantity,
                price=current_mark,
            )

            logger.info(f"Would place {side} order: {quantized_quantity} {symbol} @ market")

    elif decision.action == Action.CLOSE and context.position_state:
        close_side = "sell" if context.position_state.get("side") == "long" else "buy"
        quantity = decision.quantity
        if quantity is None:
            quantity = _quantity_from_position(context.position_state)

        if quantity is not None:
            from bot_v2.features.brokerages.coinbase.utilities import enforce_perp_rules

            quantized_quantity, _ = enforce_perp_rules(
                product=product,
                quantity=quantity,
                price=current_mark,
            )
            logger.info(
                f"Would place reduce-only {close_side} order: {quantized_quantity} {symbol}"
            )
        else:
            logger.info(f"Skipping close for {symbol}; no quantity resolved from decision/state")


def _handle_symbol_error(symbol: str, exc: Exception) -> Decision:
    logger.error(f"Error processing {symbol}: {exc}")
    return Decision(action=Action.HOLD, reason=f"Error: {str(exc)}")


def _build_template_product(symbol: str) -> Any:
    from bot_v2.features.brokerages.core.interfaces import MarketType, Product

    base_asset = symbol.split("-")[0]
    return Product(
        symbol=symbol,
        base_asset=base_asset,
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
    )


def disconnect() -> None:
    """Disconnect from broker."""
    try:
        _disconnect()
        logger.info("Disconnected from broker")
        print("Disconnected from broker")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Error during disconnect: %s", exc)
        print("Force disconnected from broker")


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
