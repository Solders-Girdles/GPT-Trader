"""
Legacy live trading orchestration (equities-oriented template).

Important: The active production path is Coinbase Perpetual Futures via the perps bot
(`bot_v2.orchestration.perps_bot` and CLI `perps-bot`). This module remains for
legacy coverage and tests and uses template broker classes (Alpaca/IBKR/Simulated).
"""

import logging
from decimal import Decimal
from typing import Any

from ...errors import ExecutionError, NetworkError, ValidationError, log_error
from ...errors.handler import RecoveryStrategy, get_error_handler
from ...validation import PositiveNumberValidator, SymbolValidator

# Import core interfaces instead of local types
from ..brokerages.core.interfaces import (
    IBrokerage,
    Order,
    OrderSide,
    OrderType,
    Position,
    Quote,
    TimeInForce,
)
from .brokers import AlpacaBroker, IBKRBroker, SimulatedBroker
from .execution import ExecutionEngine
from .risk import LiveRiskManager
from .strategies.perps_baseline import Action, BaselinePerpsStrategy, Decision

# Keep local types that don't exist in core
from .types import AccountInfo, BrokerConnection, MarketHours

logger = logging.getLogger(__name__)


def _quantity_from_position(position: Any | None) -> Decimal:
    if position is None:
        return Decimal("0")

    raw = getattr(position, "quantity", None)
    if raw is None:
        raw = getattr(position, "qty", None)

    try:
        return Decimal(str(raw)) if raw is not None else Decimal("0")
    except Exception:
        return Decimal("0")


# Global broker connection
_broker_connection: BrokerConnection | None = None
_broker_client = None
_risk_manager = None
_execution_engine = None


def connect_broker(
    broker_name: str = "alpaca",
    api_key: str = "",
    api_secret: str = "",
    is_paper: bool = True,
    base_url: str | None = None,
) -> BrokerConnection:
    """
    Connect to a broker.

    Args:
        broker_name: Name of broker ('alpaca', 'ibkr', 'simulated')
        api_key: API key
        api_secret: API secret
        is_paper: Use paper trading account
        base_url: Optional base URL override

    Returns:
        BrokerConnection object

    Raises:
        ValidationError: If invalid broker configuration
        NetworkError: If connection fails
    """
    global _broker_connection, _broker_client, _risk_manager, _execution_engine

    try:
        # Validate inputs
        valid_brokers = ["alpaca", "ibkr", "simulated"]
        if broker_name not in valid_brokers:
            raise ValidationError(
                f"Invalid broker name: {broker_name}", field="broker_name", value=broker_name
            )

        # Create connection
        _broker_connection = BrokerConnection(
            broker_name=broker_name,
            api_key=api_key,
            api_secret=api_secret,
            is_paper=is_paper,
            is_connected=False,
            account_id=None,
            base_url=base_url,
        )

        # Initialize broker client with error handling
        error_handler = get_error_handler()

        def _create_broker_client() -> IBrokerage:
            if broker_name == "alpaca":
                return AlpacaBroker(api_key, api_secret, is_paper, base_url)
            elif broker_name == "ibkr":
                return IBKRBroker(api_key, api_secret, is_paper)
            else:
                return SimulatedBroker()

        _broker_client = _create_broker_client()

        # Connect with retry logic
        def _connect_with_validation() -> bool:
            if not _broker_client.connect():
                raise NetworkError(
                    f"Failed to establish connection to {broker_name}",
                    context={"broker": broker_name, "is_paper": is_paper},
                )
            return True

        success = error_handler.with_retry(
            _connect_with_validation, recovery_strategy=RecoveryStrategy.RETRY
        )

        if success:
            _broker_connection.is_connected = True
            _broker_connection.account_id = _broker_client.get_account_id()

            # Initialize risk manager and execution engine
            _risk_manager = LiveRiskManager()
            _execution_engine = ExecutionEngine(_broker_client, _risk_manager)

            logger.info(f"Connected to {broker_name} ({'paper' if is_paper else 'live'} mode)")
            logger.info(f"Account ID: {_broker_connection.account_id}")
            print(f"✅ Connected to {broker_name} ({'paper' if is_paper else 'live'} mode)")
            print(f"   Account ID: {_broker_connection.account_id}")
        else:
            raise NetworkError(
                f"Connection to {broker_name} failed",
                context={"broker": broker_name, "is_paper": is_paper},
            )

        return _broker_connection

    except Exception as e:
        if isinstance(e, ValidationError | NetworkError):
            log_error(e)
            logger.error(f"Broker connection failed: {e.message}")
            print(f"❌ Failed to connect to {broker_name}: {e.message}")
            raise
        else:
            network_error = NetworkError(
                f"Unexpected error connecting to {broker_name}",
                context={"broker": broker_name, "original_error": str(e)},
            )
            log_error(network_error)
            logger.error(f"Unexpected connection error: {network_error.message}")
            print(f"❌ Failed to connect to {broker_name}: {network_error.message}")
            raise network_error from e


def place_order(
    symbol: str,
    side: OrderSide,
    quantity: int,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    stop_price: float | None = None,
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
    try:
        # Validate connection state
        if not _broker_connection or not _broker_connection.is_connected:
            raise NetworkError("Not connected to broker")

        if not _execution_engine:
            raise ExecutionError("Execution engine not initialized")

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
        order = _execution_engine.place_order(
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
        if not _broker_client:
            raise NetworkError("Broker client not initialized")

        error_handler = get_error_handler()

        def _get_positions_from_broker() -> list[Position]:
            return _broker_client.get_positions()

        positions = error_handler.with_retry(
            _get_positions_from_broker, recovery_strategy=RecoveryStrategy.RETRY
        )

        if positions:
            logger.info(f"Retrieved {len(positions)} positions")
            print("📊 Current Positions:")
            for pos in positions:
                pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                print(f"   {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")
                print(f"      Current: ${pos.current_price:.2f}")
                print(
                    f"      P&L: {pnl_sign}${pos.unrealized_pnl:.2f} ({pnl_sign}{pos.unrealized_pnl_pct:.2%})"
                )
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


def get_account() -> AccountInfo | None:
    """
    Get account information.

    Returns:
        AccountInfo object or None

    Raises:
        NetworkError: If broker connection issues
    """
    try:
        if not _broker_client:
            raise NetworkError("Broker client not initialized")

        error_handler = get_error_handler()

        def _get_account_from_broker() -> AccountInfo:
            return _broker_client.get_account()

        account = error_handler.with_retry(
            _get_account_from_broker, recovery_strategy=RecoveryStrategy.RETRY
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


def get_orders(status: str = "open") -> list[Order]:
    """
    Get orders.

    Args:
        status: 'open', 'closed', 'all'

    Returns:
        List of Order objects
    """
    if not _broker_client:
        return []

    return _broker_client.get_orders(status)


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

        if not _broker_client:
            raise NetworkError("Broker client not initialized")

        error_handler = get_error_handler()

        def _cancel_order_with_broker() -> bool:
            return _broker_client.cancel_order(order_id)

        success = error_handler.with_retry(
            _cancel_order_with_broker, recovery_strategy=RecoveryStrategy.RETRY
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
    """
    Get real-time quote.

    Args:
        symbol: Trading symbol

    Returns:
        Quote object or None
    """
    if not _broker_client:
        return None

    return _broker_client.get_quote(symbol)


def get_market_hours() -> MarketHours:
    """
    Get market hours information.

    Returns:
        MarketHours object
    """
    if not _broker_client:
        return MarketHours(
            is_open=False, open_time=None, close_time=None, extended_hours_open=False
        )

    return _broker_client.get_market_hours()


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
                close_side = "sell" if position.side == "long" else "buy"

                # Place market order to close
                order = place_order(
                    symbol=position.symbol,
                    side=close_side,
                    quantity=abs(position.quantity),
                    order_type="market",
                )

                if not order:
                    success = False
                    failed_positions.append(position.symbol)
                    logger.error(f"Failed to close position: {position.symbol}")
                    print(f"❌ Failed to close {position.symbol}")
                else:
                    logger.info(f"Close order placed for {position.symbol}: {order.order_id}")

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
    strategy: BaselinePerpsStrategy,
    symbols: list[str],
    mark_cache: dict[str, Decimal] | None = None,
    mark_windows: dict[str, list[Decimal]] | None = None,
) -> dict[str, Decision]:
    """
    Run strategy for given symbols.

    Args:
        strategy: Strategy instance
        symbols: List of symbols to trade
        mark_cache: Optional mark price cache (symbol -> price)
        mark_windows: Optional mark price windows for MAs

    Returns:
        Dict of symbol -> Decision
    """
    if not _broker_client:
        logger.error("Broker not connected")
        return {}

    decisions = {}

    try:
        # Get account info
        account = get_account()
        if not account:
            logger.error("Unable to get account info")
            return {}

        equity = Decimal(str(account.equity))

        # Get current positions
        positions = get_positions()
        position_map = {pos.symbol: pos for pos in positions}

        for symbol in symbols:
            try:
                # Get current mark price
                current_mark = None
                if mark_cache and symbol in mark_cache:
                    current_mark = mark_cache[symbol]
                else:
                    # Fall back to quote
                    quote = get_quote(symbol)
                    if quote:
                        current_mark = Decimal(str(quote.last_price))

                if not current_mark:
                    logger.warning(f"No mark price for {symbol}")
                    continue

                # Get position state
                position_state = None
                if symbol in position_map:
                    pos = position_map[symbol]
                    position_state = {
                        "quantity": Decimal(str(_quantity_from_position(pos))),
                        "qty": Decimal(str(_quantity_from_position(pos))),
                        "side": pos.side,
                        "entry": Decimal(str(pos.avg_cost)),
                    }

                # Get recent marks for MAs
                recent_marks = None
                if mark_windows and symbol in mark_windows:
                    recent_marks = mark_windows[symbol]

                # Generate decision
                # Note: product would come from ProductCatalog in real implementation
                from ..brokerages.core.interfaces import MarketType, Product

                product = Product(
                    symbol=symbol,
                    base_asset=symbol.split("-")[0],
                    quote_asset="USD",
                    market_type=MarketType.PERPETUAL,
                    step_size=Decimal("0.001"),
                    min_size=Decimal("0.001"),
                    price_increment=Decimal("0.01"),
                    min_notional=Decimal("10"),
                )

                decision = strategy.decide(
                    symbol=symbol,
                    current_mark=current_mark,
                    position_state=position_state,
                    recent_marks=recent_marks,
                    equity=equity,
                    product=product,
                )

                decisions[symbol] = decision
                logger.info(f"{symbol} decision: {decision.action.value} - {decision.reason}")

                # Execute decision if actionable
                if decision.action in [Action.BUY, Action.SELL]:
                    # Entry order
                    side = "buy" if decision.action == Action.BUY else "sell"

                    quantity = decision.quantity
                    if quantity is None and decision.target_notional:
                        quantity = decision.target_notional / current_mark

                    if quantity is not None:
                        from ..brokerages.coinbase.utilities import enforce_perp_rules

                        quantized_qty, quantized_price = enforce_perp_rules(
                            product=product,
                            quantity=quantity,
                            price=current_mark,
                        )

                        logger.info(f"Would place {side} order: {quantized_qty} {symbol} @ market")

                elif decision.action == Action.CLOSE:
                    if position_state:
                        close_side = "sell" if position_state.get("side") == "long" else "buy"
                        quantity = decision.quantity
                        if quantity is None and decision.qty is not None:
                            quantity = decision.qty
                        if quantity is None:
                            quantity = _quantity_from_position(position_state)

                        if quantity is not None:
                            from ..brokerages.coinbase.utilities import enforce_perp_rules

                            quantized_qty, quantized_price = enforce_perp_rules(
                                product=product,
                                quantity=quantity,
                                price=current_mark,
                            )
                            logger.info(
                                f"Would place reduce-only {close_side} order: {quantized_qty} {symbol}"
                            )
                        else:
                            logger.info(
                                f"Skipping close for {symbol}; no quantity resolved from decision/state"
                            )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                decisions[symbol] = Decision(action=Action.HOLD, reason=f"Error: {str(e)}")

    except Exception as e:
        logger.error(f"Strategy runner error: {e}")

    return decisions


def disconnect() -> None:
    """Disconnect from broker."""
    global _broker_connection, _broker_client, _risk_manager, _execution_engine

    try:
        if _broker_client:
            _broker_client.disconnect()
            logger.info("Disconnected from broker")

        # Clean up global state
        _broker_connection = None
        _broker_client = None
        _risk_manager = None
        _execution_engine = None

        print("Disconnected from broker")

    except Exception as e:
        logger.warning(f"Error during disconnect: {e}")
        # Force cleanup even if disconnect fails
        _broker_connection = None
        _broker_client = None
        _risk_manager = None
        _execution_engine = None
        print("Force disconnected from broker")
