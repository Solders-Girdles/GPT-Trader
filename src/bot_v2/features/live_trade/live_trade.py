"""
Main live trading orchestration - entry point for the slice.

Complete isolation - everything needed is local.
WARNING: This is a template - actual broker integration requires API credentials.
"""

from datetime import datetime
from typing import Dict, List, Optional
from .types import (
    BrokerConnection, Order, Position, AccountInfo,
    OrderStatus, OrderType, OrderSide, Quote, MarketHours
)
from .brokers import AlpacaBroker, IBKRBroker, SimulatedBroker
from .risk import LiveRiskManager
from .execution import ExecutionEngine


# Global broker connection
_broker_connection: Optional[BrokerConnection] = None
_broker_client = None
_risk_manager = None
_execution_engine = None


def connect_broker(
    broker_name: str = "alpaca",
    api_key: str = "",
    api_secret: str = "",
    is_paper: bool = True,
    base_url: Optional[str] = None
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
    """
    global _broker_connection, _broker_client, _risk_manager, _execution_engine
    
    # Create connection
    _broker_connection = BrokerConnection(
        broker_name=broker_name,
        api_key=api_key,
        api_secret=api_secret,
        is_paper=is_paper,
        is_connected=False,
        account_id=None,
        base_url=base_url
    )
    
    # Initialize broker client
    if broker_name == "alpaca":
        _broker_client = AlpacaBroker(api_key, api_secret, is_paper, base_url)
    elif broker_name == "ibkr":
        _broker_client = IBKRBroker(api_key, api_secret, is_paper)
    else:
        _broker_client = SimulatedBroker()
    
    # Connect
    if _broker_client.connect():
        _broker_connection.is_connected = True
        _broker_connection.account_id = _broker_client.get_account_id()
        
        # Initialize risk manager and execution engine
        _risk_manager = LiveRiskManager()
        _execution_engine = ExecutionEngine(_broker_client, _risk_manager)
        
        print(f"✅ Connected to {broker_name} ({'paper' if is_paper else 'live'} mode)")
        print(f"   Account ID: {_broker_connection.account_id}")
    else:
        print(f"❌ Failed to connect to {broker_name}")
    
    return _broker_connection


def place_order(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    time_in_force: str = "day"
) -> Optional[Order]:
    """
    Place an order.
    
    Args:
        symbol: Stock symbol
        side: 'buy' or 'sell'
        quantity: Number of shares
        order_type: 'market', 'limit', 'stop', 'stop_limit'
        limit_price: Limit price (for limit orders)
        stop_price: Stop price (for stop orders)
        time_in_force: 'day', 'gtc', 'ioc', 'fok'
        
    Returns:
        Order object or None if failed
    """
    if not _broker_connection or not _broker_connection.is_connected:
        print("❌ Not connected to broker")
        return None
    
    if not _execution_engine:
        print("❌ Execution engine not initialized")
        return None
    
    # Validate with risk manager
    account = get_account()
    if not _risk_manager.validate_order(symbol, side, quantity, account):
        print(f"❌ Order rejected by risk manager")
        return None
    
    # Place order through execution engine
    order = _execution_engine.place_order(
        symbol=symbol,
        side=OrderSide[side.upper()],
        quantity=quantity,
        order_type=OrderType[order_type.upper()],
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=time_in_force
    )
    
    if order:
        print(f"✅ Order placed: {order.order_id}")
        print(f"   {side.upper()} {quantity} {symbol} @ {order_type.upper()}")
    else:
        print(f"❌ Failed to place order")
    
    return order


def get_positions() -> List[Position]:
    """
    Get current positions.
    
    Returns:
        List of Position objects
    """
    if not _broker_client:
        return []
    
    positions = _broker_client.get_positions()
    
    if positions:
        print(f"📊 Current Positions:")
        for pos in positions:
            pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
            print(f"   {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")
            print(f"      Current: ${pos.current_price:.2f}")
            print(f"      P&L: {pnl_sign}${pos.unrealized_pnl:.2f} ({pnl_sign}{pos.unrealized_pnl_pct:.2%})")
    else:
        print("📊 No open positions")
    
    return positions


def get_account() -> Optional[AccountInfo]:
    """
    Get account information.
    
    Returns:
        AccountInfo object or None
    """
    if not _broker_client:
        return None
    
    account = _broker_client.get_account()
    
    if account:
        print(f"💰 Account Summary:")
        print(f"   Equity: ${account.equity:,.2f}")
        print(f"   Cash: ${account.cash:,.2f}")
        print(f"   Buying Power: ${account.buying_power:,.2f}")
        print(f"   Positions Value: ${account.positions_value:,.2f}")
        if account.pattern_day_trader:
            print(f"   Day Trades Remaining: {account.day_trades_remaining}")
    
    return account


def get_orders(status: str = "open") -> List[Order]:
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
    """
    if not _broker_client:
        return False
    
    success = _broker_client.cancel_order(order_id)
    
    if success:
        print(f"✅ Order {order_id} cancelled")
    else:
        print(f"❌ Failed to cancel order {order_id}")
    
    return success


def get_quote(symbol: str) -> Optional[Quote]:
    """
    Get real-time quote.
    
    Args:
        symbol: Stock symbol
        
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
            is_open=False,
            open_time=None,
            close_time=None,
            extended_hours_open=False
        )
    
    return _broker_client.get_market_hours()


def close_all_positions() -> bool:
    """
    Close all open positions.
    
    Returns:
        True if all positions closed successfully
    """
    positions = get_positions()
    
    if not positions:
        print("No positions to close")
        return True
    
    success = True
    for position in positions:
        # Determine side for closing
        close_side = "sell" if position.side == "long" else "buy"
        
        # Place market order to close
        order = place_order(
            symbol=position.symbol,
            side=close_side,
            quantity=abs(position.quantity),
            order_type="market"
        )
        
        if not order:
            success = False
            print(f"❌ Failed to close {position.symbol}")
    
    return success


def disconnect():
    """Disconnect from broker."""
    global _broker_connection, _broker_client
    
    if _broker_client:
        _broker_client.disconnect()
    
    _broker_connection = None
    _broker_client = None
    
    print("Disconnected from broker")