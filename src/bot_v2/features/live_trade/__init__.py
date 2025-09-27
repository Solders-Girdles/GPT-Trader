"""
Live trading feature slice - real broker integration.

Complete isolation - no external dependencies.
"""

from .live_trade import connect_broker, place_order, get_positions, get_account, disconnect
# Import core types
from ..brokerages.core.interfaces import Order, Position, Quote, OrderStatus
# Keep local types that don't exist in core
from .types import BrokerConnection, AccountInfo
from typing import Dict, Any, Optional

# Module-level broker connection (singleton pattern)
_broker_connection: Optional[BrokerConnection] = None

def execute_live_trade(
    symbol: str,
    action: str,
    quantity: int,
    strategy_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Facade function for orchestrator compatibility.
    
    Args:
        symbol: Trading symbol to trade
        action: 'buy' or 'sell'
        quantity: Number of shares to trade
        strategy_info: Strategy metadata (strategy name, confidence, etc.)
        
    Returns:
        Dict with trade execution result
    """
    global _broker_connection
    
    try:
        # Connect to broker if not already connected
        if _broker_connection is None:
            _broker_connection = connect_broker()
        
        # Note: This needs proper implementation with actual broker methods
        # For now, return a placeholder result
        
        # Place order through actual broker methods
        order_side = 'buy' if action == 'buy' else 'sell'
        order = place_order(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
            order_type='market'
        )
        
        if not order:
            raise Exception("Order placement failed")
        
        # Use core Order fields
        result = {
            'filled': order.status == OrderStatus.FILLED,
            'order_id': order.id,
            'filled_price': float(order.avg_fill_price) if order.avg_fill_price else 0.0
        }
        
        return {
            'status': 'executed' if result.get('filled', False) else 'pending',
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_id': result.get('order_id', 'unknown'),
            'filled_price': result.get('filled_price', 0.0),
            'strategy': strategy_info.get('strategy', 'unknown'),
            'confidence': strategy_info.get('confidence', 0.5),
            'message': f"Live trade {'executed' if result.get('filled', False) else 'submitted'}: {action} {quantity} units of {symbol}"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'error': str(e),
            'message': f"Failed to execute live trade: {e}"
        }

__all__ = [
    'connect_broker',
    'place_order',
    'get_positions', 
    'get_account',
    'disconnect',
    'execute_live_trade',  # Added facade function
    'BrokerConnection',
    'Order',
    'Position',
    'AccountInfo'
]
