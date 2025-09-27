"""
Local execution engine for live trading.

Complete isolation - no external dependencies.
"""

from datetime import datetime
from typing import Optional, List, Dict
from .types import Order, OrderType, OrderSide, OrderStatus, ExecutionReport
from .risk import LiveRiskManager


class ExecutionEngine:
    """Manages order execution and lifecycle."""
    
    def __init__(self, broker_client, risk_manager: LiveRiskManager):
        """
        Initialize execution engine.
        
        Args:
            broker_client: Broker client instance
            risk_manager: Risk manager instance
        """
        self.broker = broker_client
        self.risk_manager = risk_manager
        self.pending_orders: Dict[str, Order] = {}
        self.execution_history: List[ExecutionReport] = []
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day"
    ) -> Optional[Order]:
        """
        Place an order through the broker.
        
        Args:
            symbol: Stock symbol
            side: Order side
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Order time in force
            
        Returns:
            Order object or None if failed
        """
        # Validate order type requirements
        if order_type == OrderType.LIMIT and limit_price is None:
            print("Error: Limit order requires limit_price")
            return None
        
        if order_type == OrderType.STOP and stop_price is None:
            print("Error: Stop order requires stop_price")
            return None
        
        if order_type == OrderType.STOP_LIMIT and (limit_price is None or stop_price is None):
            print("Error: Stop-limit order requires both limit_price and stop_price")
            return None
        
        # Place order through broker
        try:
            order = self.broker.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
            
            if order:
                # Track pending order
                self.pending_orders[order.order_id] = order
                
                # Log execution
                self._log_execution(order, "ORDER_PLACED")
                
            return order
            
        except Exception as e:
            print(f"Order placement failed: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        if order_id not in self.pending_orders:
            print(f"Order {order_id} not found in pending orders")
            return False
        
        success = self.broker.cancel_order(order_id)
        
        if success:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            self._log_execution(order, "ORDER_CANCELLED")
        
        return success
    
    def update_order_status(self):
        """Update status of pending orders."""
        if not self.pending_orders:
            return
        
        # Get current orders from broker
        current_orders = self.broker.get_orders("all")
        
        for current_order in current_orders:
            if current_order.order_id in self.pending_orders:
                pending_order = self.pending_orders[current_order.order_id]
                
                # Check for status change
                if pending_order.status != current_order.status:
                    old_status = pending_order.status
                    pending_order.status = current_order.status
                    pending_order.filled_qty = current_order.filled_qty
                    pending_order.avg_fill_price = current_order.avg_fill_price
                    
                    # Handle filled orders
                    if current_order.status == OrderStatus.FILLED:
                        pending_order.filled_at = datetime.now()
                        del self.pending_orders[current_order.order_id]
                        self._log_execution(current_order, "ORDER_FILLED")
                    
                    # Handle partial fills
                    elif current_order.status == OrderStatus.PARTIALLY_FILLED:
                        self._log_execution(current_order, "ORDER_PARTIAL_FILL")
                    
                    # Handle rejections
                    elif current_order.status == OrderStatus.REJECTED:
                        del self.pending_orders[current_order.order_id]
                        self._log_execution(current_order, "ORDER_REJECTED")
    
    def get_pending_orders(self) -> List[Order]:
        """Get list of pending orders."""
        return list(self.pending_orders.values())
    
    def get_execution_history(self) -> List[ExecutionReport]:
        """Get execution history."""
        return self.execution_history
    
    def _log_execution(self, order: Order, event_type: str):
        """
        Log an execution event.
        
        Args:
            order: Order object
            event_type: Type of event
        """
        report = ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.avg_fill_price or order.price or 0,
            commission=order.commission,
            timestamp=datetime.now(),
            execution_id=f"{order.order_id}_{event_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        self.execution_history.append(report)
        
        # Print execution message
        if event_type == "ORDER_PLACED":
            print(f"📝 Order Placed: {order.side.value.upper()} {order.quantity} {order.symbol}")
        elif event_type == "ORDER_FILLED":
            print(f"✅ Order Filled: {order.side.value.upper()} {order.quantity} {order.symbol} @ ${order.avg_fill_price:.2f}")
        elif event_type == "ORDER_PARTIAL_FILL":
            print(f"⚠️ Partial Fill: {order.filled_qty}/{order.quantity} {order.symbol}")
        elif event_type == "ORDER_CANCELLED":
            print(f"❌ Order Cancelled: {order.order_id}")
        elif event_type == "ORDER_REJECTED":
            print(f"🚫 Order Rejected: {order.order_id}")
    
    def calculate_slippage(self, expected_price: float, actual_price: float, side: OrderSide) -> float:
        """
        Calculate slippage from execution.
        
        Args:
            expected_price: Expected execution price
            actual_price: Actual execution price
            side: Order side
            
        Returns:
            Slippage amount (negative is unfavorable)
        """
        if side == OrderSide.BUY:
            # For buys, higher actual price is unfavorable
            slippage = expected_price - actual_price
        else:
            # For sells, lower actual price is unfavorable
            slippage = actual_price - expected_price
        
        return slippage
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics."""
        if not self.execution_history:
            return {
                'total_orders': 0,
                'filled_orders': 0,
                'cancelled_orders': 0,
                'rejected_orders': 0,
                'total_commission': 0.0,
                'avg_fill_time': 0.0
            }
        
        filled = [r for r in self.execution_history if 'FILLED' in r.execution_id]
        cancelled = [r for r in self.execution_history if 'CANCELLED' in r.execution_id]
        rejected = [r for r in self.execution_history if 'REJECTED' in r.execution_id]
        
        total_commission = sum(r.commission for r in filled)
        
        return {
            'total_orders': len(self.execution_history),
            'filled_orders': len(filled),
            'cancelled_orders': len(cancelled),
            'rejected_orders': len(rejected),
            'total_commission': total_commission,
            'avg_fill_time': 0.0  # Would calculate from timestamps
        }