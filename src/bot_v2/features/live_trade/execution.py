"""
Local execution engine for live trading.

Complete isolation - no external dependencies.
"""

from datetime import datetime
from typing import Optional, List, Dict
import logging
# Import core interfaces instead of local types
from ..brokerages.core.interfaces import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce
)

# Keep local types that don't exist in core
from .types import ExecutionReport
from .risk import LiveRiskManager
from ...errors import ExecutionError, NetworkError, ValidationError, handle_error, log_error
from ...errors.handler import get_error_handler, with_error_handling, RecoveryStrategy
from ...validation import (
    SymbolValidator, PositiveNumberValidator, ChoiceValidator,
    CompositeValidator, validate_inputs
)
from ...config import get_config

logger = logging.getLogger(__name__)


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
        symbol: Trading symbol
            side: Order side
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Order time in force
            
        Returns:
            Order object or None if failed
        """
        # Validate inputs
        try:
            self._validate_order_inputs(
                symbol, side, quantity, order_type, limit_price, stop_price, time_in_force
            )
        except ValidationError as e:
            log_error(e)
            logger.error(f"Order validation failed: {e.message}")
            return None
        
        # Place order through broker with error handling
        error_handler = get_error_handler()
        
        def _place_order_with_broker():
            return self.broker.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force
            )
        
        try:
            order = error_handler.with_retry(
                _place_order_with_broker,
                recovery_strategy=RecoveryStrategy.RETRY
            )
            
            if order:
                # Track pending order
                self.pending_orders[order.id] = order
                
                # Log execution
                self._log_execution(order, "ORDER_PLACED")
                logger.info(f"Order placed successfully: {order.id}")
            else:
                raise ExecutionError(
                    "Broker returned None for order placement",
                    context={'symbol': symbol, 'side': side.value, 'quantity': quantity}
                )
                
            return order
            
        except Exception as e:
            execution_error = ExecutionError(
                f"Order placement failed for {symbol}",
                context={
                    'symbol': symbol,
                    'side': side.value if hasattr(side, 'value') else str(side),
                    'quantity': quantity,
                    'order_type': order_type.value if hasattr(order_type, 'value') else str(order_type),
                    'original_error': str(e)
                }
            )
            log_error(execution_error)
            logger.error(f"Order placement failed: {execution_error.message}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        # Validate order ID
        try:
            if not order_id or not isinstance(order_id, str):
                raise ValidationError("Order ID must be a non-empty string", field="order_id", value=order_id)
        except ValidationError as e:
            log_error(e)
            logger.error(f"Invalid order ID: {e.message}")
            return False
        
        if order_id not in self.pending_orders:
            logger.warning(f"Order {order_id} not found in pending orders")
            return False
        
        error_handler = get_error_handler()
        
        def _cancel_order_with_broker():
            return self.broker.cancel_order(order_id)
        
        try:
            success = error_handler.with_retry(
                _cancel_order_with_broker,
                recovery_strategy=RecoveryStrategy.RETRY
            )
            
            if success:
                order = self.pending_orders[order_id]
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order_id]
                self._log_execution(order, "ORDER_CANCELLED")
                logger.info(f"Order cancelled successfully: {order_id}")
            else:
                logger.warning(f"Failed to cancel order: {order_id}")
            
            return success
            
        except Exception as e:
            execution_error = ExecutionError(
                f"Order cancellation failed for {order_id}",
                order_id=order_id,
                context={'original_error': str(e)}
            )
            log_error(execution_error)
            logger.error(f"Order cancellation failed: {execution_error.message}")
            return False
    
    def update_order_status(self):
        """Update status of pending orders."""
        if not self.pending_orders:
            return
        
        error_handler = get_error_handler()
        
        def _get_orders_from_broker():
            return self.broker.get_orders("all")
        
        try:
            # Get current orders from broker with error handling
            current_orders = error_handler.with_retry(
                _get_orders_from_broker,
                recovery_strategy=RecoveryStrategy.RETRY
            )
            
            for current_order in current_orders:
                if current_order.id in self.pending_orders:
                    pending_order = self.pending_orders[current_order.id]
                    
                    # Check for status change
                    if pending_order.status != current_order.status:
                        old_status = pending_order.status
                        pending_order.status = current_order.status
                        pending_order.filled_qty = current_order.filled_qty
                        pending_order.avg_fill_price = current_order.avg_fill_price
                        
                        # Handle filled orders
                        if current_order.status == OrderStatus.FILLED:
                            pending_order.updated_at = datetime.now()
                            del self.pending_orders[current_order.id]
                            self._log_execution(current_order, "ORDER_FILLED")
                        
                        # Handle partial fills
                        elif current_order.status == OrderStatus.PARTIALLY_FILLED:
                            self._log_execution(current_order, "ORDER_PARTIAL_FILL")
                        
                        # Handle rejections
                        elif current_order.status == OrderStatus.REJECTED:
                            del self.pending_orders[current_order.id]
                            self._log_execution(current_order, "ORDER_REJECTED")
                            
                            # Log rejection as error for monitoring
                            execution_error = ExecutionError(
                                f"Order rejected by broker: {current_order.id}",
                                order_id=current_order.id,
                                context={'symbol': current_order.symbol}
                            )
                            log_error(execution_error)
            
        except Exception as e:
            network_error = NetworkError(
                "Failed to update order status from broker",
                context={'pending_orders': len(self.pending_orders), 'original_error': str(e)}
            )
            log_error(network_error)
            logger.warning(f"Order status update failed: {network_error.message}")
    
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
        # Convert Decimal qty to int for ExecutionReport
        qty_int = int(order.qty)
        price_float = float(order.avg_fill_price or order.price or 0)
        
        report = ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=qty_int,
            price=price_float,
            commission=0.0,  # Core Order doesn't have commission
            timestamp=datetime.now(),
            execution_id=f"{order.id}_{event_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        
        self.execution_history.append(report)
        
        # Print execution message
        if event_type == "ORDER_PLACED":
            print(f"📝 Order Placed: {order.side.value.upper()} {qty_int} {order.symbol}")
        elif event_type == "ORDER_FILLED":
            avg_price = float(order.avg_fill_price) if order.avg_fill_price else 0
            print(f"✅ Order Filled: {order.side.value.upper()} {qty_int} {order.symbol} @ ${avg_price:.2f}")
        elif event_type == "ORDER_PARTIAL_FILL":
            filled_int = int(order.filled_qty)
            print(f"⚠️ Partial Fill: {filled_int}/{qty_int} {order.symbol}")
        elif event_type == "ORDER_CANCELLED":
            print(f"❌ Order Cancelled: {order.id}")
        elif event_type == "ORDER_REJECTED":
            print(f"🚫 Order Rejected: {order.id}")
    
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
    
    def _validate_order_inputs(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        limit_price: Optional[float],
        stop_price: Optional[float],
        time_in_force: str
    ) -> None:
        """
        Validate order inputs before submission.
        
        Args:
        symbol: Trading symbol
            side: Order side
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price
            stop_price: Stop price
            time_in_force: Order time in force
            
        Raises:
            ValidationError: If any input is invalid
        """
        # Validate symbol
        symbol_validator = SymbolValidator()
        symbol_validator.validate(symbol, "symbol")
        
        # Validate quantity
        quantity_validator = PositiveNumberValidator(allow_zero=False)
        quantity_validator.validate(quantity, "quantity")
        
        # Validate time in force
        tif_validator = ChoiceValidator(['day', 'gtc', 'ioc', 'fok'])
        # Handle both string and enum types
        tif_value = time_in_force.value.lower() if hasattr(time_in_force, 'value') else str(time_in_force).lower()
        tif_validator.validate(tif_value, "time_in_force")
        
        # Validate order type requirements
        if order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValidationError(
                    "Limit order requires limit_price",
                    field="limit_price",
                    value=limit_price
                )
            PositiveNumberValidator(allow_zero=False).validate(limit_price, "limit_price")
        
        if order_type == OrderType.STOP:
            if stop_price is None:
                raise ValidationError(
                    "Stop order requires stop_price",
                    field="stop_price",
                    value=stop_price
                )
            PositiveNumberValidator(allow_zero=False).validate(stop_price, "stop_price")
        
        if order_type == OrderType.STOP_LIMIT:
            if limit_price is None or stop_price is None:
                raise ValidationError(
                    "Stop-limit order requires both limit_price and stop_price",
                    field="limit_price,stop_price",
                    value={'limit_price': limit_price, 'stop_price': stop_price}
                )
            PositiveNumberValidator(allow_zero=False).validate(limit_price, "limit_price")
            PositiveNumberValidator(allow_zero=False).validate(stop_price, "stop_price")
        
        # Additional business logic validation
        config = get_config('live_trade')
        max_quantity = config.get('max_order_quantity', 10000)
        if quantity > max_quantity:
            raise ValidationError(
                f"Order quantity exceeds maximum allowed: {max_quantity}",
                field="quantity",
                value=quantity
            )
    
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
