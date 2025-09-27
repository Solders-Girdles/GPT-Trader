"""
Local broker implementations for live trading (legacy equities templates).

Note: Coinbase Perpetual Futures is the primary, production focus for live trading.
These Alpaca/IBKR classes are retained as templates and for legacy tests only.
Actual integrations require broker APIs and credentials.
"""

from datetime import datetime
from typing import List, Optional, Dict
import random
import logging
# Import core interfaces instead of local types
from ..brokerages.core.interfaces import (
    Order, Position, Quote, 
    OrderType, OrderSide, TimeInForce,
    Balance, IBrokerage, InvalidRequestError, 
    InsufficientFunds, BrokerageError
)

# Import core types with proper status enum
from ..brokerages.core.interfaces import OrderStatus as CoreOrderStatus

# Keep local types that don't exist in core
from .types import (
    AccountInfo, MarketHours, Bar
)
from ...errors import NetworkError, ExecutionError, ValidationError, handle_error, log_error
from ...errors.handler import get_error_handler, with_error_handling, RecoveryStrategy
from ...validation import SymbolValidator, PositiveNumberValidator
from ...config import get_config

logger = logging.getLogger(__name__)


class BrokerInterface:
    """Base interface for all brokers."""
    
    def connect(self) -> bool:
        """Connect to broker."""
        raise NotImplementedError
    
    def validate_connection(self) -> bool:
        """Validate broker connection is active."""
        raise NotImplementedError
    
    def disconnect(self):
        """Disconnect from broker."""
        raise NotImplementedError
    
    def get_account_id(self) -> str:
        """Get account ID."""
        raise NotImplementedError
    
    def get_account(self) -> AccountInfo:
        """Get account information."""
        raise NotImplementedError
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        raise NotImplementedError
    
    def place_order(self, **kwargs) -> Order:
        """Place an order."""
        raise NotImplementedError
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        raise NotImplementedError
    
    def get_orders(self, status: str) -> List[Order]:
        """Get orders."""
        raise NotImplementedError
    
    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote."""
        raise NotImplementedError
    
    def get_market_hours(self) -> MarketHours:
        """Get market hours."""
        raise NotImplementedError


class AlpacaBroker(BrokerInterface):
    """Alpaca broker implementation."""
    
    def __init__(self, api_key: str, api_secret: str, is_paper: bool = True, base_url: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_paper = is_paper
        self.base_url = base_url or ("https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets")
        self.connected = False
        self.account_id = None
    
    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            # Validate configuration
            if not self.api_key or not self.api_secret:
                raise ValidationError(
                    "API key and secret are required for Alpaca connection",
                    field="api_credentials"
                )
            
            # In real implementation, would authenticate with API
            error_handler = get_error_handler()
            
            def _authenticate():
                # Simulate API authentication
                if self.api_key and self.api_secret:
                    return True
                raise NetworkError("Authentication failed")
            
            success = error_handler.with_retry(
                _authenticate,
                recovery_strategy=RecoveryStrategy.RETRY
            )
            
            if success:
                self.connected = True
                self.account_id = "ALPACA_" + ("PAPER" if self.is_paper else "LIVE")
                logger.info(f"Connected to Alpaca ({'paper' if self.is_paper else 'live'} mode)")
                return True
            
            return False
            
        except Exception as e:
            network_error = NetworkError(
                "Failed to connect to Alpaca",
                url=self.base_url,
                context={'is_paper': self.is_paper, 'original_error': str(e)}
            )
            log_error(network_error)
            logger.error(f"Alpaca connection failed: {network_error.message}")
            return False
    
    def validate_connection(self) -> bool:
        """Validate Alpaca connection is active."""
        if not self.connected:
            raise NetworkError("Not connected to Alpaca", url=self.base_url)
        return True
    
    def disconnect(self):
        """Disconnect from Alpaca."""
        self.connected = False
    
    def get_account_id(self) -> str:
        """Get account ID."""
        return self.account_id or ""
    
    @with_error_handling(recovery_strategy=RecoveryStrategy.RETRY)
    def get_account(self) -> AccountInfo:
        """Get account information from Alpaca."""
        self.validate_connection()
        
        # In real implementation, would make API call
        # Template implementation with rate limiting consideration
        config = get_config('live_trade')
        
        return AccountInfo(
            account_id=self.account_id or "",
            cash=config.get('initial_capital', 100000.0),
            portfolio_value=config.get('initial_capital', 100000.0),
            buying_power=config.get('initial_capital', 100000.0) * 2,  # 2x margin
            positions_value=0.0,
            margin_used=0.0,
            pattern_day_trader=False,
            day_trades_remaining=3,
            equity=config.get('initial_capital', 100000.0),
            last_equity=config.get('initial_capital', 100000.0)
        )
    
    @with_error_handling(recovery_strategy=RecoveryStrategy.RETRY)
    def get_positions(self) -> List[Position]:
        """Get positions from Alpaca."""
        self.validate_connection()
        
        # In real implementation, would make API call
        # Template implementation
        return []
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day"
    ) -> Order:
        """Place order through Alpaca."""
        self.validate_connection()
        
        # Validate inputs
        SymbolValidator().validate(symbol, "symbol")
        PositiveNumberValidator(allow_zero=False).validate(quantity, "quantity")
        
        error_handler = get_error_handler()
        
        def _submit_order():
            # In real implementation, would make API call
            # Simulate potential network issues
            order_id = f"ALPACA_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"
            
            # Check for rate limiting
            config = get_config('live_trade')
            if hasattr(self, '_last_order_time'):
                min_interval = config.get('min_order_interval', 1.0)  # seconds
                time_since_last = (datetime.now() - self._last_order_time).total_seconds()
                if time_since_last < min_interval:
                    raise NetworkError(
                        "Rate limit exceeded - too many orders",
                        url=self.base_url,
                        status_code=429
                    )
            
            self._last_order_time = datetime.now()
            
            # Create and return core Order directly
            from decimal import Decimal
            from .adapters import to_core_tif
            
            return Order(
                id=order_id,
                client_id=None,
                symbol=symbol,
                side=side,
                type=order_type,
                qty=Decimal(str(quantity)),
                price=Decimal(str(limit_price)) if limit_price else None,
                stop_price=Decimal(str(stop_price)) if stop_price else None,
                tif=to_core_tif(time_in_force),
                status=CoreOrderStatus.SUBMITTED,
                filled_qty=Decimal('0'),
                avg_fill_price=None,
                submitted_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        try:
            return error_handler.with_retry(
                _submit_order,
                recovery_strategy=RecoveryStrategy.RETRY
            )
            
        except Exception as e:
            execution_error = ExecutionError(
                f"Failed to place order with Alpaca",
                context={
                    'symbol': symbol,
                    'side': side.value,
                    'quantity': quantity,
                    'broker': 'alpaca',
                    'original_error': str(e)
                }
            )
            log_error(execution_error)
            raise execution_error
    
    @with_error_handling(recovery_strategy=RecoveryStrategy.RETRY)
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order through Alpaca."""
        self.validate_connection()
        
        if not order_id:
            raise ValidationError("Order ID is required", field="order_id", value=order_id)
        
        # In real implementation, would make API call
        # Template implementation
        logger.info(f"Cancelled order {order_id} with Alpaca")
        return True
    
    @with_error_handling(recovery_strategy=RecoveryStrategy.RETRY)
    def get_orders(self, status: str = "open") -> List[Order]:
        """Get orders from Alpaca."""
        self.validate_connection()
        
        # Validate status parameter
        valid_statuses = ['open', 'closed', 'all']
        if status not in valid_statuses:
            raise ValidationError(
                f"Invalid order status: {status}",
                field="status",
                value=status
            )
        
        # In real implementation, would make API call
        # Template implementation
        return []
    
    @with_error_handling(recovery_strategy=RecoveryStrategy.RETRY)
    def get_quote(self, symbol: str) -> Quote:
        """Get quote from Alpaca."""
        self.validate_connection()
        
        # Validate symbol
        SymbolValidator().validate(symbol, "symbol")
        
        # In real implementation, would make API call
        # Template implementation - random prices
        base_price = 100.0
        spread = 0.01
        
        # Return core Quote directly
        from decimal import Decimal
        return Quote(
            symbol=symbol,
            bid=Decimal(str(base_price - spread/2)),
            ask=Decimal(str(base_price + spread/2)),
            last=Decimal(str(base_price)),
            ts=datetime.now()
        )
    
    @with_error_handling(recovery_strategy=RecoveryStrategy.RETRY)
    def get_market_hours(self) -> MarketHours:
        """Get market hours from Alpaca."""
        self.validate_connection()
        
        # In real implementation, would make API call
        now = datetime.now()
        is_weekday = now.weekday() < 5
        hour = now.hour
        
        # Simple market hours check
        is_open = is_weekday and 9 <= hour < 16
        
        return MarketHours(
            is_open=is_open,
            open_time=now.replace(hour=9, minute=30),
            close_time=now.replace(hour=16, minute=0),
            extended_hours_open=is_weekday and 4 <= hour < 20
        )


class IBKRBroker(BrokerInterface):
    """Interactive Brokers implementation."""
    
    def __init__(self, api_key: str, api_secret: str, is_paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_paper = is_paper
        self.connected = False
        self.account_id = None
    
    def connect(self) -> bool:
        """Connect to IBKR."""
        try:
            # Validate configuration
            if not self.api_key:
                raise ValidationError(
                    "API key is required for IBKR connection",
                    field="api_key"
                )
            
            # Template implementation with error handling
            error_handler = get_error_handler()
            
            def _authenticate():
                if self.api_key:
                    return True
                raise NetworkError("IBKR authentication failed")
            
            success = error_handler.with_retry(
                _authenticate,
                recovery_strategy=RecoveryStrategy.RETRY
            )
            
            if success:
                self.connected = True
                self.account_id = "IBKR_" + ("PAPER" if self.is_paper else "LIVE")
                logger.info(f"Connected to IBKR ({'paper' if self.is_paper else 'live'} mode)")
                return True
            
            return False
            
        except Exception as e:
            network_error = NetworkError(
                "Failed to connect to IBKR",
                context={'is_paper': self.is_paper, 'original_error': str(e)}
            )
            log_error(network_error)
            logger.error(f"IBKR connection failed: {network_error.message}")
            return False
    
    def validate_connection(self) -> bool:
        """Validate IBKR connection is active."""
        if not self.connected:
            raise NetworkError("Not connected to IBKR")
        return True
    
    def disconnect(self):
        """Disconnect from IBKR."""
        self.connected = False
    
    def get_account_id(self) -> str:
        """Get account ID."""
        return self.account_id or ""
    
    def get_account(self) -> AccountInfo:
        """Get account from IBKR."""
        # Template implementation
        return AccountInfo(
            account_id=self.account_id or "",
            cash=250000.0,
            portfolio_value=250000.0,
            buying_power=1000000.0,  # 4x margin
            positions_value=0.0,
            margin_used=0.0,
            pattern_day_trader=True,
            day_trades_remaining=999,
            equity=250000.0,
            last_equity=250000.0
        )
    
    def get_positions(self) -> List[Position]:
        """Get positions from IBKR."""
        return []
    
    def place_order(self, **kwargs) -> Order:
        """Place order through IBKR."""
        # Similar to Alpaca implementation
        return AlpacaBroker.place_order(self, **kwargs)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order through IBKR."""
        return True
    
    def get_orders(self, status: str) -> List[Order]:
        """Get orders from IBKR."""
        return []
    
    def get_quote(self, symbol: str) -> Quote:
        """Get quote from IBKR."""
        return AlpacaBroker.get_quote(self, symbol)
    
    def get_market_hours(self) -> MarketHours:
        """Get market hours from IBKR."""
        return AlpacaBroker.get_market_hours(self)


class SimulatedBroker(BrokerInterface):
    """Simulated broker for testing without real connection."""
    
    def __init__(self):
        self.connected = False
        self.account_id = "SIM_001"
        self.cash = 100000.0
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_counter = 0
    
    def connect(self) -> bool:
        """Connect to simulated broker."""
        try:
            self.connected = True
            logger.info("Connected to simulated broker")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to simulated broker: {e}")
            return False
    
    def validate_connection(self) -> bool:
        """Validate simulated broker connection is active."""
        if not self.connected:
            raise NetworkError("Not connected to simulated broker")
        return True
    
    def disconnect(self):
        """Disconnect from simulated broker."""
        self.connected = False
    
    def get_account_id(self) -> str:
        """Get account ID."""
        return self.account_id
    
    def get_account(self) -> AccountInfo:
        """Get simulated account."""
        positions_value = sum(p.market_value for p in self.positions.values())
        equity = self.cash + positions_value
        
        return AccountInfo(
            account_id=self.account_id,
            cash=self.cash,
            portfolio_value=equity,
            buying_power=self.cash * 2,
            positions_value=positions_value,
            margin_used=positions_value,
            pattern_day_trader=False,
            day_trades_remaining=3,
            equity=equity,
            last_equity=equity
        )
    
    def get_positions(self) -> List[Position]:
        """Get simulated positions."""
        return list(self.positions.values())
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day"
    ) -> Order:
        """Place simulated order."""
        self.order_counter += 1
        order_id = f"SIM_{self.order_counter:06d}"
        
        # Simulate immediate fill for market orders
        if order_type == OrderType.MARKET:
            fill_price = 100.0 + random.uniform(-5, 5)  # Random price
            
            # Create and return core Order directly
            from decimal import Decimal
            from .adapters import to_core_tif
            
            order = Order(
                id=order_id,
                client_id=None,
                symbol=symbol,
                side=side,
                type=order_type,
                qty=Decimal(str(quantity)),
                price=Decimal(str(limit_price)) if limit_price else None,
                stop_price=Decimal(str(stop_price)) if stop_price else None,
                tif=to_core_tif(time_in_force),
                status=CoreOrderStatus.FILLED,
                filled_qty=Decimal(str(quantity)),
                avg_fill_price=Decimal(str(fill_price)),
                submitted_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Update positions
            if side == OrderSide.BUY:
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    new_qty = pos.quantity + quantity
                    new_cost = ((pos.avg_cost * pos.quantity) + (fill_price * quantity)) / new_qty
                    pos.quantity = new_qty
                    pos.avg_cost = new_cost
                else:
                    # Create core Position directly
                    from decimal import Decimal
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        qty=Decimal(str(quantity)),
                        entry_price=Decimal(str(fill_price)),
                        mark_price=Decimal(str(fill_price)),
                        unrealized_pnl=Decimal('0'),
                        realized_pnl=Decimal('0'),
                        leverage=None,
                        side='long'
                    )
                self.cash -= (fill_price * quantity + quantity * 0.01)  # Include commission
            
            elif side == OrderSide.SELL:
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    if quantity >= pos.quantity:
                        # Close position
                        realized_pnl = (fill_price - pos.avg_cost) * pos.quantity
                        del self.positions[symbol]
                    else:
                        # Partial sell
                        pos.quantity -= quantity
                        realized_pnl = (fill_price - pos.avg_cost) * quantity
                    
                    self.cash += (fill_price * quantity - order.commission)
        else:
            # Limit/stop orders stay pending
            from decimal import Decimal
            from .adapters import to_core_tif
            
            order = Order(
                id=order_id,
                client_id=None,
                symbol=symbol,
                side=side,
                type=order_type,
                qty=Decimal(str(quantity)),
                price=Decimal(str(limit_price)) if limit_price else None,
                stop_price=Decimal(str(stop_price)) if stop_price else None,
                tif=to_core_tif(time_in_force),
                status=CoreOrderStatus.SUBMITTED,
                filled_qty=Decimal('0'),
                avg_fill_price=None,
                submitted_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        self.orders.append(order)
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel simulated order."""
        for order in self.orders:
            if order.id == order_id and order.status in [CoreOrderStatus.PENDING, CoreOrderStatus.SUBMITTED, CoreOrderStatus.PARTIALLY_FILLED]:
                order.status = CoreOrderStatus.CANCELLED
                return True
        return False
    
    def get_orders(self, status: str = "open") -> List[Order]:
        """Get simulated orders."""
        if status == "open":
            return [o for o in self.orders if o.status in [CoreOrderStatus.PENDING, CoreOrderStatus.SUBMITTED, CoreOrderStatus.PARTIALLY_FILLED]]
        elif status == "closed":
            return [o for o in self.orders if o.status in [CoreOrderStatus.FILLED, CoreOrderStatus.CANCELLED, CoreOrderStatus.REJECTED]]
        else:
            return self.orders
    
    def get_quote(self, symbol: str) -> Quote:
        """Get simulated quote."""
        base_price = 100.0 + random.uniform(-10, 10)
        spread = 0.02
        
        # Return core Quote directly
        from decimal import Decimal
        return Quote(
            symbol=symbol,
            bid=Decimal(str(base_price - spread/2)),
            ask=Decimal(str(base_price + spread/2)),
            last=Decimal(str(base_price)),
            ts=datetime.now()
        )
    
    def get_market_hours(self) -> MarketHours:
        """Get simulated market hours."""
        now = datetime.now()
        is_weekday = now.weekday() < 5
        hour = now.hour
        
        return MarketHours(
            is_open=is_weekday and 9 <= hour < 16,
            open_time=now.replace(hour=9, minute=30),
            close_time=now.replace(hour=16, minute=0),
            extended_hours_open=is_weekday and 4 <= hour < 20
        )
