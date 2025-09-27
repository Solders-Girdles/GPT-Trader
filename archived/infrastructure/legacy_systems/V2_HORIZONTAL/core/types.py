"""
Core data types that flow between system components.

These types define the data structures used throughout the system,
ensuring consistency and type safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd


class SignalType(Enum):
    """Types of trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0
    

class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status states."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionStatus(Enum):
    """Position status states."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class MarketData:
    """Market data for a single symbol."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask
        }


@dataclass
class Signal:
    """Trading signal from a strategy."""
    strategy_name: str
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    strength: float  # Signal strength/confidence [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_actionable(self) -> bool:
        """Check if signal requires action."""
        return self.signal_type != SignalType.HOLD


@dataclass
class Position:
    """A trading position."""
    position_id: str
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.status == PositionStatus.OPEN:
            return (self.current_price - self.entry_price) * self.quantity
        return 0.0
    
    @property
    def realized_pnl(self) -> float:
        """Calculate realized P&L."""
        if self.exit_price:
            return (self.exit_price - self.entry_price) * self.quantity
        return 0.0
    
    @property
    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.status == PositionStatus.OPEN:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        elif self.exit_price:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        return 0.0


@dataclass
class Order:
    """A trading order."""
    order_id: str
    symbol: str
    quantity: float
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    commission: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """An executed trade."""
    trade_id: str
    order_id: str
    symbol: str
    quantity: float
    price: float
    side: str
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        return (self.price * self.quantity) + self.commission


@dataclass
class Portfolio:
    """Portfolio state."""
    cash: float
    positions: Dict[str, Position]
    timestamp: datetime
    initial_capital: float
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values() 
            if pos.status == PositionStatus.OPEN
        )
        return self.cash + position_value
    
    @property
    def total_return(self) -> float:
        """Calculate total return."""
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100
    
    @property
    def position_count(self) -> int:
        """Count open positions."""
        return sum(1 for pos in self.positions.values() if pos.status == PositionStatus.OPEN)
    
    def get_allocation(self) -> Dict[str, float]:
        """Get current allocation percentages."""
        total = self.total_value
        if total == 0:
            return {}
        
        allocations = {'cash': self.cash / total}
        for symbol, pos in self.positions.items():
            if pos.status == PositionStatus.OPEN:
                allocations[symbol] = (pos.quantity * pos.current_price) / total
        
        return allocations


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio."""
    timestamp: datetime
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    beta: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    position_concentration: float  # Largest position as % of portfolio
    leverage: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_hold_time: float  # in days
    best_trade: float
    worst_trade: float
    recovery_factor: float
    calmar_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)