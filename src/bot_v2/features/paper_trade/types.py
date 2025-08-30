"""
Local types for paper trading - no external dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Literal
import pandas as pd


@dataclass
class Position:
    """Current position in paper trading."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float
    unrealized_pnl: float
    value: float


@dataclass
class TradeLog:
    """Record of a completed trade."""
    id: int
    symbol: str
    side: Literal['buy', 'sell']
    quantity: int
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    

@dataclass  
class AccountStatus:
    """Current account status."""
    cash: float
    positions_value: float
    total_equity: float
    buying_power: float
    margin_used: float
    day_trades_remaining: int


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades_count: int
    

@dataclass
class PaperTradeResult:
    """Complete paper trading session results."""
    start_time: datetime
    end_time: Optional[datetime]
    account_status: AccountStatus
    positions: List[Position]
    trade_log: List[TradeLog]
    performance: PerformanceMetrics
    equity_curve: pd.Series
    
    def summary(self) -> str:
        """Generate summary report."""
        duration = self.end_time - self.start_time if self.end_time else "Ongoing"
        
        return f"""
Paper Trading Summary
====================
Duration: {duration}
Total Equity: ${self.account_status.total_equity:,.2f}
Total Return: {self.performance.total_return:.2%}
Sharpe Ratio: {self.performance.sharpe_ratio:.2f}
Max Drawdown: {self.performance.max_drawdown:.2%}
Win Rate: {self.performance.win_rate:.2%}
Total Trades: {self.performance.trades_count}
Open Positions: {len(self.positions)}
        """.strip()