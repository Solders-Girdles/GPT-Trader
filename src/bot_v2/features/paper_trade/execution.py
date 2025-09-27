"""
Order execution simulation for paper trading.

Complete isolation - no external dependencies.
"""

from datetime import datetime
from typing import List, Dict, Optional, Literal
from .types import Position, TradeLog, AccountStatus


class PaperExecutor:
    """Simulated order executor for paper trading."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_positions: int = 10
    ):
        """
        Initialize paper executor.
        
        Args:
            initial_capital: Starting cash
            commission: Commission rate
            slippage: Slippage rate
            max_positions: Maximum concurrent positions
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_positions = max_positions
        
        self.positions: Dict[str, Position] = {}
        self.trade_log: List[TradeLog] = []
        self.trade_id = 0
        self.day_trades = 0
        self.last_trade_date = None
    
    def execute_signal(
        self,
        symbol: str,
        signal: int,
        current_price: float,
        timestamp: datetime,
        position_size: float = 0.95
    ) -> Optional[TradeLog]:
        """
        Execute a trading signal.
        
        Args:
            symbol: Trading symbol
            signal: 1 (buy), -1 (sell), 0 (hold)
            current_price: Current market price
            timestamp: Current time
            position_size: Fraction of capital to use
            
        Returns:
            TradeLog if trade executed, None otherwise
        """
        if signal == 0:
            return None
        
        # Reset day trade counter if new day
        if self.last_trade_date and timestamp.date() != self.last_trade_date:
            self.day_trades = 0
        
        # Check if we have a position
        has_position = symbol in self.positions
        
        # Process buy signal
        if signal == 1 and not has_position:
            return self._execute_buy(symbol, current_price, timestamp, position_size)
        
        # Process sell signal
        elif signal == -1 and has_position:
            return self._execute_sell(symbol, current_price, timestamp)
        
        return None
    
    def _execute_buy(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        position_size: float
    ) -> Optional[TradeLog]:
        """Execute a buy order."""
        # Check position limit
        if len(self.positions) >= self.max_positions:
            return None
        
        # Apply slippage
        execution_price = price * (1 + self.slippage)
        
        # Calculate position size
        available_capital = self.cash * position_size
        max_shares = int(available_capital / execution_price)
        
        if max_shares <= 0:
            return None
        
        # Calculate costs
        cost = max_shares * execution_price
        commission_cost = cost * self.commission
        total_cost = cost + commission_cost
        
        if total_cost > self.cash:
            # Adjust shares if needed
            max_shares = int((self.cash * 0.99) / (execution_price * (1 + self.commission)))
            if max_shares <= 0:
                return None
            cost = max_shares * execution_price
            commission_cost = cost * self.commission
            total_cost = cost + commission_cost
        
        # Execute trade
        self.cash -= total_cost
        
        # Create position
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=max_shares,
            entry_price=execution_price,
            entry_date=timestamp,
            current_price=execution_price,
            unrealized_pnl=0,
            value=cost
        )
        
        # Log trade
        trade = TradeLog(
            id=self.trade_id,
            symbol=symbol,
            side='buy',
            quantity=max_shares,
            price=execution_price,
            timestamp=timestamp,
            commission=commission_cost,
            slippage=price * self.slippage
        )
        
        self.trade_log.append(trade)
        self.trade_id += 1
        self.last_trade_date = timestamp.date()
        
        return trade
    
    def _execute_sell(
        self,
        symbol: str,
        price: float,
        timestamp: datetime
    ) -> Optional[TradeLog]:
        """Execute a sell order."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Apply slippage
        execution_price = price * (1 - self.slippage)
        
        # Calculate proceeds
        proceeds = position.quantity * execution_price
        commission_cost = proceeds * self.commission
        net_proceeds = proceeds - commission_cost
        
        # Update cash
        self.cash += net_proceeds
        
        # Log trade
        trade = TradeLog(
            id=self.trade_id,
            symbol=symbol,
            side='sell',
            quantity=position.quantity,
            price=execution_price,
            timestamp=timestamp,
            commission=commission_cost,
            slippage=price * self.slippage
        )
        
        self.trade_log.append(trade)
        self.trade_id += 1
        
        # Remove position
        del self.positions[symbol]
        
        # Track day trades
        if timestamp.date() == position.entry_date.date():
            self.day_trades += 1
        
        self.last_trade_date = timestamp.date()
        
        return trade
    
    def update_positions(self, prices: Dict[str, float]):
        """
        Update position values with current prices.
        
        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.current_price = current_price
                position.value = position.quantity * current_price
                position.unrealized_pnl = (
                    (current_price - position.entry_price) * position.quantity
                )
    
    def get_account_status(self) -> AccountStatus:
        """Get current account status."""
        positions_value = sum(p.value for p in self.positions.values())
        total_equity = self.cash + positions_value
        
        return AccountStatus(
            cash=self.cash,
            positions_value=positions_value,
            total_equity=total_equity,
            buying_power=self.cash * 2,  # Simplified margin
            margin_used=positions_value,
            day_trades_remaining=max(0, 3 - self.day_trades)  # PDT rule
        )
    
    def close_all_positions(self, prices: Dict[str, float], timestamp: datetime):
        """
        Close all open positions.
        
        Args:
            prices: Current prices for all symbols
            timestamp: Current time
        """
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            if symbol in prices:
                self._execute_sell(symbol, prices[symbol], timestamp)
