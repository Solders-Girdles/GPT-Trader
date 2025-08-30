"""
Minimal executor module - Simple position management.
Converts signals to position changes and tracks holdings.
"""

import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """Represents a position in a symbol."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    
    def current_value(self, current_price: float) -> float:
        """Calculate current value of position."""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        return (current_price - self.entry_price) * self.quantity


class SimpleExecutor:
    """
    Simple executor that manages positions based on signals.
    Only allows one position per symbol at a time.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.position_history: List[Tuple[datetime, str, str, float, int]] = []
        
    def process_signal(
        self, 
        symbol: str, 
        signal: int, 
        price: float, 
        date: datetime
    ) -> Dict:
        """
        Process a trading signal and update positions.
        
        Args:
            symbol: Stock symbol
            signal: 1 = buy, -1 = sell, 0 = hold
            price: Current price
            date: Current date
            
        Returns:
            Dict with action taken and details
        """
        action = {'type': 'hold', 'symbol': symbol, 'date': date}
        
        if signal == 1:  # Buy signal
            # Only buy if we don't have a position
            if symbol not in self.positions:
                # Use all available cash (simple sizing)
                max_shares = int(self.cash / price)
                if max_shares > 0:
                    cost = max_shares * price
                    self.cash -= cost
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=max_shares,
                        entry_price=price,
                        entry_date=date
                    )
                    self.position_history.append(
                        (date, symbol, 'buy', price, max_shares)
                    )
                    action = {
                        'type': 'buy',
                        'symbol': symbol,
                        'date': date,
                        'price': price,
                        'quantity': max_shares,
                        'cost': cost
                    }
                    
        elif signal == -1:  # Sell signal
            # Only sell if we have a position
            if symbol in self.positions:
                position = self.positions[symbol]
                proceeds = position.quantity * price
                self.cash += proceeds
                realized_pnl = (price - position.entry_price) * position.quantity
                
                self.position_history.append(
                    (date, symbol, 'sell', price, position.quantity)
                )
                
                action = {
                    'type': 'sell',
                    'symbol': symbol,
                    'date': date,
                    'price': price,
                    'quantity': position.quantity,
                    'proceeds': proceeds,
                    'realized_pnl': realized_pnl
                }
                
                # Remove position
                del self.positions[symbol]
                
        return action
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Dict of symbol -> current price
            
        Returns:
            Total portfolio value (cash + positions)
        """
        total = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += position.current_value(current_prices[symbol])
                
        return total
    
    def get_position_summary(self) -> Dict:
        """Get summary of current positions."""
        return {
            'cash': self.cash,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'entry_date': pos.entry_date
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def reset(self):
        """Reset executor to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.position_history = []