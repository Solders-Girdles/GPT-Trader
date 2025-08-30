"""
Minimal ledger module - Clear trade recording that makes sense.
Tracks BOTH transactions (fills) AND completed trades (round-trips).
"""

import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Transaction:
    """A single buy or sell transaction."""
    date: datetime
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: int
    price: float
    cost: float  # Negative for buys, positive for sells
    
    
@dataclass 
class CompletedTrade:
    """A completed round-trip trade (buy + sell)."""
    symbol: str
    entry_date: datetime
    entry_price: float
    entry_quantity: int
    exit_date: datetime
    exit_price: float
    exit_quantity: int
    pnl: float
    return_pct: float
    holding_days: int


class TradeLedger:
    """
    Ledger that properly tracks both transactions and completed trades.
    This makes sense - you can see both your transaction history AND 
    your completed round-trip trades.
    """
    
    def __init__(self):
        self.transactions: List[Transaction] = []
        self.completed_trades: List[CompletedTrade] = []
        self.open_positions: dict = {}  # Track open positions for matching
        
    def record_transaction(
        self,
        date: datetime,
        symbol: str,
        action: str,
        quantity: int,
        price: float
    ):
        """
        Record a buy or sell transaction.
        
        Args:
            date: Transaction date
            symbol: Stock symbol
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Price per share
        """
        # Calculate cost (negative for buys, positive for sells)
        cost = quantity * price * (-1 if action == 'buy' else 1)
        
        # Record transaction
        transaction = Transaction(
            date=date,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            cost=cost
        )
        self.transactions.append(transaction)
        
        # Update positions and check for completed trades
        if action == 'buy':
            if symbol not in self.open_positions:
                self.open_positions[symbol] = []
            self.open_positions[symbol].append({
                'date': date,
                'quantity': quantity,
                'price': price
            })
            
        elif action == 'sell' and symbol in self.open_positions:
            # Match with open positions (FIFO)
            remaining_to_sell = quantity
            positions = self.open_positions[symbol]
            
            while remaining_to_sell > 0 and positions:
                position = positions[0]
                
                # Calculate how much we can close from this position
                close_quantity = min(remaining_to_sell, position['quantity'])
                
                # Record completed trade
                pnl = close_quantity * (price - position['price'])
                return_pct = ((price - position['price']) / position['price']) * 100
                holding_days = (date - position['date']).days
                
                completed = CompletedTrade(
                    symbol=symbol,
                    entry_date=position['date'],
                    entry_price=position['price'],
                    entry_quantity=close_quantity,
                    exit_date=date,
                    exit_price=price,
                    exit_quantity=close_quantity,
                    pnl=pnl,
                    return_pct=return_pct,
                    holding_days=holding_days
                )
                self.completed_trades.append(completed)
                
                # Update position
                position['quantity'] -= close_quantity
                remaining_to_sell -= close_quantity
                
                # Remove position if fully closed
                if position['quantity'] == 0:
                    positions.pop(0)
                    
            # Clean up if no positions left
            if not positions:
                del self.open_positions[symbol]
                
    def get_transaction_history(self) -> pd.DataFrame:
        """Get all transactions as a DataFrame."""
        if not self.transactions:
            return pd.DataFrame()
            
        data = [
            {
                'date': t.date,
                'symbol': t.symbol,
                'action': t.action,
                'quantity': t.quantity,
                'price': t.price,
                'cost': t.cost
            }
            for t in self.transactions
        ]
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    def get_completed_trades(self) -> pd.DataFrame:
        """Get all completed trades as a DataFrame."""
        if not self.completed_trades:
            return pd.DataFrame()
            
        data = [
            {
                'symbol': t.symbol,
                'entry_date': t.entry_date,
                'entry_price': t.entry_price,
                'entry_quantity': t.entry_quantity,
                'exit_date': t.exit_date,
                'exit_price': t.exit_price,
                'exit_quantity': t.exit_quantity,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'holding_days': t.holding_days
            }
            for t in self.completed_trades
        ]
        
        return pd.DataFrame(data)
    
    def get_open_positions_summary(self) -> pd.DataFrame:
        """Get summary of current open positions."""
        if not self.open_positions:
            return pd.DataFrame()
            
        data = []
        for symbol, positions in self.open_positions.items():
            total_quantity = sum(p['quantity'] for p in positions)
            avg_price = sum(p['price'] * p['quantity'] for p in positions) / total_quantity
            
            data.append({
                'symbol': symbol,
                'quantity': total_quantity,
                'avg_price': avg_price,
                'oldest_entry': min(p['date'] for p in positions)
            })
            
        return pd.DataFrame(data)
    
    def calculate_statistics(self) -> dict:
        """Calculate trading statistics."""
        stats = {
            'total_transactions': len(self.transactions),
            'total_completed_trades': len(self.completed_trades),
            'open_positions': len(self.open_positions)
        }
        
        if self.completed_trades:
            trades_df = self.get_completed_trades()
            stats.update({
                'total_pnl': trades_df['pnl'].sum(),
                'avg_pnl': trades_df['pnl'].mean(),
                'win_rate': (trades_df['pnl'] > 0).mean() * 100,
                'avg_return_pct': trades_df['return_pct'].mean(),
                'avg_holding_days': trades_df['holding_days'].mean(),
                'best_trade': trades_df['pnl'].max(),
                'worst_trade': trades_df['pnl'].min()
            })
            
        return stats
    
    def reset(self):
        """Clear all records."""
        self.transactions = []
        self.completed_trades = []
        self.open_positions = {}