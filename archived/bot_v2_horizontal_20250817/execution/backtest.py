"""
Minimal backtest module - Basic backtesting engine.
Clear execution flow: Data → Strategy → Executor → Ledger → Results.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from data import DataLoader
from strategy import SimpleMAStrategy
from executor import SimpleExecutor
from ledger import TradeLedger


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float
    buy_and_hold_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_transactions: int
    completed_trades: int
    win_rate: float
    portfolio_values: pd.Series
    transactions: pd.DataFrame
    completed_trades_df: pd.DataFrame
    trade_statistics: dict


class SimpleBacktest:
    """
    Simple backtesting engine with clear execution flow.
    No hidden complexity, everything is transparent.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0  # Can add later if needed
    ):
        """
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (flat fee)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.data_loader = DataLoader()
        
    def run(
        self,
        strategy: SimpleMAStrategy,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResults:
        """
        Run a backtest for a single symbol.
        
        Args:
            strategy: Strategy to test
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            BacktestResults with all metrics
        """
        # Step 1: Load data
        print(f"Loading data for {symbol}...")
        data = self.data_loader.get_data(symbol, start_date, end_date)
        
        if data is None or data.empty:
            raise ValueError(f"No data available for {symbol}")
            
        # Step 2: Generate signals
        print("Generating signals...")
        signals = strategy.generate_signals(data)
        
        # Step 3: Initialize executor and ledger
        executor = SimpleExecutor(self.initial_capital)
        ledger = TradeLedger()
        
        # Step 4: Simulate trading
        print("Simulating trades...")
        portfolio_values = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i]
            price = row['Close']
            
            # Process signal
            action = executor.process_signal(symbol, signal, price, date)
            
            # Record transaction if we did something
            if action['type'] in ['buy', 'sell']:
                ledger.record_transaction(
                    date=date,
                    symbol=symbol,
                    action=action['type'],
                    quantity=action['quantity'],
                    price=price
                )
                
            # Calculate portfolio value
            current_prices = {symbol: price}
            portfolio_value = executor.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            
        # Step 5: Calculate metrics
        print("Calculating metrics...")
        portfolio_series = pd.Series(portfolio_values, index=data.index)
        
        # Basic metrics
        final_value = portfolio_values[-1]
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Buy and hold comparison
        buy_hold_shares = int(self.initial_capital / data['Close'].iloc[0])
        buy_hold_value = buy_hold_shares * data['Close'].iloc[-1]
        buy_hold_return = ((buy_hold_value - self.initial_capital) / self.initial_capital) * 100
        
        # Risk metrics
        returns = portfolio_series.pct_change().dropna()
        sharpe_ratio = self._calculate_sharpe(returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_series)
        
        # Trade statistics
        trade_stats = ledger.calculate_statistics()
        
        # Create results
        results = BacktestResults(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            buy_and_hold_return=buy_hold_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_transactions=trade_stats.get('total_transactions', 0),
            completed_trades=trade_stats.get('total_completed_trades', 0),
            win_rate=trade_stats.get('win_rate', 0),
            portfolio_values=portfolio_series,
            transactions=ledger.get_transaction_history(),
            completed_trades_df=ledger.get_completed_trades(),
            trade_statistics=trade_stats
        )
        
        return results
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
            
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100  # Return as percentage
    
    def print_results(self, results: BacktestResults):
        """Print formatted results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Symbol: {results.symbol}")
        print(f"Period: {results.start_date.date()} to {results.end_date.date()}")
        print(f"Initial Capital: ${results.initial_capital:,.2f}")
        print(f"Final Value: ${results.final_value:,.2f}")
        print(f"Total Return: ${results.total_return:,.2f} ({results.total_return_pct:.2f}%)")
        print(f"Buy & Hold Return: {results.buy_and_hold_return:.2f}%")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"\nTrade Statistics:")
        print(f"Total Transactions: {results.total_transactions}")
        print(f"Completed Trades: {results.completed_trades}")
        print(f"Win Rate: {results.win_rate:.1f}%")
        
        if results.trade_statistics:
            stats = results.trade_statistics
            if 'avg_return_pct' in stats:
                print(f"Avg Return per Trade: {stats['avg_return_pct']:.2f}%")
            if 'avg_holding_days' in stats:
                print(f"Avg Holding Period: {stats['avg_holding_days']:.1f} days")