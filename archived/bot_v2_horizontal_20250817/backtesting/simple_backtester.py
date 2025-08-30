"""
Simple backtesting engine for strategy evaluation.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from core.interfaces import IBacktester, IStrategy, IDataProvider, ComponentConfig
from core.types import (
    Signal, SignalType, Position, Trade, Portfolio, 
    PerformanceMetrics, PositionStatus
)
from core.events import Event, EventType, get_event_bus


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    metrics: PerformanceMetrics = None
    signals: pd.Series = field(default_factory=pd.Series)
    
    def summary(self) -> str:
        """Get a summary of results."""
        if self.metrics:
            return (
                f"Total Return: {self.metrics.total_return:.2f}%\n"
                f"Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}\n"
                f"Max Drawdown: {self.metrics.max_drawdown:.2f}%\n"
                f"Win Rate: {self.metrics.win_rate:.2f}%\n"
                f"Total Trades: {self.metrics.total_trades}"
            )
        return "No results available"


class SimpleBacktester(IBacktester):
    """
    Simple backtesting engine that runs strategies through historical data.
    
    Features:
    - Runs strategy through historical data
    - Tracks positions and trades
    - Calculates performance metrics
    - Supports commission and slippage
    """
    
    def __init__(self, config: ComponentConfig):
        """Initialize the backtester."""
        super().__init__(config)
        self.commission = config.config.get('commission', 0.001)  # 0.1% default
        self.slippage = config.config.get('slippage', 0.0005)  # 0.05% default
        self.position_size = config.config.get('position_size', 0.95)  # Use 95% of capital
        self._event_bus = get_event_bus()
        
    def initialize(self) -> None:
        """Initialize the backtester."""
        pass
    
    def shutdown(self) -> None:
        """Cleanup and shutdown."""
        pass
    
    def run(
        self,
        strategy: IStrategy,
        data_provider: IDataProvider,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """
        Run a backtest.
        
        Args:
            strategy: Strategy to test
            data_provider: Data source
            start_date: Start of backtest period
            end_date: End of backtest period
            initial_capital: Starting capital
            
        Returns:
            Dict with backtest results
        """
        # Fetch historical data
        # For simplicity, assume we're testing on a single symbol
        # In production, this would handle multiple symbols
        symbol = self.config.config.get('symbol', 'AAPL')
        
        data = data_provider.get_historical_data(
            symbol=symbol,
            start=start_date,
            end=end_date,
            interval='1d'
        )
        
        if data.empty:
            raise ValueError("No data available for backtesting")
        
        # Run strategy to get signals
        signals = strategy.analyze(data)
        
        # Simulate trading
        result = self._simulate_trading(
            data=data,
            signals=signals,
            initial_capital=initial_capital
        )
        
        # Calculate metrics
        result.metrics = self._calculate_metrics(result)
        
        # Return as dictionary
        return {
            'result': result,
            'trades': result.trades,
            'metrics': result.metrics,
            'equity_curve': result.equity_curve,
            'returns': result.returns,
            'signals': result.signals
        }
    
    def _simulate_trading(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        initial_capital: float
    ) -> BacktestResult:
        """
        Simulate trading based on signals.
        
        Simple logic:
        - Buy signal (1): Buy if not in position
        - Sell signal (-1): Sell if in position
        - Hold signal (0): Do nothing
        """
        result = BacktestResult()
        result.signals = signals
        
        # Initialize portfolio
        cash = initial_capital
        position_qty = 0
        position_entry = None
        equity_curve = []
        daily_returns = []
        
        # Track trades
        trade_id = 0
        position_id = 0
        
        # Process each day
        for i, (date, signal) in enumerate(signals.items()):
            current_price = data.loc[date, 'Close']
            
            # Calculate current equity
            current_equity = cash + (position_qty * current_price)
            equity_curve.append(current_equity)
            
            # Calculate daily return
            if i > 0:
                prev_equity = equity_curve[i-1]
                daily_return = (current_equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0)
            
            # Process signal
            if signal == 1 and position_qty == 0:
                # Buy signal - open position
                buy_price = current_price * (1 + self.slippage)
                max_qty = int((cash * self.position_size) / buy_price)
                
                if max_qty > 0:
                    position_qty = max_qty
                    cost = max_qty * buy_price * (1 + self.commission)
                    cash -= cost
                    
                    # Record trade
                    trade = Trade(
                        trade_id=f"T{trade_id}",
                        order_id=f"O{trade_id}",
                        symbol=data.index.name or "UNKNOWN",
                        quantity=position_qty,
                        price=buy_price,
                        side='buy',
                        timestamp=date,
                        commission=cost * self.commission
                    )
                    result.trades.append(trade)
                    trade_id += 1
                    
                    # Record position
                    position_entry = Position(
                        position_id=f"P{position_id}",
                        symbol=data.index.name or "UNKNOWN",
                        quantity=position_qty,
                        entry_price=buy_price,
                        entry_time=date,
                        current_price=current_price,
                        status=PositionStatus.OPEN
                    )
                    result.positions.append(position_entry)
                    position_id += 1
                    
            elif signal == -1 and position_qty > 0:
                # Sell signal - close position
                sell_price = current_price * (1 - self.slippage)
                proceeds = position_qty * sell_price * (1 - self.commission)
                cash += proceeds
                
                # Record trade
                trade = Trade(
                    trade_id=f"T{trade_id}",
                    order_id=f"O{trade_id}",
                    symbol=data.index.name or "UNKNOWN",
                    quantity=position_qty,
                    price=sell_price,
                    side='sell',
                    timestamp=date,
                    commission=proceeds * self.commission
                )
                result.trades.append(trade)
                trade_id += 1
                
                # Update position
                if position_entry:
                    position_entry.exit_price = sell_price
                    position_entry.exit_time = date
                    position_entry.status = PositionStatus.CLOSED
                
                position_qty = 0
                position_entry = None
        
        # Close any open position at end
        if position_qty > 0:
            final_price = data.iloc[-1]['Close']
            proceeds = position_qty * final_price * (1 - self.commission)
            cash += proceeds
            
            if position_entry:
                position_entry.exit_price = final_price
                position_entry.exit_time = data.index[-1]
                position_entry.status = PositionStatus.CLOSED
        
        # Create series
        result.equity_curve = pd.Series(equity_curve, index=data.index)
        result.returns = pd.Series(daily_returns, index=data.index)
        
        return result
    
    def _calculate_metrics(self, result: BacktestResult) -> PerformanceMetrics:
        """Calculate performance metrics from backtest results."""
        
        # Basic return metrics
        total_return = 0
        if len(result.equity_curve) > 1:
            initial = result.equity_curve.iloc[0]
            final = result.equity_curve.iloc[-1]
            total_return = ((final - initial) / initial) * 100
        
        # Sharpe ratio (simplified - assuming 0 risk-free rate)
        sharpe_ratio = 0
        if len(result.returns) > 1:
            returns_std = result.returns.std()
            if returns_std > 0:
                sharpe_ratio = (result.returns.mean() / returns_std) * np.sqrt(252)
        
        # Maximum drawdown
        max_drawdown = 0
        if len(result.equity_curve) > 1:
            rolling_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
            max_drawdown = abs(drawdown.min())
        
        # Trade statistics
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for position in result.positions:
            if position.status == PositionStatus.CLOSED:
                pnl = position.realized_pnl
                if pnl > 0:
                    winning_trades += 1
                    total_profit += pnl
                else:
                    losing_trades += 1
                    total_loss += abs(pnl)
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Annualized return (simplified)
        days = len(result.equity_curve)
        years = days / 252
        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=0,  # Not implemented yet
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_hold_time=0,  # Not implemented yet
            best_trade=max([p.realized_pnl for p in result.positions if p.status == PositionStatus.CLOSED], default=0),
            worst_trade=min([p.realized_pnl for p in result.positions if p.status == PositionStatus.CLOSED], default=0),
            recovery_factor=0,  # Not implemented yet
            calmar_ratio=0  # Not implemented yet
        )
    
    def optimize(
        self,
        strategy: IStrategy,
        parameter_ranges: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.
        
        Note: Not implemented in simple version.
        """
        raise NotImplementedError("Parameter optimization not yet implemented")
    
    def walk_forward_analysis(
        self,
        strategy: IStrategy,
        data_provider: IDataProvider,
        window_size: int,
        step_size: int
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis.
        
        Note: Not implemented in simple version.
        """
        raise NotImplementedError("Walk-forward analysis not yet implemented")