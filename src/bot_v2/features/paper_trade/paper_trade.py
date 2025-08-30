"""
Main paper trading orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import time
import threading

from .types import PaperTradeResult, PerformanceMetrics
from .strategies import create_paper_strategy
from .data import DataFeed
from .execution import PaperExecutor
from .risk import RiskManager


class PaperTradingSession:
    """Manages a paper trading session."""
    
    def __init__(
        self,
        strategy: str,
        symbols: List[str],
        initial_capital: float = 100000,
        **kwargs
    ):
        """
        Initialize paper trading session.
        
        Args:
            strategy: Strategy name
            symbols: List of symbols to trade
            initial_capital: Starting capital
            **kwargs: Strategy parameters and settings
        """
        self.strategy_name = strategy
        self.symbols = symbols
        self.initial_capital = initial_capital
        
        # Extract settings
        self.commission = kwargs.pop('commission', 0.001)
        self.slippage = kwargs.pop('slippage', 0.0005)
        self.max_positions = kwargs.pop('max_positions', 10)
        self.position_size = kwargs.pop('position_size', 0.95)
        self.update_interval = kwargs.pop('update_interval', 60)
        
        # Initialize components
        self.strategy = create_paper_strategy(strategy, **kwargs)
        self.data_feed = DataFeed(symbols)
        self.executor = PaperExecutor(
            initial_capital=initial_capital,
            commission=self.commission,
            slippage=self.slippage,
            max_positions=self.max_positions
        )
        self.risk_manager = RiskManager()
        
        # Session state
        self.start_time = None
        self.end_time = None
        self.is_running = False
        self.thread = None
        self.equity_history = []
        
    def start(self):
        """Start paper trading session."""
        if self.is_running:
            return
        
        self.start_time = datetime.now()
        self.is_running = True
        
        # Start trading loop in background thread
        self.thread = threading.Thread(target=self._trading_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print(f"Paper trading started at {self.start_time}")
        print(f"Strategy: {self.strategy_name}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
    
    def stop(self) -> PaperTradeResult:
        """
        Stop paper trading session.
        
        Returns:
            Final results
        """
        if not self.is_running:
            return self.get_results()
        
        self.is_running = False
        self.end_time = datetime.now()
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5)
        
        # Close all positions
        prices = {s: self.data_feed.get_latest_price(s) for s in self.symbols}
        prices = {s: p for s, p in prices.items() if p is not None}
        self.executor.close_all_positions(prices, self.end_time)
        
        print(f"Paper trading stopped at {self.end_time}")
        
        return self.get_results()
    
    def _trading_loop(self):
        """Main trading loop (runs in background thread)."""
        while self.is_running:
            try:
                # Update data
                self.data_feed.update()
                
                # Process each symbol
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                
                # Update positions
                prices = {s: self.data_feed.get_latest_price(s) for s in self.symbols}
                prices = {s: p for s, p in prices.items() if p is not None}
                self.executor.update_positions(prices)
                
                # Record equity
                status = self.executor.get_account_status()
                self.equity_history.append({
                    'timestamp': datetime.now(),
                    'equity': status.total_equity
                })
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(self.update_interval)
    
    def _process_symbol(self, symbol: str):
        """Process trading logic for a symbol."""
        # Get historical data
        data = self.data_feed.get_historical(
            symbol, 
            self.strategy.get_required_periods()
        )
        
        if data.empty or len(data) < self.strategy.get_required_periods():
            return
        
        # Generate signal
        signal = self.strategy.analyze(data)
        
        # Check risk limits
        if signal != 0:
            current_price = self.data_feed.get_latest_price(symbol)
            if current_price:
                # Apply risk checks
                account = self.executor.get_account_status()
                if not self.risk_manager.check_trade(
                    symbol, signal, current_price, account
                ):
                    return
                
                # Execute signal
                self.executor.execute_signal(
                    symbol=symbol,
                    signal=signal,
                    current_price=current_price,
                    timestamp=datetime.now(),
                    position_size=self.position_size
                )
    
    def get_results(self) -> PaperTradeResult:
        """Get current results."""
        account = self.executor.get_account_status()
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Build equity curve
        if self.equity_history:
            equity_df = pd.DataFrame(self.equity_history)
            equity_curve = pd.Series(
                equity_df['equity'].values,
                index=equity_df['timestamp']
            )
        else:
            equity_curve = pd.Series([self.initial_capital])
        
        return PaperTradeResult(
            start_time=self.start_time or datetime.now(),
            end_time=self.end_time,
            account_status=account,
            positions=list(self.executor.positions.values()),
            trade_log=self.executor.trade_log,
            performance=metrics,
            equity_curve=equity_curve
        )
    
    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics."""
        account = self.executor.get_account_status()
        trades = self.executor.trade_log
        
        # Total return
        total_return = (account.total_equity - self.initial_capital) / self.initial_capital
        
        # Daily return (simplified)
        if self.equity_history and len(self.equity_history) > 1:
            daily_returns = []
            for i in range(1, len(self.equity_history)):
                prev = self.equity_history[i-1]['equity']
                curr = self.equity_history[i]['equity']
                daily_returns.append((curr - prev) / prev)
            
            if daily_returns:
                avg_daily_return = sum(daily_returns) / len(daily_returns)
                sharpe_ratio = (avg_daily_return * 252) / (pd.Series(daily_returns).std() * (252 ** 0.5)) if pd.Series(daily_returns).std() > 0 else 0
            else:
                avg_daily_return = 0
                sharpe_ratio = 0
        else:
            avg_daily_return = 0
            sharpe_ratio = 0
        
        # Max drawdown
        if self.equity_history:
            equity_values = [h['equity'] for h in self.equity_history]
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0
        
        # Win rate
        if trades:
            buy_trades = {t.id: t for t in trades if t.side == 'buy'}
            sell_trades = [t for t in trades if t.side == 'sell']
            
            wins = 0
            losses = 0
            
            for sell in sell_trades:
                # Find corresponding buy
                buy_id = sell.id - 1  # Simplified pairing
                if buy_id in buy_trades:
                    buy = buy_trades[buy_id]
                    pnl = (sell.price - buy.price) * sell.quantity
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
            
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            profit_factor = wins / losses if losses > 0 else float('inf') if wins > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            daily_return=avg_daily_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_count=len(trades)
        )


# Global session management
_active_session: Optional[PaperTradingSession] = None


def start_paper_trading(
    strategy: str,
    symbols: List[str],
    initial_capital: float = 100000,
    **kwargs
) -> None:
    """
    Start a paper trading session.
    
    Args:
        strategy: Strategy name
        symbols: List of symbols to trade
        initial_capital: Starting capital
        **kwargs: Additional parameters
    """
    global _active_session
    
    if _active_session and _active_session.is_running:
        raise RuntimeError("A paper trading session is already running")
    
    _active_session = PaperTradingSession(
        strategy=strategy,
        symbols=symbols,
        initial_capital=initial_capital,
        **kwargs
    )
    
    _active_session.start()


def stop_paper_trading() -> PaperTradeResult:
    """
    Stop the active paper trading session.
    
    Returns:
        Final results
    """
    global _active_session
    
    if not _active_session:
        raise RuntimeError("No active paper trading session")
    
    results = _active_session.stop()
    _active_session = None
    
    return results


def get_status() -> Optional[PaperTradeResult]:
    """
    Get current status of paper trading session.
    
    Returns:
        Current results or None if no session
    """
    global _active_session
    
    if not _active_session:
        return None
    
    return _active_session.get_results()