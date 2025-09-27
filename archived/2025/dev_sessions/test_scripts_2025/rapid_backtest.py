#!/usr/bin/env python3
"""
Rapid Backtesting System
Quickly test strategies on historical data without waiting for real-time.
"""

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class RapidBacktester:
    """Fast backtesting engine for strategy validation."""
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
    def fetch_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical data with caching."""
        cache_file = self.cache_dir / f"{symbol}_{days}d.pkl"
        
        # Check cache (valid for 1 hour)
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < 3600:  # 1 hour
                return pd.read_pickle(cache_file)
        
        # Fetch from Yahoo Finance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval='15m')
        
        # Cache the data
        data.to_pickle(cache_file)
        
        return data
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Simple Moving Averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_30'] = data['Close'].rolling(window=30).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        return data
    
    def run_strategy_backtest(self, strategy_name: str, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Run backtest for a specific strategy."""
        
        # Strategy implementations
        if strategy_name == 'momentum':
            signals = self.momentum_signals(data)
        elif strategy_name == 'mean_reversion':
            signals = self.mean_reversion_signals(data)
        elif strategy_name == 'breakout':
            signals = self.breakout_signals(data)
        elif strategy_name == 'ma_crossover':
            signals = self.ma_crossover_signals(data)
        elif strategy_name == 'volatility':
            signals = self.volatility_signals(data)
        else:
            signals = pd.Series(0, index=data.index)
        
        # Simulate trading
        return self.simulate_trading(data, signals, initial_capital)
    
    def momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum trading signals."""
        signals = pd.Series(0, index=data.index)
        
        # Buy when price momentum is strong
        momentum = data['Close'].pct_change(periods=10)
        signals[momentum > 0.05] = 1  # Buy signal
        signals[momentum < -0.05] = -1  # Sell signal
        
        return signals
    
    def mean_reversion_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals using Bollinger Bands."""
        signals = pd.Series(0, index=data.index)
        
        # Buy at lower band, sell at upper band
        signals[data['Close'] < data['BB_Lower']] = 1
        signals[data['Close'] > data['BB_Upper']] = -1
        
        return signals
    
    def breakout_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate breakout signals."""
        signals = pd.Series(0, index=data.index)
        
        # Rolling high/low
        rolling_high = data['High'].rolling(window=20).max()
        rolling_low = data['Low'].rolling(window=20).min()
        
        # Buy on breakout above high
        signals[data['Close'] > rolling_high.shift(1)] = 1
        # Sell on breakdown below low
        signals[data['Close'] < rolling_low.shift(1)] = -1
        
        return signals
    
    def ma_crossover_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MA crossover signals."""
        signals = pd.Series(0, index=data.index)
        
        # Golden cross / Death cross
        signals[(data['SMA_10'] > data['SMA_30']) & 
                (data['SMA_10'].shift(1) <= data['SMA_30'].shift(1))] = 1
        signals[(data['SMA_10'] < data['SMA_30']) & 
                (data['SMA_10'].shift(1) >= data['SMA_30'].shift(1))] = -1
        
        return signals
    
    def volatility_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate volatility-based signals."""
        signals = pd.Series(0, index=data.index)
        
        # Trade when volatility is low
        vol_threshold = data['ATR'].quantile(0.3)
        low_vol = data['ATR'] < vol_threshold
        
        # Trend following in low volatility
        signals[low_vol & (data['Close'] > data['SMA_10'])] = 1
        signals[low_vol & (data['Close'] < data['SMA_10'])] = -1
        
        return signals
    
    def simulate_trading(self, data: pd.DataFrame, signals: pd.Series, initial_capital: float) -> Dict:
        """Simulate trading based on signals."""
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            price = data['Close'].iloc[i]
            signal = signals.iloc[i]
            
            # Execute trades
            if signal == 1 and position == 0:
                # Buy
                position = capital * 0.95 / price  # Use 95% of capital
                capital = capital * 0.05  # Keep 5% cash
                trades.append({
                    'time': data.index[i],
                    'side': 'buy',
                    'price': price,
                    'quantity': position
                })
                
            elif signal == -1 and position > 0:
                # Sell
                capital += position * price * 0.994  # 0.6% fee
                trades.append({
                    'time': data.index[i],
                    'side': 'sell',
                    'price': price,
                    'quantity': position,
                    'pnl': (position * price * 0.994) - (position * trades[-1]['price'])
                })
                position = 0
            
            # Calculate equity
            equity = capital + (position * price if position > 0 else 0)
            equity_curve.append(equity)
        
        # Close final position
        if position > 0:
            final_price = data['Close'].iloc[-1]
            capital += position * final_price * 0.994
            position = 0
        
        # Calculate metrics
        final_equity = capital
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Calculate max drawdown
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len([t for t in trades if t['side'] == 'buy']),
            'win_rate': win_rate,
            'final_equity': final_equity
        }


def backtest_strategy_worker(args: Tuple[str, str, int]) -> Dict:
    """Worker function for parallel backtesting."""
    strategy, symbol, days = args
    
    backtester = RapidBacktester()
    
    # Fetch data
    data = backtester.fetch_historical_data(symbol, days)
    
    # Calculate indicators
    data = backtester.calculate_indicators(data)
    
    # Run backtest
    result = backtester.run_strategy_backtest(strategy, data)
    
    return {
        'strategy': strategy,
        'symbol': symbol,
        'days': days,
        **result
    }


def run_parallel_backtests(strategies: List[str], symbols: List[str], days: int = 30):
    """Run backtests in parallel across strategies and symbols."""
    
    print("=" * 70)
    print("RAPID PARALLEL BACKTESTING")
    print("=" * 70)
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {days} days")
    print("=" * 70)
    
    # Create work items
    work_items = [
        (strategy, symbol, days)
        for strategy in strategies
        for symbol in symbols
    ]
    
    print(f"\n📊 Running {len(work_items)} backtests in parallel...")
    start_time = time.time()
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(backtest_strategy_worker, work_items))
    
    elapsed = time.time() - start_time
    
    # Aggregate results by strategy
    strategy_results = {}
    for result in results:
        strategy = result['strategy']
        if strategy not in strategy_results:
            strategy_results[strategy] = {
                'returns': [],
                'sharpe_ratios': [],
                'drawdowns': [],
                'trades': [],
                'win_rates': []
            }
        
        strategy_results[strategy]['returns'].append(result['total_return'])
        strategy_results[strategy]['sharpe_ratios'].append(result['sharpe_ratio'])
        strategy_results[strategy]['drawdowns'].append(result['max_drawdown'])
        strategy_results[strategy]['trades'].append(result['num_trades'])
        strategy_results[strategy]['win_rates'].append(result['win_rate'])
    
    # Display results
    print(f"\n✅ Completed {len(results)} backtests in {elapsed:.1f} seconds")
    print(f"   ({len(results)/elapsed:.1f} backtests per second)")
    
    print("\n" + "=" * 70)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Strategy':<15} {'Avg Return':<12} {'Avg Sharpe':<12} {'Avg DD':<12} {'Avg Trades':<12} {'Win Rate':<10}")
    print("-" * 80)
    
    sorted_strategies = sorted(
        strategy_results.items(),
        key=lambda x: np.mean(x[1]['returns']),
        reverse=True
    )
    
    for strategy, metrics in sorted_strategies:
        avg_return = np.mean(metrics['returns'])
        avg_sharpe = np.mean(metrics['sharpe_ratios'])
        avg_dd = np.mean(metrics['drawdowns'])
        avg_trades = np.mean(metrics['trades'])
        avg_wr = np.mean(metrics['win_rates'])
        
        print(f"{strategy:<15} {avg_return:>+11.2f}% {avg_sharpe:>11.2f} "
              f"{avg_dd:>11.2f}% {avg_trades:>11.0f} {avg_wr:>9.1f}%")
    
    # Best performing combinations
    print("\n" + "=" * 70)
    print("TOP 5 STRATEGY-SYMBOL COMBINATIONS")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda x: x['total_return'], reverse=True)[:5]
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['strategy']}-{result['symbol']}: "
              f"{result['total_return']:+.2f}% return, "
              f"Sharpe: {result['sharpe_ratio']:.2f}")
    
    # Save results
    results_file = Path(__file__).parent.parent / 'results' / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                strategy: {
                    'avg_return': float(np.mean(metrics['returns'])),
                    'avg_sharpe': float(np.mean(metrics['sharpe_ratios'])),
                    'avg_drawdown': float(np.mean(metrics['drawdowns'])),
                    'avg_trades': float(np.mean(metrics['trades'])),
                    'avg_win_rate': float(np.mean(metrics['win_rates']))
                }
                for strategy, metrics in strategy_results.items()
            },
            'detailed_results': results,
            'metadata': {
                'strategies': strategies,
                'symbols': symbols,
                'days': days,
                'elapsed_seconds': elapsed,
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file.name}")
    
    return strategy_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rapid Backtesting System")
    parser.add_argument('--strategies', type=str,
                       default='momentum,mean_reversion,breakout,ma_crossover,volatility',
                       help='Comma-separated strategies')
    parser.add_argument('--symbols', type=str,
                       default='BTC-USD,ETH-USD,SOL-USD,LINK-USD,MATIC-USD',
                       help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to backtest')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer combinations')
    
    args = parser.parse_args()
    
    strategies = args.strategies.split(',')
    symbols = args.symbols.split(',')
    
    if args.quick:
        # Quick test with fewer combinations
        strategies = strategies[:3]
        symbols = symbols[:2]
        days = 7
    else:
        days = args.days
    
    # Run parallel backtests
    run_parallel_backtests(strategies, symbols, days)


if __name__ == "__main__":
    main()