#!/usr/bin/env python3
"""
Strategy parameter optimization framework.

This script:
1. Tests different parameter combinations
2. Compares against buy-and-hold benchmark
3. Finds optimal parameters for each strategy
4. Validates across different time periods
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Any
import itertools
import json
import logging

from bot.strategy import get_strategy
from bot.strategy.mean_reversion import MeanReversionParams, MeanReversionStrategy
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy
# from bot.strategy.volatility import VolatilityParams, VolatilityStrategy  # May not exist
from bot.strategy.demo_ma import DemoMAStrategy
from bot.integration.orchestrator import IntegratedOrchestrator
from bot.integration.orchestrator import BacktestConfig

logging.basicConfig(level=logging.WARNING)  # Reduce noise during optimization
logger = logging.getLogger(__name__)


def calculate_buy_hold_return(symbol: str, start_date: str, end_date: str) -> float:
    """Calculate buy-and-hold return for benchmark."""
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        return 0.0
    
    start_price = float(data['Close'].iloc[0])
    end_price = float(data['Close'].iloc[-1])
    return ((end_price - start_price) / start_price) * 100


def run_backtest_with_params(
    strategy_name: str,
    params: dict,
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 10000
) -> Dict[str, float]:
    """Run backtest with specific parameters."""
    try:
        # Create strategy with parameters
        if strategy_name == "mean_reversion":
            strategy_params = MeanReversionParams(**params)
            strategy = MeanReversionStrategy(strategy_params)
        elif strategy_name == "trend_breakout":
            # Map parameters to expected names
            mapped_params = {
                'donchian_lookback': params.get('channel_period', 20),
                'atr_period': params.get('atr_period', 14),
                'atr_k': params.get('atr_multiplier', 2.0)
            }
            strategy_params = TrendBreakoutParams(**mapped_params)
            strategy = TrendBreakoutStrategy(strategy_params)
        elif strategy_name == "demo_ma":
            # DemoMAStrategy takes direct parameters
            strategy = DemoMAStrategy(
                fast=params.get('fast', 10),
                slow=params.get('slow', 30),
                atr_period=params.get('atr_period', 14)
            )
        else:
            return {'return': -100, 'sharpe': -1}
        
        # Configure backtest
        config = BacktestConfig(
            start_date=datetime.strptime(start_date, '%Y-%m-%d'),
            end_date=datetime.strptime(end_date, '%Y-%m-%d'),
            initial_capital=initial_capital
        )
        
        # Run backtest
        orchestrator = IntegratedOrchestrator(config)
        result = orchestrator.run_backtest(strategy, [symbol])
        
        # Extract metrics
        metrics = result.get('metrics', {})
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        num_trades = metrics.get('total_trades', 0)
        
        return {
            'return': total_return,
            'sharpe': sharpe_ratio,
            'max_dd': max_drawdown,
            'trades': num_trades
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {'return': -100, 'sharpe': -1, 'max_dd': 1.0, 'trades': 0}


def optimize_mean_reversion(symbol: str, start_date: str, end_date: str) -> Dict:
    """Optimize mean reversion strategy parameters."""
    print("\n" + "="*60)
    print("OPTIMIZING MEAN REVERSION STRATEGY")
    print("="*60)
    
    # Calculate benchmark
    buy_hold = calculate_buy_hold_return(symbol, start_date, end_date)
    print(f"Buy-and-Hold Benchmark: {buy_hold:.2f}%")
    
    # Define parameter grid
    param_grid = {
        'rsi_period': [7, 14, 21, 28],
        'oversold_threshold': [20, 25, 30, 35],
        'overbought_threshold': [65, 70, 75, 80],
        'exit_rsi_threshold': [45, 50, 55],
        'atr_period': [14, 20]
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    
    best_params = None
    best_return = -float('inf')
    best_sharpe = -float('inf')
    results = []
    
    # Test each combination
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Skip invalid combinations
        if params['oversold_threshold'] >= params['overbought_threshold']:
            continue
        if params['exit_rsi_threshold'] <= params['oversold_threshold']:
            continue
        if params['exit_rsi_threshold'] >= params['overbought_threshold']:
            continue
        
        # Run backtest
        result = run_backtest_with_params('mean_reversion', params, symbol, start_date, end_date)
        
        # Track results
        results.append({
            'params': params,
            'return': result['return'],
            'sharpe': result['sharpe'],
            'max_dd': result['max_dd'],
            'trades': result['trades'],
            'beats_benchmark': result['return'] > buy_hold
        })
        
        # Update best if this beats benchmark and has better Sharpe
        if result['return'] > buy_hold and result['sharpe'] > best_sharpe:
            best_params = params
            best_return = result['return']
            best_sharpe = result['sharpe']
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Tested {i+1}/{len(combinations)} combinations...")
    
    # Sort results by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    # Display top 5 results
    print("\nTop 5 Parameter Sets:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Return':<10} {'Sharpe':<10} {'Max DD':<10} {'Trades':<10} {'Beats BH'}")
    print("-" * 80)
    
    for i, r in enumerate(results[:5]):
        beats = "✅" if r['beats_benchmark'] else "❌"
        print(f"{i+1:<5} {r['return']:<10.2f}% {r['sharpe']:<10.2f} {r['max_dd']:<10.2%} {r['trades']:<10} {beats}")
        if i == 0:
            print(f"  Parameters: {r['params']}")
    
    if best_params:
        print(f"\n✅ Best Parameters (Return: {best_return:.2f}%, Sharpe: {best_sharpe:.2f}):")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"  Outperformance vs Buy-Hold: {best_return - buy_hold:.2f}%")
    else:
        print(f"\n❌ No parameters beat buy-and-hold ({buy_hold:.2f}%)")
        
    return best_params if best_params else results[0]['params'] if results else None


def optimize_trend_breakout(symbol: str, start_date: str, end_date: str) -> Dict:
    """Optimize trend breakout strategy parameters."""
    print("\n" + "="*60)
    print("OPTIMIZING TREND BREAKOUT STRATEGY")
    print("="*60)
    
    buy_hold = calculate_buy_hold_return(symbol, start_date, end_date)
    print(f"Buy-and-Hold Benchmark: {buy_hold:.2f}%")
    
    param_grid = {
        'channel_period': [10, 15, 20, 25, 30],
        'atr_period': [14, 20],
        'atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'min_channel_width_atr': [0.5, 1.0, 1.5]
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    
    best_params = None
    best_return = -float('inf')
    best_sharpe = -float('inf')
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        result = run_backtest_with_params('trend_breakout', params, symbol, start_date, end_date)
        
        results.append({
            'params': params,
            'return': result['return'],
            'sharpe': result['sharpe'],
            'max_dd': result['max_dd'],
            'trades': result['trades'],
            'beats_benchmark': result['return'] > buy_hold
        })
        
        if result['return'] > buy_hold and result['sharpe'] > best_sharpe:
            best_params = params
            best_return = result['return']
            best_sharpe = result['sharpe']
        
        if (i + 1) % 10 == 0:
            print(f"  Tested {i+1}/{len(combinations)} combinations...")
    
    results.sort(key=lambda x: x['return'], reverse=True)
    
    print("\nTop 5 Parameter Sets:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Return':<10} {'Sharpe':<10} {'Max DD':<10} {'Trades':<10} {'Beats BH'}")
    print("-" * 80)
    
    for i, r in enumerate(results[:5]):
        beats = "✅" if r['beats_benchmark'] else "❌"
        print(f"{i+1:<5} {r['return']:<10.2f}% {r['sharpe']:<10.2f} {r['max_dd']:<10.2%} {r['trades']:<10} {beats}")
        if i == 0:
            print(f"  Parameters: {r['params']}")
    
    if best_params:
        print(f"\n✅ Best Parameters (Return: {best_return:.2f}%, Sharpe: {best_sharpe:.2f}):")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"  Outperformance vs Buy-Hold: {best_return - buy_hold:.2f}%")
    else:
        print(f"\n❌ No parameters beat buy-and-hold ({buy_hold:.2f}%)")
        
    return best_params if best_params else results[0]['params'] if results else None


def optimize_demo_ma(symbol: str, start_date: str, end_date: str) -> Dict:
    """Optimize demo MA strategy parameters."""
    print("\n" + "="*60)
    print("OPTIMIZING DEMO MA STRATEGY")
    print("="*60)
    
    buy_hold = calculate_buy_hold_return(symbol, start_date, end_date)
    print(f"Buy-and-Hold Benchmark: {buy_hold:.2f}%")
    
    param_grid = {
        'fast': [5, 10, 15, 20],
        'slow': [20, 30, 40, 50, 60],
        'atr_period': [14, 20]
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    
    best_params = None
    best_return = -float('inf')
    best_sharpe = -float('inf')
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Skip invalid combinations
        if params['fast'] >= params['slow']:
            continue
        
        result = run_backtest_with_params('demo_ma', params, symbol, start_date, end_date)
        
        results.append({
            'params': params,
            'return': result['return'],
            'sharpe': result['sharpe'],
            'max_dd': result['max_dd'],
            'trades': result['trades'],
            'beats_benchmark': result['return'] > buy_hold
        })
        
        if result['return'] > buy_hold and result['sharpe'] > best_sharpe:
            best_params = params
            best_return = result['return']
            best_sharpe = result['sharpe']
        
        if (i + 1) % 10 == 0:
            print(f"  Tested {i+1}/{len(combinations)} combinations...")
    
    results.sort(key=lambda x: x['return'], reverse=True)
    
    print("\nTop 5 Parameter Sets:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Return':<10} {'Sharpe':<10} {'Max DD':<10} {'Trades':<10} {'Beats BH'}")
    print("-" * 80)
    
    for i, r in enumerate(results[:5]):
        beats = "✅" if r['beats_benchmark'] else "❌"
        print(f"{i+1:<5} {r['return']:<10.2f}% {r['sharpe']:<10.2f} {r['max_dd']:<10.2%} {r['trades']:<10} {beats}")
        if i == 0:
            print(f"  Parameters: {r['params']}")
    
    if best_params:
        print(f"\n✅ Best Parameters (Return: {best_return:.2f}%, Sharpe: {best_sharpe:.2f}):")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"  Outperformance vs Buy-Hold: {best_return - buy_hold:.2f}%")
    else:
        print(f"\n❌ No parameters beat buy-and-hold ({buy_hold:.2f}%)")
        
    return best_params if best_params else results[0]['params'] if results else None


def save_optimal_parameters(results: Dict[str, Dict]) -> None:
    """Save optimal parameters to file."""
    output_file = "optimal_strategy_parameters.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Optimal parameters saved to {output_file}")


def main():
    """Run parameter optimization for all strategies."""
    print("="*60)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Optimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test periods
    optimization_period = {
        'symbol': 'AAPL',
        'start': '2023-01-01',
        'end': '2023-12-31'
    }
    
    validation_period = {
        'symbol': 'AAPL',
        'start': '2024-01-01',
        'end': '2024-06-30'
    }
    
    print(f"\nOptimization Period: {optimization_period['start']} to {optimization_period['end']}")
    print(f"Validation Period: {validation_period['start']} to {validation_period['end']}")
    
    # Optimize each strategy
    optimal_params = {}
    
    # 1. Mean Reversion
    mean_rev_params = optimize_mean_reversion(
        optimization_period['symbol'],
        optimization_period['start'],
        optimization_period['end']
    )
    if mean_rev_params:
        optimal_params['mean_reversion'] = mean_rev_params
    
    # 2. Trend Breakout  
    trend_params = optimize_trend_breakout(
        optimization_period['symbol'],
        optimization_period['start'],
        optimization_period['end']
    )
    if trend_params:
        optimal_params['trend_breakout'] = trend_params
    
    # 3. Demo MA
    ma_params = optimize_demo_ma(
        optimization_period['symbol'],
        optimization_period['start'],
        optimization_period['end']
    )
    if ma_params:
        optimal_params['demo_ma'] = ma_params
    
    # Save results
    if optimal_params:
        save_optimal_parameters(optimal_params)
    
    # Validate on out-of-sample period
    print("\n" + "="*60)
    print("VALIDATION ON OUT-OF-SAMPLE PERIOD")
    print("="*60)
    
    buy_hold_validation = calculate_buy_hold_return(
        validation_period['symbol'],
        validation_period['start'],
        validation_period['end']
    )
    print(f"Validation Buy-Hold: {buy_hold_validation:.2f}%")
    
    for strategy_name, params in optimal_params.items():
        print(f"\nValidating {strategy_name}...")
        result = run_backtest_with_params(
            strategy_name,
            params,
            validation_period['symbol'],
            validation_period['start'],
            validation_period['end']
        )
        
        if result['return'] > buy_hold_validation:
            print(f"  ✅ Return: {result['return']:.2f}% (beats buy-hold by {result['return'] - buy_hold_validation:.2f}%)")
        else:
            print(f"  ❌ Return: {result['return']:.2f}% (underperforms by {buy_hold_validation - result['return']:.2f}%)")
        print(f"  Sharpe: {result['sharpe']:.2f}")
        print(f"  Max DD: {result['max_dd']:.2%}")
        print(f"  Trades: {result['trades']}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()