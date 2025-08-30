#!/usr/bin/env python3
"""
Quick optimization test for mean reversion strategy.
Smaller parameter grid for faster results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import logging

from bot.strategy.mean_reversion import MeanReversionParams, MeanReversionStrategy
from bot.integration.orchestrator import IntegratedOrchestrator, BacktestConfig

logging.basicConfig(level=logging.WARNING)


def quick_test():
    """Quick test with limited parameters."""
    print("Quick Strategy Optimization Test")
    print("=" * 60)
    
    # Download data
    symbol = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-06-30"
    
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        print("Failed to download data")
        return
    
    # Calculate buy-and-hold
    start_price = float(data['Close'].iloc[0])
    end_price = float(data['Close'].iloc[-1])
    buy_hold = ((end_price - start_price) / start_price) * 100
    print(f"Buy-and-Hold Return: {buy_hold:.2f}%")
    print(f"Period: {start_date} to {end_date}")
    
    # Test a few parameter combinations
    test_params = [
        # Current default
        {'rsi_period': 10, 'oversold_threshold': 40, 'overbought_threshold': 60},
        # More selective
        {'rsi_period': 14, 'oversold_threshold': 30, 'overbought_threshold': 70},
        # Very selective
        {'rsi_period': 21, 'oversold_threshold': 25, 'overbought_threshold': 75},
        # Fast signals
        {'rsi_period': 7, 'oversold_threshold': 35, 'overbought_threshold': 65},
    ]
    
    print("\nTesting Parameter Sets:")
    print("-" * 60)
    
    best_return = -float('inf')
    best_params = None
    
    for i, params in enumerate(test_params, 1):
        print(f"\nTest {i}: RSI={params['rsi_period']}, "
              f"Oversold={params['oversold_threshold']}, "
              f"Overbought={params['overbought_threshold']}")
        
        try:
            # Create strategy
            strategy_params = MeanReversionParams(
                rsi_period=params['rsi_period'],
                oversold_threshold=params['oversold_threshold'],
                overbought_threshold=params['overbought_threshold'],
                exit_rsi_threshold=50.0,
                atr_period=14
            )
            strategy = MeanReversionStrategy(strategy_params)
            
            # Run backtest
            config = BacktestConfig(
                start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                initial_capital=10000,
                quiet_mode=True
            )
            
            orchestrator = IntegratedOrchestrator(config)
            result = orchestrator.run_backtest(strategy, [symbol])
            
            # Get metrics
            metrics = result.get('metrics', {})
            total_return = metrics.get('total_return', 0)
            num_trades = metrics.get('total_trades', 0)
            sharpe = metrics.get('sharpe_ratio', 0)
            
            print(f"  Return: {total_return:.2f}%")
            print(f"  Trades: {num_trades}")
            print(f"  Sharpe: {sharpe:.2f}")
            
            if total_return > buy_hold:
                print(f"  ✅ Beats buy-hold by {total_return - buy_hold:.2f}%")
            else:
                print(f"  ❌ Underperforms by {buy_hold - total_return:.2f}%")
            
            if total_return > best_return:
                best_return = total_return
                best_params = params
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    if best_params and best_return > buy_hold:
        print(f"✅ FOUND PROFITABLE PARAMETERS!")
        print(f"   Return: {best_return:.2f}% (vs {buy_hold:.2f}% buy-hold)")
        print(f"   Params: {best_params}")
    else:
        print(f"❌ No parameters beat buy-hold ({buy_hold:.2f}%)")
        if best_params:
            print(f"   Best attempt: {best_return:.2f}%")


if __name__ == "__main__":
    quick_test()