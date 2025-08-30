#!/usr/bin/env python3
"""
Debug momentum strategy signal generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies import create_strategy


def create_simple_test_data():
    """Create simple test data for debugging."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    # Simple upward trend
    prices = [100 + i * 0.5 for i in range(50)]
    
    data = pd.DataFrame({
        'Close': prices,
        'Open': [p * 0.999 for p in prices],
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Volume': [1000000] * 50
    }, index=dates)
    
    return data


def debug_momentum_strategy():
    """Debug the momentum strategy."""
    
    print("="*60)
    print("DEBUGGING MOMENTUM STRATEGY")
    print("="*60)
    
    # Create simple test data
    data = create_simple_test_data()
    print(f"Test data: {len(data)} days, prices from {data['Close'].iloc[0]:.2f} to {data['Close'].iloc[-1]:.2f}")
    
    # Create strategy
    momentum_strategy = create_strategy("MomentumStrategy", lookback_period=5, momentum_smoothing=2)
    print(f"Strategy: {momentum_strategy}")
    print(f"Required periods: {momentum_strategy.get_required_periods()}")
    
    # Generate signals manually to see what's happening
    print("\n" + "-"*40)
    print("MANUAL SIGNAL GENERATION")
    print("-"*40)
    
    signals = momentum_strategy.generate_signals(data)
    print(f"Raw signals generated: {len(signals)}")
    print(f"Signal values: {sorted(signals.unique())}")
    print(f"Signal counts: {signals.value_counts().sort_index()}")
    
    # Check signal validation
    print("\n" + "-"*40)
    print("SIGNAL VALIDATION ANALYSIS")
    print("-"*40)
    
    # Check for valid values
    valid_values = {-1, 0, 1}
    invalid_signals = signals[~signals.isin(valid_values)]
    print(f"Invalid signal values: {invalid_signals.tolist()}")
    
    # Check signal rate
    signal_rate = (signals != 0).mean()
    print(f"Signal rate: {signal_rate:.2%}")
    
    if signal_rate > 0.5:
        print("⚠️  Signal rate too high (>50%)")
        non_zero_dates = data.index[signals != 0]
        print(f"Non-zero signals on: {non_zero_dates[:10].tolist()}")
    
    # Show the actual ROC calculation
    print("\n" + "-"*40)
    print("ROC CALCULATION ANALYSIS")
    print("-"*40)
    
    lookback = momentum_strategy.get_parameter('lookback_period')
    smoothing = momentum_strategy.get_parameter('momentum_smoothing')
    buy_threshold = momentum_strategy.get_parameter('buy_threshold')
    sell_threshold = momentum_strategy.get_parameter('sell_threshold')
    
    print(f"Lookback: {lookback}, Smoothing: {smoothing}")
    print(f"Buy threshold: {buy_threshold}%, Sell threshold: {sell_threshold}%")
    
    # Calculate ROC
    roc = data['Close'].pct_change(periods=lookback) * 100
    roc_smooth = roc.rolling(window=smoothing).mean()
    roc_change = roc_smooth.diff()
    
    print(f"\nROC stats:")
    print(f"  Min: {roc_smooth.min():.2f}%")
    print(f"  Max: {roc_smooth.max():.2f}%")
    print(f"  Mean: {roc_smooth.mean():.2f}%")
    
    print(f"\nROC change stats:")
    print(f"  Min: {roc_change.min():.2f}%")
    print(f"  Max: {roc_change.max():.2f}%")
    print(f"  Mean: {roc_change.mean():.2f}%")
    
    # Show conditions that trigger signals
    print("\n" + "-"*40)
    print("SIGNAL TRIGGER ANALYSIS")
    print("-"*40)
    
    buy_condition = (roc_smooth > buy_threshold) & (roc_change > 0)
    sell_condition1 = roc_smooth < sell_threshold
    sell_condition2 = (roc_smooth > 0) & (roc_change < -0.5)
    
    print(f"Buy conditions met: {buy_condition.sum()} times")
    print(f"Sell condition 1 (ROC < {sell_threshold}%): {sell_condition1.sum()} times")
    print(f"Sell condition 2 (deceleration): {sell_condition2.sum()} times")
    
    return True


if __name__ == "__main__":
    debug_momentum_strategy()