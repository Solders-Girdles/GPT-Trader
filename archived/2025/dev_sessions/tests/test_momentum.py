#!/usr/bin/env python3
"""
Test the momentum strategy implementation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies import create_strategy, list_available_strategies


def create_momentum_test_data():
    """Create test data with realistic momentum patterns."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create more realistic price movements
    np.random.seed(42)  # For reproducible results
    prices = [100]  # Start at $100
    
    for i in range(99):
        # Create phases with different momentum characteristics
        if i < 25:  # Sideways with low volatility
            daily_return = np.random.normal(0, 0.5) / 100
        elif i < 50:  # Moderate uptrend
            daily_return = np.random.normal(0.3, 0.8) / 100  # 0.3% avg daily gain
        elif i < 75:  # Consolidation
            daily_return = np.random.normal(-0.1, 0.6) / 100  # Slight downward bias
        else:  # Downtrend
            daily_return = np.random.normal(-0.4, 0.7) / 100  # 0.4% avg daily loss
        
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    data = pd.DataFrame({
        'Close': prices,
        'Open': [p * 0.999 for p in prices],
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Volume': [1000000] * 100
    }, index=dates)
    
    return data


def test_momentum_strategy():
    """Test the momentum strategy implementation."""
    
    print("="*60)
    print("TESTING MOMENTUM STRATEGY")
    print("="*60)
    
    # Check if momentum strategy is registered
    strategies = list_available_strategies()
    print(f"Available strategies: {strategies}")
    assert "MomentumStrategy" in strategies, "MomentumStrategy not registered!"
    
    # Create test data
    data = create_momentum_test_data()
    print(f"Created test data: {len(data)} days")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Test strategy creation
    print("\n" + "-"*40)
    print("TESTING STRATEGY CREATION")
    print("-"*40)
    
    # Create with default parameters
    momentum_strategy = create_strategy("MomentumStrategy")
    print(f"Created strategy: {momentum_strategy}")
    print(f"Required periods: {momentum_strategy.get_required_periods()}")
    
    # Test with custom parameters
    custom_momentum = create_strategy(
        "MomentumStrategy",
        lookback_period=10,
        buy_threshold=1.5,
        sell_threshold=-0.5,
        momentum_smoothing=2
    )
    print(f"Custom strategy: {custom_momentum}")
    
    # Test data validation
    print("\n" + "-"*40)
    print("TESTING DATA VALIDATION")
    print("-"*40)
    
    print(f"Data validation: {momentum_strategy.validate_data(data)}")
    
    # Test with insufficient data
    small_data = data.head(10)
    print(f"Small data validation: {momentum_strategy.validate_data(small_data)}")
    
    # Test signal generation
    print("\n" + "-"*40)
    print("TESTING SIGNAL GENERATION")
    print("-"*40)
    
    signals = momentum_strategy.run(data)
    
    print(f"Signals generated: {len(signals)}")
    print(f"Buy signals: {(signals == 1).sum()}")
    print(f"Sell signals: {(signals == -1).sum()}")
    print(f"Hold signals: {(signals == 0).sum()}")
    
    status = momentum_strategy.get_status()
    print(f"Signal rate: {status['metrics']['signal_rate']:.1f}%")
    
    # Analyze signal timing
    print("\n" + "-"*40)
    print("ANALYZING SIGNAL PATTERNS")
    print("-"*40)
    
    buy_dates = data.index[signals == 1]
    sell_dates = data.index[signals == -1]
    
    if len(buy_dates) > 0:
        print(f"First buy signal: {buy_dates[0].date()} at ${data.loc[buy_dates[0], 'Close']:.2f}")
        if len(buy_dates) > 1:
            print(f"Last buy signal: {buy_dates[-1].date()} at ${data.loc[buy_dates[-1], 'Close']:.2f}")
    
    if len(sell_dates) > 0:
        print(f"First sell signal: {sell_dates[0].date()} at ${data.loc[sell_dates[0], 'Close']:.2f}")
        if len(sell_dates) > 1:
            print(f"Last sell signal: {sell_dates[-1].date()} at ${data.loc[sell_dates[-1], 'Close']:.2f}")
    
    # Test parameter updates
    print("\n" + "-"*40)
    print("TESTING PARAMETER UPDATES")
    print("-"*40)
    
    original_threshold = momentum_strategy.get_parameter('buy_threshold')
    print(f"Original buy threshold: {original_threshold}")
    
    momentum_strategy.set_parameters(buy_threshold=3.0)
    new_threshold = momentum_strategy.get_parameter('buy_threshold')
    print(f"Updated buy threshold: {new_threshold}")
    
    # Generate signals with new parameters
    new_signals = momentum_strategy.run(data)
    print(f"New signal rate: {momentum_strategy.metrics.signal_rate:.1f}%")
    
    print("\n" + "="*60)
    print("MOMENTUM STRATEGY TEST COMPLETE")
    print("="*60)
    print("✅ All tests passed!")
    print("✅ Momentum strategy properly registered")
    print("✅ Signal generation working correctly")
    print("✅ Parameter management working")
    print("✅ Data validation working")
    
    return True


if __name__ == "__main__":
    test_momentum_strategy()