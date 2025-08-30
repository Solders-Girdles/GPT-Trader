#!/usr/bin/env python3
"""
Debug breakout strategy to understand why no signals are generated.
"""

import pandas as pd
import numpy as np
from strategies import create_strategy


def create_simple_breakout_data():
    """Create simple data with clear breakout pattern."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    
    prices = []
    volumes = []
    
    # Clear consolidation and breakout pattern
    for i in range(50):
        if i < 20:  # Consolidation between 98-102
            price = 100 + np.random.uniform(-2, 2)
            volume = 1000000
        elif i < 25:  # Breakout to 105-110
            price = 105 + (i - 20)
            volume = 2000000  # Double volume
        else:  # Continue higher
            price = 110 + (i - 25) * 0.2
            volume = 1200000
        
        prices.append(price)
        volumes.append(volume)
    
    data = pd.DataFrame({
        'Close': prices,
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Volume': volumes
    }, index=dates)
    
    return data


def debug_breakout():
    """Debug the breakout strategy."""
    
    print("="*60)
    print("DEBUGGING BREAKOUT STRATEGY")
    print("="*60)
    
    # Create simple test data
    data = create_simple_breakout_data()
    print(f"Test data: {len(data)} days")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Create strategy with more lenient parameters
    strategy = create_strategy(
        "BreakoutStrategy",
        lookback_period=10,
        breakout_threshold=0.1,  # Very low threshold
        volume_multiplier=1.2,    # Lower volume requirement
        retracement_threshold=5.0  # Higher retracement allowed
    )
    
    print(f"\nStrategy parameters:")
    print(f"  Lookback: {strategy.get_parameter('lookback_period')}")
    print(f"  Breakout threshold: {strategy.get_parameter('breakout_threshold')}%")
    print(f"  Volume multiplier: {strategy.get_parameter('volume_multiplier')}")
    
    # Calculate levels manually
    print(f"\n{'-'*40}")
    print("MANUAL LEVEL CALCULATION")
    print(f"{'-'*40}")
    
    lookback = 10
    resistance = data['High'].rolling(window=lookback).max().shift(1)
    support = data['Low'].rolling(window=lookback).min().shift(1)
    volume_ma = data['Volume'].rolling(window=20).mean()
    volume_surge = data['Volume'] / volume_ma
    
    # Check conditions at key points
    for i in [19, 20, 21, 22, 23, 24, 25]:
        if i < len(data):
            print(f"\nDay {i} ({data.index[i].date()}):")
            print(f"  Close: ${data['Close'].iloc[i]:.2f}")
            print(f"  Resistance: ${resistance.iloc[i]:.2f}" if not pd.isna(resistance.iloc[i]) else "  Resistance: NaN")
            print(f"  Support: ${support.iloc[i]:.2f}" if not pd.isna(support.iloc[i]) else "  Support: NaN")
            print(f"  Volume surge: {volume_surge.iloc[i]:.2f}x" if not pd.isna(volume_surge.iloc[i]) else "  Volume surge: NaN")
            
            if not pd.isna(resistance.iloc[i]):
                breakout_price = resistance.iloc[i] * 1.001  # 0.1% above resistance
                above_resistance = data['Close'].iloc[i] > breakout_price
                print(f"  Above resistance? {above_resistance} (need > ${breakout_price:.2f})")
    
    # Try to generate signals
    print(f"\n{'-'*40}")
    print("SIGNAL GENERATION")
    print(f"{'-'*40}")
    
    try:
        signals = strategy.run(data)
        print(f"Signals generated: {len(signals)}")
        print(f"Buy signals: {(signals == 1).sum()}")
        print(f"Sell signals: {(signals == -1).sum()}")
        
        if (signals != 0).any():
            print("\nSignal dates:")
            for i in range(len(signals)):
                if signals.iloc[i] != 0:
                    signal_type = "BUY" if signals.iloc[i] == 1 else "SELL"
                    print(f"  {signal_type} on {data.index[i].date()} at ${data['Close'].iloc[i]:.2f}")
    except Exception as e:
        print(f"Error: {e}")
    
    return True


if __name__ == "__main__":
    debug_breakout()