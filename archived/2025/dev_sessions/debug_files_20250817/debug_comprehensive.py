#!/usr/bin/env python3
"""
Debug comprehensive test data with momentum strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies import create_strategy


def create_comprehensive_test_data():
    """Create test data that will trigger different strategy types."""
    dates = pd.date_range('2024-01-01', periods=120, freq='D')
    
    np.random.seed(42)
    prices = [100]  # Start at $100
    
    for i in range(119):
        # Create different market phases
        if i < 30:  # Trending up (good for momentum)
            daily_return = np.random.normal(0.4, 0.6) / 100
        elif i < 60:  # Oscillating/mean reverting
            cycle_pos = ((i - 30) % 20) / 20.0
            oscillation = np.sin(cycle_pos * 2 * np.pi) * 2.5
            daily_return = (oscillation + np.random.normal(0, 0.5)) / 100
        elif i < 90:  # Trending down 
            daily_return = np.random.normal(-0.3, 0.7) / 100
        else:  # Recovery/crossover phase
            daily_return = np.random.normal(0.2, 0.8) / 100
        
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 50))  # Floor at $50
    
    data = pd.DataFrame({
        'Close': prices,
        'Open': [p * 0.999 for p in prices],
        'High': [p * 1.015 for p in prices],
        'Low': [p * 0.985 for p in prices],
        'Volume': [1000000] * 120
    }, index=dates)
    
    return data


def debug_momentum_with_comprehensive_data():
    """Debug momentum strategy with comprehensive test data."""
    
    print("="*60)
    print("DEBUGGING MOMENTUM WITH COMPREHENSIVE DATA")
    print("="*60)
    
    # Create data
    data = create_comprehensive_test_data()
    print(f"Test data: {len(data)} days")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"Total return: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.1f}%")
    
    # Look at price movements
    print(f"\nPrice statistics:")
    print(f"  Daily returns mean: {data['Close'].pct_change().mean()*100:.3f}%")
    print(f"  Daily returns std: {data['Close'].pct_change().std()*100:.3f}%")
    print(f"  Max daily gain: {data['Close'].pct_change().max()*100:.2f}%")
    print(f"  Max daily loss: {data['Close'].pct_change().min()*100:.2f}%")
    
    # Create momentum strategy
    momentum_strategy = create_strategy("MomentumStrategy")
    print(f"\nStrategy: {momentum_strategy}")
    print(f"Required periods: {momentum_strategy.get_required_periods()}")
    
    # Get parameters
    lookback = momentum_strategy.get_parameter('lookback_period')
    smoothing = momentum_strategy.get_parameter('momentum_smoothing')
    buy_threshold = momentum_strategy.get_parameter('buy_threshold')
    sell_threshold = momentum_strategy.get_parameter('sell_threshold')
    
    print(f"Parameters: lookback={lookback}, smoothing={smoothing}")
    print(f"Thresholds: buy={buy_threshold}%, sell={sell_threshold}%")
    
    # Calculate ROC manually
    print(f"\n{'-'*40}")
    print("ROC ANALYSIS")
    print(f"{'-'*40}")
    
    roc = data['Close'].pct_change(periods=lookback) * 100
    roc_smooth = roc.rolling(window=smoothing).mean()
    roc_change = roc_smooth.diff()
    
    # Remove NaN values for analysis
    valid_roc = roc_smooth.dropna()
    valid_change = roc_change.dropna()
    
    print(f"ROC stats (valid values only):")
    print(f"  Count: {len(valid_roc)}")
    print(f"  Min: {valid_roc.min():.3f}%")
    print(f"  Max: {valid_roc.max():.3f}%")
    print(f"  Mean: {valid_roc.mean():.3f}%")
    print(f"  Std: {valid_roc.std():.3f}%")
    
    print(f"\nROC change stats:")
    print(f"  Count: {len(valid_change)}")
    print(f"  Min: {valid_change.min():.3f}%")
    print(f"  Max: {valid_change.max():.3f}%")
    print(f"  Mean: {valid_change.mean():.3f}%")
    
    # Check signal conditions
    print(f"\n{'-'*40}")
    print("SIGNAL CONDITION ANALYSIS")
    print(f"{'-'*40}")
    
    # Start analysis from the minimum required period
    start_idx = smoothing + lookback
    
    buy_conditions = 0
    sell_conditions = 0
    
    print(f"Analyzing from index {start_idx} to {len(data)-1}")
    
    for i in range(start_idx, len(data)):
        current_roc = roc_smooth.iloc[i]
        roc_delta = roc_change.iloc[i]
        
        if pd.isna(current_roc) or pd.isna(roc_delta):
            continue
            
        # Buy condition
        if current_roc > buy_threshold and roc_delta > 0:
            buy_conditions += 1
            if buy_conditions <= 3:  # Show first few
                print(f"  Buy condition at {i}: ROC={current_roc:.2f}%, delta={roc_delta:.3f}%")
        
        # Sell condition
        if current_roc < sell_threshold or (current_roc > 0 and roc_delta < -0.5):
            sell_conditions += 1
            if sell_conditions <= 3:  # Show first few
                print(f"  Sell condition at {i}: ROC={current_roc:.2f}%, delta={roc_delta:.3f}%")
    
    print(f"\nTotal buy conditions: {buy_conditions}")
    print(f"Total sell conditions: {sell_conditions}")
    print(f"Total analyzed periods: {len(data) - start_idx}")
    
    expected_signal_rate = (buy_conditions + sell_conditions) / (len(data) - start_idx) * 100
    print(f"Expected signal rate: {expected_signal_rate:.1f}%")
    
    if expected_signal_rate > 50:
        print("⚠️  Signal rate will be too high!")
        print("   This will cause validation to fail")
        print("   Consider adjusting thresholds or smoothing")
    else:
        print("✅ Signal rate should be acceptable")
    
    # Try generating signals
    print(f"\n{'-'*40}")
    print("ATTEMPTING SIGNAL GENERATION")
    print(f"{'-'*40}")
    
    try:
        signals = momentum_strategy.generate_signals(data)
        print(f"✅ Signals generated successfully")
        print(f"   Buy signals: {(signals == 1).sum()}")
        print(f"   Sell signals: {(signals == -1).sum()}")
        print(f"   Signal rate: {((signals != 0).sum() / len(signals)) * 100:.1f}%")
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        return False
    
    return True


if __name__ == "__main__":
    debug_momentum_with_comprehensive_data()