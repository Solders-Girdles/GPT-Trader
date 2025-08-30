#!/usr/bin/env python3
"""
Test different momentum parameters to find ones that work.
"""

import pandas as pd
import numpy as np
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


def test_momentum_parameters():
    """Test different momentum parameters to find working ones."""
    
    data = create_comprehensive_test_data()
    print(f"Test data: {len(data)} days, return: {((data['Close'].iloc[-1]/data['Close'].iloc[0])-1)*100:.1f}%")
    
    # Test different parameter combinations
    parameter_sets = [
        {"buy_threshold": 2.0, "sell_threshold": -1.0, "momentum_smoothing": 3},  # Default
        {"buy_threshold": 3.5, "sell_threshold": -2.0, "momentum_smoothing": 5},  # More conservative
        {"buy_threshold": 5.0, "sell_threshold": -3.0, "momentum_smoothing": 7},  # Very conservative
        {"buy_threshold": 7.0, "sell_threshold": -4.0, "momentum_smoothing": 10}, # Ultra conservative
    ]
    
    print(f"\n{'Parameters':<35} | {'Signals':<15} | {'Rate':<8} | {'Valid':<5}")
    print("-" * 70)
    
    for i, params in enumerate(parameter_sets):
        try:
            strategy = create_strategy("MomentumStrategy", **params)
            signals = strategy.run(data)
            
            buy_signals = (signals == 1).sum()
            sell_signals = (signals == -1).sum()
            total_signals = buy_signals + sell_signals
            signal_rate = (total_signals / len(signals)) * 100
            
            param_str = f"B:{params['buy_threshold']}, S:{params['sell_threshold']}, Sm:{params['momentum_smoothing']}"
            signal_str = f"{buy_signals}B + {sell_signals}S = {total_signals}"
            
            print(f"{param_str:<35} | {signal_str:<15} | {signal_rate:5.1f}%  | ✅")
            
            if signal_rate <= 50:
                print(f"  ✅ This parameter set works! Signal rate: {signal_rate:.1f}%")
                return params
                
        except Exception as e:
            param_str = f"B:{params['buy_threshold']}, S:{params['sell_threshold']}, Sm:{params['momentum_smoothing']}"
            print(f"{param_str:<35} | {'ERROR':<15} | {'N/A':<8} | ❌")
    
    print("\n❌ No parameter set worked with this test data")
    return None


if __name__ == "__main__":
    working_params = test_momentum_parameters()
    if working_params:
        print(f"\n🎯 Use these parameters: {working_params}")
    else:
        print("\n🔧 Need to adjust test data or validation rules")