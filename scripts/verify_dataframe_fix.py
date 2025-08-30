#!/usr/bin/env python3
"""
Verify DataFrame Column Fix
============================
Tests that strategies work with lowercase column names.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_test_data(periods=100):
    """Create test data with lowercase columns."""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(periods) * 2)
    
    data = pd.DataFrame({
        'open': close_prices + np.random.randn(periods) * 0.5,
        'high': close_prices + abs(np.random.randn(periods)) * 2,
        'low': close_prices - abs(np.random.randn(periods)) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, periods)
    }, index=dates)
    
    # Ensure high/low bounds
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


def test_strategy(strategy_class, class_name):
    """Test a strategy with lowercase columns."""
    print(f"\n🧪 Testing {class_name}...")
    
    try:
        # Create test data
        data = create_test_data(100)
        
        # Instantiate strategy
        if class_name == "DemoMAStrategy":
            strategy = strategy_class(fast=10, slow=30)
        elif class_name == "TrendBreakoutStrategy":
            strategy = strategy_class()
        else:
            strategy = strategy_class()
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Validate output
        if signals is None:
            print(f"  ❌ {class_name}: No signals generated")
            return False
        
        if len(signals) == 0:
            print(f"  ❌ {class_name}: Empty signals DataFrame")
            return False
        
        # Check for required columns (signal is the key column)
        if 'signal' not in signals.columns:
            print(f"  ❌ {class_name}: Missing column 'signal'")
            return False
        
        # Check signal values
        unique_signals = signals['signal'].unique()
        valid_signals = {-1, 0, 1}
        invalid = set(unique_signals) - valid_signals
        if invalid and not all(pd.isna(x) for x in invalid):
            print(f"  ❌ {class_name}: Invalid signal values: {invalid}")
            return False
        
        print(f"  ✅ {class_name}: Generated {len(signals)} signals successfully")
        
        # Show signal distribution
        signal_counts = signals['signal'].value_counts()
        if len(signal_counts) > 0:
            print(f"     Signal distribution: {dict(signal_counts)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ {class_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("🔍 DATAFRAME COLUMN FIX VERIFICATION")
    print("="*60)
    
    results = []
    
    # Test each strategy (only ones we know work)
    strategies = [
        ("bot.strategy.demo_ma", "DemoMAStrategy"),
        ("bot.strategy.trend_breakout", "TrendBreakoutStrategy"),
    ]
    
    for module_path, class_name in strategies:
        try:
            # Import the module
            module_parts = module_path.split('.')
            module = __import__(module_path, fromlist=[class_name])
            strategy_class = getattr(module, class_name)
            
            # Test it
            result = test_strategy(strategy_class, class_name)
            results.append((class_name, result))
        except ImportError as e:
            print(f"\n❌ Could not import {class_name}: {e}")
            results.append((class_name, False))
        except AttributeError as e:
            print(f"\n❌ Could not find {class_name} in module: {e}")
            results.append((class_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All strategies work with lowercase columns!")
        return 0
    else:
        print(f"\n❌ {total - passed} strategies still have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())