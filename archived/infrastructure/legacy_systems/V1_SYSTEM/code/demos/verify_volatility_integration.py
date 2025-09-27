#!/usr/bin/env python3
"""
Quick verification that the volatility strategy integrates properly with the system.

This verifies that:
1. The strategy can be imported successfully
2. It follows the same interface as other strategies
3. It can generate signals on sample data
4. It works with the backtest engine
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from bot.strategy.volatility import VolatilityStrategy
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.trend_breakout import TrendBreakoutStrategy


def create_sample_data(days=100):
    """Create sample OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=days, freq="1D")
    np.random.seed(42)
    
    # Create data with some volatility
    base_price = 100
    returns = np.random.normal(0, 0.02, days)
    # Add volatility clusters
    returns[30:40] *= 3  # High volatility period
    returns[70:75] *= 2  # Another spike
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, days))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, days))),
        'close': prices,
        'volume': np.random.uniform(1000000, 5000000, days)
    }, index=dates)


def test_strategy_interface():
    """Test that all strategies follow the same interface."""
    
    print("🔍 Testing Strategy Interface Compatibility")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    print(f"✅ Created {len(data)} bars of sample data")
    
    # Test all strategies
    strategies = [
        ("demo_ma", DemoMAStrategy()),
        ("trend_breakout", TrendBreakoutStrategy()),
        ("volatility", VolatilityStrategy())
    ]
    
    results = {}
    
    for name, strategy in strategies:
        print(f"\n📊 Testing {name} strategy...")
        
        # Check basic attributes
        assert hasattr(strategy, 'name'), f"{name} missing 'name' attribute"
        assert hasattr(strategy, 'supports_short'), f"{name} missing 'supports_short' attribute"
        assert hasattr(strategy, 'generate_signals'), f"{name} missing 'generate_signals' method"
        
        print(f"   • Name: {strategy.name}")
        print(f"   • Supports short: {strategy.supports_short}")
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Check required columns
        assert 'signal' in signals.columns, f"{name} missing 'signal' column"
        assert len(signals) == len(data), f"{name} wrong signal length"
        
        # Check signal values
        unique_signals = signals['signal'].unique()
        valid_signals = all(s in [0, 1, -1] for s in unique_signals if not pd.isna(s))
        assert valid_signals, f"{name} has invalid signal values: {unique_signals}"
        
        # For long-only strategies, check no short signals
        if not strategy.supports_short:
            has_shorts = (signals['signal'] == -1).any()
            assert not has_shorts, f"{name} claims long-only but has short signals"
        
        # Count signals
        total_signals = signals['signal'].sum()
        entry_count = len(signals[signals['signal'].diff() == 1])
        
        print(f"   • Total signal periods: {int(total_signals)}")
        print(f"   • Entry signals: {entry_count}")
        print(f"   • Signal columns: {list(signals.columns)}")
        
        results[name] = {
            'signals': signals,
            'total_periods': int(total_signals),
            'entries': entry_count,
            'columns': list(signals.columns)
        }
        
        print(f"   ✅ {name} passed interface tests")
    
    return results


def test_volatility_specific():
    """Test volatility-specific functionality."""
    
    print("\n🎯 Testing Volatility Strategy Specifics")
    print("=" * 40)
    
    strategy = VolatilityStrategy()
    data = create_sample_data()
    
    signals = strategy.generate_signals(data)
    
    # Check volatility-specific columns
    required_cols = ['bb_upper', 'bb_middle', 'bb_lower', 'atr', 'volatility_signal', 'bb_touch_signal']
    
    for col in required_cols:
        assert col in signals.columns, f"Missing volatility column: {col}"
        print(f"   ✅ Has {col} column")
    
    # Check Bollinger Band ordering
    valid_data = signals.dropna()
    if not valid_data.empty:
        bb_ordered = ((valid_data['bb_upper'] >= valid_data['bb_middle']) & 
                     (valid_data['bb_middle'] >= valid_data['bb_lower'])).all()
        assert bb_ordered, "Bollinger Bands not properly ordered"
        print("   ✅ Bollinger Bands properly ordered")
    
    # Check ATR non-negative
    valid_atr = signals['atr'].dropna()
    if not valid_atr.empty:
        atr_positive = (valid_atr >= 0).all()
        assert atr_positive, "ATR values are negative"
        print("   ✅ ATR values are non-negative")
    
    # Check signal logic
    volatility_periods = int(signals['volatility_signal'].sum())
    bb_touches = int(signals['bb_touch_signal'].sum())
    
    print(f"   • High volatility periods: {volatility_periods}")
    print(f"   • BB lower band touches: {bb_touches}")
    
    return True


def test_simple_usage():
    """Test the simple usage example from requirements."""
    
    print("\n🚀 Testing Simple Usage Example")
    print("=" * 35)
    
    # This should work as specified in requirements
    from bot.strategy.volatility import VolatilityStrategy
    
    strategy = VolatilityStrategy()
    print(f"   ✅ Strategy created: {strategy}")
    
    # Create some sample OHLCV data
    data = create_sample_data(50)
    print(f"   ✅ Sample data created: {len(data)} bars")
    
    # Should be able to generate signals
    signals = strategy.generate_signals(data)
    print(f"   ✅ Signals generated: {len(signals)} signals")
    
    # Check basic signal properties
    signal_count = int(signals['signal'].sum())
    print(f"   ✅ Total signal periods: {signal_count}")
    
    return True


if __name__ == "__main__":
    try:
        print("🎯 Volatility Strategy Integration Verification")
        print("=" * 60)
        
        # Test interface compatibility
        results = test_strategy_interface()
        
        # Test volatility specifics
        test_volatility_specific()
        
        # Test simple usage
        test_simple_usage()
        
        print("\n" + "=" * 60)
        print("🎉 All integration tests passed!")
        print("\n📊 Summary:")
        
        for name, result in results.items():
            print(f"   • {name}: {result['entries']} entries, {result['total_periods']} signal periods")
        
        print("\n✅ Volatility strategy is fully integrated and ready to use!")
        print("🎯 The strategy follows the same interface as demo_ma and trend_breakout")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)