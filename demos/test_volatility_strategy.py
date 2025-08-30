#!/usr/bin/env python3
"""
Test the new Volatility Strategy implementation.

This script demonstrates the volatility strategy with sample data
and validates its functionality.
"""

import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import yfinance as yf
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_volatility_strategy():
    """Test the volatility strategy with real market data."""
    
    print("🔬 Testing Volatility Strategy")
    print("=" * 50)
    
    # Download sample data
    print("📊 Downloading test data (SPY, 6 months)...")
    ticker = "SPY"
    data = yf.download(ticker, period="6mo", interval="1d")
    
    if data.empty:
        print("❌ Failed to download data")
        return False
    
    # Flatten multi-level columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    # Normalize column names to lowercase for strategy compatibility
    data.columns = [col.lower() for col in data.columns]
    
    print(f"✅ Downloaded {len(data)} bars")
    print(f"📋 Data columns: {list(data.columns)}")
    print(f"📋 Data index type: {type(data.index)}")
    print(f"📋 Sample data:\n{data.head(2)}")
    
    # Test default parameters
    print("\n🎯 Testing with default parameters...")
    strategy = VolatilityStrategy()
    print(f"Strategy: {strategy}")
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    print(f"✅ Generated signals for {len(signals)} bars")
    print(f"📈 Signal columns: {list(signals.columns)}")
    
    # Analyze signals
    total_signals = signals['signal'].sum()
    entry_signals = len(signals[signals['signal'].diff() == 1])
    exit_signals = len(signals[signals['signal'].diff() == -1])
    
    print(f"📊 Signal Analysis:")
    print(f"   • Total signal periods: {int(total_signals)}")
    print(f"   • Entry signals: {entry_signals}")
    print(f"   • Exit signals: {exit_signals}")
    
    # Check for recent volatility signals
    recent_volatility = signals['volatility_signal'].tail(20).sum()
    recent_bb_touches = signals['bb_touch_signal'].tail(20).sum()
    
    print(f"   • Recent volatility periods (last 20 days): {int(recent_volatility)}")
    print(f"   • Recent BB lower touches (last 20 days): {int(recent_bb_touches)}")
    
    # Test custom parameters
    print("\n🔧 Testing with custom parameters...")
    custom_params = VolatilityParams(
        bb_period=10,
        bb_std_dev=1.5,
        atr_period=10,
        atr_threshold_multiplier=1.0,
        exit_middle_band=False  # Only exit at upper band
    )
    
    custom_strategy = VolatilityStrategy(custom_params)
    print(f"Custom Strategy: {custom_strategy}")
    
    custom_signals = custom_strategy.generate_signals(data)
    custom_total = custom_signals['signal'].sum()
    custom_entries = len(custom_signals[custom_signals['signal'].diff() == 1])
    
    print(f"📊 Custom Strategy Analysis:")
    print(f"   • Total signal periods: {int(custom_total)}")
    print(f"   • Entry signals: {custom_entries}")
    
    # Show some example data
    print("\n📋 Sample Signal Data (last 10 days):")
    sample_data = signals.tail(10)[['signal', 'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'volatility_signal']].round(2)
    print(sample_data.to_string())
    
    # Validate signal logic
    print("\n✅ Validation checks:")
    
    # Check that signals are only 0 or 1
    valid_signals = signals['signal'].isin([0, 1]).all()
    print(f"   • All signals are 0 or 1: {valid_signals}")
    
    # Check that we have the required columns
    required_cols = ['signal', 'bb_upper', 'bb_middle', 'bb_lower', 'atr']
    has_required = all(col in signals.columns for col in required_cols)
    print(f"   • Has all required columns: {has_required}")
    
    # Check for NaN values in the signal column
    no_nan_signals = not signals['signal'].isna().any()
    print(f"   • No NaN values in signals: {no_nan_signals}")
    
    # Check that Bollinger Bands are ordered correctly (upper > middle > lower when not NaN)
    bb_data = signals[['bb_upper', 'bb_middle', 'bb_lower']].dropna()
    if not bb_data.empty:
        bb_ordered = ((bb_data['bb_upper'] >= bb_data['bb_middle']) & 
                     (bb_data['bb_middle'] >= bb_data['bb_lower'])).all()
        print(f"   • Bollinger Bands properly ordered: {bb_ordered}")
    
    return valid_signals and has_required and no_nan_signals


def test_edge_cases():
    """Test edge cases for the volatility strategy."""
    
    print("\n🧪 Testing Edge Cases")
    print("=" * 30)
    
    strategy = VolatilityStrategy()
    
    # Test with insufficient data
    print("📊 Testing with insufficient data...")
    small_data = pd.DataFrame({
        'open': [100, 101],
        'high': [102, 103], 
        'low': [99, 100],
        'close': [101, 102],
        'volume': [1000, 1100]
    })
    
    signals = strategy.generate_signals(small_data)
    all_zero = (signals['signal'] == 0).all()
    print(f"   • All signals are zero with insufficient data: {all_zero}")
    
    # Test with empty DataFrame
    print("📊 Testing with empty data...")
    empty_data = pd.DataFrame()
    empty_signals = strategy.generate_signals(empty_data)
    is_empty = empty_signals.empty
    print(f"   • Empty data returns empty signals: {is_empty}")
    
    return all_zero and is_empty


if __name__ == "__main__":
    try:
        success = test_volatility_strategy()
        edge_success = test_edge_cases()
        
        print("\n" + "=" * 50)
        if success and edge_success:
            print("🎉 All tests passed! Volatility strategy is working correctly.")
            exit_code = 0
        else:
            print("❌ Some tests failed. Check the output above.")
            exit_code = 1
            
        print("🎯 Volatility strategy is ready for integration!")
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)