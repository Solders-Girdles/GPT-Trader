#!/usr/bin/env python3
"""Debug the 4 failing strategies to find the same systematic issues we've seen before."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy
from bot.strategy.mean_reversion import MeanReversionStrategy  
from bot.strategy.momentum import MomentumStrategy
from bot.strategy.optimized_ma import OptimizedMAStrategy
from bot.dataflow.pipeline import DataPipeline

def debug_strategy_signals(strategy_name, strategy):
    """Debug a strategy's signal generation - same analysis we did before."""
    
    print(f"\n🔍 DEBUGGING {strategy_name.upper()}")
    print("-" * 50)
    
    # Get market data (same as working strategies)
    pipeline = DataPipeline()
    try:
        data = pipeline.get_data(["AAPL"], datetime(2024, 1, 1), datetime(2024, 1, 31))
        aapl_data = data["AAPL"]
        print(f"  ✅ Data loaded: {len(aapl_data)} rows")
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return
    
    # Generate signals 
    try:
        signals = strategy.generate_signals(aapl_data)
        print(f"  ✅ Signals generated successfully")
        
        # Analyze signal output (same as we did before)
        if hasattr(signals, 'columns'):
            print(f"  📊 Signal columns: {list(signals.columns)}")
            
            # Check for 'signal' column (our standard)
            if 'signal' in signals.columns:
                signal_values = signals['signal']
                active_signals = signal_values[signal_values > 0]
                print(f"  📈 Signal column found: {len(active_signals)} active signals out of {len(signal_values)}")
                print(f"  📈 Signal range: {signal_values.min():.3f} to {signal_values.max():.3f}")
                
                if len(active_signals) > 0:
                    print(f"  ✅ Strategy generates signals - should work!")
                else:
                    print(f"  ⚠️ Strategy generates 0 active signals")
            else:
                print(f"  ❌ No 'signal' column - this is the issue!")
                print(f"  🔍 Available columns: {list(signals.columns)}")
        else:
            # Signals might be a Series
            if hasattr(signals, 'index'):
                active_signals = signals[signals > 0] if hasattr(signals, '__getitem__') else []
                print(f"  📈 Signal series: {len(active_signals)} active signals")
            else:
                print(f"  ❓ Unknown signal format: {type(signals)}")
                
    except Exception as e:
        print(f"  ❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()

def test_allocator_compatibility(strategy_name, strategy):
    """Test if strategy signals are compatible with allocator (same test as before)."""
    
    print(f"\n🔗 ALLOCATOR COMPATIBILITY TEST: {strategy_name}")
    print("-" * 40)
    
    try:
        # Get data and signals
        pipeline = DataPipeline()
        data = pipeline.get_data(["AAPL"], datetime(2024, 1, 1), datetime(2024, 1, 31))
        aapl_data = data["AAPL"]
        
        signals = strategy.generate_signals(aapl_data)
        
        # Create combined data like the bridge does
        combined_data = aapl_data.copy()
        
        if hasattr(signals, 'columns'):
            for col in signals.columns:
                combined_data[col] = signals[col]
        else:
            combined_data['signal'] = signals
        
        # Test what allocator sees (same check as before)
        from bot.portfolio.allocator import allocate_signals, PortfolioRules
        
        rules = PortfolioRules()
        signals_dict = {"AAPL": combined_data}
        
        # Test with $1K portfolio
        allocations = allocate_signals(signals_dict, 1000, rules)
        
        print(f"  📊 Allocator input columns: {list(combined_data.columns)}")
        print(f"  📈 Allocator output: {allocations}")
        
        if allocations.get("AAPL", 0) > 0:
            print(f"  ✅ Allocator working! Should execute trades")
        else:
            print(f"  ❌ Allocator returns 0 - same issue as before!")
            
            # Debug why (same analysis as before)
            if 'signal' in combined_data.columns:
                last_signal = combined_data['signal'].iloc[-1]
                recent_signals = combined_data['signal'].iloc[-120:]
                active_recent = recent_signals[recent_signals > 0]
                
                print(f"  🔍 Last signal: {last_signal}")
                print(f"  🔍 Recent active signals: {len(active_recent)}")
                
                if len(active_recent) == 0:
                    print(f"  💡 Same issue: No recent signals in 120-bar window!")
            else:
                print(f"  💡 Missing 'signal' column - format issue!")
                
    except Exception as e:
        print(f"  ❌ Compatibility test failed: {e}")

def comprehensive_failing_strategy_analysis():
    """Analyze all failing strategies for the same issues we've seen."""
    
    print("🚨 ANALYZING FAILING STRATEGIES FOR SAME SYSTEMATIC ISSUES")
    print("="*70)
    
    failing_strategies = [
        ("enhanced_trend_breakout", EnhancedTrendBreakoutStrategy()),
        ("mean_reversion", MeanReversionStrategy()),
        ("momentum", MomentumStrategy()),
        ("optimized_ma", OptimizedMAStrategy()),
    ]
    
    issue_patterns = {
        'signal_generation': [],
        'signal_format': [],
        'allocator_compatibility': [],
        'recent_signals': []
    }
    
    for strategy_name, strategy in failing_strategies:
        print(f"\n{'='*20} {strategy_name.upper()} {'='*20}")
        
        debug_strategy_signals(strategy_name, strategy)
        test_allocator_compatibility(strategy_name, strategy)
    
    print(f"\n\n📋 SYSTEMATIC ISSUE SUMMARY")
    print("="*50)
    print("We likely found the same patterns as before:")
    print("1. Strategies might use different signal column names")
    print("2. Signal formats might be incompatible with allocator")  
    print("3. No recent signals in 120-bar window")
    print("4. Same fixes needed as for the working strategies")

if __name__ == "__main__":
    comprehensive_failing_strategy_analysis()