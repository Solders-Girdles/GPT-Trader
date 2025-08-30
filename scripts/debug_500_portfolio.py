#!/usr/bin/env python3
"""Debug why $500 portfolios don't execute trades."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def debug_500_portfolio():
    """Debug the $500 portfolio trading issue step by step."""
    
    print("🔍 DEBUGGING $500 PORTFOLIO ISSUE")
    print("="*50)
    
    # Test both $500 (broken) and $1000 (working) for comparison
    test_amounts = [500, 1000]
    
    strategy = TrendBreakoutStrategy()
    
    for amount in test_amounts:
        print(f"\n💰 TESTING ${amount} PORTFOLIO")
        print("-" * 40)
        
        orchestrator = IntegratedOrchestrator()
        
        # Hook into the key methods to see what's happening
        original_bridge_process = orchestrator.strategy_allocator_bridge.process_signals
        original_allocator_allocate = orchestrator.strategy_allocator_bridge.allocator.allocate_positions
        
        allocations_received = []
        bridge_outputs = []
        
        def debug_bridge_process(daily_data, current_equity):
            """Debug the bridge process_signals method."""
            print(f"  📊 Bridge received equity: ${current_equity:,.2f}")
            result = original_bridge_process(daily_data, current_equity)
            bridge_outputs.append(result)
            print(f"  📤 Bridge output allocations: {result}")
            return result
        
        def debug_allocator_allocate(signals, available_capital, current_prices, existing_positions=None):
            """Debug the allocator allocate_positions method."""
            print(f"  🎯 Allocator inputs:")
            print(f"    - Signals: {signals}")
            print(f"    - Available capital: ${available_capital:,.2f}")
            print(f"    - Current prices: {current_prices}")
            
            result = original_allocator_allocate(signals, available_capital, current_prices, existing_positions)
            allocations_received.append(result)
            
            print(f"  📋 Allocator output: {result}")
            
            # Check if any allocations are too small
            for symbol, shares in result.items():
                if shares > 0:
                    position_value = shares * current_prices.get(symbol, 0)
                    position_pct = position_value / available_capital * 100 if available_capital > 0 else 0
                    print(f"    - {symbol}: {shares} shares = ${position_value:.2f} ({position_pct:.1f}% of capital)")
                    
                    # Check if this might be getting filtered out somewhere
                    if position_value < 50:  # Less than $50 position
                        print(f"      ⚠️  WARNING: Very small position size might be filtered")
            
            return result
        
        # Apply debug hooks
        orchestrator.strategy_allocator_bridge.process_signals = debug_bridge_process
        orchestrator.strategy_allocator_bridge.allocator.allocate_positions = debug_allocator_allocate
        
        # Run backtest with very short period to see immediate results
        print(f"  🚀 Running backtest...")
        results = orchestrator.run_backtest(
            symbols=["AAPL"],
            start_date=datetime(2024, 1, 10),
            end_date=datetime(2024, 1, 12),  # Just 2 trading days
            strategy=strategy,
            initial_capital=amount
        )
        
        print(f"\n  📊 RESULTS:")
        print(f"    - Total trades: {results.total_trades}")
        print(f"    - Allocations made: {len(bridge_outputs)} times")
        print(f"    - Non-zero allocations: {sum(1 for a in bridge_outputs if any(v > 0 for v in a.values()))}")
        
        # Check risk management limits
        risk_config = orchestrator.risk_integration.risk_config
        dynamic_limit = risk_config.calculate_dynamic_position_limit(amount)
        max_position_value = amount * dynamic_limit
        
        print(f"\n  🛡️ RISK ANALYSIS:")
        print(f"    - Portfolio size: ${amount}")
        print(f"    - Dynamic limit: {dynamic_limit:.1%}")
        print(f"    - Max position value: ${max_position_value:.2f}")
        print(f"    - AAPL price: ~$185")
        print(f"    - Max AAPL shares: {int(max_position_value / 185)}")
        
        if amount == 500:
            print(f"\n  🔍 $500 PORTFOLIO DIAGNOSIS:")
            if results.total_trades == 0:
                print(f"    ❌ NO TRADES EXECUTED")
                print(f"    🔎 Checking for potential issues:")
                
                # Check if signals are being generated
                if len(bridge_outputs) == 0:
                    print(f"      - No bridge outputs recorded (signals not reaching allocator)")
                elif all(all(v == 0 for v in a.values()) for a in bridge_outputs):
                    print(f"      - Bridge outputs all zero (allocator returning no positions)")
                else:
                    print(f"      - Bridge outputs look normal, issue likely in execution or risk management")
                
                # Check if minimum position sizes are the issue
                if max_position_value < 100:
                    print(f"      - Max position value (${max_position_value:.2f}) might be below minimum thresholds")
            else:
                print(f"    ✅ TRADES WORKING")

def find_minimum_viable_portfolio():
    """Find the exact minimum portfolio size that works."""
    
    print(f"\n\n🎯 FINDING MINIMUM VIABLE PORTFOLIO SIZE")
    print("="*60)
    
    strategy = TrendBreakoutStrategy()
    
    # Binary search approach to find the exact threshold
    test_amounts = [300, 400, 500, 600, 700, 800, 900, 1000]
    
    print(f"{'Amount':<8} {'Trades':<8} {'Status':<15}")
    print("-" * 35)
    
    threshold = None
    
    for amount in test_amounts:
        try:
            orchestrator = IntegratedOrchestrator()
            
            results = orchestrator.run_backtest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 10),
                end_date=datetime(2024, 1, 12),  # Just 2 trading days
                strategy=strategy,
                initial_capital=amount
            )
            
            trades = results.total_trades
            status = "✅ Works" if trades > 0 else "❌ Broken"
            
            print(f"${amount:<7} {trades:<8} {status:<15}")
            
            if trades > 0 and threshold is None:
                threshold = amount
                
        except Exception as e:
            print(f"${amount:<7} {'ERROR':<8} ❌ {str(e)[:10]}...")
    
    print(f"\n🎯 CONCLUSION:")
    if threshold:
        print(f"   📊 Minimum viable portfolio: ${threshold}")
        print(f"   💡 Portfolios below ${threshold} fail to execute trades")
    else:
        print(f"   ❌ No working portfolios found in test range")

if __name__ == "__main__":
    debug_500_portfolio()
    find_minimum_viable_portfolio()