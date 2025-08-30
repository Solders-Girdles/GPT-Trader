#!/usr/bin/env python3
"""Simple debug of $500 vs $1000 portfolio comparison."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def compare_portfolio_sizes():
    """Compare $500 vs $1000 portfolios to find the difference."""
    
    print("🔍 SIMPLE $500 vs $1000 PORTFOLIO COMPARISON")
    print("="*60)
    
    strategy = TrendBreakoutStrategy()
    
    for amount in [500, 1000]:
        print(f"\n💰 TESTING ${amount} PORTFOLIO")
        print("-" * 40)
        
        orchestrator = IntegratedOrchestrator()
        
        # Test with very short period
        results = orchestrator.run_backtest(
            symbols=["AAPL"],
            start_date=datetime(2024, 1, 10),
            end_date=datetime(2024, 1, 12),  # Just 2 trading days
            strategy=strategy,
            initial_capital=amount
        )
        
        print(f"Results:")
        print(f"  📊 Total trades: {results.total_trades}")
        print(f"  💰 Total return: {results.total_return*100:.2f}%")
        print(f"  📈 Final value: ${amount * (1 + results.total_return):,.2f}")
        
        # Calculate expected dynamic risk limits
        if amount < 1000:
            expected_limit = 0.50  # 50% for micro portfolios
        elif amount < 5000:
            expected_limit = 0.25  # 25% for small portfolios
        else:
            expected_limit = 0.15  # 15% for medium portfolios
            
        max_position_value = amount * expected_limit
        aapl_price = 185  # Approximate
        max_shares = int(max_position_value / aapl_price)
        
        print(f"  🛡️ Risk analysis:")
        print(f"    - Dynamic limit: {expected_limit:.0%}")
        print(f"    - Max position: ${max_position_value:.2f}")
        print(f"    - Max AAPL shares: {max_shares}")
        print(f"    - Viable?: {'✅ YES' if max_shares >= 1 else '❌ NO'}")

def test_minimum_thresholds():
    """Test a range of portfolio sizes to find the exact threshold."""
    
    print(f"\n\n🎯 FINDING EXACT THRESHOLD")
    print("="*40)
    
    strategy = TrendBreakoutStrategy()
    
    # Test range from $400 to $1200 in $100 increments
    test_amounts = range(400, 1300, 100)
    
    print(f"{'Amount':<8} {'Trades':<8} {'Max Pos':<10} {'Max Shares':<12} {'Status':<10}")
    print("-" * 55)
    
    working_threshold = None
    
    for amount in test_amounts:
        try:
            orchestrator = IntegratedOrchestrator()
            
            results = orchestrator.run_backtest(
                symbols=["AAPL"],
                start_date=datetime(2024, 1, 10),
                end_date=datetime(2024, 1, 12),
                strategy=strategy,
                initial_capital=amount
            )
            
            # Calculate expected limits
            if amount < 1000:
                limit = 0.50
            elif amount < 5000:
                limit = 0.25
            else:
                limit = 0.15
                
            max_pos = amount * limit
            max_shares = int(max_pos / 185)
            
            trades = results.total_trades
            status = "✅ Works" if trades > 0 else "❌ Fails"
            
            print(f"${amount:<7} {trades:<8} ${max_pos:<9.0f} {max_shares:<12} {status:<10}")
            
            if trades > 0 and working_threshold is None:
                working_threshold = amount
                
        except Exception as e:
            print(f"${amount:<7} {'ERROR':<8} {'N/A':<10} {'N/A':<12} ❌ Error")
    
    print(f"\n🎯 ANALYSIS:")
    if working_threshold:
        print(f"   📊 First working portfolio: ${working_threshold}")
        print(f"   ⚠️  Portfolios under ${working_threshold} fail to execute trades")
        
        # Analyze why
        failed_amount = working_threshold - 100
        if failed_amount < 1000:
            failed_limit = 0.50
        else:
            failed_limit = 0.25
            
        failed_max_pos = failed_amount * failed_limit
        failed_max_shares = int(failed_max_pos / 185)
        
        print(f"\n   🔍 Failed portfolio analysis (${failed_amount}):")
        print(f"     - Max position: ${failed_max_pos:.2f}")
        print(f"     - Max shares: {failed_max_shares}")
        print(f"     - Issue: {'Insufficient shares for minimum position' if failed_max_shares < 1 else 'Unknown - need deeper investigation'}")
    else:
        print(f"   ❌ No working portfolios found in test range")

if __name__ == "__main__":
    compare_portfolio_sizes()
    test_minimum_thresholds()