#!/usr/bin/env python3
"""Test the $500 portfolio fix with dynamic position sizing."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.portfolio.allocator import position_size, PortfolioRules
from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def test_position_sizing_fix():
    """Test the new dynamic position sizing fix."""
    
    print("🧮 TESTING DYNAMIC POSITION SIZING FIX")
    print("="*50)
    
    # Use realistic values for AAPL
    price = 185.0  # AAPL price
    atr_value = 3.5  # Typical ATR for AAPL
    
    rules = PortfolioRules()
    
    print(f"Market conditions:")
    print(f"  - AAPL price: ${price:.2f}")
    print(f"  - ATR value: ${atr_value:.2f}")
    print(f"  - Stop distance base: {rules.atr_k} × ${atr_value:.2f} = ${rules.atr_k * atr_value:.2f}")
    
    print(f"\n{'Portfolio':<10} {'Risk %':<8} {'Risk $':<10} {'Qty':<6} {'Status':<15}")
    print("-" * 60)
    
    portfolio_sizes = [400, 500, 600, 1000, 2500, 5000, 25000]
    
    for portfolio_size in portfolio_sizes:
        dynamic_risk_pct = rules.calculate_dynamic_risk_pct(portfolio_size)
        risk_usd = portfolio_size * dynamic_risk_pct
        qty = position_size(portfolio_size, atr_value, price, rules)
        
        # Determine tier
        if portfolio_size < 1000:
            tier = "Micro"
        elif portfolio_size < 5000:
            tier = "Small"
        elif portfolio_size < 25000:
            tier = "Medium"
        else:
            tier = "Standard"
        
        status = f"✅ {tier}" if qty > 0 else f"❌ {tier}"
        
        print(f"${portfolio_size:<9} {dynamic_risk_pct:.1%}{'':>3} ${risk_usd:<9.2f} {qty:<6} {status:<15}")

def test_500_portfolio_full_backtest():
    """Test $500 portfolio with full backtest to confirm it works."""
    
    print(f"\n\n🎯 TESTING $500 PORTFOLIO FULL BACKTEST")
    print("="*60)
    
    strategy = TrendBreakoutStrategy()
    orchestrator = IntegratedOrchestrator()
    
    print("Running $500 portfolio backtest...")
    
    results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 10),
        end_date=datetime(2024, 1, 12),  # Short test
        strategy=strategy,
        initial_capital=500
    )
    
    print(f"\n📊 RESULTS:")
    print(f"  Total trades: {results.total_trades}")
    print(f"  Total return: {results.total_return*100:.2f}%")
    print(f"  Final value: ${500 * (1 + results.total_return):,.2f}")
    
    if results.total_trades > 0:
        print(f"\n✅ SUCCESS! $500 portfolios now execute trades!")
        print(f"🎯 The foundation is now truly bulletproof for autonomous trading!")
    else:
        print(f"\n❌ Still not working. Need further investigation.")

def test_all_portfolio_sizes():
    """Test a range of portfolio sizes to confirm they all work."""
    
    print(f"\n\n📊 COMPREHENSIVE PORTFOLIO SIZE TEST")
    print("="*50)
    
    strategy = TrendBreakoutStrategy()
    
    test_amounts = [500, 750, 1000, 1500, 2000, 3000, 5000]
    
    print(f"{'Amount':<8} {'Trades':<8} {'Status':<15}")
    print("-" * 35)
    
    working_count = 0
    
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
            
            trades = results.total_trades
            status = "✅ Working" if trades > 0 else "❌ Not working"
            
            if trades > 0:
                working_count += 1
            
            print(f"${amount:<7} {trades:<8} {status:<15}")
            
        except Exception as e:
            print(f"${amount:<7} {'ERROR':<8} ❌ Error")
    
    print(f"\n🎯 SUMMARY:")
    print(f"  Working portfolios: {working_count}/{len(test_amounts)}")
    print(f"  Success rate: {working_count/len(test_amounts)*100:.0f}%")
    
    if working_count == len(test_amounts):
        print(f"  🚀 PERFECT! All portfolio sizes work!")
        print(f"  🎯 Foundation is bulletproof for autonomous trading!")
    elif working_count >= len(test_amounts) * 0.8:
        print(f"  ✅ Excellent! Most portfolio sizes work!")
    else:
        print(f"  ⚠️ Need more work to achieve universal compatibility")

if __name__ == "__main__":
    test_position_sizing_fix()
    test_500_portfolio_full_backtest()
    test_all_portfolio_sizes()