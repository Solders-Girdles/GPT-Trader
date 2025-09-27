#!/usr/bin/env python3
"""Test position sizing math to find the $600 threshold issue."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.portfolio.allocator import position_size, PortfolioRules

def test_position_sizing_math():
    """Test the position sizing math that's causing the $600 threshold."""
    
    print("🧮 POSITION SIZING MATH TEST")
    print("="*50)
    
    # Use realistic values for AAPL
    price = 185.0  # AAPL price
    atr_value = 3.5  # Typical ATR for AAPL
    
    rules = PortfolioRules()
    print(f"Rules:")
    print(f"  - Risk per trade: {rules.per_trade_risk_pct:.1%}")
    print(f"  - ATR multiplier: {rules.atr_k}x")
    print(f"  - Max positions: {rules.max_positions}")
    
    print(f"\nMarket conditions:")
    print(f"  - AAPL price: ${price:.2f}")
    print(f"  - ATR value: ${atr_value:.2f}")
    print(f"  - Stop distance: {rules.atr_k} × ${atr_value:.2f} = ${rules.atr_k * atr_value:.2f}")
    
    print(f"\n{'Portfolio':<10} {'Risk $':<10} {'Stop Dist':<12} {'Raw Calc':<12} {'Qty':<6} {'Status':<10}")
    print("-" * 70)
    
    portfolio_sizes = [400, 500, 550, 600, 650, 700, 1000]
    
    for portfolio_size in portfolio_sizes:
        risk_usd = portfolio_size * rules.per_trade_risk_pct
        stop_dist = rules.atr_k * atr_value
        raw_calc = risk_usd / stop_dist
        qty = position_size(portfolio_size, atr_value, price, rules)
        
        status = "✅ Works" if qty > 0 else "❌ Fails"
        
        print(f"${portfolio_size:<9} ${risk_usd:<9.2f} ${stop_dist:<11.2f} {raw_calc:<11.3f} {qty:<6} {status:<10}")
    
    print(f"\n🔍 ANALYSIS:")
    print(f"The threshold occurs when: risk_usd / stop_distance >= 1.0")
    
    # Calculate exact threshold
    stop_dist = rules.atr_k * atr_value
    min_risk_needed = stop_dist * 1.0  # Need at least 1.0 to get qty=1
    min_portfolio = min_risk_needed / rules.per_trade_risk_pct
    
    print(f"")
    print(f"  Required: ${min_risk_needed:.2f} risk ÷ {rules.per_trade_risk_pct:.1%} = ${min_portfolio:.0f} minimum portfolio")
    print(f"  Actual threshold found: $600")
    print(f"  Math suggests: ${min_portfolio:.0f}")

def test_solution_approaches():
    """Test different approaches to fix the small portfolio issue."""
    
    print(f"\n\n💡 SOLUTION APPROACHES")
    print("="*50)
    
    price = 185.0
    atr_value = 3.5
    portfolio_size = 500
    
    print(f"Target: Make $500 portfolio viable")
    print(f"Current: {position_size(portfolio_size, atr_value, price, PortfolioRules())} shares")
    
    print(f"\n🎯 Approach 1: Increase risk per trade for small portfolios")
    risk_levels = [0.005, 0.01, 0.015, 0.02, 0.025]
    
    for risk_pct in risk_levels:
        rules = PortfolioRules(per_trade_risk_pct=risk_pct)
        qty = position_size(portfolio_size, atr_value, price, rules)
        risk_usd = portfolio_size * risk_pct
        print(f"  {risk_pct:.1%} risk → ${risk_usd:.2f} → {qty} shares")
    
    print(f"\n🎯 Approach 2: Reduce ATR multiplier for small portfolios")
    atr_multipliers = [2.0, 1.5, 1.0, 0.5]
    
    for atr_k in atr_multipliers:
        rules = PortfolioRules(atr_k=atr_k)
        qty = position_size(portfolio_size, atr_value, price, rules)
        stop_dist = atr_k * atr_value
        print(f"  {atr_k}x ATR → ${stop_dist:.2f} stop → {qty} shares")
    
    print(f"\n🎯 Approach 3: Portfolio-size-aware risk scaling")
    
    def dynamic_risk_pct(portfolio_size):
        """Scale risk percentage based on portfolio size."""
        if portfolio_size < 1000:
            return 0.02  # 2% for micro portfolios
        elif portfolio_size < 5000:
            return 0.01  # 1% for small portfolios
        else:
            return 0.005  # 0.5% for standard portfolios
    
    test_portfolios = [500, 750, 1000, 2500, 5000]
    
    for portfolio in test_portfolios:
        risk_pct = dynamic_risk_pct(portfolio)
        rules = PortfolioRules(per_trade_risk_pct=risk_pct)
        qty = position_size(portfolio, atr_value, price, rules)
        risk_usd = portfolio * risk_pct
        print(f"  ${portfolio:,} → {risk_pct:.1%} risk → ${risk_usd:.2f} → {qty} shares")

if __name__ == "__main__":
    test_position_sizing_math()
    test_solution_approaches()