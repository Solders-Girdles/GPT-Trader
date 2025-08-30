#!/usr/bin/env python3
"""Debug the position sizing logic that's causing small portfolios to fail."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.dataflow.pipeline import DataPipeline
from bot.portfolio.allocator import position_size, PortfolioRules, allocate_signals

def debug_position_sizing():
    """Debug the position sizing calculation for small portfolios."""
    
    print("🔍 DEBUGGING POSITION SIZING LOGIC")
    print("="*50)
    
    # Get real market data
    pipeline = DataPipeline()
    data = pipeline.get_data(["AAPL"], datetime(2024, 1, 10), datetime(2024, 1, 12))
    aapl_data = data["AAPL"]
    
    print(f"📊 Market Data Sample:")
    print(f"  AAPL close: ${aapl_data['close'].iloc[-1]:.2f}")
    print(f"  AAPL open: ${aapl_data['open'].iloc[-1]:.2f}")
    
    # Generate signals using trend breakout strategy
    strategy = TrendBreakoutStrategy()
    signals = strategy.generate_signals(aapl_data)
    
    # Check if ATR column exists and get its value
    if "atr" in aapl_data.columns:
        atr_value = float(aapl_data["atr"].iloc[-1])
        print(f"  ATR value: ${atr_value:.2f}")
    else:
        print(f"  ⚠️ ATR column missing, calculating manually...")
        # Calculate ATR manually (simplified)
        high_low = aapl_data['high'] - aapl_data['low']
        atr_value = high_low.rolling(14).mean().iloc[-1]
        print(f"  Calculated ATR: ${atr_value:.2f}")
    
    price = float(aapl_data["close"].iloc[-1])
    
    # Test position sizing for different portfolio sizes
    portfolio_sizes = [400, 500, 600, 700, 1000]
    rules = PortfolioRules()
    
    print(f"\n🧮 POSITION SIZE CALCULATIONS:")
    print(f"  Rules: {rules.per_trade_risk_pct:.1%} risk per trade, {rules.atr_k}x ATR stop")
    print(f"  Price: ${price:.2f}, ATR: ${atr_value:.2f}")
    print(f"  Stop distance: {rules.atr_k} × ${atr_value:.2f} = ${rules.atr_k * atr_value:.2f}")
    
    print(f"\n{'Portfolio':<10} {'Risk $':<8} {'Position Size':<12} {'Viable?':<8}")
    print("-" * 45)
    
    threshold_found = None
    
    for portfolio_size in portfolio_sizes:
        risk_usd = portfolio_size * rules.per_trade_risk_pct
        stop_dist = rules.atr_k * atr_value
        qty = position_size(portfolio_size, atr_value, price, rules)
        
        viable = "✅ YES" if qty > 0 else "❌ NO"
        
        print(f"${portfolio_size:<9} ${risk_usd:<7.2f} {qty:<12} {viable:<8}")
        
        if qty > 0 and threshold_found is None:
            threshold_found = portfolio_size
    
    # Now test the full allocation pipeline
    print(f"\n🎯 FULL ALLOCATION PIPELINE TEST:")
    print("-" * 40)
    
    # Add signals to the dataframe
    combined_data = aapl_data.copy()
    if hasattr(signals, 'items'):
        for col, values in signals.items():
            combined_data[col] = values
    else:
        combined_data['signal'] = signals
    
    for portfolio_size in [500, 600]:
        print(f"\n💰 ${portfolio_size} Portfolio:")
        
        signals_dict = {"AAPL": combined_data}
        allocations = allocate_signals(signals_dict, portfolio_size, rules)
        
        print(f"  Allocations: {allocations}")
        total_allocated = sum(allocations.values())
        print(f"  Total positions: {total_allocated}")
        
        if total_allocated == 0:
            # Debug why allocation failed
            print(f"  🔍 Debug:")
            print(f"    - Signal value: {combined_data['signal'].iloc[-1] if 'signal' in combined_data.columns else 'N/A'}")
            print(f"    - Position size calc: {position_size(portfolio_size, atr_value, price, rules)}")
            print(f"    - Risk amount: ${portfolio_size * rules.per_trade_risk_pct:.2f}")
            print(f"    - Stop distance: ${rules.atr_k * atr_value:.2f}")

def test_different_risk_levels():
    """Test what happens with different risk per trade levels."""
    
    print(f"\n\n🎛️ TESTING DIFFERENT RISK LEVELS")
    print("="*50)
    
    # Get market data
    pipeline = DataPipeline()
    data = pipeline.get_data(["AAPL"], datetime(2024, 1, 10), datetime(2024, 1, 12))
    aapl_data = data["AAPL"]
    
    price = float(aapl_data["close"].iloc[-1])
    atr_value = float(aapl_data["atr"].iloc[-1]) if "atr" in aapl_data.columns else 3.0
    
    portfolio_size = 500  # Test with $500 portfolio
    risk_levels = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1% to 5%
    
    print(f"Testing with ${portfolio_size} portfolio, AAPL @ ${price:.2f}, ATR ${atr_value:.2f}")
    print(f"\n{'Risk %':<8} {'Risk $':<8} {'Stop Dist':<10} {'Qty':<6} {'Position Value':<15}")
    print("-" * 60)
    
    for risk_pct in risk_levels:
        rules = PortfolioRules(per_trade_risk_pct=risk_pct)
        risk_usd = portfolio_size * risk_pct
        stop_dist = rules.atr_k * atr_value
        qty = position_size(portfolio_size, atr_value, price, rules)
        position_value = qty * price
        
        print(f"{risk_pct:.1%}{'':>4} ${risk_usd:<7.2f} ${stop_dist:<9.2f} {qty:<6} ${position_value:<14.2f}")
    
    print(f"\n💡 SOLUTION: Increase risk per trade for small portfolios!")

if __name__ == "__main__":
    debug_position_sizing()
    test_different_risk_levels()