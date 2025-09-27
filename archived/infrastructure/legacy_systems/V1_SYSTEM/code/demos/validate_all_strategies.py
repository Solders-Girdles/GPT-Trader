#!/usr/bin/env python3
"""
VALIDATION: All 4 Strategies Available

Quick validation that all four strategies can be imported and initialized:
1. Demo MA Strategy
2. Trend Breakout Strategy  
3. Mean Reversion Strategy
4. Momentum Strategy (NEW)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def validate_all_strategies():
    """Validate that all 4 strategies can be imported and initialized."""
    
    print("🔍 VALIDATING ALL STRATEGIES")
    print("=" * 50)
    
    strategies = {}
    
    # 1. Demo MA Strategy
    try:
        from bot.strategy.demo_ma import DemoMAStrategy
        strategies["Demo MA"] = DemoMAStrategy(fast=10, slow=20)
        print("✅ Demo MA Strategy: OK")
    except Exception as e:
        print(f"❌ Demo MA Strategy: {e}")
    
    # 2. Trend Breakout Strategy
    try:
        from bot.strategy.trend_breakout import TrendBreakoutStrategy, TrendBreakoutParams
        params = TrendBreakoutParams(donchian_lookback=20, atr_period=14)
        strategies["Trend Breakout"] = TrendBreakoutStrategy(params)
        print("✅ Trend Breakout Strategy: OK")
    except Exception as e:
        print(f"❌ Trend Breakout Strategy: {e}")
    
    # 3. Mean Reversion Strategy
    try:
        from bot.strategy.mean_reversion import MeanReversionStrategy, MeanReversionParams
        params = MeanReversionParams(rsi_period=14, oversold_threshold=30)
        strategies["Mean Reversion"] = MeanReversionStrategy(params)
        print("✅ Mean Reversion Strategy: OK")
    except Exception as e:
        print(f"❌ Mean Reversion Strategy: {e}")
    
    # 4. Momentum Strategy (NEW)
    try:
        from bot.strategy.momentum import MomentumStrategy, MomentumParams
        params = MomentumParams(roc_period=20, momentum_threshold=0.03)
        strategies["Momentum"] = MomentumStrategy(params)
        print("✅ Momentum Strategy: OK")
    except Exception as e:
        print(f"❌ Momentum Strategy: {e}")
    
    print("\n📊 STRATEGY SUMMARY")
    print("=" * 50)
    print(f"Total strategies available: {len(strategies)}/4")
    
    for name, strategy in strategies.items():
        print(f"  • {name}: {strategy.name}")
        print(f"    Supports short: {strategy.supports_short}")
    
    if len(strategies) == 4:
        print("\n🎉 SUCCESS: All 4 strategies are available!")
        print("✅ Demo MA (trend-following)")
        print("✅ Trend Breakout (momentum)")
        print("✅ Mean Reversion (contrarian)")
        print("✅ Momentum (trend acceleration)")
        print("\n📈 Strategy portfolio is now diversified across:")
        print("  - Trend following")
        print("  - Momentum/breakout")
        print("  - Mean reversion")
        print("  - Momentum acceleration")
        return True
    else:
        print(f"\n⚠️  WARNING: Only {len(strategies)}/4 strategies available")
        return False

if __name__ == "__main__":
    success = validate_all_strategies()
    sys.exit(0 if success else 1)