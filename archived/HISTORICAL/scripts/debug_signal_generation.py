#!/usr/bin/env python3
"""Debug signal generation for strategies to understand why no trades occur."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.dataflow.pipeline import DataPipeline
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.trend_breakout import TrendBreakoutStrategy
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def analyze_volatility_signals(df: pd.DataFrame, params: VolatilityParams) -> dict:
    """Analyze why volatility strategy isn't generating signals."""
    
    # Calculate indicators same as strategy
    from bot.indicators.optimized import OptimizedIndicators
    from bot.strategy.volatility import _safe_atr
    
    bb_upper, bb_middle, bb_lower = OptimizedIndicators.bollinger_bands(
        df["close"], period=params.bb_period, std_dev=params.bb_std_dev
    )
    
    atr_values = _safe_atr(df, params.atr_period)
    atr_sma = atr_values.rolling(window=params.atr_period, min_periods=1).mean()
    atr_threshold = atr_sma * params.atr_threshold_multiplier
    
    # Analyze conditions
    price_below_lower = df["close"] <= bb_lower
    atr_above_threshold = atr_values > atr_threshold
    both_conditions = price_below_lower & atr_above_threshold
    
    stats = {
        "total_bars": len(df),
        "price_below_lower_count": price_below_lower.sum(),
        "price_below_lower_pct": (price_below_lower.sum() / len(df)) * 100,
        "atr_above_threshold_count": atr_above_threshold.sum(),
        "atr_above_threshold_pct": (atr_above_threshold.sum() / len(df)) * 100,
        "both_conditions_count": both_conditions.sum(),
        "both_conditions_pct": (both_conditions.sum() / len(df)) * 100,
        "avg_atr": atr_values.mean(),
        "avg_atr_threshold": atr_threshold.mean(),
        "min_price": df["close"].min(),
        "max_price": df["close"].max(),
        "avg_price": df["close"].mean(),
        "bb_lower_avg": bb_lower.mean(),
        "bb_upper_avg": bb_upper.mean(),
        "bb_width_avg": (bb_upper - bb_lower).mean(),
    }
    
    # Find examples of when conditions are met
    if both_conditions.sum() > 0:
        examples = []
        for idx in df[both_conditions].index[:5]:  # First 5 examples
            examples.append({
                "date": str(idx),
                "price": df.loc[idx, "close"],
                "bb_lower": bb_lower.loc[idx],
                "atr": atr_values.loc[idx],
                "atr_threshold": atr_threshold.loc[idx]
            })
        stats["signal_examples"] = examples
    
    return stats

def test_strategy_signals(strategy_name: str, strategy_instance, df: pd.DataFrame) -> dict:
    """Test a strategy and analyze its signals."""
    
    print(f"\n{'='*60}")
    print(f"Testing {strategy_name}")
    print('='*60)
    
    # Generate signals
    signals_df = strategy_instance.generate_signals(df)
    
    # Count signals
    buy_signals = (signals_df["signal"] == 1).sum()
    sell_signals = (signals_df["signal"] == -1).sum() if "signal" in signals_df else 0
    
    # Find signal transitions (entries)
    signal_diff = signals_df["signal"].diff()
    entries = (signal_diff == 1).sum()
    exits = (signal_diff == -1).sum()
    
    stats = {
        "strategy": strategy_name,
        "total_bars": len(df),
        "buy_signals": int(buy_signals),
        "sell_signals": int(sell_signals),
        "entries": int(entries),
        "exits": int(exits),
        "signal_rate": (buy_signals / len(df)) * 100 if len(df) > 0 else 0,
        "has_signals": buy_signals > 0
    }
    
    # Get first few signal examples
    if buy_signals > 0:
        signal_dates = signals_df[signals_df["signal"] == 1].index[:5]
        stats["first_signals"] = [str(d) for d in signal_dates]
    
    print(f"📊 Results:")
    print(f"  - Total bars: {stats['total_bars']}")
    print(f"  - Buy signals: {stats['buy_signals']} ({stats['signal_rate']:.2f}%)")
    print(f"  - Entry points: {stats['entries']}")
    print(f"  - Exit points: {stats['exits']}")
    
    if stats.get("first_signals"):
        print(f"  - First signals at: {', '.join(stats['first_signals'][:3])}")
    
    return stats

def main():
    """Debug signal generation for all strategies."""
    
    print("🔍 Signal Generation Debugger")
    print("="*60)
    
    # Load data
    pipeline = DataPipeline()
    
    # Test with AAPL for 6 months
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    print(f"\n📈 Loading data for AAPL from {start_date.date()} to {end_date.date()}")
    df = pipeline.get_data(["AAPL"], start_date, end_date)["AAPL"]
    print(f"✅ Loaded {len(df)} bars")
    
    results = {}
    
    # Test volatility with different parameters
    print("\n" + "="*60)
    print("VOLATILITY STRATEGY ANALYSIS")
    print("="*60)
    
    # Default parameters
    default_params = VolatilityParams()
    print(f"\n📋 Default Parameters:")
    print(f"  - BB Period: {default_params.bb_period}")
    print(f"  - BB Std Dev: {default_params.bb_std_dev}")
    print(f"  - ATR Period: {default_params.atr_period}")
    print(f"  - ATR Threshold Multiplier: {default_params.atr_threshold_multiplier}x")
    
    vol_analysis = analyze_volatility_signals(df, default_params)
    
    print(f"\n📊 Signal Condition Analysis:")
    print(f"  - Price below lower BB: {vol_analysis['price_below_lower_count']} times ({vol_analysis['price_below_lower_pct']:.1f}%)")
    print(f"  - ATR above threshold: {vol_analysis['atr_above_threshold_count']} times ({vol_analysis['atr_above_threshold_pct']:.1f}%)")
    print(f"  - BOTH conditions met: {vol_analysis['both_conditions_count']} times ({vol_analysis['both_conditions_pct']:.1f}%)")
    
    print(f"\n📈 Market Statistics:")
    print(f"  - Avg ATR: {vol_analysis['avg_atr']:.2f}")
    print(f"  - Avg ATR Threshold: {vol_analysis['avg_atr_threshold']:.2f}")
    print(f"  - Price Range: ${vol_analysis['min_price']:.2f} - ${vol_analysis['max_price']:.2f}")
    print(f"  - Avg BB Width: ${vol_analysis['bb_width_avg']:.2f}")
    
    if vol_analysis.get("signal_examples"):
        print(f"\n🎯 Signal Examples:")
        for ex in vol_analysis["signal_examples"]:
            print(f"  - {ex['date']}: Price=${ex['price']:.2f}, BB_Lower=${ex['bb_lower']:.2f}, ATR={ex['atr']:.2f}")
    
    # Test with original parameters
    vol_strategy = VolatilityStrategy(default_params)
    results["volatility_default"] = test_strategy_signals("Volatility (Default)", vol_strategy, df)
    
    # Test with relaxed parameters
    print("\n📋 Testing with Relaxed Parameters:")
    relaxed_params = VolatilityParams(
        bb_period=20,
        bb_std_dev=2.0,
        atr_period=14,
        atr_threshold_multiplier=1.0  # Reduced from 1.2
    )
    print(f"  - ATR Threshold Multiplier: {relaxed_params.atr_threshold_multiplier}x (was 1.2x)")
    
    vol_relaxed = VolatilityStrategy(relaxed_params)
    results["volatility_relaxed"] = test_strategy_signals("Volatility (Relaxed)", vol_relaxed, df)
    
    # Test with very relaxed parameters
    print("\n📋 Testing with Very Relaxed Parameters:")
    very_relaxed_params = VolatilityParams(
        bb_period=20,
        bb_std_dev=1.5,  # Reduced from 2.0
        atr_period=14,
        atr_threshold_multiplier=0.8  # Much lower threshold
    )
    print(f"  - BB Std Dev: {very_relaxed_params.bb_std_dev} (was 2.0)")
    print(f"  - ATR Threshold Multiplier: {very_relaxed_params.atr_threshold_multiplier}x (was 1.2x)")
    
    vol_very_relaxed = VolatilityStrategy(very_relaxed_params)
    results["volatility_very_relaxed"] = test_strategy_signals("Volatility (Very Relaxed)", vol_very_relaxed, df)
    
    # Test other strategies for comparison
    print("\n" + "="*60)
    print("TESTING OTHER STRATEGIES FOR COMPARISON")
    print("="*60)
    
    # Demo MA
    ma_strategy = DemoMAStrategy(fast=10, slow=20)
    results["demo_ma"] = test_strategy_signals("Demo MA", ma_strategy, df)
    
    # Trend Breakout
    trend_strategy = TrendBreakoutStrategy()
    results["trend_breakout"] = test_strategy_signals("Trend Breakout", trend_strategy, df)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for name, stats in results.items():
        status = "✅" if stats["has_signals"] else "❌"
        print(f"{status} {name}: {stats['buy_signals']} signals ({stats['signal_rate']:.2f}% of bars)")
    
    # Save results
    output_file = Path("data/outputs/signal_debug_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Recommendations
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    if vol_analysis['both_conditions_count'] == 0:
        print("❌ Volatility strategy conditions are too restrictive!")
        print("   Solutions:")
        print("   1. Lower ATR threshold multiplier (try 0.8-1.0)")
        print("   2. Reduce BB standard deviation (try 1.5)")
        print("   3. Use OR logic instead of AND for conditions")
        print("   4. Add alternative entry conditions")
    elif vol_analysis['both_conditions_count'] < 5:
        print("⚠️  Volatility strategy generates very few signals")
        print("   Consider relaxing parameters slightly")
    else:
        print("✅ Volatility strategy parameters seem reasonable")
        print("   Check allocator and position sizing next")

if __name__ == "__main__":
    main()