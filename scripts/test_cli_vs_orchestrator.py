#!/usr/bin/env python3
"""Compare CLI backtest vs direct orchestrator usage."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.strategy.trend_breakout import TrendBreakoutStrategy
from bot.integration.orchestrator import IntegratedOrchestrator

def test_cli_vs_orchestrator():
    """Test both the CLI approach and direct orchestrator approach."""
    
    print("🔄 CLI vs ORCHESTRATOR COMPARISON")
    print("="*60)
    
    # Test 1: Direct orchestrator (like CLI does)
    print("\n🚀 Test 1: Direct IntegratedOrchestrator")
    print("-" * 40)
    
    # Use trend_breakout (same as CLI default)
    trend_strategy = TrendBreakoutStrategy()
    
    orchestrator = IntegratedOrchestrator()
    
    results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        strategy=trend_strategy,
        initial_capital=100000
    )
    
    print(f"Results:")
    print(f"  Total trades: {results.total_trades}")
    print(f"  Total return: {results.total_return*100:.2f}%")
    print(f"  Max positions: {results.max_positions}")
    print(f"  Winning trades: {results.winning_trades}")
    print(f"  Losing trades: {results.losing_trades}")
    
    if results.trades is not None and not results.trades.empty:
        print(f"  First 3 trades:")
        for i, trade in results.trades.head(3).iterrows():
            print(f"    {i+1}. {trade.get('symbol', 'N/A')}: "
                  f"Entry ${trade.get('entry_price', 0):.2f} → "
                  f"Exit ${trade.get('exit_price', 0):.2f}, "
                  f"Qty: {trade.get('qty', 0)}")
    
    print(f"\n📊 Equity curve points: {len(results.equity_curve) if results.equity_curve is not None else 0}")
    if results.equity_curve is not None and not results.equity_curve.empty:
        first_equity = results.equity_curve.iloc[0]
        last_equity = results.equity_curve.iloc[-1]
        print(f"  First: ${first_equity:,.2f}")
        print(f"  Last: ${last_equity:,.2f}")
    
    # Test 2: Volatility strategy for comparison
    print("\n🌪️ Test 2: Volatility Strategy")
    print("-" * 40)
    
    vol_strategy = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,
        atr_threshold_multiplier=0.8
    ))
    
    vol_results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        strategy=vol_strategy,
        initial_capital=100000
    )
    
    print(f"Results:")
    print(f"  Total trades: {vol_results.total_trades}")
    print(f"  Total return: {vol_results.total_return*100:.2f}%")
    print(f"  Max positions: {vol_results.max_positions}")
    
    # Summary
    print("\n📊 COMPARISON SUMMARY")
    print("="*40)
    print(f"Trend Breakout: {results.total_trades} trades, {results.total_return*100:.2f}% return")
    print(f"Volatility:     {vol_results.total_trades} trades, {vol_results.total_return*100:.2f}% return")
    
    if results.total_trades == 0 and vol_results.total_trades == 0:
        print("\n❌ Both strategies show 0 trades!")
        print("   This suggests a systematic issue in the orchestrator")
    elif results.total_trades > 0 or vol_results.total_trades > 0:
        print("\n✅ At least one strategy is working!")
    
    # Check if there are errors/warnings in results
    if hasattr(results, 'errors') and results.errors:
        print(f"\nTrend Breakout Errors: {results.errors}")
    if hasattr(vol_results, 'errors') and vol_results.errors:
        print(f"Volatility Errors: {vol_results.errors}")

if __name__ == "__main__":
    test_cli_vs_orchestrator()