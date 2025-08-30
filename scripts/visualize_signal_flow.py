#!/usr/bin/env python3
"""Visualize the complete signal flow from strategy to trades."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.dataflow.pipeline import DataPipeline
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.strategy.demo_ma import DemoMAStrategy
from bot.portfolio.allocator import PortfolioRules, allocate_signals
from bot.integration.strategy_allocator_bridge import StrategyAllocatorBridge
from bot.risk.integration import RiskIntegration
from bot.integration.orchestrator import IntegratedOrchestrator
from bot.config import get_config
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def trace_signal_flow(strategy_name: str, strategy_instance, symbol: str = "AAPL"):
    """Trace signals through the complete trading pipeline."""
    
    print(f"\n{'='*80}")
    print(f"🔍 SIGNAL FLOW TRACE: {strategy_name}")
    print('='*80)
    
    # Initialize components
    config = get_config()
    pipeline = DataPipeline()
    portfolio_rules = PortfolioRules()
    bridge = StrategyAllocatorBridge(strategy_instance, portfolio_rules)
    risk = RiskIntegration()
    
    # Load data
    start_date = datetime(2024, 1, 1) 
    end_date = datetime(2024, 6, 30)
    
    print(f"\n📈 Step 1: Load Data")
    print(f"  Symbol: {symbol}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    
    df = pipeline.get_data([symbol], start_date, end_date)[symbol]
    print(f"  ✅ Loaded {len(df)} bars")
    
    # Generate signals
    print(f"\n📊 Step 2: Generate Strategy Signals")
    signals_df = strategy_instance.generate_signals(df)
    
    # Count signals
    buy_signals = (signals_df["signal"] == 1).sum()
    signal_diff = signals_df["signal"].diff()
    entries = (signal_diff == 1).sum()
    exits = (signal_diff == -1).sum()
    
    print(f"  Buy signals: {buy_signals} ({buy_signals/len(df)*100:.1f}% of bars)")
    print(f"  Entry points: {entries}")
    print(f"  Exit points: {exits}")
    
    if entries > 0:
        # Get entry details
        entry_mask = signal_diff == 1
        entry_dates = signals_df[entry_mask].index
        print(f"\n  📍 Entry Points:")
        for i, date in enumerate(entry_dates[:3]):  # Show first 3
            price = df.loc[date, "close"]
            print(f"    {i+1}. {date}: Price=${price:.2f}")
    
    # Process signals through bridge
    print(f"\n💼 Step 3: Process Signals Through Bridge")
    
    # Prepare market data for bridge
    market_data = {symbol: df}
    portfolio_value = 100000  # Default initial capital
    
    print(f"  Portfolio value: ${portfolio_value:,.2f}")
    print(f"  Max positions: {portfolio_rules.max_positions}")
    print(f"  Risk per trade: {portfolio_rules.per_trade_risk_pct*100:.1f}%")
    
    # Get allocations through bridge
    try:
        allocations = bridge.process_signals(market_data, portfolio_value)
        
        print(f"\n  📊 Bridge Output (Position Sizes):")
        if allocations:
            total_allocated = 0
            for sym, qty in allocations.items():
                if qty > 0:
                    position_value = qty * df["close"].iloc[-1]
                    total_allocated += position_value
                    print(f"    {sym}: {qty} shares (${position_value:,.2f})")
            
            if total_allocated > 0:
                print(f"    Total allocated: ${total_allocated:,.2f} ({total_allocated/portfolio_value*100:.1f}%)")
            else:
                print(f"    ⚠️  No positions allocated!")
        else:
            print(f"    ⚠️  No allocations generated!")
            
    except Exception as e:
        print(f"  ❌ Bridge error: {e}")
        import traceback
        traceback.print_exc()
        allocations = {}
    
    # Risk check
    print(f"\n🛡️ Step 4: Risk Management Check")
    
    if allocations:
        for sym, qty in allocations.items():
            if qty > 0:
                position_value = qty * df["close"].iloc[-1]
                
                print(f"  Checking {sym}: {qty} shares (${position_value:,.2f})")
                
                # Check position limits
                max_position = risk.config.position_limits.max_position_size
                print(f"    Max position size: ${max_position:,.2f}")
                
                if position_value > max_position:
                    print(f"    ❌ Position too large: ${position_value:,.2f} > ${max_position:,.2f}")
                else:
                    print(f"    ✅ Position within limits: ${position_value:,.2f}")
                
                # Check exposure limits
                max_exposure = risk.config.portfolio_limits.max_total_exposure
                exposure_pct = position_value / portfolio_value
                print(f"    Max portfolio exposure: {max_exposure*100:.0f}%")
                print(f"    Actual exposure: {exposure_pct*100:.1f}%")
                
                if exposure_pct > max_exposure:
                    print(f"    ❌ Allocation exceeds exposure limit")
                else:
                    print(f"    ✅ Allocation within exposure limits")
    else:
        print(f"  ⚠️  No allocations to check")
    
    # Run full backtest to see actual trades
    print(f"\n🎯 Step 5: Run Full Backtest")
    
    orchestrator = IntegratedOrchestrator()
    results = orchestrator.run_backtest(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        strategy=strategy_instance,
        initial_capital=100000
    )
    
    print(f"\n  📊 Backtest Results:")
    print(f"    Total trades: {results.total_trades}")
    print(f"    Final value: ${100000 * (1 + results.total_return):,.2f}")
    print(f"    Total return: {results.total_return*100:.2f}%")
    
    if results.trades is not None and not results.trades.empty:
        print(f"\n  📈 Actual Trades:")
        for i, trade in results.trades.iterrows():
            if i >= 3:  # Show first 3 trades
                break
            print(f"    {i+1}. {trade['symbol']}: Entry ${trade['entry_price']:.2f} → Exit ${trade.get('exit_price', 'Open')}")
            print(f"       Qty: {trade['qty']}, PnL: ${trade.get('pnl', 0):.2f}")
    else:
        print(f"  ⚠️  No trades executed!")
    
    return {
        "strategy": strategy_name,
        "signals_generated": int(buy_signals),
        "entry_points": int(entries),
        "allocations_made": sum(1 for qty in allocations.values() if qty > 0) if allocations else 0,
        "trades_executed": results.total_trades,
        "final_return": results.total_return
    }

def main():
    """Trace signal flow for multiple strategies."""
    
    print("🔄 SIGNAL FLOW VISUALIZER")
    print("="*80)
    print("Tracing the journey from signal generation to trade execution")
    
    results = []
    
    # Test with default volatility
    vol_default = VolatilityStrategy(VolatilityParams())
    results.append(trace_signal_flow("Volatility (Default)", vol_default))
    
    # Test with relaxed volatility
    vol_relaxed = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,
        atr_threshold_multiplier=0.8
    ))
    results.append(trace_signal_flow("Volatility (Relaxed)", vol_relaxed))
    
    # Test with Demo MA
    ma_strategy = DemoMAStrategy(fast=10, slow=20)
    results.append(trace_signal_flow("Demo MA", ma_strategy))
    
    # Summary
    print("\n" + "="*80)
    print("📊 SIGNAL FLOW SUMMARY")
    print("="*80)
    
    print(f"\n{'Strategy':<25} {'Signals':<10} {'Entries':<10} {'Allocs':<10} {'Trades':<10} {'Return':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['strategy']:<25} {r['signals_generated']:<10} {r['entry_points']:<10} "
              f"{r['allocations_made']:<10} {r['trades_executed']:<10} "
              f"{r['final_return']*100:>6.2f}%")
    
    # Identify bottlenecks
    print("\n" + "="*80)
    print("🔍 BOTTLENECK ANALYSIS")
    print("="*80)
    
    for r in results:
        print(f"\n{r['strategy']}:")
        
        if r['signals_generated'] == 0:
            print("  ❌ No signals generated - Check strategy parameters")
        elif r['entry_points'] == 0:
            print("  ❌ Signals but no entries - Check entry conditions")
        elif r['allocations_made'] == 0:
            print("  ❌ Entries but no allocations - Check allocator logic")
        elif r['trades_executed'] == 0:
            print("  ❌ Allocations but no trades - Check risk limits or position sizing")
        elif r['trades_executed'] < r['entry_points']:
            print(f"  ⚠️  Some signals filtered: {r['entry_points']} entries → {r['trades_executed']} trades")
            print("     Check risk limits, position sizing, or capital constraints")
        else:
            print(f"  ✅ Signal flow working: {r['trades_executed']} trades executed")
    
    # Save results
    output_file = Path("data/outputs/signal_flow_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis saved to {output_file}")

if __name__ == "__main__":
    main()