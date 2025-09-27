#!/usr/bin/env python3
"""Debug ledger trade recording issue."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.strategy.volatility import VolatilityStrategy, VolatilityParams
from bot.integration.orchestrator import IntegratedOrchestrator

def debug_ledger():
    """Debug what's happening in the ledger during backtest."""
    
    print("🔍 LEDGER DEBUG")
    print("="*50)
    
    # Create strategy
    strategy = VolatilityStrategy(VolatilityParams(
        bb_std_dev=1.5,  # Relaxed parameters  
        atr_threshold_multiplier=0.8
    ))
    
    # Create orchestrator
    orchestrator = IntegratedOrchestrator()
    
    # Hook into the ledger to see what's happening
    original_submit_and_fill = orchestrator.ledger.submit_and_fill
    
    trades_submitted = []
    
    def debug_submit_and_fill(symbol, new_qty, price, ts, reason, cost_usd):
        """Debug wrapper for submit_and_fill."""
        trades_submitted.append({
            'symbol': symbol,
            'new_qty': new_qty,
            'price': price,
            'ts': ts,
            'reason': reason,
            'cost_usd': cost_usd,
            'current_position': orchestrator.ledger.positions.get(symbol, None)
        })
        
        print(f"  📈 LEDGER: {symbol} new_qty={new_qty}, price=${price:.2f}, reason={reason}")
        if symbol in orchestrator.ledger.positions:
            pos = orchestrator.ledger.positions[symbol]
            print(f"      Current pos: qty={pos.qty}, avg_price=${pos.avg_price:.2f}")
        
        # Call original method
        original_submit_and_fill(symbol, new_qty, price, ts, reason, cost_usd)
        
        # Check what happened
        if symbol in orchestrator.ledger.positions:
            pos = orchestrator.ledger.positions[symbol]
            print(f"      After: qty={pos.qty}, avg_price=${pos.avg_price:.2f}")
        
        print(f"      Total trades recorded: {len(orchestrator.ledger.trades)}")
    
    orchestrator.ledger.submit_and_fill = debug_submit_and_fill
    
    # Run short backtest
    print("Running short backtest to see ledger activity...")
    results = orchestrator.run_backtest(
        symbols=["AAPL"],
        start_date=datetime(2024, 2, 1),
        end_date=datetime(2024, 2, 29),  # Just one month
        strategy=strategy,
        initial_capital=100000
    )
    
    print(f"\n📊 RESULTS:")
    print(f"  Total trades (from ledger): {results.total_trades}")
    print(f"  Ledger trades list length: {len(orchestrator.ledger.trades)}")
    print(f"  Trades submitted to ledger: {len(trades_submitted)}")
    
    print(f"\n📈 TRADES SUBMITTED:")
    for i, trade in enumerate(trades_submitted):
        print(f"  {i+1}. {trade['symbol']}: {trade['new_qty']} shares @ ${trade['price']:.2f} ({trade['reason']})")
    
    print(f"\n💼 FINAL POSITIONS:")
    for symbol, pos in orchestrator.ledger.positions.items():
        if pos.qty > 0:
            print(f"  {symbol}: {pos.qty} shares @ ${pos.avg_price:.2f} (STILL OPEN)")
    
    print(f"\n🎯 DIAGNOSIS:")
    if len(trades_submitted) > 0 and results.total_trades == 0:
        print("  ❌ FOUND THE BUG: Trades submitted but never closed!")
        print("  💡 SOLUTION: Positions stay open, so no completed trades recorded")
        if any(pos.qty > 0 for pos in orchestrator.ledger.positions.values()):
            print("  📋 ACTION: Need to force close all positions at end of backtest")

if __name__ == "__main__":
    debug_ledger()