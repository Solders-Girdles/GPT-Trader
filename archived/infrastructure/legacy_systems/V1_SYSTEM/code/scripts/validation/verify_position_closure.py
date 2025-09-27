#!/usr/bin/env python3
"""
Verify that positions are being closed at the end of backtest.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from bot.integration.orchestrator import IntegratedOrchestrator, BacktestConfig
from bot.integration.unified_optimizer import UnifiedOptimizer, UnifiedOptimizationConfig
from bot.strategy.mean_reversion import MeanReversionStrategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_position_closure():
    """Test if positions are closed at end of backtest."""
    
    print("="*80)
    print("TESTING POSITION CLOSURE AT BACKTEST END")
    print("="*80)
    
    # Test base orchestrator
    print("\n1. BASE ORCHESTRATOR TEST")
    print("-"*40)
    
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
        initial_capital=10000,
        quiet_mode=False
    )
    
    orch = IntegratedOrchestrator(config)
    strategy = MeanReversionStrategy()
    
    # Trace close_all_positions
    original_close = orch._close_all_positions
    close_called = []
    
    def traced_close(final_date):
        print(f"\n_close_all_positions called with date: {final_date}")
        print(f"  Positions before: {orch.current_positions}")
        result = original_close(final_date)
        print(f"  Positions after: {orch.current_positions}")
        print(f"  Trades closed: {result}")
        close_called.append(True)
        return result
    
    orch._close_all_positions = traced_close
    
    result = orch.run_backtest(strategy, ['AAPL'])
    
    print(f"\nBase Orchestrator Results:")
    print(f"  close_all_positions called: {len(close_called) > 0}")
    print(f"  Final positions: {orch.current_positions}")
    print(f"  Total trades: {result.total_trades}")
    print(f"  Ledger trades: {len(orch.ledger.trades)}")
    print(f"  Ledger fills: {len(orch.ledger.fills)}")
    
    # Test UnifiedOptimizer
    print("\n2. UNIFIED OPTIMIZER TEST")
    print("-"*40)
    
    opt_config = UnifiedOptimizationConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 10),
        initial_capital=10000,
        quiet_mode=False,
        use_optimization=False  # Keep it simple
    )
    
    optimizer = UnifiedOptimizer(opt_config)
    
    # Check if UnifiedOptimizer overrides _close_all_positions
    has_own_close = '_close_all_positions' in UnifiedOptimizer.__dict__
    print(f"  UnifiedOptimizer overrides _close_all_positions: {has_own_close}")
    
    # Trace if it's called
    close_called_opt = []
    if hasattr(optimizer, '_close_all_positions'):
        original_close_opt = optimizer._close_all_positions
        
        def traced_close_opt(final_date):
            print(f"\nOptimizer _close_all_positions called with date: {final_date}")
            print(f"  Positions before: {optimizer.current_positions}")
            result = original_close_opt(final_date)
            print(f"  Positions after: {optimizer.current_positions}")
            close_called_opt.append(True)
            return result
        
        optimizer._close_all_positions = traced_close_opt
    
    opt_result = optimizer.run_backtest(strategy, ['AAPL'])
    
    print(f"\nUnified Optimizer Results:")
    print(f"  close_all_positions called: {len(close_called_opt) > 0}")
    print(f"  Final positions: {optimizer.current_positions}")
    print(f"  Total trades: {opt_result['metrics']['total_trades']}")
    print(f"  Ledger trades: {len(optimizer.ledger.trades)}")
    print(f"  Ledger fills: {len(optimizer.ledger.fills)}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if len(orch.ledger.trades) > 0 and len(optimizer.ledger.trades) == 0:
        print("❌ UnifiedOptimizer not closing positions properly!")
    elif len(orch.ledger.trades) == 0:
        print("⚠️ Even base orchestrator not creating completed trades")
        print("   This suggests positions aren't being closed at all")
    else:
        print("✅ Both systems handling position closure correctly")
    
    return result, opt_result


if __name__ == "__main__":
    test_position_closure()