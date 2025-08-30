#!/usr/bin/env python3
"""
Debug why trades aren't being recorded in the ledger.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from bot.integration.orchestrator import IntegratedOrchestrator, BacktestConfig
from bot.strategy.mean_reversion import MeanReversionStrategy
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_base_orchestrator():
    """Test that the base IntegratedOrchestrator records trades correctly."""
    
    print("="*80)
    print("TESTING BASE ORCHESTRATOR TRADE RECORDING")
    print("="*80)
    
    # Simple configuration
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),  # Short period
        initial_capital=10000,
        quiet_mode=False
    )
    
    orchestrator = IntegratedOrchestrator(config)
    strategy = MeanReversionStrategy()
    
    print(f"\nBefore Backtest:")
    print(f"  Ledger entries: {len(orchestrator.ledger.to_trades_dataframe())}")
    
    # Run backtest
    result = orchestrator.run_backtest(strategy, ['AAPL'])
    
    print(f"\nAfter Backtest:")
    print(f"  Ledger entries: {len(orchestrator.ledger.to_trades_dataframe())}")
    print(f"  Result total_trades: {result.total_trades}")
    print(f"  Result total_return: {result.total_return*100:.2f}%")
    print(f"  Final equity: ${orchestrator.current_equity:,.2f}")
    
    # Check ledger directly
    trades_df = orchestrator.ledger.to_trades_dataframe()
    if not trades_df.empty:
        print(f"\nFirst few trades from ledger:")
        print(trades_df.head())
    else:
        print("\nNo trades in ledger!")
    
    return result


def test_unified_optimizer():
    """Test that UnifiedOptimizer records trades correctly."""
    
    print("\n" + "="*80)
    print("TESTING UNIFIED OPTIMIZER TRADE RECORDING")
    print("="*80)
    
    from bot.integration.unified_optimizer import UnifiedOptimizer, UnifiedOptimizationConfig
    
    # Simple configuration - NO optimizations
    config = UnifiedOptimizationConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        initial_capital=10000,
        quiet_mode=False,
        auto_apply_optimal_params=False,  # Disable all optimizations
        apply_signal_filters=False,
        use_regime_detection=False,
        use_trailing_stops=False,
        use_realistic_costs=False  # This should use parent's _execute_trades
    )
    
    optimizer = UnifiedOptimizer(config)
    strategy = MeanReversionStrategy()
    
    print(f"\nBefore Backtest:")
    print(f"  Ledger entries: {len(optimizer.ledger.to_trades_dataframe())}")
    
    # Run backtest
    result = optimizer.run_backtest(strategy, ['AAPL'])
    
    print(f"\nAfter Backtest:")
    print(f"  Ledger entries: {len(optimizer.ledger.to_trades_dataframe())}")
    print(f"  Result total_trades: {result['metrics']['total_trades']}")
    print(f"  Result total_return: {result['metrics']['total_return']:.2f}%")
    print(f"  Final equity: ${optimizer.current_equity:,.2f}")
    
    # Check ledger directly
    trades_df = optimizer.ledger.to_trades_dataframe()
    if not trades_df.empty:
        print(f"\nFirst few trades from ledger:")
        print(trades_df.head())
    else:
        print("\nNo trades in ledger!")
    
    # Check if positions were taken
    print(f"\nCurrent positions: {optimizer.current_positions}")
    
    return result


def main():
    """Run both tests and compare."""
    
    print("TRADE RECORDING DEBUG")
    print("="*80)
    
    # Test base orchestrator
    base_result = test_base_orchestrator()
    
    # Test unified optimizer
    unified_result = test_unified_optimizer()
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Base Orchestrator trades: {base_result.total_trades}")
    print(f"Unified Optimizer trades: {unified_result['metrics']['total_trades']}")
    
    if base_result.total_trades > 0 and unified_result['metrics']['total_trades'] == 0:
        print("\n❌ BUG CONFIRMED: UnifiedOptimizer not recording trades!")
        print("   Even though positions are being taken")
    else:
        print("\n✅ Both systems recording trades correctly")


if __name__ == "__main__":
    main()