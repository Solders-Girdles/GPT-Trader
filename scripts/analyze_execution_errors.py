#!/usr/bin/env python3
"""
Analyze the 13 execution errors reported in full optimization runs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from bot.integration.unified_optimizer import UnifiedOptimizer, UnifiedOptimizationConfig
from bot.strategy.mean_reversion import MeanReversionStrategy
import logging

# Capture ALL logs including errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def capture_errors():
    """Run optimization and capture all errors."""
    
    print("="*80)
    print("CAPTURING EXECUTION ERRORS")
    print("="*80)
    
    # Configuration that previously showed 13 errors
    config = UnifiedOptimizationConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),  # 6 months like before
        initial_capital=10000,
        quiet_mode=False,
        auto_apply_optimal_params=True,
        apply_signal_filters=True,
        use_regime_detection=True,
        use_trailing_stops=True,
        use_realistic_costs=True,
        log_optimization_actions=True
    )
    
    optimizer = UnifiedOptimizer(config)
    strategy = MeanReversionStrategy()
    
    print("\nRunning full optimization to capture errors...")
    result = optimizer.run_backtest(strategy, ['AAPL'])
    
    # Check for errors in result
    errors = result.get('errors', [])
    warnings = result.get('warnings', [])
    
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    print(f"\nTotal Errors: {len(errors)}")
    if errors:
        print("\nErrors found:")
        for i, error in enumerate(errors, 1):
            print(f"\n{i}. {error}")
    else:
        print("No errors recorded in result")
    
    print(f"\nTotal Warnings: {len(warnings)}")
    if warnings:
        print("\nWarnings found:")
        for i, warning in enumerate(warnings[:5], 1):  # First 5
            print(f"{i}. {warning}")
    
    # Check optimization stats for anomalies
    opt_stats = result.get('optimization', {})
    print("\n" + "="*80)
    print("OPTIMIZATION STATISTICS")
    print("="*80)
    print(f"Signals Generated: {opt_stats.get('signals_generated', 0)}")
    print(f"Signals Filtered: {opt_stats.get('signals_filtered', 0)}")
    print(f"Regime Changes: {opt_stats.get('regime_changes', 0)}")
    print(f"Trailing Stops Hit: {opt_stats.get('trailing_stops_hit', 0)}")
    print(f"Profit Targets Hit: {opt_stats.get('profit_targets_hit', 0)}")
    print(f"Transaction Costs: ${opt_stats.get('total_transaction_costs', 0):.2f}")
    
    # Check for NaN or inf values
    metrics = result.get('metrics', {})
    print("\n" + "="*80)
    print("METRICS CHECK")
    print("="*80)
    
    import math
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if math.isnan(value):
                print(f"⚠️ {key}: NaN")
            elif math.isinf(value):
                print(f"⚠️ {key}: Infinity")
            else:
                print(f"✓ {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    return result, errors


def test_minimal_config():
    """Test with minimal configuration to isolate error source."""
    
    print("\n" + "="*80)
    print("TESTING MINIMAL CONFIGURATION")
    print("="*80)
    
    config = UnifiedOptimizationConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),  # Just 1 month
        initial_capital=10000,
        quiet_mode=False,
        auto_apply_optimal_params=False,  # Disable everything
        apply_signal_filters=False,
        use_regime_detection=False,
        use_trailing_stops=False,
        use_realistic_costs=False
    )
    
    optimizer = UnifiedOptimizer(config)
    strategy = MeanReversionStrategy()
    
    result = optimizer.run_backtest(strategy, ['AAPL'])
    
    errors = result.get('errors', [])
    print(f"\nMinimal config errors: {len(errors)}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    return len(errors)


def main():
    """Analyze execution errors."""
    
    # Test minimal first
    minimal_errors = test_minimal_config()
    
    # Then test full
    result, errors = capture_errors()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Minimal configuration: {minimal_errors} errors")
    print(f"Full optimization: {len(errors)} errors")
    
    if len(errors) > minimal_errors:
        print("\n⚠️ Optimization components are introducing errors")
    elif len(errors) > 0:
        print("\n⚠️ Base system has errors that need fixing")
    else:
        print("\n✅ No errors detected in this run")
    
    # Check if the 13 errors might have been from a specific component
    if len(errors) == 0:
        print("\nNote: The 13 errors reported earlier may have been:")
        print("1. Already fixed by our changes")
        print("2. Intermittent/data-dependent")
        print("3. From a different configuration")
        print("4. Warnings miscounted as errors")


if __name__ == "__main__":
    main()