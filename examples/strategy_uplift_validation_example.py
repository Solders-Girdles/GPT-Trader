"""
Example: Strategy Uplift Validation

Demonstrates how to validate that an enhanced strategy beats the baseline
MA crossover across different market regimes and out-of-sample periods.

Phase 2 Exit Criteria:
- Enhanced must beat baseline in ALL 3 regimes (trend/range/high-vol)
- Enhanced must win majority (>=60%) of out-of-sample CV splits
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from bot_v2.features.live_trade.strategies.enhanced_strategy import (
    EnhancedStrategy,
    EnhancedStrategyConfig,
)
from bot_v2.features.live_trade.strategies.perps_baseline.config import StrategyConfig
from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.features.optimize.regime_detection import RegimeDetector
from bot_v2.features.optimize.strategy_comparison import StrategyComparator
from bot_v2.features.optimize.types_v2 import BacktestConfig


def create_sample_data(days: int = 180) -> pd.DataFrame:
    """Create sample price data with different regimes."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=days * 24, freq="1h")

    # Create data with three distinct phases
    n_bars = len(dates)
    third = n_bars // 3

    # Phase 1: Trending (first third)
    trend1 = np.linspace(40000, 48000, third)
    noise1 = np.cumsum(np.random.normal(0, 150, third))
    phase1 = trend1 + noise1

    # Phase 2: Ranging (middle third)
    range_base = 48000
    range_noise = np.random.normal(0, 800, third)
    phase2 = range_base + range_noise

    # Phase 3: High volatility trending (last third)
    trend3 = np.linspace(48000, 52000, third)
    high_vol_noise = np.random.normal(0, 1200, third)
    noise3 = np.cumsum(np.random.normal(0, 200, third))
    phase3 = trend3 + high_vol_noise + noise3

    # Combine phases
    close = np.concatenate([phase1, phase2, phase3])

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close * 0.9995,
            "high": close * 1.008,
            "low": close * 0.992,
            "close": close,
            "volume": np.random.uniform(500, 2000, len(dates)),
        }
    )


def main():
    """Run strategy uplift validation."""
    print("=" * 80)
    print("STRATEGY UPLIFT VALIDATION")
    print("=" * 80)
    print()

    # 1. Create sample data
    print("1. Creating sample data...")
    data = create_sample_data(days=180)
    print(f"   Data period: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
    print(f"   Total bars: {len(data)}")
    print()

    # 2. Configure baseline strategy
    print("2. Configuring baseline strategy...")
    baseline_config = StrategyConfig(
        short_ma_period=5,
        long_ma_period=20,
        position_fraction=0.20,
        target_leverage=3,
        enable_shorts=True,
        trailing_stop_pct=0.02,
    )
    baseline_strategy = BaselinePerpsStrategy(config=baseline_config, environment="validation")
    print(f"   Strategy: MA({baseline_config.short_ma_period}/{baseline_config.long_ma_period})")
    print(f"   Leverage: {baseline_config.target_leverage}x")
    print(f"   Position size: {baseline_config.position_fraction * 100}%")
    print()

    # 3. Configure enhanced strategy
    print("3. Configuring enhanced strategy...")
    enhanced_config = EnhancedStrategyConfig(
        # Same baseline parameters
        short_ma_period=5,
        long_ma_period=20,
        position_fraction=0.20,
        target_leverage=3,
        enable_shorts=True,
        trailing_stop_pct=0.02,
        # Enhancement parameters
        enable_volatility_filter=True,
        atr_period=14,
        atr_high_threshold_multiplier=2.0,
        atr_low_threshold_multiplier=0.5,
        enable_trend_filter=True,
        adx_period=14,
        adx_min_threshold=20.0,
        enable_dynamic_stops=False,  # Phase 2: keep simple
    )
    enhanced_strategy = EnhancedStrategy(config=enhanced_config, environment="validation")
    print(f"   Base strategy: Same as baseline")
    print(f"   + Volatility filter: ATR threshold {enhanced_config.atr_high_threshold_multiplier}x")
    print(f"   + Trend filter: ADX >= {enhanced_config.adx_min_threshold}")
    print()

    # 4. Configure backtest
    print("4. Configuring backtest...")
    backtest_config = BacktestConfig(
        initial_capital=Decimal("50000"),
        commission_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        enable_decision_logging=False,
    )
    print(f"   Initial capital: ${backtest_config.initial_capital}")
    print(f"   Commission: {float(backtest_config.commission_rate) * 100}%")
    print()

    # 5. Create comparator
    print("5. Initializing strategy comparator...")
    comparator = StrategyComparator(
        baseline_strategy=baseline_strategy,
        enhanced_strategy=enhanced_strategy,
        backtest_config=backtest_config,
    )
    print("   Comparator ready")
    print()

    # 6. Compare on full dataset
    print("=" * 80)
    print("FULL DATASET COMPARISON")
    print("=" * 80)
    print()

    full_result = comparator.compare_full_dataset(data=data, symbol="BTC-PERP")

    print("Results:")
    print(f"  Baseline - Return: {full_result.baseline_performance.total_return:.2%}, "
          f"Sharpe: {full_result.baseline_performance.sharpe_ratio:.2f}")
    print(f"  Enhanced - Return: {full_result.enhanced_performance.total_return:.2%}, "
          f"Sharpe: {full_result.enhanced_performance.sharpe_ratio:.2f}")
    print(f"  Winner: {'Enhanced ✅' if full_result.enhanced_wins() else 'Baseline ❌'}")
    print()

    # 7. Compare by regime
    print("=" * 80)
    print("REGIME-SPECIFIC COMPARISON")
    print("=" * 80)
    print()

    detector = RegimeDetector(
        adx_trend_threshold=25.0,
        adx_range_threshold=20.0,
        atr_volatility_multiplier=1.5,
        min_regime_bars=20,
    )

    regime_results = comparator.compare_by_regime(data=data, symbol="BTC-PERP", detector=detector)

    print(f"Found {len(regime_results)} regimes:")
    for regime, result in regime_results.items():
        print(f"\n  {regime.upper()}:")
        print(f"    Baseline - Return: {result.baseline_performance.total_return:.2%}, "
              f"Sharpe: {result.baseline_performance.sharpe_ratio:.2f}")
        print(f"    Enhanced - Return: {result.enhanced_performance.total_return:.2%}, "
              f"Sharpe: {result.enhanced_performance.sharpe_ratio:.2f}")
        print(f"    Winner: {'Enhanced ✅' if result.enhanced_wins() else 'Baseline ❌'}")

    regime_wins = sum(1 for r in regime_results.values() if r.enhanced_wins())
    regime_total = len(regime_results)
    print(f"\n  Enhanced wins {regime_wins}/{regime_total} regimes")
    print()

    # 8. Walk-forward cross-validation
    print("=" * 80)
    print("WALK-FORWARD CROSS-VALIDATION (Out-of-Sample)")
    print("=" * 80)
    print()

    cv_results = comparator.compare_walk_forward(data=data, symbol="BTC-PERP", n_splits=5)

    print(f"Ran {len(cv_results)} CV splits:")
    for result in cv_results:
        winner = "Enhanced ✅" if result.enhanced_wins() else "Baseline ❌"
        sharpe_imp = result.improvement_pct()["sharpe"]
        print(f"  Split {result.split_id}: {winner} "
              f"(Sharpe improvement: {sharpe_imp:+.2f}%)")

    cv_wins = sum(1 for r in cv_results if r.enhanced_wins())
    cv_total = len(cv_results)
    print(f"\n  Enhanced wins {cv_wins}/{cv_total} splits")
    print()

    # 9. Generate comprehensive report
    print("=" * 80)
    print("COMPREHENSIVE REPORT")
    print("=" * 80)
    print()

    report = comparator.generate_report(
        full_result=full_result,
        regime_results=regime_results,
        cv_results=cv_results,
    )

    print(report)

    # 10. Phase 2 exit criteria check
    print()
    print("=" * 80)
    print("PHASE 2 EXIT CRITERIA")
    print("=" * 80)
    print()

    # Criterion 1: Enhanced beats baseline in ALL regimes
    regime_criterion = regime_wins == regime_total
    print(f"1. Enhanced beats baseline across ALL regimes: {regime_wins}/{regime_total}")
    if regime_criterion:
        print("   ✅ PASSED")
    else:
        print("   ❌ FAILED - Must win all regimes")

    # Criterion 2: Enhanced wins majority of CV splits
    cv_criterion = cv_wins >= cv_total * 0.6
    print(f"2. Enhanced wins majority (>=60%) of out-of-sample splits: {cv_wins}/{cv_total}")
    if cv_criterion:
        print("   ✅ PASSED")
    else:
        print("   ❌ FAILED - Must win >=60% of splits")

    print()
    if regime_criterion and cv_criterion:
        print("✅✅✅ ALL EXIT CRITERIA PASSED ✅✅✅")
        print()
        print("Strategy uplift validated! Enhanced strategy beats baseline across:")
        print("  - All market regimes (trend/range/high-vol)")
        print("  - Majority of out-of-sample periods")
        print()
        print("Next steps:")
        print("  1. Combine with stress testing (Workstream 1)")
        print("  2. Verify INTX eligibility (Workstream 2)")
        print("  3. If all pass → Phase 2 complete → Ready for Phase 3")
    else:
        print("❌ SOME CRITERIA FAILED")
        print()
        print("Required actions:")
        print("  1. Review which regimes/splits failed")
        print("  2. Adjust enhancement parameters:")
        if not regime_criterion:
            print("     - ADX threshold (try lower, e.g., 15)")
            print("     - ATR thresholds (wider range)")
        if not cv_criterion:
            print("     - Re-run with different CV splits")
            print("     - Check if overfitting to full dataset")
        print("  3. Re-run validation until criteria pass")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
