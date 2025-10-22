"""
Example: Derivatives Stress Testing

Demonstrates how to run comprehensive stress tests on a derivatives trading strategy
to validate system resilience before live deployment.

Tests covered:
- Gap moves (overnight gaps, flash crashes)
- High volatility periods
- Funding rate shocks
- Liquidation scenarios
"""

from decimal import Decimal

import numpy as np
import pandas as pd

from bot_v2.features.live_trade.strategies.perps_baseline.config import StrategyConfig
from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.features.optimize.stress_validator import StressTestCriteria, StressTestValidator
from bot_v2.features.optimize.types_v2 import BacktestConfig


def create_sample_perps_data(days: int = 60) -> pd.DataFrame:
    """Create sample perpetuals price data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=days * 24, freq="1h")

    # Create realistic price movement with trend and volatility
    trend = np.linspace(40000, 48000, len(dates))
    volatility = np.random.normal(0, 800, len(dates))
    noise = np.cumsum(np.random.normal(0, 200, len(dates)))

    close = trend + volatility + noise

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
    """Run derivatives stress tests."""
    print("=" * 80)
    print("DERIVATIVES STRESS TESTING SUITE")
    print("=" * 80)
    print()

    # 1. Create sample data
    print("1. Creating sample perpetuals data...")
    data = create_sample_perps_data(days=60)
    print(f"   Data period: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
    print(f"   Total bars: {len(data)}")
    print()

    # 2. Configure strategy
    print("2. Configuring perpetuals strategy...")
    strategy_config = StrategyConfig(
        short_ma_period=5,
        long_ma_period=20,
        position_fraction=0.20,  # 20% of equity per trade
        target_leverage=5,  # 5x leverage
        enable_shorts=True,  # Allow shorts on perps
        trailing_stop_pct=0.02,  # 2% trailing stop
    )

    strategy = BaselinePerpsStrategy(config=strategy_config, environment="stress_test")
    print(f"   MA Periods: {strategy_config.short_ma_period}/{strategy_config.long_ma_period}")
    print(f"   Target Leverage: {strategy_config.target_leverage}x")
    print(f"   Position Size: {strategy_config.position_fraction * 100}% of equity")
    print()

    # 3. Configure stress test criteria
    print("3. Setting stress test criteria...")
    criteria = StressTestCriteria(
        max_liquidations_allowed=0,  # MUST avoid liquidation
        max_liquidation_warnings=5,  # Allow up to 5 warnings
        max_drawdown_pct=0.30,  # Max 30% drawdown
        max_drawdown_warning_pct=0.20,  # Warning at 20%
        max_margin_utilization=0.90,  # Max 90% margin used
        max_leverage=10.0,  # Max 10x leverage
        max_negative_return_pct=-0.50,  # Can't lose more than 50%
    )
    print("   Liquidations Allowed: 0 (MUST avoid)")
    print("   Max Drawdown: 30%")
    print("   Max Leverage: 10x")
    print()

    # 4. Configure backtest
    print("4. Configuring backtest parameters...")
    backtest_config = BacktestConfig(
        initial_capital=Decimal("50000"),
        commission_rate=Decimal("0.001"),  # 0.1%
        slippage_rate=Decimal("0.001"),  # 0.1% (higher for stress)
        enable_decision_logging=False,  # Disable for speed
    )
    print(f"   Initial Capital: ${backtest_config.initial_capital}")
    print(f"   Commission: {float(backtest_config.commission_rate) * 100}%")
    print(f"   Slippage: {float(backtest_config.slippage_rate) * 100}%")
    print()

    # 5. Create validator
    print("5. Initializing stress test validator...")
    validator = StressTestValidator(
        strategy=strategy,
        criteria=criteria,
        backtest_config=backtest_config,
    )
    print("   Validator ready")
    print()

    # 6. Run standard stress test suite
    print("=" * 80)
    print("RUNNING STRESS TESTS")
    print("=" * 80)
    print()

    results = validator.run_standard_suite(data=data, symbol="BTC-PERP")

    print()
    print("=" * 80)
    print("STRESS TEST RESULTS")
    print("=" * 80)
    print()

    # 7. Display results
    for i, result in enumerate(results, 1):
        print(f"Test {i}: {result.scenario_name}")
        print(f"   Type: {result.scenario_type.value}")
        print(f"   Status: {result.status.value.upper()}")
        print(f"   Metrics:")
        print(f"      Return: {result.metrics.get('total_return', 0):.2%}")
        print(f"      Max DD: {result.metrics.get('max_drawdown', 0):.2%}")
        print(f"      Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"      Liquidations: {result.metrics.get('liquidations', 0)}")
        print(f"      Warnings: {result.metrics.get('liquidation_warnings', 0)}")

        if result.failures:
            print(f"   ❌ Failures:")
            for failure in result.failures:
                print(f"      - {failure}")

        if result.warnings:
            print(f"   ⚠️  Warnings:")
            for warning in result.warnings:
                print(f"      - {warning}")

        print()

    # 8. Generate final report
    print()
    print(validator.generate_report())

    # 9. Exit criteria check
    print()
    print("=" * 80)
    print("PHASE 2 EXIT CRITERIA CHECK")
    print("=" * 80)
    print()

    if validator.passed_all():
        print("✅ ALL STRESS TESTS PASSED")
        print("✅ System meets Phase 2 exit criteria for derivatives readiness")
        print("✅ Safe to proceed with INTX eligibility verification")
        print()
        print("Next Steps:")
        print("   1. Verify INTX eligibility and permissions")
        print("   2. Run canary deployment on spot (if not already done)")
        print("   3. Enable derivatives with CONSERVATIVE limits")
        print("   4. Monitor for 48 hours before scaling")
    else:
        print("❌ SOME STRESS TESTS FAILED")
        print("❌ System NOT ready for derivatives deployment")
        print()
        print("Required Actions:")
        print("   1. Review failures above")
        print("   2. Adjust strategy parameters (MA periods, stops, position sizing)")
        print("   3. Adjust risk parameters (max leverage, liquidation buffer)")
        print("   4. Re-run stress tests until all pass")
        print()
        print("Common Fixes:")
        print("   - Reduce target_leverage (e.g., 5x → 3x)")
        print("   - Reduce position_fraction (e.g., 0.20 → 0.10)")
        print("   - Tighten trailing_stop_pct (e.g., 0.02 → 0.015)")
        print("   - Increase MA periods for less frequent trades")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
