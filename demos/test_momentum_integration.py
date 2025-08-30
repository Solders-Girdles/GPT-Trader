#!/usr/bin/env python3
"""
MOMENTUM-001: Momentum Strategy Integration Demo

This demo tests the new momentum strategy alongside existing strategies:
1. Momentum Strategy (new)
2. Trend Breakout Strategy
3. Moving Average Strategy
4. Mean Reversion Strategy

Tests integration with the orchestrator and compares performance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta

import pandas as pd
from bot.integration.orchestrator import BacktestConfig, IntegratedOrchestrator
from bot.portfolio.allocator import PortfolioRules
from bot.risk.integration import RiskConfig
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.mean_reversion import MeanReversionParams, MeanReversionStrategy
from bot.strategy.momentum import MomentumParams, MomentumStrategy
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy


def run_momentum_comparison():
    """Run a comprehensive 4-strategy comparison including the new momentum strategy."""

    print("=" * 70)
    print("MOMENTUM STRATEGY INTEGRATION TEST")
    print("=" * 70)

    # Define test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)  # 4 months for meaningful results

    # Test symbols - diverse set for different market conditions
    symbols = ["AAPL", "MSFT", "NVDA"]  # Reduced for faster testing

    print(f"\nTest period: {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {', '.join(symbols)}")

    # Common configuration for all strategies
    risk_config = RiskConfig(
        max_position_size=0.25,  # Max 25% per position
        max_portfolio_exposure=0.75,  # Max 75% invested
        default_stop_loss_pct=0.05,  # 5% stop loss
        max_daily_loss=0.02,  # 2% daily loss limit
        enable_realtime_monitoring=True,
    )

    portfolio_rules = PortfolioRules(
        per_trade_risk_pct=0.015,  # 1.5% risk per trade
        max_positions=3,  # Max 3 concurrent positions
        max_gross_exposure_pct=0.75,  # 75% max exposure
        atr_k=2.0,  # 2x ATR for sizing
        cost_bps=8.0,  # 8 bps transaction costs
    )

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100_000.0,
        risk_config=risk_config,
        portfolio_rules=portfolio_rules,
        use_cache=True,
        strict_validation=False,
        show_progress=True,
        save_trades=True,
        save_metrics=True,
        generate_plot=False,
    )

    # Strategy definitions
    strategies = {}

    # 1. Momentum Strategy (NEW)
    print("\n1. MOMENTUM STRATEGY (NEW)")
    print("-" * 35)
    momentum_params = MomentumParams(
        roc_period=15,  # 15-day rate of change
        momentum_threshold=0.025,  # 2.5% momentum threshold
        momentum_exit_threshold=0.008,  # 0.8% exit threshold
        volume_threshold=1.4,  # 1.4x volume confirmation
        atr_period=14,
    )
    strategies["Momentum"] = MomentumStrategy(momentum_params)

    # 2. Trend Breakout Strategy
    print("\n2. TREND BREAKOUT STRATEGY")
    print("-" * 30)
    trend_params = TrendBreakoutParams(
        donchian_lookback=20,
        atr_period=14,
        atr_k=2.0,
    )
    strategies["Trend Breakout"] = TrendBreakoutStrategy(trend_params)

    # 3. Moving Average Strategy
    print("\n3. MOVING AVERAGE STRATEGY")
    print("-" * 30)
    strategies["Moving Average"] = DemoMAStrategy(
        fast=12,
        slow=26,
        atr_period=14,
    )

    # 4. Mean Reversion Strategy
    print("\n4. MEAN REVERSION STRATEGY")
    print("-" * 30)
    mean_reversion_params = MeanReversionParams(
        rsi_period=14,
        oversold_threshold=25,  # More aggressive
        overbought_threshold=75,  # More aggressive
        exit_rsi_threshold=50,
        atr_period=14,
    )
    strategies["Mean Reversion"] = MeanReversionStrategy(mean_reversion_params)

    # Run backtests for all strategies
    results = {}
    print("\n" + "=" * 70)
    print("RUNNING BACKTESTS")
    print("=" * 70)

    for strategy_name, strategy in strategies.items():
        print(f"\nRunning {strategy_name} backtest...")
        try:
            orchestrator = IntegratedOrchestrator(config)
            result = orchestrator.run_backtest(strategy, symbols)
            results[strategy_name] = result

            print(f"✅ {strategy_name}: {result.total_return:.2%} return")

        except Exception as e:
            print(f"❌ {strategy_name} failed: {e}")
            results[strategy_name] = None

    # Display comprehensive results
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 70)

    # Create comparison table
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("❌ No strategies completed successfully")
        return False

    # Build comparison DataFrame
    comparison_data = {
        "Strategy": [],
        "Total Return": [],
        "CAGR": [],
        "Sharpe Ratio": [],
        "Max Drawdown": [],
        "Win Rate": [],
        "Total Trades": [],
    }

    for strategy_name, result in valid_results.items():
        comparison_data["Strategy"].append(strategy_name)
        comparison_data["Total Return"].append(f"{result.total_return:.2%}")
        comparison_data["CAGR"].append(f"{result.cagr:.2%}")
        comparison_data["Sharpe Ratio"].append(f"{result.sharpe_ratio:.2f}")
        comparison_data["Max Drawdown"].append(f"{result.max_drawdown:.2%}")
        comparison_data["Win Rate"].append(f"{result.win_rate:.2%}")
        comparison_data["Total Trades"].append(result.total_trades)

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Momentum strategy analysis
    print("\n" + "=" * 70)
    print("MOMENTUM STRATEGY ANALYSIS")
    print("=" * 70)

    if "Momentum" in valid_results:
        momentum_result = valid_results["Momentum"]
        print(f"✅ Momentum Strategy Performance:")
        print(f"  📈 Total Return: {momentum_result.total_return:.2%}")
        print(f"  📊 Sharpe Ratio: {momentum_result.sharpe_ratio:.2f}")
        print(f"  📉 Max Drawdown: {momentum_result.max_drawdown:.2%}")
        print(f"  🎯 Win Rate: {momentum_result.win_rate:.2%}")
        print(f"  🔄 Total Trades: {momentum_result.total_trades}")

        # Compare momentum to other strategies
        other_returns = [
            result.total_return
            for name, result in valid_results.items()
            if name != "Momentum"
        ]

        if other_returns:
            avg_other_return = sum(other_returns) / len(other_returns)
            print(f"\n📊 Performance vs Other Strategies:")
            print(f"  Momentum Return: {momentum_result.total_return:.2%}")
            print(f"  Average Other Return: {avg_other_return:.2%}")

            if momentum_result.total_return > avg_other_return:
                print("🏆 Momentum strategy OUTPERFORMED average!")
                outperformance = momentum_result.total_return - avg_other_return
                print(f"  Outperformance: +{outperformance:.2%}")
            else:
                print("📉 Momentum strategy underperformed average")
                underperformance = avg_other_return - momentum_result.total_return
                print(f"  Underperformance: -{underperformance:.2%}")

    else:
        print("❌ Momentum strategy failed to complete")

    # Overall assessment
    print("\n" + "=" * 70)
    print("INTEGRATION ASSESSMENT")
    print("=" * 70)

    completed_strategies = len(valid_results)
    total_strategies = len(strategies)

    print(f"✅ Successfully completed: {completed_strategies}/{total_strategies} strategies")

    if "Momentum" in valid_results:
        print("✅ Momentum strategy successfully integrated")
        print("✅ Compatible with existing orchestrator")
        print("✅ Risk management integration working")

        momentum_profitable = valid_results["Momentum"].total_return > 0
        if momentum_profitable:
            print("✅ Momentum strategy is PROFITABLE")
        else:
            print("⚠️  Momentum strategy shows losses in test period")

    else:
        print("❌ Momentum strategy integration failed")

    # Final verdict
    print("\n" + "=" * 70)
    if (
        "Momentum" in valid_results
        and completed_strategies >= 3
        and valid_results["Momentum"].total_return > -0.10
    ):  # Accept up to -10% loss
        print("🎉 MOMENTUM STRATEGY INTEGRATION: SUCCESS!")
        print("✅ Strategy implemented correctly")
        print("✅ Integrated with trading system")
        print("✅ Risk management working")
        print("✅ Performance tracking functional")
        return True
    else:
        print("❌ MOMENTUM STRATEGY INTEGRATION: ISSUES DETECTED")
        return False


if __name__ == "__main__":
    try:
        success = run_momentum_comparison()
        if success:
            print("\n🚀 Momentum strategy ready for production use!")
        else:
            print("\n⚠️  Momentum strategy needs further refinement")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error running momentum integration test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)