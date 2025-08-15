#!/usr/bin/env python3
"""
Mean Reversion Strategy Demo

This demo shows the new Mean Reversion strategy using RSI for mean reversion trading.
Compares performance against existing strategies.
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
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy


def run_mean_reversion_demo() -> bool:
    """Run a mean reversion strategy demo comparing with existing strategies."""

    print("=" * 60)
    print("MEAN REVERSION STRATEGY DEMONSTRATION")
    print("=" * 60)

    # Define test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months

    # Select symbols for testing (mix of volatile and stable stocks)
    symbols = ["AAPL", "TSLA", "SPY"]  # Tech growth, volatile, and stable index

    # Configure risk management
    risk_config = RiskConfig(
        max_position_size=0.30,  # Max 30% per position
        max_portfolio_exposure=0.80,  # Max 80% invested
        default_stop_loss_pct=0.05,  # 5% stop loss
        max_daily_loss=0.03,  # 3% daily loss limit
        enable_realtime_monitoring=True,
    )

    # Portfolio allocation rules
    portfolio_rules = PortfolioRules(
        per_trade_risk_pct=0.015,  # 1.5% risk per trade
        max_positions=3,  # Max 3 concurrent positions
        max_gross_exposure_pct=0.80,  # 80% max exposure
        atr_k=2.0,  # 2x ATR for sizing
        cost_bps=10.0,  # 10 bps transaction costs
    )

    # Backtest configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=50_000.0,
        risk_config=risk_config,
        portfolio_rules=portfolio_rules,
        use_cache=True,
        strict_validation=False,
        show_progress=True,
        save_trades=True,
        save_metrics=True,
        generate_plot=False,
    )

    print(f"\nRunning backtests from {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial capital: ${config.initial_capital:,.0f}")

    # Strategy 1: Mean Reversion (new strategy)
    print("\n1. MEAN REVERSION STRATEGY")
    print("-" * 30)

    # Optimized mean reversion parameters
    mr_params = MeanReversionParams(
        rsi_period=14,
        oversold_threshold=25,  # More aggressive oversold
        overbought_threshold=75,  # More aggressive overbought
        atr_period=14,
        exit_rsi_threshold=50,  # Exit when RSI returns to neutral
    )

    mr_strategy = MeanReversionStrategy(mr_params)
    print(f"Parameters: {mr_strategy}")

    orchestrator1 = IntegratedOrchestrator(config)
    mr_results = orchestrator1.run_backtest(mr_strategy, symbols)

    print("\n📊 MEAN REVERSION RESULTS:")
    print(f"  Total Return: {mr_results.total_return:.2%}")
    print(f"  CAGR: {mr_results.cagr:.2%}")
    print(f"  Sharpe Ratio: {mr_results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {mr_results.max_drawdown:.2%}")
    print(f"  Win Rate: {mr_results.win_rate:.2%}")
    print(f"  Total Trades: {mr_results.total_trades}")

    # Strategy 2: Trend Breakout (for comparison)
    print("\n2. TREND BREAKOUT STRATEGY (Comparison)")
    print("-" * 40)

    trend_params = TrendBreakoutParams(
        donchian_lookback=20,
        atr_period=14,
        atr_k=2.0,
    )

    trend_strategy = TrendBreakoutStrategy(trend_params)
    orchestrator2 = IntegratedOrchestrator(config)
    trend_results = orchestrator2.run_backtest(trend_strategy, symbols)

    print("\n📊 TREND BREAKOUT RESULTS:")
    print(f"  Total Return: {trend_results.total_return:.2%}")
    print(f"  CAGR: {trend_results.cagr:.2%}")
    print(f"  Sharpe Ratio: {trend_results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {trend_results.max_drawdown:.2%}")
    print(f"  Win Rate: {trend_results.win_rate:.2%}")
    print(f"  Total Trades: {trend_results.total_trades}")

    # Strategy 3: Moving Average (for comparison)
    print("\n3. MOVING AVERAGE STRATEGY (Comparison)")
    print("-" * 39)

    ma_strategy = DemoMAStrategy(fast=10, slow=30, atr_period=14)
    orchestrator3 = IntegratedOrchestrator(config)
    ma_results = orchestrator3.run_backtest(ma_strategy, symbols)

    print("\n📊 MOVING AVERAGE RESULTS:")
    print(f"  Total Return: {ma_results.total_return:.2%}")
    print(f"  CAGR: {ma_results.cagr:.2%}")
    print(f"  Sharpe Ratio: {ma_results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {ma_results.max_drawdown:.2%}")
    print(f"  Win Rate: {ma_results.win_rate:.2%}")
    print(f"  Total Trades: {ma_results.total_trades}")

    # Strategy comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Total Return",
                "CAGR",
                "Sharpe Ratio",
                "Max Drawdown",
                "Win Rate",
                "Total Trades",
            ],
            "Mean Reversion": [
                f"{mr_results.total_return:.2%}",
                f"{mr_results.cagr:.2%}",
                f"{mr_results.sharpe_ratio:.2f}",
                f"{mr_results.max_drawdown:.2%}",
                f"{mr_results.win_rate:.2%}",
                mr_results.total_trades,
            ],
            "Trend Breakout": [
                f"{trend_results.total_return:.2%}",
                f"{trend_results.cagr:.2%}",
                f"{trend_results.sharpe_ratio:.2f}",
                f"{trend_results.max_drawdown:.2%}",
                f"{trend_results.win_rate:.2%}",
                trend_results.total_trades,
            ],
            "Moving Average": [
                f"{ma_results.total_return:.2%}",
                f"{ma_results.cagr:.2%}",
                f"{ma_results.sharpe_ratio:.2f}",
                f"{ma_results.max_drawdown:.2%}",
                f"{ma_results.win_rate:.2%}",
                ma_results.total_trades,
            ],
        }
    )

    print(comparison.to_string(index=False))

    # Analysis
    print("\n" + "=" * 70)
    print("STRATEGY ANALYSIS")
    print("=" * 70)

    strategies = {
        "Mean Reversion": mr_results,
        "Trend Breakout": trend_results,
        "Moving Average": ma_results,
    }

    # Find best performing strategy
    best_return = max(strategies.items(), key=lambda x: x[1].total_return)
    best_sharpe = max(strategies.items(), key=lambda x: x[1].sharpe_ratio)
    best_drawdown = min(strategies.items(), key=lambda x: abs(x[1].max_drawdown))

    print(f"🏆 Best Total Return: {best_return[0]} ({best_return[1].total_return:.2%})")
    print(f"📈 Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")
    print(f"🛡️  Best Drawdown: {best_drawdown[0]} ({best_drawdown[1].max_drawdown:.2%})")

    # Mean reversion strategy analysis
    print("\n📊 MEAN REVERSION STRATEGY ANALYSIS:")
    if mr_results.total_return > 0:
        print("✅ Mean reversion strategy is profitable")
        if mr_results.sharpe_ratio > 1.0:
            print("✅ Excellent risk-adjusted returns")
        elif mr_results.sharpe_ratio > 0.5:
            print("⚠️  Good risk-adjusted returns")
        else:
            print("❌ Poor risk-adjusted returns")
    else:
        print("❌ Mean reversion strategy lost money")

    print("\n📈 STRATEGY CHARACTERISTICS:")
    print("  • Mean Reversion works best in: Range-bound, volatile markets")
    print("  • Trend Breakout works best in: Strong trending markets")
    print("  • Moving Average works best in: Moderate trend markets")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE - Mean Reversion Strategy Successfully Added!")
    print("=" * 70)

    return mr_results.total_return > 0


if __name__ == "__main__":
    try:
        success = run_mean_reversion_demo()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error running mean reversion demo: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
