"""
Example: Production-Parity Backtesting

This example demonstrates how to run a backtest using the production strategy code,
ensuring perfect alignment with live trading decisions.
"""

from decimal import Decimal

import numpy as np
import pandas as pd
from gpt_trader.features.live_trade.strategies.perps_baseline.config import StrategyConfig
from gpt_trader.features.optimize.backtest_engine import run_backtest_production
from gpt_trader.features.optimize.types_v2 import BacktestConfig

from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy


def create_sample_data(days: int = 30) -> pd.DataFrame:
    """Create sample price data for demonstration."""
    dates = pd.date_range("2024-01-01", periods=days * 24, freq="1h")

    # Create trending data with volatility
    np.random.seed(42)
    trend = np.linspace(40000, 45000, len(dates))
    volatility = np.random.normal(0, 500, len(dates))
    close = trend + volatility

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.uniform(100, 1000, len(dates)),
        }
    )


def main() -> None:
    """Run production-parity backtest example."""

    # 1. Create sample historical data
    print("Creating sample data...")
    data = create_sample_data(days=30)
    print(f"  Data period: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
    print(f"  Total bars: {len(data)}")

    # 2. Configure strategy (same config you'd use in production)
    print("\nConfiguring strategy...")
    strategy_config = StrategyConfig(
        short_ma_period=5,
        long_ma_period=20,
        position_fraction=0.1,  # 10% of equity per trade
        enable_shorts=False,  # Long only for spot
        trailing_stop_pct=0.01,  # 1% trailing stop
    )

    # 3. Create strategy instance
    strategy = BaselinePerpsStrategy(config=strategy_config, environment="backtest")

    # 4. Configure backtest parameters
    backtest_config = BacktestConfig(
        initial_capital=Decimal("10000"),
        commission_rate=Decimal("0.001"),  # 0.1% commission
        slippage_rate=Decimal("0.0005"),  # 0.05% slippage
        enable_decision_logging=True,
        log_directory="backtesting/decision_logs",
    )

    # 5. Run backtest
    print("\nRunning backtest...")
    result = run_backtest_production(
        strategy=strategy,
        data=data,
        symbol="BTC-USD",
        config=backtest_config,
    )

    # 6. Display results
    print("\n" + "=" * 80)
    print(result.summary())
    print("=" * 80)

    # 7. Detailed analysis
    print("\nDecision Analysis:")
    print(f"  Total decisions: {len(result.decisions)}")

    # Count decisions by action
    from collections import Counter

    action_counts = Counter(d.decision.action.value for d in result.decisions)
    for action, count in action_counts.items():
        print(f"  {action.upper()}: {count}")

    # Analyze filled trades
    filled_decisions = [d for d in result.decisions if d.execution.filled]
    if filled_decisions:
        print(f"\n  Filled trades: {len(filled_decisions)}")

        total_commission = sum(float(d.execution.commission or 0) for d in filled_decisions)
        print(f"  Total commission: ${total_commission:.2f}")

        avg_slippage = sum(float(d.execution.slippage or 0) for d in filled_decisions) / len(
            filled_decisions
        )
        print(f"  Average slippage: ${avg_slippage:.2f}")

    # 8. Show equity curve
    print("\nEquity Curve (sampled):")
    equity_curve = result.equity_curve
    sample_points = [0, len(equity_curve) // 4, len(equity_curve) // 2, -1]

    for i in sample_points:
        timestamp, equity = equity_curve[i]
        print(f"  {timestamp.strftime('%Y-%m-%d %H:%M')}: ${equity:,.2f}")

    # 9. Decision log location
    if backtest_config.enable_decision_logging:
        log_dir = backtest_config.log_directory
        print(f"\nDecision log saved to: {log_dir}/")
        print(f"  Run ID: {result.run_id}")

    # 10. Example: Inspect a specific decision
    if result.decisions:
        print("\nSample Decision (first trade):")
        first_trade = next((d for d in result.decisions if d.execution.filled), None)

        if first_trade:
            print(f"  Timestamp: {first_trade.context.timestamp}")
            print(f"  Action: {first_trade.decision.action.value}")
            print(f"  Reason: {first_trade.decision.reason}")
            print(f"  Quantity: {first_trade.decision.quantity}")
            print(f"  Fill Price: ${first_trade.execution.fill_price}")
            print(f"  Commission: ${first_trade.execution.commission}")


if __name__ == "__main__":
    main()
