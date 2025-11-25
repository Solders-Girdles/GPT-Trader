"""
Example: Running a backtest with the simulation harness.

This example demonstrates how to:
1. Set up a simulated broker
2. Configure historical data fetching
3. Run a backtest with the strategy coordinator
4. Generate performance reports
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from gpt_trader.backtesting import ClockedBarRunner, SimulatedBroker
from gpt_trader.backtesting.data import CoinbaseHistoricalFetcher, HistoricalDataManager
from gpt_trader.backtesting.types import ClockSpeed, FeeTier
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.brokerages.core.interfaces import MarketType, Product


async def run_backtest_example() -> None:
    """Run a simple backtest example."""

    # 1. Initialize Coinbase client for data fetching
    # Note: For backtesting, you can use read-only API keys
    client = CoinbaseClient(
        api_key="your_api_key",
        api_secret="your_api_secret",
    )

    # 2. Set up historical data manager
    data_dir = Path("data/backtest_cache")
    fetcher = CoinbaseHistoricalFetcher(client=client, rate_limit_rps=10)
    data_manager = HistoricalDataManager(fetcher=fetcher, cache_dir=data_dir)

    # 3. Create simulated broker
    broker = SimulatedBroker(
        initial_equity_usd=Decimal("100000"),  # Start with $100k
        fee_tier=FeeTier.TIER_2,  # $50K-$100K volume tier
        slippage_bps={
            "BTC-PERP-USDC": Decimal("2"),  # 2 bps slippage for BTC
            "ETH-PERP-USDC": Decimal("2"),  # 2 bps slippage for ETH
        },
        spread_impact_pct=Decimal("0.5"),  # Apply 50% of spread
        enable_funding_pnl=True,  # Track funding for perps
    )

    # 4. Register products with broker
    btc_perp = Product(
        symbol="BTC-PERP-USDC",
        base_asset="BTC",
        quote_asset="USDC",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.0001"),
        step_size=Decimal("0.0001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=10,
        funding_rate=Decimal("0.0001"),  # 0.01% per 8 hours
    )

    eth_perp = Product(
        symbol="ETH-PERP-USDC",
        base_asset="ETH",
        quote_asset="USDC",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=10,
        funding_rate=Decimal("0.0001"),
    )

    broker.register_product(btc_perp)
    broker.register_product(eth_perp)

    # 5. Set up bar runner
    symbols = ["BTC-PERP-USDC", "ETH-PERP-USDC"]
    runner = ClockedBarRunner(
        data_provider=data_manager,
        symbols=symbols,
        granularity="FIVE_MINUTE",  # 5-minute bars
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),  # One month backtest
        clock_speed=ClockSpeed.INSTANT,  # Run as fast as possible
    )

    # 6. Run backtest
    print("Starting backtest...")
    print(f"Period: {runner.start_date} to {runner.end_date}")
    print(f"Symbols: {symbols}")
    print(f"Initial Equity: ${broker._initial_equity:,.2f}")
    print("-" * 60)

    bars_processed = 0

    async for bar_time, bars, quotes in runner.run():
        # Update broker with current market data
        broker.update_market_data(bar_time, bars, quotes)

        # Here you would normally call:
        # await strategy_coordinator.run_cycle()
        #
        # For this example, we'll simulate a simple strategy:
        # - Buy BTC if we don't have a position
        # - Hold for 10 bars, then sell

        # Simple buy-and-hold example
        positions = broker.list_positions()
        btc_position = next((p for p in positions if p.symbol == "BTC-PERP-USDC"), None)

        if btc_position is None and bars_processed == 10:
            # Buy 0.01 BTC
            print(f"[{bar_time}] Buying 0.01 BTC at {bars['BTC-PERP-USDC'].close}")
            try:
                broker.place_order(
                    symbol="BTC-PERP-USDC",
                    side="BUY",
                    order_type="MARKET",
                    quantity="0.01",
                )
            except Exception as e:
                print(f"Order failed: {e}")

        elif btc_position is not None and bars_processed == 100:
            # Sell position
            print(f"[{bar_time}] Selling BTC position at {bars['BTC-PERP-USDC'].close}")
            try:
                broker.place_order(
                    symbol="BTC-PERP-USDC",
                    side="SELL",
                    order_type="MARKET",
                    quantity=str(btc_position.quantity),
                )
            except Exception as e:
                print(f"Order failed: {e}")

        bars_processed += 1

        # Print progress every 100 bars
        if bars_processed % 100 == 0:
            equity = broker.get_equity()
            print(
                f"Progress: {runner.progress_pct:.1f}% | Bars: {bars_processed} | Equity: ${equity:,.2f}"
            )

    # 7. Generate performance report
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    report = broker.generate_report()

    print(f"\nPeriod: {report.start_date} to {report.end_date}")
    print(f"Duration: {report.duration_days} days")
    print("\nEquity:")
    print(f"  Initial: ${report.initial_equity:,.2f}")
    print(f"  Final:   ${report.final_equity:,.2f}")
    print(f"  Return:  {report.total_return:+.2f}% (${report.total_return_usd:+,.2f})")
    print("\nPnL Breakdown:")
    print(f"  Realized:    ${report.realized_pnl:+,.2f}")
    print(f"  Unrealized:  ${report.unrealized_pnl:+,.2f}")
    print(f"  Funding:     ${report.funding_pnl:+,.2f}")
    print(f"  Fees:        ${report.fees_paid:,.2f}")
    print("\nTrade Statistics:")
    print(f"  Total Trades:   {report.total_trades}")
    print(f"  Winning:        {report.winning_trades}")
    print(f"  Losing:         {report.losing_trades}")
    print(f"  Win Rate:       {report.win_rate:.1f}%")
    print("\nRisk Metrics:")
    print(f"  Max Drawdown:   {report.max_drawdown:.2f}% (${report.max_drawdown_usd:,.2f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(run_backtest_example())
