#!/usr/bin/env python3
"""
Paper Trading Example for GPT-Trader

This example demonstrates how to run paper trading with the GPT-Trader platform.
Make sure you have set up your Alpaca API credentials in your environment variables.

Environment Variables Required:
- ALPACA_API_KEY_ID: Your Alpaca API key
- ALPACA_API_SECRET_KEY: Your Alpaca secret key

Usage:
    python examples/paper_trading_example.py
"""

import asyncio

from bot.config import settings

# Run from project root with poetry so package imports work
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.live.trading_engine import LiveTradingEngine
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy


async def run_paper_trading_example():
    """Run a paper trading example with a small set of symbols."""

    # Check for Alpaca credentials
    if not settings.alpaca.api_key_id or not settings.alpaca.api_secret_key:
        print("❌ Alpaca credentials not found!")
        print("Please set the following environment variables:")
        print("  ALPACA_API_KEY_ID=your_api_key")
        print("  ALPACA_API_SECRET_KEY=your_secret_key")
        return

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Small set for demo
    risk_pct = 0.5  # 0.5% risk per trade
    max_positions = 3  # Limit positions for demo
    rebalance_interval = 300  # 5 minutes

    print("🚀 Starting Paper Trading Example")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"⚠️  Risk per trade: {risk_pct}%")
    print(f"📈 Max positions: {max_positions}")
    print(f"⏰ Rebalance interval: {rebalance_interval} seconds")
    print()

    try:
        # Initialize broker
        print("🔗 Connecting to Alpaca Paper Trading...")
        broker = AlpacaPaperBroker(
            api_key=settings.alpaca.api_key_id,
            secret_key=settings.alpaca.api_secret_key,
            base_url=settings.alpaca.paper_base_url,
        )

        # Get account information
        account = broker.get_account()
        print(f"💰 Account: {account.account_number}")
        print(f"💵 Cash: ${account.cash:,.2f}")
        print(f"📊 Portfolio Value: ${account.portfolio_value:,.2f}")
        print(f"💳 Buying Power: ${account.buying_power:,.2f}")
        print()

        # Initialize strategy
        print("🧠 Initializing Trend Breakout Strategy...")
        strategy = TrendBreakoutStrategy(
            TrendBreakoutParams(
                donchian_lookback=55,
                atr_period=20,
                atr_k=2.0,
            )
        )

        # Initialize portfolio rules
        rules = PortfolioRules(
            per_trade_risk_pct=risk_pct / 100.0,
            atr_k=2.0,
            max_positions=max_positions,
            max_gross_exposure_pct=0.60,
            cost_bps=5.0,  # 5 bps transaction cost
        )

        # Initialize trading engine
        print("⚙️  Initializing Trading Engine...")
        engine = LiveTradingEngine(
            broker=broker,
            strategy=strategy,
            rules=rules,
            symbols=symbols,
            rebalance_interval=rebalance_interval,
            max_positions=max_positions,
        )

        print("✅ Paper trading engine ready!")
        print("📈 Starting trading loop...")
        print("Press Ctrl+C to stop")
        print("-" * 50)

        # Start the trading engine
        await engine.start()

    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal, stopping...")
        await engine.stop()
        print("✅ Paper trading stopped safely")

        # Print final summary
        summary = engine.get_trading_summary()
        print("\n📊 Final Trading Summary:")
        print(f"   Trading decisions: {summary['trading_decisions']}")
        print(f"   Pending orders: {summary['pending_orders']}")

        portfolio_summary = summary["portfolio_summary"]
        print(f"   Portfolio value: ${portfolio_summary['portfolio_value']:,.2f}")
        print(f"   Unrealized P&L: ${portfolio_summary['unrealized_pl']:,.2f}")
        print(f"   Position count: {portfolio_summary['position_count']}")

    except Exception as e:
        print(f"❌ Error in paper trading: {e}")
        raise


def main():
    """Main entry point."""
    print("🤖 GPT-Trader Paper Trading Example")
    print("=" * 50)

    # Run the example
    asyncio.run(run_paper_trading_example())


if __name__ == "__main__":
    main()
