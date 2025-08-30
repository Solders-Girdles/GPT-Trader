#!/usr/bin/env python3
"""Demonstration of Strategy-Allocator Bridge Integration.

This example shows how to use the StrategyAllocatorBridge to connect
trading strategies with portfolio allocation in a production-like setup.
"""

import pandas as pd
import yfinance as yf
from bot.integration.strategy_allocator_bridge import StrategyAllocatorBridge
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.demo_ma import DemoMAStrategy


def fetch_real_market_data(symbols: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    """Fetch real market data using yfinance."""
    market_data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if not data.empty:
                # Rename columns to match expected format
                data = data.rename(
                    columns={
                        "Open": "Open",
                        "High": "High",
                        "Low": "Low",
                        "Close": "Close",
                        "Volume": "Volume",
                    }
                )
                market_data[symbol] = data
            else:
                print(f"Warning: No data retrieved for {symbol}")

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    return market_data


def main():
    """Run the integration demonstration."""
    print("Strategy-Allocator Bridge Integration Demo")
    print("==========================================\n")

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    portfolio_equity = 100000.0  # $100k portfolio

    print("Portfolio Setup:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Equity: ${portfolio_equity:,.2f}")
    print()

    # 1. Set up strategy
    print("1. Setting up moving average strategy...")
    strategy = DemoMAStrategy(
        fast=10,  # 10-day fast MA
        slow=30,  # 30-day slow MA
        atr_period=14,  # 14-day ATR for position sizing
    )
    print(f"   Created {strategy.name} strategy (fast={strategy.fast}, slow={strategy.slow})")

    # 2. Set up portfolio rules
    print("\n2. Configuring portfolio rules...")
    rules = PortfolioRules(
        per_trade_risk_pct=0.015,  # 1.5% risk per trade
        max_positions=4,  # Max 4 positions
        max_gross_exposure_pct=0.75,  # Max 75% of equity deployed
        atr_k=2.5,  # 2.5x ATR for stop distance
        cost_bps=5.0,  # 5bps transaction costs
    )
    print(f"   Risk per trade: {rules.per_trade_risk_pct*100}%")
    print(f"   Max positions: {rules.max_positions}")
    print(f"   Max exposure: {rules.max_gross_exposure_pct*100}%")
    print(f"   ATR multiplier: {rules.atr_k}x")

    # 3. Create integration bridge
    print("\n3. Creating integration bridge...")
    bridge = StrategyAllocatorBridge(strategy, rules)

    # Validate configuration
    if not bridge.validate_configuration():
        print("   ERROR: Bridge configuration validation failed!")
        return
    print("   Bridge configured and validated successfully")

    # 4. Fetch market data
    print("\n4. Fetching market data...")
    try:
        market_data = fetch_real_market_data(symbols, period="3mo")
        print(f"   Retrieved data for {len(market_data)} symbols")

        # Show latest prices
        print("   Latest prices:")
        for symbol, data in market_data.items():
            if not data.empty:
                latest_price = data["Close"].iloc[-1]
                prev_price = data["Close"].iloc[-5]  # 5 days ago
                change_pct = (latest_price - prev_price) / prev_price * 100
                print(f"     {symbol}: ${latest_price:.2f} ({change_pct:+.1f}% 5d)")

    except Exception as e:
        print(f"   Error fetching data: {e}")
        print("   This demo requires internet connection for real data.")
        return

    if not market_data:
        print("   No market data available. Exiting.")
        return

    # 5. Process signals and generate allocations
    print("\n5. Processing signals and allocating capital...")

    try:
        allocations = bridge.process_signals(market_data, portfolio_equity)

        # Display results
        print("\n   Allocation Results:")
        print("   " + "=" * 50)

        total_allocated_value = 0
        active_positions = 0

        for symbol in symbols:
            qty = allocations.get(symbol, 0)

            if qty > 0:
                price = market_data[symbol]["Close"].iloc[-1]
                value = qty * price
                total_allocated_value += value
                active_positions += 1

                # Calculate position as % of portfolio
                position_pct = (value / portfolio_equity) * 100

                print(
                    f"   {symbol:6}: {qty:4d} shares @ ${price:7.2f} = ${value:9,.2f} ({position_pct:4.1f}%)"
                )
            else:
                print(f"   {symbol:6}: No position (no signal or filtered out)")

        print("   " + "-" * 50)
        print(f"   Total: {active_positions} positions, ${total_allocated_value:,.2f} allocated")
        print(f"   Cash remaining: ${portfolio_equity - total_allocated_value:,.2f}")
        print(f"   Portfolio utilization: {(total_allocated_value/portfolio_equity)*100:.1f}%")

        # 6. Show signal analysis
        print("\n6. Signal Analysis:")
        print("   " + "=" * 40)

        for symbol, data in market_data.items():
            if symbol in allocations and allocations[symbol] > 0:
                # Generate signals to analyze
                signals = strategy.generate_signals(data)

                if not signals.empty:
                    latest_signal = signals["signal"].iloc[-1]
                    latest_sma_fast = signals["sma_fast"].iloc[-1]
                    latest_sma_slow = signals["sma_slow"].iloc[-1]
                    latest_atr = signals["atr"].iloc[-1]

                    print(
                        f"   {symbol}: Signal={latest_signal:.0f}, Fast MA=${latest_sma_fast:.2f}, Slow MA=${latest_sma_slow:.2f}, ATR=${latest_atr:.2f}"
                    )

    except Exception as e:
        print(f"   Error processing signals: {e}")
        return

    print("\n" + "=" * 60)
    print("Integration Demo Complete!")
    print("\nKey takeaways:")
    print("• Strategy generates signals based on moving average crossovers")
    print("• Bridge seamlessly connects strategy signals to position allocation")
    print("• Portfolio rules control risk and position sizing automatically")
    print("• System handles real market data and multiple symbols efficiently")
    print("\nNext steps:")
    print("• Integrate with execution engine for actual trading")
    print("• Add more sophisticated strategies")
    print("• Implement dynamic risk management")
    print("• Add performance monitoring and reporting")


if __name__ == "__main__":
    main()
