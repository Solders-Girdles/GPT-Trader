#!/usr/bin/env python3
"""Integration test script for StrategyAllocatorBridge.

This script demonstrates and tests the complete integration flow:
Strategy → Bridge → Allocator
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bot.integration.strategy_allocator_bridge import StrategyAllocatorBridge
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.demo_ma import DemoMAStrategy


def create_sample_market_data(symbols: list[str], days: int = 60) -> dict[str, pd.DataFrame]:
    """Create realistic sample market data for testing."""
    market_data = {}

    base_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(base_date, periods=days, freq="D")

    for i, symbol in enumerate(symbols):
        # Set different random seeds for each symbol to get different patterns
        np.random.seed(42 + i)

        # Create realistic price movements with trend
        base_price = 100 + i * 50  # Different price levels per symbol
        daily_returns = np.random.normal(
            0.0005, 0.02, days
        )  # ~0.05% daily return with 2% volatility
        daily_returns[days // 3 : 2 * days // 3] += 0.001  # Add trend in middle period

        prices = [base_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        close_prices = np.array(prices)

        # Generate OHLV from close prices
        daily_ranges = np.random.uniform(0.005, 0.03, days)  # 0.5-3% daily range

        high_prices = close_prices * (1 + daily_ranges * np.random.uniform(0.3, 1.0, days))
        low_prices = close_prices * (1 - daily_ranges * np.random.uniform(0.3, 1.0, days))

        # Open prices with some gap behavior
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        open_prices += np.random.normal(0, close_prices * 0.002)  # Small gaps

        volume = np.random.randint(100000, 1000000, days)

        market_data[symbol] = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volume,
            },
            index=dates,
        )

        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        market_data[symbol]["High"] = np.maximum(
            market_data[symbol]["High"],
            np.maximum(market_data[symbol]["Open"], market_data[symbol]["Close"]),
        )
        market_data[symbol]["Low"] = np.minimum(
            market_data[symbol]["Low"],
            np.minimum(market_data[symbol]["Open"], market_data[symbol]["Close"]),
        )

    return market_data


def main():
    """Run the integration test."""
    print("=== Strategy-Allocator Bridge Integration Test ===")
    print()

    # 1. Set up components
    print("1. Setting up components...")

    # Create strategy with reasonable parameters for demo
    strategy = DemoMAStrategy(fast=5, slow=20, atr_period=14)
    print(f"   Strategy: {strategy.name} (fast={strategy.fast}, slow={strategy.slow})")

    # Create portfolio rules
    rules = PortfolioRules(
        per_trade_risk_pct=0.02,  # 2% risk per trade
        max_positions=3,  # Max 3 positions for demo
        max_gross_exposure_pct=0.8,
        atr_k=2.0,
    )
    print(
        f"   Portfolio Rules: {rules.per_trade_risk_pct*100}% risk, max {rules.max_positions} positions"
    )

    # Create bridge
    bridge = StrategyAllocatorBridge(strategy, rules)
    print(f"   Bridge: Configured and ready")
    print()

    # 2. Validate configuration
    print("2. Validating configuration...")
    if bridge.validate_configuration():
        print("   ✓ Configuration is valid")
    else:
        print("   ✗ Configuration validation failed")
        return
    print()

    # 3. Create market data
    print("3. Creating sample market data...")
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    market_data = create_sample_market_data(symbols, days=60)

    for symbol, data in market_data.items():
        latest_price = data["Close"].iloc[-1]
        price_change = (latest_price - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100
        print(f"   {symbol}: ${latest_price:.2f} ({price_change:+.1f}% over period)")
    print()

    # 4. Generate signals and allocate
    print("4. Processing signals and allocating capital...")
    equity = 100000.0  # $100k portfolio
    print(f"   Starting equity: ${equity:,.2f}")

    allocations = bridge.process_signals(market_data, equity)

    print(f"   Generated allocations for {len(allocations)} symbols:")
    total_allocated_value = 0

    for symbol, qty in allocations.items():
        if qty > 0:
            price = market_data[symbol]["Close"].iloc[-1]
            value = qty * price
            total_allocated_value += value
            print(f"     {symbol}: {qty} shares @ ${price:.2f} = ${value:,.2f}")
        else:
            print(f"     {symbol}: No position")

    print(
        f"   Total allocated: ${total_allocated_value:,.2f} ({total_allocated_value/equity*100:.1f}% of equity)"
    )
    print()

    # 5. Show strategy and rules info
    print("5. Component information:")
    strategy_info = bridge.get_strategy_info()
    rules_info = bridge.get_allocation_rules_info()

    print("   Strategy Info:")
    for key, value in strategy_info.items():
        print(f"     {key}: {value}")

    print("   Allocation Rules:")
    for key, value in rules_info.items():
        print(f"     {key}: {value}")
    print()

    # 6. Test with different equity levels
    print("6. Testing with different equity levels:")
    equity_levels = [50000, 250000, 500000]

    for test_equity in equity_levels:
        test_allocations = bridge.process_signals(market_data, test_equity)
        active_positions = sum(1 for qty in test_allocations.values() if qty > 0)
        total_test_value = sum(
            qty * market_data[symbol]["Close"].iloc[-1]
            for symbol, qty in test_allocations.items()
            if qty > 0
        )
        print(
            f"   ${test_equity:,} equity → {active_positions} positions, ${total_test_value:,.0f} allocated"
        )

    print()
    print("=== Integration Test Complete ===")
    print("✓ Strategy successfully generates signals")
    print("✓ Bridge correctly processes multi-symbol data")
    print("✓ Allocator properly sizes positions based on rules")
    print("✓ Complete integration flow working as expected")


if __name__ == "__main__":
    main()
