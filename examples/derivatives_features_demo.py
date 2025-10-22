"""
Demonstration of October 2025 Derivatives Features

This script demonstrates the new optional features added for derivatives trading:
1. Funding accrual history tracking
2. Liquidation price calculator
3. Scaled order placement

Note: This is a demonstration/example file, not production code.
"""

from datetime import datetime, timedelta
from decimal import Decimal

from bot_v2.features.brokerages.coinbase.utilities import (
    FundingCalculator,
    FundingEvent,
    LiquidationCalculator,
)
from bot_v2.features.brokerages.core.interfaces import OrderSide


def demo_funding_history():
    """Demonstrate funding accrual history tracking."""
    print("\n" + "=" * 70)
    print("1. FUNDING ACCRUAL HISTORY TRACKING")
    print("=" * 70)

    # Initialize calculator with history tracking enabled
    calc = FundingCalculator(track_history=True, max_history_per_symbol=100)

    # Simulate funding accruals
    symbol = "BTC-PERP"
    position_size = Decimal("1.0")
    position_side = "long"
    mark_price = Decimal("50000")

    # Simulate 5 hourly funding events
    base_time = datetime(2025, 10, 22, 0, 0, 0)
    for hour in range(5):
        funding_time = base_time + timedelta(hours=hour)
        funding_rate = Decimal("0.0001")  # 0.01% per hour

        funding_amount = calc.accrue_if_due(
            symbol=symbol,
            position_size=position_size,
            position_side=position_side,
            mark_price=mark_price,
            funding_rate=funding_rate,
            next_funding_time=funding_time,
            now=funding_time + timedelta(minutes=1),  # Slightly after funding time
        )

        if funding_amount != 0:
            print(f"Hour {hour}: Funding accrued = ${funding_amount:.2f}")

    # Get funding history
    print(f"\n📊 Total funding paid: ${calc.get_total_funding(symbol):.2f}")

    history = calc.get_funding_history(symbol, limit=10)
    print(f"\n📜 Recent funding events (last {len(history)}):")
    for i, event in enumerate(history, 1):
        print(
            f"  {i}. {event.timestamp} | Rate: {event.rate*100:.3f}% | Amount: ${event.amount:.2f}"
        )


def demo_liquidation_calculator():
    """Demonstrate liquidation price calculations."""
    print("\n" + "=" * 70)
    print("2. LIQUIDATION PRICE CALCULATOR")
    print("=" * 70)

    calc = LiquidationCalculator()

    # Example 1: Long position
    entry_price = Decimal("50000")
    current_price = Decimal("51000")
    leverage = 10
    side = "long"

    liq_price = calc.calculate_liquidation_price(entry_price, leverage, side)
    print(f"\n📈 LONG Position Example:")
    print(f"   Entry Price: ${entry_price:,.2f}")
    print(f"   Current Price: ${current_price:,.2f}")
    print(f"   Leverage: {leverage}x")
    print(f"   Liquidation Price: ${liq_price:,.2f}")

    distances = calc.calculate_liquidation_distance(
        entry_price, current_price, leverage, side
    )
    print(f"   Distance to Liquidation: ${distances['price_distance']:,.2f}")
    print(f"   Buffer: {distances['buffer_percent']:.2f}%")

    is_risky = calc.is_at_risk(
        entry_price, current_price, leverage, side, risk_threshold_percent=Decimal("20")
    )
    print(f"   At Risk (< 20% buffer): {'⚠️  YES' if is_risky else '✅ NO'}")

    # Example 2: Short position
    entry_price = Decimal("50000")
    current_price = Decimal("49000")
    side = "short"

    liq_price = calc.calculate_liquidation_price(entry_price, leverage, side)
    print(f"\n📉 SHORT Position Example:")
    print(f"   Entry Price: ${entry_price:,.2f}")
    print(f"   Current Price: ${current_price:,.2f}")
    print(f"   Leverage: {leverage}x")
    print(f"   Liquidation Price: ${liq_price:,.2f}")

    distances = calc.calculate_liquidation_distance(
        entry_price, current_price, leverage, side
    )
    print(f"   Distance to Liquidation: ${distances['price_distance']:,.2f}")
    print(f"   Buffer: {distances['buffer_percent']:.2f}%")


def demo_scaled_orders():
    """Demonstrate scaled order calculation (without actually placing orders)."""
    print("\n" + "=" * 70)
    print("3. SCALED ORDER PLACEMENT")
    print("=" * 70)

    # Example configuration
    symbol = "BTC-PERP"
    side = OrderSide.BUY
    total_quantity = Decimal("1.0")
    price_levels = [
        Decimal("45000"),
        Decimal("46000"),
        Decimal("47000"),
        Decimal("48000"),
        Decimal("50000"),
    ]

    print(f"\n📊 Scaled Order Configuration:")
    print(f"   Symbol: {symbol}")
    print(f"   Side: {side.value}")
    print(f"   Total Quantity: {total_quantity} BTC")
    print(f"   Price Levels: {len(price_levels)}")

    # Linear distribution
    print(f"\n1️⃣  LINEAR Distribution (equal at each level):")
    num_levels = len(price_levels)
    linear_qty = total_quantity / Decimal(str(num_levels))
    for i, price in enumerate(price_levels, 1):
        print(f"   Level {i}: {linear_qty:.4f} BTC @ ${price:,.0f}")
    print(f"   Average Price: ${sum(price_levels) / len(price_levels):,.2f}")

    # Weighted distribution
    print(f"\n2️⃣  WEIGHTED Distribution (more at better prices):")
    weights = [Decimal(str(num_levels - i)) for i in range(num_levels)]
    total_weight = sum(weights)
    for i, (price, weight) in enumerate(zip(price_levels, weights), 1):
        qty = (weight / total_weight) * total_quantity
        print(f"   Level {i}: {qty:.4f} BTC @ ${price:,.0f} (weight: {weight})")

    # Calculate weighted average price
    weighted_qty = [(w / total_weight) * total_quantity for w in weights]
    weighted_avg = sum(p * q for p, q in zip(price_levels, weighted_qty)) / total_quantity
    print(f"   Weighted Average Price: ${weighted_avg:,.2f}")

    print(
        f"\n💡 Note: To actually place orders, use rest_service.place_scaled_order()"
    )


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("GPT-TRADER: OCTOBER 2025 DERIVATIVES FEATURES DEMO")
    print("=" * 70)
    print("\nThis demonstrates the new optional features for derivatives trading:")
    print("  • Funding accrual history tracking")
    print("  • Liquidation price calculations")
    print("  • Scaled order placement")

    demo_funding_history()
    demo_liquidation_calculator()
    demo_scaled_orders()

    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print("\nFor production usage, see:")
    print("  • src/bot_v2/features/brokerages/coinbase/utilities.py")
    print("  • src/bot_v2/features/brokerages/coinbase/rest/orders.py")
    print("\n")


if __name__ == "__main__":
    main()
