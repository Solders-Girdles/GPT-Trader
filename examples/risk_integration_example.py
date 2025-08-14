#!/usr/bin/env python3
"""Example demonstrating Risk Integration usage in GPT-Trader.

This example shows how to integrate the risk management system
into a typical trading workflow.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports (in real usage, this would be installed)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def mock_market_data():
    """Create mock market data for demonstration."""
    import pandas as pd

    # Create 60 days of mock price data
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")

    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    market_data = {}

    for i, symbol in enumerate(symbols):
        base_price = [150, 2500, 300, 200, 800][i]
        # Create some realistic price movement
        price_changes = pd.Series(range(60)) * 0.1 + (pd.Series(range(60)) % 7 - 3) * 2
        prices = base_price + price_changes

        market_data[symbol] = pd.DataFrame(
            {"Close": prices, "Volume": [1000000 + i * 100000] * 60}, index=dates
        )

    return market_data


def create_sample_positions():
    """Create sample position data."""
    return {
        "AAPL": {
            "position_value": 25000,
            "position_size_pct": 0.125,  # 12.5%
            "total_risk": 1250,
            "unrealized_pnl": 500,
            "unrealized_pnl_pct": 0.02,
            "current_price": 155.0,
            "entry_price": 150.0,
            "highest_price": 160.0,
        },
        "GOOGL": {
            "position_value": 20000,
            "position_size_pct": 0.10,
            "total_risk": 1000,
            "unrealized_pnl": -300,
            "unrealized_pnl_pct": -0.015,
            "current_price": 2480.0,
            "entry_price": 2500.0,
            "highest_price": 2520.0,
        },
        "MSFT": {
            "position_value": 15000,
            "position_size_pct": 0.075,
            "total_risk": 750,
            "unrealized_pnl": 200,
            "unrealized_pnl_pct": 0.013,
            "current_price": 305.0,
            "entry_price": 300.0,
            "highest_price": 308.0,
        },
    }


def demonstrate_risk_integration():
    """Demonstrate the risk integration system."""
    print("=" * 60)
    print("GPT-TRADER RISK INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # This would normally import from the actual modules
    # For this demo, we'll define simplified versions

    class MockRiskConfig:
        def __init__(self):
            self.max_position_size = 0.10
            self.max_portfolio_exposure = 0.95
            self.default_stop_loss_pct = 0.05
            self.max_daily_loss = 0.03
            self.take_profit_pct = 0.10

    class MockRiskIntegration:
        def __init__(self, risk_config):
            self.risk_config = risk_config

        def validate_allocations(self, allocations, current_prices, portfolio_value, **kwargs):
            # Simplified validation logic
            adjusted = allocations.copy()
            warnings = {}

            # Check position sizes
            for symbol, shares in allocations.items():
                if symbol in current_prices:
                    position_value = shares * current_prices[symbol]
                    position_pct = position_value / portfolio_value

                    if position_pct > self.risk_config.max_position_size:
                        max_shares = int(
                            (portfolio_value * self.risk_config.max_position_size)
                            / current_prices[symbol]
                        )
                        adjusted[symbol] = max_shares
                        warnings[symbol] = f"Position reduced from {shares} to {max_shares} shares"

            # Check total exposure
            total_exposure = sum(
                adjusted[sym] * current_prices[sym] for sym in adjusted if sym in current_prices
            )
            exposure_pct = total_exposure / portfolio_value

            if exposure_pct > self.risk_config.max_portfolio_exposure:
                scale_factor = self.risk_config.max_portfolio_exposure / exposure_pct
                for symbol in adjusted:
                    adjusted[symbol] = int(adjusted[symbol] * scale_factor)
                warnings["portfolio"] = f"Portfolio scaled down by {scale_factor:.1%}"

            # Calculate stop levels
            stop_levels = {}
            for symbol, shares in adjusted.items():
                if symbol in current_prices and shares > 0:
                    price = current_prices[symbol]
                    stop_levels[symbol] = {
                        "stop_loss": price * (1 - self.risk_config.default_stop_loss_pct),
                        "take_profit": price * (1 + self.risk_config.take_profit_pct),
                        "current_price": price,
                    }

            return type(
                "Result",
                (),
                {
                    "original_allocations": allocations,
                    "adjusted_allocations": adjusted,
                    "warnings": warnings,
                    "stop_levels": stop_levels,
                    "passed_validation": len(warnings) == 0
                    or all("reduced" in w or "scaled" in w for w in warnings.values()),
                },
            )()

    # 1. Initialize Risk System
    print("\n1. INITIALIZING RISK SYSTEM")
    print("-" * 40)

    risk_config = MockRiskConfig()
    risk_integration = MockRiskIntegration(risk_config)

    print(f"Risk Configuration:")
    print(f"  Max Position Size: {risk_config.max_position_size:.1%}")
    print(f"  Max Portfolio Exposure: {risk_config.max_portfolio_exposure:.1%}")
    print(f"  Default Stop Loss: {risk_config.default_stop_loss_pct:.1%}")
    print(f"  Max Daily Loss: {risk_config.max_daily_loss:.1%}")

    # 2. Portfolio Allocation Scenario
    print("\n2. PORTFOLIO ALLOCATION VALIDATION")
    print("-" * 40)

    # Proposed allocations (some exceed limits)
    allocations = {
        "AAPL": 200,  # $30k position = 15% (exceeds 10% limit)
        "GOOGL": 12,  # $30k position = 15% (exceeds 10% limit)
        "MSFT": 100,  # $30k position = 15% (exceeds 10% limit)
        "TSLA": 75,  # $15k position = 7.5% (OK)
        "NVDA": 25,  # $20k position = 10% (OK)
    }

    current_prices = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0, "TSLA": 200.0, "NVDA": 800.0}

    portfolio_value = 200000.0  # $200k portfolio

    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    print(f"\nProposed Allocations:")
    total_proposed = 0
    for symbol, shares in allocations.items():
        value = shares * current_prices[symbol]
        pct = value / portfolio_value
        total_proposed += value
        print(
            f"  {symbol}: {shares} shares @ ${current_prices[symbol]} = ${value:,.0f} ({pct:.1%})"
        )

    print(
        f"\nTotal Proposed Exposure: ${total_proposed:,.0f} ({total_proposed/portfolio_value:.1%})"
    )

    # 3. Risk Validation
    print("\n3. RISK VALIDATION RESULTS")
    print("-" * 40)

    result = risk_integration.validate_allocations(
        allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
    )

    print(f"Validation Result: {'PASSED' if result.passed_validation else 'FAILED'}")

    if result.warnings:
        print(f"\nRisk Adjustments Made:")
        for symbol, warning in result.warnings.items():
            print(f"  {symbol}: {warning}")

    print(f"\nFinal Allocations:")
    total_final = 0
    for symbol, shares in result.adjusted_allocations.items():
        value = shares * current_prices[symbol]
        pct = value / portfolio_value
        total_final += value
        original_shares = allocations[symbol]
        change = "" if shares == original_shares else f" (was {original_shares})"
        print(f"  {symbol}: {shares} shares = ${value:,.0f} ({pct:.1%}){change}")

    print(f"\nFinal Total Exposure: ${total_final:,.0f} ({total_final/portfolio_value:.1%})")

    # 4. Stop-Loss Levels
    print("\n4. STOP-LOSS AND TAKE-PROFIT LEVELS")
    print("-" * 40)

    for symbol, stop_data in result.stop_levels.items():
        if symbol in result.adjusted_allocations and result.adjusted_allocations[symbol] > 0:
            current = stop_data["current_price"]
            stop = stop_data["stop_loss"]
            target = stop_data["take_profit"]
            risk_per_share = current - stop
            reward_per_share = target - current
            risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0

            print(f"{symbol}:")
            print(f"  Current Price: ${current:.2f}")
            print(f"  Stop Loss: ${stop:.2f} ({-((stop-current)/current):.1%})")
            print(f"  Take Profit: ${target:.2f} ({((target-current)/current):.1%})")
            print(f"  Risk/Reward: 1:{risk_reward:.1f}")
            print()

    # 5. Risk Monitoring Simulation
    print("\n5. RISK MONITORING SIMULATION")
    print("-" * 40)

    positions = create_sample_positions()

    print("Current Portfolio Positions:")
    total_value = sum(pos["position_value"] for pos in positions.values())
    total_pnl = sum(pos["unrealized_pnl"] for pos in positions.values())

    for symbol, pos in positions.items():
        print(
            f"  {symbol}: ${pos['position_value']:,.0f} ({pos['position_size_pct']:.1%}) "
            f"P&L: ${pos['unrealized_pnl']:+,.0f} ({pos['unrealized_pnl_pct']:+.1%})"
        )

    print(f"\nPortfolio Summary:")
    print(f"  Total Value: ${total_value:,.0f}")
    print(f"  Total P&L: ${total_pnl:+,.0f} ({total_pnl/total_value:+.1%})")

    # Check risk alerts
    alerts = []

    # Position size alerts
    for symbol, pos in positions.items():
        if pos["position_size_pct"] > risk_config.max_position_size:
            alerts.append(
                f"⚠️ {symbol} position ({pos['position_size_pct']:.1%}) exceeds limit ({risk_config.max_position_size:.1%})"
            )

    # Daily loss check
    daily_loss_pct = abs(total_pnl) / total_value if total_pnl < 0 else 0
    if daily_loss_pct > risk_config.max_daily_loss:
        alerts.append(
            f"🔴 Daily loss ({daily_loss_pct:.1%}) exceeds limit ({risk_config.max_daily_loss:.1%})"
        )

    if alerts:
        print(f"\nRisk Alerts:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print(f"\n✅ All risk limits within acceptable ranges")

    # 6. Risk Metrics
    print("\n6. PORTFOLIO RISK METRICS")
    print("-" * 40)

    position_values = [pos["position_value"] for pos in positions.values()]
    largest_position_pct = max(position_values) / total_value

    # Herfindahl concentration index
    weights = [pv / total_value for pv in position_values]
    concentration_ratio = sum(w**2 for w in weights)

    total_risk = sum(pos["total_risk"] for pos in positions.values())

    print(f"Risk Metrics:")
    print(f"  Number of Positions: {len(positions)}")
    print(f"  Largest Position: {largest_position_pct:.1%}")
    print(f"  Concentration Ratio: {concentration_ratio:.3f}")
    print(f"  Total Risk: ${total_risk:,.0f} ({total_risk/total_value:.1%})")
    print(f"  Average Position Size: {(total_value/len(positions))/total_value:.1%}")

    # Risk limit utilization
    print(f"\nRisk Limit Utilization:")
    print(f"  Position Size: {largest_position_pct/risk_config.max_position_size:.1%} of limit")
    print(
        f"  Portfolio Exposure: {(total_value/(total_value/0.85))/risk_config.max_portfolio_exposure:.1%} of limit"
    )
    if daily_loss_pct > 0:
        print(f"  Daily Loss: {daily_loss_pct/risk_config.max_daily_loss:.1%} of limit")

    print("\n" + "=" * 60)
    print("RISK INTEGRATION DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\nKey Features Demonstrated:")
    print("✓ Position size limit enforcement")
    print("✓ Portfolio exposure validation")
    print("✓ Stop-loss and take-profit calculation")
    print("✓ Real-time risk monitoring")
    print("✓ Risk metrics calculation")
    print("✓ Alert generation and reporting")

    print("\nThe risk integration system is protecting your portfolio by:")
    print("• Preventing oversized positions")
    print("• Limiting total portfolio exposure")
    print("• Calculating appropriate stop-loss levels")
    print("• Monitoring for risk limit violations")
    print("• Providing comprehensive risk reporting")

    print("\n🛡️ Your portfolio is protected! 🛡️")


if __name__ == "__main__":
    demonstrate_risk_integration()
