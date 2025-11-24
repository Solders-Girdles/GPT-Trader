#!/usr/bin/env python3
"""
Reproduction script for BaselinePerpsStrategy logic.
"""
import sys
from decimal import Decimal
from unittest.mock import MagicMock

# Add src to path
sys.path.append("src")

from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from gpt_trader.features.live_trade.strategies.perps_baseline.config import StrategyConfig
from gpt_trader.features.brokerages.core.interfaces import Product, MarketType


def run_repro():
    print("--- Starting Reproduction ---")

    # 1. Setup Config
    config = StrategyConfig(
        short_ma_period=5,
        long_ma_period=20,
        position_fraction=0.1,
    )
    print(f"Config: Short MA={config.short_ma_period}, Long MA={config.long_ma_period}")

    # 2. Initialize Strategy
    strategy = BaselinePerpsStrategy(config=config)

    # 3. Create Mock Data (Trending Scenario - No Fresh Cross)
    # Price consistently above MA.
    recent_marks = [Decimal("110") for _ in range(25)]
    current_mark = Decimal("112")  # Higher than recent to ensure Short > Long

    # Enable force_entry_on_trend
    config.force_entry_on_trend = True

    print(f"Recent Marks (last 5): {recent_marks[-5:]}")
    print(f"Current Mark: {current_mark}")

    # 4. Mock Product
    product = MagicMock(spec=Product)
    product.symbol = "BTC-USD"
    product.min_size = Decimal("0.001")
    product.base_increment = Decimal("0.001")
    product.quote_increment = Decimal("0.01")
    product.market_type = MarketType.PERPETUAL

    # 5. Decide
    decision = strategy.decide(
        symbol="BTC-USD",
        current_mark=current_mark,
        position_state=None,  # No position
        recent_marks=recent_marks,
        equity=Decimal("1000"),
        product=product,
    )

    print(f"\nDecision: {decision.action} - {decision.reason}")

    if decision.action == "hold":
        print("\n[FAIL] Strategy returned HOLD despite bullish data.")
    else:
        print(f"\n[SUCCESS] Strategy returned {decision.action}!")


if __name__ == "__main__":
    run_repro()
