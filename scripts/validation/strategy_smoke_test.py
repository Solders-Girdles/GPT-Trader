#!/usr/bin/env python3
"""
Strategy Smoke Test for BaselinePerpsStrategy

Validates simplified behavior on synthetic data:
- Bullish crossover → BUY with predictable sizing
- Bearish crossover → SELL when shorts enabled
- Trailing stop triggers after peak move

Run: python scripts/validation/strategy_smoke_test.py
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, Any

from bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy, StrategyConfig, Action
)
from bot_v2.features.brokerages.core.interfaces import Product, MarketType


def make_product(symbol: str = "BTC-PERP") -> Product:
    return Product(
        symbol=symbol,
        base_asset=symbol.split('-')[0],
        quote_asset='USD',
        market_type=MarketType.PERPETUAL,
        min_size=Decimal('0.001'),
        step_size=Decimal('0.001'),
        min_notional=Decimal('10'),
        price_increment=Decimal('0.01'),
        leverage_max=3,
        contract_size=Decimal('1')
    )


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def run_smoke() -> None:
    symbol = "BTC-PERP"
    equity = Decimal('10000')
    product = make_product(symbol)

    cfg = StrategyConfig(
        short_ma_period=3,
        long_ma_period=5,
        position_fraction=0.05,
        max_trade_usd=Decimal('600'),  # cap 5%*10k=500 < 600 → 500
        enable_shorts=True,
        trailing_stop_pct=0.01,
    )
    strat = BaselinePerpsStrategy(config=cfg)

    # Synthetic marks: flat → dip → rise to create bullish crossover
    marks_up = [Decimal('100'), Decimal('100'), Decimal('99'), Decimal('101'), Decimal('103'), Decimal('105')]
    decision = None
    pos_state: Dict[str, Any] = {"qty": 0}
    window: list[Decimal] = []
    for m in marks_up:
        window.append(m)
        decision = strat.decide(symbol, m, pos_state, window.copy(), equity, product)

    assert_true(decision is not None, "No decision produced")
    assert_true(decision.action == Action.BUY, f"Expected BUY on bullish crossover, got {decision.action}")
    # Sizing check: 5% of 10k → 500 (capped at 600)
    assert_true(decision.target_notional == Decimal('500'), f"Unexpected notional {decision.target_notional}")

    # Simulate open long position and rising prices for trailing stop setup
    pos_state = {"qty": Decimal('0.01'), "side": "long"}
    for m in [Decimal('106'), Decimal('107'), Decimal('108')]:
        decision = strat.decide(symbol, m, pos_state, window + [m], equity, product)
        window.append(m)
    # Now drop below trailing stop (1% from peak ~108 → stop ~106.92)
    m = Decimal('106.5')
    decision = strat.decide(symbol, m, pos_state, window + [m], equity, product)
    assert_true(decision.action == Action.CLOSE, "Expected CLOSE on trailing stop breach")

    # Bearish crossover (shorts enabled)
    window = [Decimal('110'), Decimal('109'), Decimal('108'), Decimal('107'), Decimal('106'), Decimal('105')]
    pos_state = {"qty": 0}
    decision = strat.decide(symbol, window[-1], pos_state, window.copy(), equity, product)
    assert_true(decision.action == Action.SELL, "Expected SELL on bearish crossover with shorts enabled")

    print("Strategy smoke test passed ✔︎")


if __name__ == "__main__":
    run_smoke()

