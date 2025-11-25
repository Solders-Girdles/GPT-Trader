from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.orchestration.configuration import RiskConfig
from gpt_trader.features.brokerages.core.interfaces import MarketType, Product
from gpt_trader.features.live_trade.risk import LiveRiskManager, ValidationError


def make_perp(symbol: str) -> Product:
    return Product(
        symbol=symbol,
        base_asset=symbol.split("-")[0],
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
    )


@pytest.mark.xfail(reason="Time mocking for leverage windows mismatch")
def test_day_vs_night_leverage_caps_enforced(monkeypatch):
    config = RiskConfig(
        max_leverage=20,
        leverage_max_per_symbol={"BTC-PERP": 20},
        daytime_start_utc="09:00",
        daytime_end_utc="17:00",
        day_leverage_max_per_symbol={"BTC-PERP": 10},
        night_leverage_max_per_symbol={"BTC-PERP": 5},
        max_position_pct_per_symbol=100.0,
        max_exposure_pct=100.0,
        min_liquidation_buffer_pct=0.0,
        slippage_guard_bps=1_000_000,
    )
    rm = LiveRiskManager(config=config)
    product = make_perp("BTC-PERP")

    # Mock now into daytime
    rm._now_provider = lambda: datetime(2024, 1, 3, 10, 0)

    # 10x allowed in day
    equity = Decimal("10000")
    price = Decimal("5000")
    quantity = Decimal("20")  # 100k notional => 10x
    rm.pre_trade_validate(
        symbol="BTC-PERP",
        side="buy",
        quantity=quantity,
        price=price,
        product=product,
        equity=equity,
        current_positions={},
    )

    # Mock now into nighttime
    rm._now_provider = lambda: datetime(2024, 1, 3, 20, 0)
    # Same order → 10x exceeds night cap 5x
    with pytest.raises(ValidationError, match="exceeds BTC-PERP cap of 5x"):
        rm.pre_trade_validate(
            symbol="BTC-PERP",
            side="buy",
            quantity=quantity,
            price=price,
            product=product,
            equity=equity,
            current_positions={},
        )


@pytest.mark.xfail(reason="Time mocking mismatch")
def test_day_vs_night_mmr_projection(monkeypatch):
    # Night MMR higher → projected buffer insufficient at night, OK in day
    config = RiskConfig(
        max_leverage=20,
        leverage_max_per_symbol={"BTC-PERP": 20},
        daytime_start_utc="09:00",
        daytime_end_utc="17:00",
        day_mmr_per_symbol={"BTC-PERP": 0.005},
        night_mmr_per_symbol={"BTC-PERP": 0.2},
        enable_pre_trade_liq_projection=True,
        min_liquidation_buffer_pct=0.15,
        max_position_pct_per_symbol=100.0,
        max_exposure_pct=100.0,
        slippage_guard_bps=1_000_000,
    )
    rm = LiveRiskManager(config=config)
    product = make_perp("BTC-PERP")

    equity = Decimal("10000")
    price = Decimal("5000")
    quantity = Decimal("20.0")  # 100,000 notional

    # Daytime: low MMR → buffer OK
    rm._now_provider = lambda: datetime(2024, 1, 3, 10, 0)
    rm.pre_trade_validate(
        symbol="BTC-PERP",
        side="buy",
        quantity=quantity,
        price=price,
        product=product,
        equity=equity,
        current_positions={},
    )

    # Nighttime: high MMR → buffer insufficient
    rm._now_provider = lambda: datetime(2024, 1, 3, 20, 0)
    with pytest.raises(ValidationError, match="Projected liquidation buffer"):
        rm.pre_trade_validate(
            symbol="BTC-PERP",
            side="buy",
            quantity=quantity,
            price=price,
            product=product,
            equity=equity,
            current_positions={},
        )
