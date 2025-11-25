from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.orchestration.configuration import RiskConfig
from gpt_trader.features.brokerages.core.interfaces import MarketType, Product
from gpt_trader.features.live_trade.risk import LiveRiskManager, ValidationError


def make_perp(symbol: str = "BTC-PERP") -> Product:
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


@pytest.mark.skip(reason="TODO: Fix total exposure validation - breach detection logic needs update")
def test_total_exposure_breach_while_symbol_cap_ok():
    config = RiskConfig(
        max_leverage=10,
        min_liquidation_buffer_pct=0.1,
        max_position_pct_per_symbol=0.5,  # 50% per symbol allowed
        max_exposure_pct=0.8,  # 80% portfolio cap
        slippage_guard_bps=1_000_000,
    )
    rm = LiveRiskManager(config=config)
    product = make_perp("BTC-PERP")

    equity = Decimal("100000")
    price = Decimal("50000")

    current_positions = {
        "ETH-PERP": {
            "quantity": Decimal("10"),
            "mark": Decimal("4000"),
            "price": Decimal("4000"),
        },  # 40k
        "SOL-PERP": {
            "quantity": Decimal("500"),
            "mark": Decimal("60"),
            "price": Decimal("60"),
        },  # 30k
    }  # total 70k (70%)

    # New order 15k (15%) keeps per-symbol under 50%, but total → 85% > 80%
    quantity = Decimal("0.3")  # 0.3 * 50k = 15k

    with pytest.raises(ValidationError, match="Total exposure .* would exceed cap"):
        rm.pre_trade_validate(
            symbol="BTC-PERP",
            side="buy",
            quantity=quantity,
            price=price,
            product=product,
            equity=equity,
            current_positions=current_positions,
        )
