from decimal import Decimal

import pytest

from gpt_trader.features.brokerages.core.interfaces import MarketType, Product
from gpt_trader.features.live_trade.risk import LiveRiskManager, ValidationError
from gpt_trader.orchestration.configuration import RiskConfig


def make_product():
    return Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
        leverage_max=10,
    )


@pytest.mark.perps
def test_pre_trade_rejects_above_max_leverage():
    risk = LiveRiskManager(config=RiskConfig(max_leverage=5))
    product = make_product()

    # Equity 10k; notional 1.02M → 102x > 5x, should be rejected
    equity = Decimal("10000")
    price = Decimal("51000")
    quantity = Decimal("20")  # notional = 1,020,000

    with pytest.raises(ValidationError):
        risk.pre_trade_validate(
            symbol="BTC-PERP",
            side="buy",
            quantity=quantity,
            price=price,
            product=product,
            equity=equity,
            current_positions=None,
        )
