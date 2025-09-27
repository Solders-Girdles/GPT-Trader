from decimal import Decimal
import pytest

from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import Product, MarketType


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
    )


@pytest.mark.perps
def test_pre_trade_rejects_above_max_leverage():
    risk = LiveRiskManager(config=RiskConfig(max_leverage=5))
    product = make_product()

    # Equity 10k; notional 1.02M → 102x > 5x, should be rejected
    equity = Decimal("10000")
    price = Decimal("51000")
    qty = Decimal("20")  # notional = 1,020,000

    with pytest.raises(ValidationError):
        risk.pre_trade_validate(
            symbol="BTC-PERP",
            side="buy",
            qty=qty,
            price=price,
            product=product,
            equity=equity,
            current_positions=None,
        )
