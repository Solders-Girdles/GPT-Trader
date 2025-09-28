from decimal import Decimal
import pytest

from bot_v2.features.brokerages.coinbase.utilities import enforce_perp_rules, InvalidRequestError
from bot_v2.features.brokerages.core.interfaces import Product, MarketType


def make_product():
    return Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.05"),  # 5c ticks to observe rounding
        min_notional=Decimal("10"),
    )


@pytest.mark.perps
def test_quantity_rounds_to_step_and_enforces_min_size():
    p = make_product()
    # Below min size should raise
    with pytest.raises(InvalidRequestError):
        enforce_perp_rules(p, qty=Decimal("0.0005"), price=Decimal("50000"))

    # Rounds down to nearest step
    q, pr = enforce_perp_rules(p, qty=Decimal("1.234567"), price=Decimal("50000.1234"))
    assert q == Decimal("1.234")
    assert pr == Decimal("50000.10")  # price increment 0.05 → rounds down


@pytest.mark.perps
def test_min_notional_enforced():
    p = make_product()
    # Small qty with valid rounding but too small notional fails
    with pytest.raises(InvalidRequestError):
        enforce_perp_rules(p, qty=Decimal("0.001"), price=Decimal("1000.00"))
