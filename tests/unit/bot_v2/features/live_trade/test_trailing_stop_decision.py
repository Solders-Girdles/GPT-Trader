from decimal import Decimal
import pytest

from bot_v2.features.live_trade.strategies.perps_baseline import BaselinePerpsStrategy, StrategyConfig, Action
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
def test_trailing_stop_emits_reduce_only_close():
    strat = BaselinePerpsStrategy(config=StrategyConfig(short_ma_period=3, long_ma_period=5, trailing_stop_pct=0.01))
    product = make_product()
    pos = {"qty": Decimal("1"), "side": "long", "entry": Decimal("50000")}

    # Peak then drop beyond 1% triggers close
    for px in [Decimal("51000"), Decimal("50900"), Decimal("50400")]:
        d = strat.decide(
            symbol="BTC-PERP",
            current_mark=px,
            position_state=pos,
            recent_marks=[Decimal("50000")] * 5,
            equity=Decimal("10000"),
            product=product,
        )

    assert d.action == Action.CLOSE
    assert d.reduce_only is True
    # Baseline emits market exit; no stop trigger fields set
    assert d.order_type is None
    assert d.time_in_force is None
    assert d.stop_trigger is None
