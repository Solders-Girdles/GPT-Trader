from decimal import Decimal
import pytest

from bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy,
    StrategyConfig,
    Action,
)
from bot_v2.features.live_trade.risk import LiveRiskManager
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
def test_reduce_only_blocks_entries_allows_exits():
    risk = LiveRiskManager(config=RiskConfig(reduce_only_mode=True))
    strat = BaselinePerpsStrategy(
        config=StrategyConfig(short_ma_period=3, long_ma_period=5), risk_manager=risk
    )
    product = make_product()

    # Rising marks would normally emit BUY, but reduce-only should HOLD
    rising = [Decimal("49000"), Decimal("49500"), Decimal("50000"), Decimal("50500")]
    d1 = strat.decide(
        symbol="BTC-PERP",
        current_mark=Decimal("51000"),
        position_state=None,
        recent_marks=rising,
        equity=Decimal("10000"),
        product=product,
    )
    assert d1.action == Action.HOLD
    assert "Reduce-only" in d1.reason

    # With an open position, reduce-only should allow CLOSE
    pos = {"quantity": Decimal("1"), "side": "long", "entry": Decimal("50000")}
    d2 = strat.decide(
        symbol="BTC-PERP",
        current_mark=Decimal("51000"),
        position_state=pos,
        recent_marks=[Decimal("50000")] * 5,
        equity=Decimal("10000"),
        product=product,
    )
    assert d2.action == Action.CLOSE
    assert d2.reduce_only is True
