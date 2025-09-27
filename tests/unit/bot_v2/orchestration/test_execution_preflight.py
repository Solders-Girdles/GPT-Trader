from decimal import Decimal
from pathlib import Path

import pytest

from bot_v2.orchestration.live_execution import LiveExecutionEngine
from tests.utils.deterministic_broker import DeterministicBroker
from bot_v2.persistence.event_store import EventStore
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType


def make_engine(tmp_path: Path) -> tuple[LiveExecutionEngine, DeterministicBroker, EventStore]:
    broker = DeterministicBroker()
    broker.connect()
    store = EventStore(root=tmp_path)
    engine = LiveExecutionEngine(broker=broker, event_store=store, bot_id="itest_preflight")
    return engine, broker, store


def tail_rejections(store: EventStore, bot_id: str):
    return store.tail(bot_id, limit=50, types=["order_rejected"])


def test_min_notional_rejection_records_metric(tmp_path):
    engine, broker, store = make_engine(tmp_path)
    # Make XRP require high min notional to trigger rejection
    p = broker.get_product("XRP-PERP")
    # Ensure size rules won't trigger before min_notional
    setattr(p, "min_size", Decimal("0.001"))
    setattr(p, "step_size", Decimal("0.001"))
    setattr(p, "min_notional", Decimal("1000"))

    with pytest.raises(Exception):
        engine.place_order(
            symbol="XRP-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1"),  # 1 * $0.50 << 1000 min_notional
        )

    rej = tail_rejections(store, "itest_preflight")
    assert any(r.get("symbol") == "XRP-PERP" and r.get("reason") == "min_notional" for r in rej)


def test_small_qty_min_size_rejection(tmp_path):
    engine, broker, store = make_engine(tmp_path)
    p = broker.get_product("ETH-PERP")
    # Configure min size/step
    setattr(p, "min_size", Decimal("0.01"))
    setattr(p, "step_size", Decimal("0.01"))
    setattr(p, "price_increment", Decimal("0.01"))
    
    with pytest.raises(Exception):
        engine.place_order(
            symbol="ETH-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("0.001"),  # Below min_size; validator should reject
        )
    
    rej = tail_rejections(store, "itest_preflight")
    assert any(r.get("symbol") == "ETH-PERP" and r.get("reason") == "min_size" for r in rej)
