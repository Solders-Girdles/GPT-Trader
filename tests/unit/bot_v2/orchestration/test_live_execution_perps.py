from decimal import Decimal
import tempfile
import pytest

from bot_v2.orchestration.live_execution import LiveExecutionEngine
from tests.support.deterministic_broker import DeterministicBroker
from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.persistence.event_store import EventStore
from pathlib import Path
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, Position


@pytest.mark.perps
def test_reduce_only_is_forced_in_execution():
    broker = DeterministicBroker()
    risk = LiveRiskManager(config=RiskConfig(reduce_only_mode=True))
    eng = LiveExecutionEngine(broker=broker, risk_manager=risk, event_store=EventStore())

    # Seed an existing long position so a SELL reduces
    broker.seed_position("BTC-PERP", side="long", quantity=Decimal("0.01"), price=Decimal("50000"))

    # Monkeypatch broker.place_order to capture reduce_only flag
    captured = {}
    orig = broker.place_order

    def _patched_place_order(**kwargs):  # type: ignore[no-redef]
        captured.update(kwargs)
        return orig(**kwargs)

    broker.place_order = _patched_place_order  # type: ignore[assignment]

    broker.connect()
    order_id = eng.place_order(
        symbol="BTC-PERP",
        side=OrderSide.SELL,  # reducing long
        order_type=OrderType.MARKET,
        quantity=Decimal("0.005"),
    )
    assert order_id is not None
    assert captured.get("reduce_only") is True


@pytest.mark.perps
def test_rejection_logs_metric_with_reason():
    broker = DeterministicBroker()
    tmp = tempfile.TemporaryDirectory()
    ev = EventStore(root=Path(tmp.name))
    risk = LiveRiskManager(config=RiskConfig(max_leverage=5, slippage_guard_bps=1_000_000))
    eng = LiveExecutionEngine(
        broker=broker, risk_manager=risk, event_store=ev, bot_id="perps_test_bot"
    )

    broker.connect()
    # Force huge quantity → leverage rejection
    with pytest.raises(ValidationError):
        eng.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("20"),  # notional >> equity
        )

    # Tail events and ensure order_rejected logged
    events = ev.tail("perps_test_bot", limit=20, types=["order_rejected"])  # type: ignore[arg-type]
    assert any("order_rejected" == e.get("type") for e in events)
    assert any("Leverage" in e.get("reason", "") for e in events)
