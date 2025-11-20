"""Unit tests for PerpsBot essentials.

Avoids network and real broker connections; focuses on safe internals.
"""

import asyncio
import threading
import time as _time
from datetime import datetime, time, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.engines.base import CoordinatorContext
from bot_v2.orchestration.engines.runtime import RuntimeEngine
from bot_v2.orchestration.engines.strategy import TradingEngine
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.perps_bot_builder import create_perps_bot
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry

BOT_ID = "coinbase_trader"


def test_runtime_coordinator_uses_deterministic_broker_for_dev(monkeypatch):
    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    registry = ServiceRegistry(config=config)
    runtime_state = PerpsBotRuntimeState(config.symbols)
    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=object(),
        orders_store=None,
        broker=None,
        risk_manager=None,
        symbols=tuple(config.symbols),
        bot_id=BOT_ID,
        runtime_state=runtime_state,
        set_running_flag=lambda _: None,
    )

    stub_broker = object()
    monkeypatch.setattr(
        "bot_v2.orchestration.engines.runtime.DeterministicBroker",
        lambda: stub_broker,
    )

    coordinator = RuntimeEngine(context)
    updated = coordinator._init_broker(context)

    assert updated.broker is stub_broker
    assert updated.registry.broker is stub_broker


def test_calculate_spread_bps():
    # 100 bid, 101 ask => spread 1 over mid 100.5 => ~0.00995 * 10000 ≈ 99.5 bps
    from bot_v2.orchestration.perps_bot import PerpsBot as _PB  # alias to access staticmethod

    bps = _PB._calculate_spread_bps(Decimal("100"), Decimal("101"))
    assert bps > 0
    # Within a small tolerance around 99-100 bps
    assert Decimal("90") < bps < Decimal("110")


def test_update_mark_window_trims() -> None:
    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], short_ma=2, long_ma=3)

    class StubState:
        def __init__(self) -> None:
            self.mark_windows: dict[str, list[Decimal]] = {}
            self.mark_lock = threading.RLock()

    state = StubState()
    runtime_state = state  # reuse stub state as runtime state
    registry = ServiceRegistry(config=config)
    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=None,
        orders_store=None,
        broker=None,
        risk_manager=None,
        symbols=tuple(config.symbols),
        bot_id=BOT_ID,
        runtime_state=runtime_state,
        set_running_flag=lambda _: None,
    )
    coordinator = TradingEngine(context)

    for i in range(50):
        coordinator.update_mark_window("BTC-PERP", Decimal(str(50000 + i)))

    max_expected = max(config.short_ma, config.long_ma) + 5
    assert len(state.mark_windows["BTC-PERP"]) == max_expected


def test_collect_account_snapshot_uses_broker(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)

    snapshot_data = {
        "key_permissions": {"can_trade": True},
        "fee_schedule": {"tier": "Advanced"},
        "limits": {"max_order": "10000"},
        "transaction_summary": {"total": "123"},
    }

    bot.account_manager.snapshot = lambda emit_metric=False: snapshot_data  # type: ignore[assignment]
    bot.account_telemetry.supports_snapshots = lambda: True  # type: ignore[assignment]
    bot.account_telemetry._broker.get_server_time = (  # type: ignore[attr-defined]
        lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)
    )

    snap = bot.account_telemetry.collect_snapshot()
    for key, value in snapshot_data.items():
        assert snap[key] == value
    assert snap["server_time"].startswith("2024-01-01")
    assert bot.account_telemetry.latest_snapshot == snap


@pytest.mark.asyncio
async def test_run_account_telemetry_emits_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)

    monkeypatch.setattr("bot_v2.orchestration.account_telemetry.RUNTIME_DATA_DIR", tmp_path)

    bot.account_manager.snapshot = lambda emit_metric=False: {
        "key_permissions": {},
        "fee_schedule": {},
    }
    bot.account_telemetry.supports_snapshots = lambda: True  # type: ignore[assignment]
    bot.account_telemetry._broker.get_server_time = lambda: None  # type: ignore[attr-defined]

    event = asyncio.Event()
    calls: list[tuple[str, dict[str, object] | None]] = []

    def fake_append_metric(bot_id, metrics=None, **kwargs):  # type: ignore[override]
        calls.append((bot_id, metrics))
        event.set()

    bot.event_store.append_metric = fake_append_metric  # type: ignore

    telemetry_task = asyncio.create_task(bot.account_telemetry.run(interval_seconds=0))
    await asyncio.wait_for(event.wait(), timeout=1)
    telemetry_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await telemetry_task

    assert calls
    bot_id, metrics = calls[0]
    assert bot_id == bot.bot_id
    assert metrics and metrics.get("event_type") == "account_snapshot"


@pytest.mark.asyncio
async def test_run_cycle_respects_trading_window(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-PERP"],
        trading_window_start=time(10, 0),
        trading_window_end=time(11, 0),
        trading_days=["monday"],
        mock_broker=True,
    )

    bot = create_perps_bot(config)

    async def noop_update():
        return None

    recorded: list[str] = []

    async def record_process(symbol: str):
        recorded.append(symbol)

    async def noop_log():
        return None

    bot.update_marks = noop_update  # type: ignore
    bot.process_symbol = record_process  # type: ignore
    bot.log_status = noop_log  # type: ignore

    bot._session_guard._now = lambda: datetime(2024, 1, 1, 8, 0)  # Monday outside window
    await bot.run_cycle()
    assert recorded == []

    expected_symbol = bot.config.symbols[0]

    bot._session_guard._now = lambda: datetime(2024, 1, 1, 10, 30)
    await bot.run_cycle()
    assert recorded == [expected_symbol]


@pytest.mark.asyncio
async def test_place_order_lock_serialises_calls(monkeypatch, tmp_path, fake_clock):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("COINBASE_ENABLE_DERIVATIVES", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = create_perps_bot(config)

    product = bot.get_product("BTC-PERP")

    class StubExecEngine:
        def __init__(self):
            self._lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.counter = 0

        def place_order(self, **kwargs):
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                _time.sleep(0.02)
                self.counter += 1
                return f"order-{self.counter}"
            finally:
                with self._lock:
                    self.active -= 1

    exec_engine = StubExecEngine()
    bot.runtime_state.exec_engine = exec_engine  # type: ignore[attr-defined]

    def make_order(order_id: str) -> Order:
        now = datetime.now(timezone.utc)
        return Order(
            id=order_id,
            client_id=order_id,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.01"),
            avg_fill_price=Decimal("50000"),
            submitted_at=now,
            updated_at=now,
        )

    bot.broker.get_order = make_order  # type: ignore

    async def submit():
        return await bot._place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            order_type=OrderType.MARKET,
            product=product,
            reduce_only=False,
            leverage=None,
            price=None,
            stop_price=None,
            tif=None,
        )

    tasks = [asyncio.create_task(submit()) for _ in range(6)]
    results = await asyncio.gather(*tasks)

    assert all(order is not None for order in results)
    assert exec_engine.max_active == 1
    assert bot.order_stats["attempted"] == 6
    assert bot.order_stats["successful"] == 6


def test_ws_failure_records_metrics_and_risk_listener(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setenv("COINBASE_ENABLE_DERIVATIVES", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = create_perps_bot(config)

    metric_records = []

    def record_metric(bot_id, metrics=None, **kwargs):  # type: ignore
        payload = metrics or kwargs
        metric_records.append(payload)

    bot.event_store.append_metric = record_metric  # type: ignore

    def boom(*args, **kwargs):
        raise RuntimeError("stream failed")

    bot.broker.stream_orderbook = boom  # type: ignore
    bot.broker.stream_trades = boom  # type: ignore

    bot._ws_stop = None
    bot._run_stream_loop(["BTC-PERP"], level=1)

    assert any(
        m.get("event_type") in {"ws_stream_error", "ws_stream_exit"}
        for m in metric_records
        if isinstance(m, dict)
    )

    # Ensure reduce-only listener still in effect after failure handling
    bot.risk_manager.set_reduce_only_mode(True, reason="ws_failure")
    assert bot.is_reduce_only_mode() is True
    bot.risk_manager.set_reduce_only_mode(False, reason="ws_recovery")
    assert bot.is_reduce_only_mode() is False
