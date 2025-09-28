"""Unit tests for PerpsBot essentials.

Avoids network and real broker connections; focuses on safe internals.
"""

from datetime import datetime, timezone, time
from decimal import Decimal

import asyncio
import threading
import time as _time

import pytest

from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig, Profile
from bot_v2.orchestration.deterministic_broker import DeterministicBroker
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderType,
    Order,
    OrderStatus,
    TimeInForce,
)


@pytest.mark.uses_mock_broker
def test_init_uses_mock_broker_in_dev(monkeypatch, tmp_path):
    # Reduce side effects: avoid spawning background threads
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    cfg = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(cfg)

    assert isinstance(bot.broker, DeterministicBroker)


def test_calculate_spread_bps():
    # 100 bid, 101 ask => spread 1 over mid 100.5 => ~0.00995 * 10000 ≈ 99.5 bps
    from bot_v2.orchestration.perps_bot import PerpsBot as _PB  # alias to access staticmethod

    bps = _PB._calculate_spread_bps(Decimal("100"), Decimal("101"))
    assert bps > 0
    # Within a small tolerance around 99-100 bps
    assert Decimal("90") < bps < Decimal("110")


@pytest.mark.uses_mock_broker
def test_update_mark_window_trims(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    cfg = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], short_ma=2, long_ma=3)
    bot = PerpsBot(cfg)

    for i in range(50):
        bot._update_mark_window("BTC-PERP", Decimal(str(50000 + i)))

    max_expected = max(cfg.short_ma, cfg.long_ma) + 5
    assert len(bot.mark_windows["BTC-PERP"]) <= max_expected


def test_collect_account_snapshot_uses_broker(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    cfg = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(cfg)

    snapshot_data = {
        "permissions": {"can_trade": True},
        "fees": {"tier": "Advanced"},
        "limits": {"max_order": "10000"},
        "summary": {"total": "123"},
    }
    bot.broker.get_key_permissions = lambda: snapshot_data["permissions"]  # type: ignore[attr-defined]
    bot.broker.get_fee_schedule = lambda: snapshot_data["fees"]  # type: ignore[attr-defined]
    bot.broker.get_account_limits = lambda: snapshot_data["limits"]  # type: ignore[attr-defined]
    bot.broker.get_transaction_summary = lambda: snapshot_data["summary"]  # type: ignore[attr-defined]
    bot.broker.get_server_time = lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)  # type: ignore[attr-defined]

    snap = bot._collect_account_snapshot()
    assert snap["key_permissions"] == snapshot_data["permissions"]
    assert snap["fee_schedule"] == snapshot_data["fees"]
    assert snap["limits"] == snapshot_data["limits"]
    assert snap["transaction_summary"] == snapshot_data["summary"]
    assert bot._latest_account_snapshot == snap


@pytest.mark.asyncio
async def test_run_account_telemetry_emits_metrics(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    cfg = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(cfg)

    calls = []
    bot.event_store.append_metric = lambda bot_id, metrics=None, **kwargs: calls.append((bot_id, metrics))  # type: ignore

    def fake_collect():
        bot.running = False
        return {"key_permissions": {}, "fee_schedule": {}, "limits": {}, "transaction_summary": {}}

    bot._collect_account_snapshot = fake_collect  # type: ignore
    bot.running = True
    await bot._run_account_telemetry(interval_seconds=0)

    assert calls
    bot_id, metrics = calls[0]
    assert bot_id == bot.bot_id
    assert metrics.get("event_type") == "account_snapshot"


@pytest.mark.asyncio
async def test_run_cycle_respects_trading_window(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    cfg = BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-PERP"],
        trading_window_start=time(10, 0),
        trading_window_end=time(11, 0),
        trading_days=["monday"],
        mock_broker=True,
    )

    bot = PerpsBot(cfg)

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

    cfg = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(cfg)

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
    bot.exec_engine = exec_engine  # type: ignore

    def make_order(order_id: str) -> Order:
        now = datetime.now(timezone.utc)
        return Order(
            id=order_id,
            client_id=order_id,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            qty=Decimal("0.01"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.FILLED,
            filled_qty=Decimal("0.01"),
            avg_fill_price=Decimal("50000"),
            submitted_at=now,
            updated_at=now,
        )

    bot.broker.get_order = make_order  # type: ignore

    async def submit():
        return await bot._place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            qty=Decimal("0.01"),
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

    cfg = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = PerpsBot(cfg)

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
