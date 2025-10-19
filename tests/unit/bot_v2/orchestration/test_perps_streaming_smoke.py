from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot_builder import create_perps_bot
from bot_v2.orchestration.service_registry import ServiceRegistry


@pytest.mark.asyncio
async def test_perps_streaming_smoke(monkeypatch, tmp_path, fake_clock):
    # Ensure streaming does not auto-start; we'll start it after patching broker
    monkeypatch.setenv("PERPS_ENABLE_STREAMING", "0")
    monkeypatch.setenv("PERPS_FORCE_MOCK", "0")
    # Set event store root to temp dir
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

    # CANARY profile to allow streaming
    config = BotConfig.from_profile("canary")

    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)

    # Start bot; streaming thread won't start due to gate
    bot = create_perps_bot(config, registry=registry)

    # Monkeypatch broker streams to yield a couple messages
    def stream_orderbook_unavail(symbols, level=1):
        raise AttributeError("no orderbook stream")

    def stream_trades_once(symbols):
        # Yield two messages then stop
        yield {
            "type": "trade",
            "product_id": symbols[0],
            "price": "100",
            "time": "2024-01-01T00:00:00Z",
        }
        yield {
            "type": "trade",
            "product_id": symbols[0],
            "price": "101",
            "time": "2024-01-01T00:00:01Z",
        }
        return

    bot.broker.stream_orderbook = stream_orderbook_unavail  # type: ignore[attr-defined]
    bot.broker.stream_trades = stream_trades_once  # type: ignore[attr-defined]

    # Start streaming now
    stream_task = await bot.telemetry_coordinator._start_streaming()
    assert stream_task is not None
    await asyncio.wait_for(stream_task, timeout=2.0)

    events = bot.event_store.tail(bot_id="perps_bot", limit=50)
    assert any(e.get("event_type") == "ws_mark_update" for e in events), events

    # Verify risk manager got a mark timestamp for at least one symbol
    assert any(sym in bot.risk_manager.last_mark_update for sym in config.symbols)

    await bot.telemetry_coordinator.shutdown()
