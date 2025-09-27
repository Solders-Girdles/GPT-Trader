from __future__ import annotations

import os
import time

from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig, Profile


def test_perps_streaming_smoke(monkeypatch, tmp_path, fake_clock):
    # Ensure streaming does not auto-start; we'll start it after patching broker
    monkeypatch.setenv('PERPS_ENABLE_STREAMING', '0')
    monkeypatch.setenv('PERPS_FORCE_MOCK', '1')
    # Set event store root to temp dir
    monkeypatch.setenv('EVENT_STORE_ROOT', str(tmp_path))

    # CANARY profile to allow streaming
    cfg = BotConfig.from_profile('canary')

    # Start bot; streaming thread won't start due to gate
    bot = PerpsBot(cfg)

    # Monkeypatch broker streams to yield a couple messages
    def stream_orderbook_unavail(symbols, level=1):
        raise AttributeError("no orderbook stream")

    def stream_trades_once(symbols):
        # Yield two messages then stop
        yield {"type": "trade", "product_id": symbols[0], "price": "100", "time": "2024-01-01T00:00:00Z"}
        yield {"type": "trade", "product_id": symbols[0], "price": "101", "time": "2024-01-01T00:00:01Z"}
        return

    bot.broker.stream_orderbook = stream_orderbook_unavail  # type: ignore[attr-defined]
    bot.broker.stream_trades = stream_trades_once  # type: ignore[attr-defined]

    # Start streaming now
    bot._start_streaming_background()

    from bot_v2.persistence.event_store import EventStore
    es = EventStore(root=tmp_path / f"perps_bot/{cfg.profile.value}")

    deadline = time.time() + 2.0
    has_ws_update = False
    events = []
    while time.time() < deadline and not has_ws_update:
        events = es.tail(bot_id='perps_bot', limit=50)
        has_ws_update = any(e.get('event_type') == 'ws_mark_update' for e in events)
        if not has_ws_update:
            time.sleep(0.05)

    assert has_ws_update, f"No ws_mark_update events found in {events}"

    # Verify risk manager got a mark timestamp for at least one symbol
    assert any(sym in bot.risk_manager.last_mark_update for sym in cfg.symbols)
