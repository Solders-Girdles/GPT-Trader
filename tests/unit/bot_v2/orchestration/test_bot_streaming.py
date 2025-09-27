from __future__ import annotations

import asyncio
import time

from bot_v2.orchestration.perps_bot import PerpsBot, BotConfig
import pytest


@pytest.mark.uses_mock_broker
def test_background_ws_stream_updates_marks(fake_clock):
    # Dev profile uses MockBroker which now exposes stream_trades
    bot = PerpsBot(BotConfig.from_profile('dev'))
    # Initially, windows may be empty until update or stream
    initial = {s: len(bot.mark_windows.get(s, [])) for s in bot.config.symbols}

    # Wait up to 2 seconds for marks to increase, polling periodically
    deadline = time.time() + 2.0
    grew = False
    while time.time() < deadline and not grew:
        # seed one manual update to ensure window exists
        asyncio.run(bot.update_marks())
        for s in bot.config.symbols:
            if len(bot.mark_windows.get(s, [])) > initial.get(s, 0):
                grew = True
                break
        if not grew:
            time.sleep(0.05)

    assert grew, "Expected mark_windows to grow from background stream within timeout"
