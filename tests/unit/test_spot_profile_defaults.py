from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_v2.orchestration.configuration import (
    BotConfig,
    Profile,
    DEFAULT_SPOT_SYMBOLS,
)
from bot_v2.orchestration.bootstrap import build_bot


def test_spot_profile_uses_top_volume_symbols(monkeypatch, tmp_path):
    monkeypatch.delenv("RISK_CONFIG_PATH", raising=False)
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    # Ensure we stay on mock broker regardless of host env
    monkeypatch.delenv("SPOT_FORCE_LIVE", raising=False)

    config = BotConfig.from_profile(Profile.SPOT.value)
    assert config.symbols == DEFAULT_SPOT_SYMBOLS

    bot, _registry = build_bot(config)
    try:
        risk_config = bot.risk_manager.config
        assert risk_config.max_leverage == 1
        for symbol in DEFAULT_SPOT_SYMBOLS:
            assert risk_config.leverage_max_per_symbol.get(symbol) == 1
            assert symbol in risk_config.max_notional_per_symbol
        assert risk_config.max_notional_per_symbol["BTC-USD"] == Decimal("10000")
        assert risk_config.max_notional_per_symbol["DOGE-USD"] == Decimal("800")
    finally:
        asyncio.run(bot.shutdown())
