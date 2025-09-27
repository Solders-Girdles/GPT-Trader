from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_v2.orchestration.perps_bot import (
    BotConfig,
    PerpsBot,
    Profile,
    DEFAULT_SPOT_SYMBOLS,
)


def test_spot_profile_uses_top_volume_symbols(monkeypatch, tmp_path):
    monkeypatch.delenv("RISK_CONFIG_PATH", raising=False)
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    # Ensure we stay on mock broker regardless of host env
    monkeypatch.delenv("SPOT_FORCE_LIVE", raising=False)

    cfg = BotConfig.from_profile(Profile.SPOT.value)
    assert cfg.symbols == DEFAULT_SPOT_SYMBOLS

    bot = PerpsBot(cfg)
    try:
        risk_cfg = bot.risk_manager.config
        assert risk_cfg.max_leverage == 1
        for symbol in DEFAULT_SPOT_SYMBOLS:
            assert risk_cfg.leverage_max_per_symbol.get(symbol) == 1
            assert symbol in risk_cfg.max_notional_per_symbol
        assert risk_cfg.max_notional_per_symbol["BTC-USD"] == Decimal("10000")
        assert risk_cfg.max_notional_per_symbol["DOGE-USD"] == Decimal("800")
    finally:
        asyncio.run(bot.shutdown())
