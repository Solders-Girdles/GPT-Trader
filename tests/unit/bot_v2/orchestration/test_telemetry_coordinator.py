"""Unit tests for the telemetry coordinator."""

from __future__ import annotations

import pytest

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.perps_bot_builder import create_perps_bot


def test_init_accounting_services_sets_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = create_perps_bot(config)

    # Clear existing instances to exercise re-initialisation path.
    bot.account_manager = None  # type: ignore[assignment]
    bot.account_telemetry = None  # type: ignore[assignment]

    bot.telemetry_coordinator.init_accounting_services()

    assert isinstance(bot.account_manager, CoinbaseAccountManager)
    assert bot.account_telemetry is not None
    assert getattr(bot.system_monitor, "_account_telemetry", None) is bot.account_telemetry


def test_init_market_services_populates_monitor(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = create_perps_bot(config)

    bot.symbols = ["BTC-PERP", "ETH-PERP"]
    bot.telemetry_coordinator.bootstrap()

    monitor = getattr(bot, "_market_monitor", None)
    assert monitor is not None
    assert set(monitor.last_update.keys()) == set(bot.symbols)


@pytest.mark.asyncio
async def test_run_account_telemetry_respects_snapshot_support(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    bot = create_perps_bot(config)

    run_calls: list[int] = []

    async def fake_run(interval_seconds: int) -> None:
        run_calls.append(interval_seconds)

    bot.account_telemetry.run = fake_run  # type: ignore[assignment]
    bot.account_telemetry.supports_snapshots = lambda: False  # type: ignore[assignment]

    await bot.telemetry_coordinator.run_account_telemetry(interval_seconds=5)
    assert run_calls == []

    bot.account_telemetry.supports_snapshots = lambda: True  # type: ignore[assignment]
    await bot.telemetry_coordinator.run_account_telemetry(interval_seconds=5)
    assert run_calls == [5]
