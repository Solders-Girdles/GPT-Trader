"""Unit tests for the telemetry coordinator."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.perps_bot_builder import create_perps_bot
from bot_v2.orchestration.service_registry import ServiceRegistry


def test_init_accounting_services_sets_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "0")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP", "ETH-PERP"], update_interval=1)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)

    context = bot.telemetry_coordinator.context.with_updates(symbols=tuple(bot.symbols))
    updated = bot.telemetry_coordinator.initialize(context)
    bot.telemetry_coordinator.update_context(updated)
    bot.registry = updated.registry

    # Clear existing instances to exercise re-initialisation path.
    bot.account_manager = None  # type: ignore[assignment]
    bot.account_telemetry = None  # type: ignore[assignment]

    updated = bot.telemetry_coordinator.initialize(bot.telemetry_coordinator.context)
    bot.telemetry_coordinator.update_context(updated)

    bot.registry = updated.registry
    extras = updated.registry.extras
    bot.account_manager = extras.get("account_manager")  # type: ignore[assignment]
    bot.account_telemetry = extras.get("account_telemetry")  # type: ignore[assignment]
    bot.intx_portfolio_service = extras.get("intx_portfolio_service")  # type: ignore[attr-defined]

    assert isinstance(bot.account_manager, CoinbaseAccountManager)
    assert bot.account_telemetry is not None
    assert getattr(bot, "intx_portfolio_service", None) is not None
    assert "intx_portfolio_service" in bot.registry.extras


def test_init_market_services_populates_monitor(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "0")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)
    updated = bot.telemetry_coordinator.initialize(bot.telemetry_coordinator.context)
    bot.telemetry_coordinator.update_context(updated)
    bot.registry = updated.registry

    monitor = bot.telemetry_coordinator._market_monitor
    assert monitor is not None
    assert set(monitor.last_update.keys()) == set(bot.symbols)


@pytest.mark.asyncio
async def test_run_account_telemetry_respects_snapshot_support(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "0")
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"], update_interval=1)
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    registry = ServiceRegistry(config=config, broker=broker)
    bot = create_perps_bot(config, registry=registry)

    run_calls: list[int] = []

    async def fake_run(interval_seconds: int) -> None:
        run_calls.append(interval_seconds)

    bot.account_telemetry.run = fake_run  # type: ignore[assignment]
    bot.account_telemetry.supports_snapshots = lambda: False  # type: ignore[assignment]

    await bot.telemetry_coordinator._run_account_telemetry(interval_seconds=5)
    assert run_calls == []

    bot.account_telemetry.supports_snapshots = lambda: True  # type: ignore[assignment]
    await bot.telemetry_coordinator._run_account_telemetry(interval_seconds=5)
    assert run_calls == [5]
