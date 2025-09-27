"""Tests for orchestration bootstrap helpers."""

from pathlib import Path

import pytest

from bot_v2.orchestration.bootstrap import build_bot, build_service_registry, bot_from_profile
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.persistence.event_store import EventStore


def test_build_bot_populates_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(
        "bot_v2.orchestration.perps_bot.PerpsBot._start_streaming_background", lambda self: None
    )

    config = BotConfig.from_profile("dev")
    bot, registry = build_bot(config)

    assert bot.config is config
    assert registry.event_store is not None
    assert registry.orders_store is not None
    assert registry.broker is not None


def test_build_service_registry_respects_injected_event_store(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    config = BotConfig.from_profile("dev")

    custom_store = EventStore(root=Path(tmp_path) / "custom")
    registry = build_service_registry(config).with_updates(event_store=custom_store)
    bot, updated_registry = build_bot(config, registry=registry)

    assert bot.event_store is custom_store
    assert updated_registry.event_store is custom_store


def test_bot_from_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setenv("PERPS_FORCE_MOCK", "1")
    monkeypatch.setattr(
        "bot_v2.orchestration.perps_bot.PerpsBot._start_streaming_background", lambda self: None
    )

    bot, registry = bot_from_profile("dev")
    assert bot.config.profile.value == "dev"
    assert registry.config is bot.config
