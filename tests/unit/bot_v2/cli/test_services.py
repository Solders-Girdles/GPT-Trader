from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import bot_v2.cli.services as services


def test_resolve_settings_prefers_call_argument(runtime_settings_factory):
    snapshot = runtime_settings_factory()
    assert services._resolve_settings(snapshot) is snapshot


def test_resolve_settings_uses_override(monkeypatch, runtime_settings_factory):
    override = runtime_settings_factory(env_overrides={"TRADING_SYMBOLS": "BTC-PERP"})
    monkeypatch.setattr(services, "OVERRIDE_SETTINGS", override)
    assert services._resolve_settings(None) is override


def test_resolve_settings_defers_to_provider(monkeypatch, runtime_settings_factory):
    provided = runtime_settings_factory(env_overrides={"TRADING_SYMBOLS": "ETH-PERP"})
    provider = SimpleNamespace(get=lambda: provided)
    monkeypatch.setattr(services, "OVERRIDE_SETTINGS", None)
    monkeypatch.setattr(services, "SETTINGS_PROVIDER", provider)
    assert services._resolve_settings(None) is provided


def test_build_config_from_args_applies_filters(monkeypatch, runtime_settings_factory):
    settings = runtime_settings_factory(env_overrides={"TRADING_SYMBOLS": "BTC-PERP; ETH-PERP"})
    monkeypatch.setattr(services, "OVERRIDE_SETTINGS", None)
    monkeypatch.setattr(services, "SETTINGS_PROVIDER", SimpleNamespace(get=lambda: settings))

    captured: dict[str, object] = {}

    def fake_from_profile(cls, profile, *, settings: object, **overrides: object):
        captured["profile"] = profile
        captured["settings"] = settings
        captured["overrides"] = overrides
        return "config"

    monkeypatch.setattr(services.BotConfig, "from_profile", classmethod(fake_from_profile))

    args = Namespace(
        profile="demo",
        command="orders",
        handler=None,
        interval=15,
        dry_run=True,
        symbols=None,
        orders_command="preview",
    )

    result = services.build_config_from_args(
        args,
        include={"interval", "dry_run", "symbols"},
        skip={"orders_command"},
    )

    assert result == "config"
    assert captured["profile"] == "demo"
    assert captured["settings"] is settings
    overrides = captured["overrides"]
    assert overrides["dry_run"] is True
    assert overrides["update_interval"] == 15
    assert overrides["symbols"] == ["BTC-PERP", "ETH-PERP"]
    assert "interval" not in overrides


def test_build_config_respects_explicit_symbols(monkeypatch, runtime_settings_factory):
    settings = runtime_settings_factory(env_overrides={"TRADING_SYMBOLS": "BTC-PERP"})
    monkeypatch.setattr(services, "OVERRIDE_SETTINGS", None)
    monkeypatch.setattr(services, "SETTINGS_PROVIDER", SimpleNamespace(get=lambda: settings))

    captured: dict[str, object] = {}

    def fake_from_profile(cls, profile, *, settings: object, **overrides: object):
        captured["overrides"] = overrides
        return "config"

    monkeypatch.setattr(services.BotConfig, "from_profile", classmethod(fake_from_profile))

    args = Namespace(
        profile="dev",
        command="run",
        handler=None,
        symbols=["DOGE-PERP"],
        dry_run=False,
    )

    services.build_config_from_args(args)

    overrides = captured["overrides"]
    assert overrides["symbols"] == ["DOGE-PERP"]


def test_build_config_include_skips_unlisted_fields(monkeypatch, runtime_settings_factory):
    settings = runtime_settings_factory(env_overrides={})
    monkeypatch.setattr(services, "OVERRIDE_SETTINGS", None)
    monkeypatch.setattr(services, "SETTINGS_PROVIDER", SimpleNamespace(get=lambda: settings))

    captured: dict[str, object] = {}

    def fake_from_profile(cls, profile, *, settings: object, **overrides: object):
        captured["overrides"] = overrides
        captured["profile"] = profile
        return "config"

    monkeypatch.setattr(services.BotConfig, "from_profile", classmethod(fake_from_profile))

    args = Namespace(
        profile="demo",
        command="orders",
        handler=None,
        dry_run=True,
        interval=30,
    )

    services.build_config_from_args(args, include={"dry_run"})

    overrides = captured["overrides"]
    assert overrides == {"dry_run": True}


def test_build_config_ignores_empty_env_symbols(monkeypatch, runtime_settings_factory):
    settings = runtime_settings_factory(env_overrides={"TRADING_SYMBOLS": " ; ,"})
    monkeypatch.setattr(services, "OVERRIDE_SETTINGS", None)
    monkeypatch.setattr(services, "SETTINGS_PROVIDER", SimpleNamespace(get=lambda: settings))

    captured: dict[str, object] = {}

    def fake_from_profile(cls, profile, *, settings: object, **overrides: object):
        captured["overrides"] = overrides
        return "config"

    monkeypatch.setattr(services.BotConfig, "from_profile", classmethod(fake_from_profile))

    args = Namespace(
        profile="dev",
        command="run",
        handler=None,
        dry_run=False,
    )

    services.build_config_from_args(args)

    assert "symbols" not in captured["overrides"]


def test_instantiate_bot_uses_builder(monkeypatch):
    config = object()
    sentinel_bot = object()
    provider = object()
    monkeypatch.setattr(services, "SETTINGS_PROVIDER", provider)

    def fake_build_bot(passed_config, *, settings_provider):
        assert passed_config is config
        assert settings_provider is provider
        return sentinel_bot, {"registry": True}

    monkeypatch.setattr(services, "build_bot", fake_build_bot)

    assert services.instantiate_bot(config) is sentinel_bot
