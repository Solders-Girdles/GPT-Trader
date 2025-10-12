from pathlib import Path

import logging

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bootstrap import (
    normalise_symbols,
    prepare_perps_bot,
    resolve_runtime_paths,
)
from bot_v2.orchestration.service_registry import empty_registry
from bot_v2.persistence.event_store import EventStore


def test_normalise_symbols_filters_unsupported_perps():
    symbols, logs = normalise_symbols(
        ["BTC-PERP", "DOGE-PERP"],
        derivatives_enabled=True,
        default_quote="USD",
        allowed_perps={"BTC-PERP"},
    )

    assert symbols == ["BTC-PERP"]
    assert any("Filtering unsupported" in record.message for record in logs)


def test_normalise_symbols_replaces_perps_when_disabled():
    symbols, logs = normalise_symbols(
        ["btc-perp", "eth-usd"],
        derivatives_enabled=False,
        default_quote="usd",
    )

    assert symbols == ["BTC-USD", "ETH-USD"]
    assert any("Derivatives disabled" in record.message for record in logs)


def test_normalise_symbols_uses_fallback_when_empty():
    fallback_bases = ["BTC", "ETH"]
    symbols, logs = normalise_symbols(
        [],
        derivatives_enabled=False,
        default_quote="eur",
        fallback_bases=fallback_bases,
    )

    assert symbols == ["BTC-EUR", "ETH-EUR"]
    assert logs and logs[-1].level == logging.INFO


def test_resolve_runtime_paths_respects_environment(tmp_path):
    env = {
        "GPT_TRADER_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "EVENT_STORE_ROOT": str(tmp_path / "events"),
    }

    paths = resolve_runtime_paths(Profile.DEV, env)

    assert paths.storage_dir == Path(env["GPT_TRADER_RUNTIME_ROOT"]) / "perps_bot/dev"
    assert paths.storage_dir.exists()
    assert paths.event_store_root == Path(env["EVENT_STORE_ROOT"]) / "perps_bot/dev"
    assert paths.event_store_root.exists()


def test_prepare_perps_bot_reuses_injected_event_store(tmp_path):
    env = {
        "COINBASE_ENABLE_DERIVATIVES": "1",
        "GPT_TRADER_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "EVENT_STORE_ROOT": str(tmp_path / "events"),
    }

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP", "DOGE-PERP"])
    custom_store = EventStore(root=tmp_path / "custom")
    registry = empty_registry(config).with_updates(event_store=custom_store)

    result = prepare_perps_bot(config, registry, env=env)

    assert result.event_store is custom_store
    assert result.registry.event_store is custom_store
    assert result.orders_store is not None
    assert config.symbols == ["BTC-PERP"]
    assert (
        result.runtime_paths.storage_dir == Path(env["GPT_TRADER_RUNTIME_ROOT"]) / "perps_bot/dev"
    )
