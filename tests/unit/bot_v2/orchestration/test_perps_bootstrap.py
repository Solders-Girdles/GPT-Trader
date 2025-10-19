import logging
from pathlib import Path

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bootstrap import (
    normalise_symbols,
    prepare_perps_bot,
    resolve_runtime_paths,
)
from bot_v2.orchestration.runtime_settings import load_runtime_settings
from bot_v2.orchestration.service_registry import empty_registry
from bot_v2.persistence.event_store import EventStore


def test_normalise_symbols_filters_unsupported_perps():
    settings = load_runtime_settings(
        {"COINBASE_ENABLE_DERIVATIVES": "1", "COINBASE_DEFAULT_QUOTE": "USD"}
    )
    symbols, logs = normalise_symbols(
        ["BTC-PERP", "DOGE-PERP"],
        settings=settings,
        allowed_perps={"BTC-PERP"},
    )

    assert symbols == ["BTC-PERP"]
    assert any("Filtering unsupported" in record.message for record in logs)


def test_normalise_symbols_replaces_perps_when_disabled():
    settings = load_runtime_settings(
        {"COINBASE_ENABLE_DERIVATIVES": "0", "COINBASE_DEFAULT_QUOTE": "USD"}
    )
    symbols, logs = normalise_symbols(
        ["btc-perp", "eth-usd"],
        settings=settings,
    )

    assert symbols == ["BTC-USD", "ETH-USD"]
    assert any("Derivatives disabled" in record.message for record in logs)


def test_normalise_symbols_uses_fallback_when_empty():
    fallback_bases = ["BTC", "ETH"]
    settings = load_runtime_settings(
        {"COINBASE_ENABLE_DERIVATIVES": "0", "COINBASE_DEFAULT_QUOTE": "eur"}
    )
    symbols, logs = normalise_symbols(
        [],
        settings=settings,
        fallback_bases=fallback_bases,
    )

    assert symbols == ["BTC-EUR", "ETH-EUR"]
    assert logs and logs[-1].level == logging.INFO


def test_resolve_runtime_paths_respects_environment(tmp_path):
    env = {
        "GPT_TRADER_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "EVENT_STORE_ROOT": str(tmp_path / "events"),
    }

    settings = load_runtime_settings(env)

    paths = resolve_runtime_paths(Profile.DEV, settings)

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


def test_prepare_perps_bot_reapplies_symbol_normalization_with_env_overrides(tmp_path):
    env_disable = {
        "COINBASE_ENABLE_DERIVATIVES": "0",
        "COINBASE_DEFAULT_QUOTE": "eur",
        "GPT_TRADER_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "EVENT_STORE_ROOT": str(tmp_path / "events"),
    }

    config = BotConfig(profile=Profile.DEV, symbols=["BTC-PERP"])

    result_disable = prepare_perps_bot(config, env=env_disable)
    disable_config = result_disable.config

    assert disable_config is config
    assert disable_config.symbols == ["BTC-EUR"]
    assert disable_config.derivatives_enabled is False
    assert disable_config.metadata.get("symbol_normalization_overrides") == {
        "quote": "EUR",
        "allow_derivatives": False,
    }
    assert any("Derivatives disabled" in record.message for record in result_disable.logs)

    env_enable = {
        "COINBASE_ENABLE_DERIVATIVES": "1",
        "COINBASE_DEFAULT_QUOTE": "usd",
        "GPT_TRADER_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "EVENT_STORE_ROOT": str(tmp_path / "events"),
    }

    result_enable = prepare_perps_bot(disable_config, env=env_enable)
    enable_config = result_enable.config

    assert enable_config is disable_config
    assert enable_config.symbols == ["BTC-PERP"]
    assert enable_config.derivatives_enabled is True
    assert enable_config.metadata.get("symbol_normalization_overrides") == {
        "quote": "USD",
        "allow_derivatives": True,
    }
