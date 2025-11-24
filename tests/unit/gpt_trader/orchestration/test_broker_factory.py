"""Unit tests for broker_factory.create_brokerage env-driven selection."""

import importlib


def _reload_factory(monkeypatch):
    # Ensure a clean import state per scenario
    if "gpt_trader.orchestration.broker_factory" in list(importlib.sys.modules):
        del importlib.sys.modules["gpt_trader.orchestration.broker_factory"]
    return importlib.import_module("gpt_trader.orchestration.broker_factory")


def test_coinbase_sandbox_exchange_mode(monkeypatch):
    # Env selection for coinbase sandbox in exchange mode
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "1")
    monkeypatch.setenv("COINBASE_SANDBOX_API_KEY", "sandbox_key")
    monkeypatch.setenv("COINBASE_SANDBOX_API_SECRET", "sandbox_secret")
    monkeypatch.delenv("COINBASE_API_MODE", raising=False)
    monkeypatch.delenv("COINBASE_API_BASE", raising=False)
    monkeypatch.delenv("COINBASE_WS_URL", raising=False)

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config, **kwargs):
            captured["api_config"] = api_config
            captured["kwargs"] = kwargs

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    broker, event_store, market_data, product_catalog = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.sandbox is True
    assert api_config.api_mode == "exchange"
    assert api_config.base_url.startswith("https://api-public.sandbox.exchange.coinbase.com")
    assert api_config.ws_url.startswith("wss://ws-feed-public.sandbox.exchange.coinbase.com")
    assert api_config.auth_type == "HMAC"


def test_coinbase_prod_advanced_jwt(monkeypatch):
    # Env for production, advanced mode with JWT available
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "0")
    monkeypatch.setenv("COINBASE_CDP_API_KEY", "k")
    monkeypatch.setenv("COINBASE_CDP_PRIVATE_KEY", "p")
    monkeypatch.setenv("COINBASE_API_MODE", "advanced")
    monkeypatch.delenv("COINBASE_API_BASE", raising=False)
    monkeypatch.delenv("COINBASE_WS_URL", raising=False)

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config, **kwargs):
            captured["api_config"] = api_config
            captured["kwargs"] = kwargs

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    broker_factory.create_brokerage()
    api_config = captured["api_config"]
    assert api_config.sandbox is False
    assert api_config.api_mode == "advanced"
    assert api_config.base_url == "https://api.coinbase.com"
    assert api_config.ws_url == "wss://advanced-trade-ws.coinbase.com"
    assert api_config.auth_type == "JWT"


def test_unsupported_broker_raises(monkeypatch):
    monkeypatch.setenv("BROKER", "unknown")
    broker_factory = _reload_factory(monkeypatch)
    try:
        broker_factory.create_brokerage()
        assert False, "Expected ValueError for unsupported broker"
    except ValueError as e:
        assert "Unsupported broker" in str(e)
