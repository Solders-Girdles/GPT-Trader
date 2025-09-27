"""Unit tests for broker_factory.create_brokerage env-driven selection."""

import importlib
from types import SimpleNamespace


def _reload_factory(monkeypatch):
    # Ensure a clean import state per scenario
    if "bot_v2.orchestration.broker_factory" in list(importlib.sys.modules):
        del importlib.sys.modules["bot_v2.orchestration.broker_factory"]
    return importlib.import_module("bot_v2.orchestration.broker_factory")


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
        def __init__(self, cfg):
            captured["cfg"] = cfg

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    cfg = captured["cfg"]

    assert cfg.sandbox is True
    assert cfg.api_mode == "exchange"
    assert cfg.base_url.startswith("https://api-public.sandbox.exchange.coinbase.com")
    assert cfg.ws_url.startswith("wss://ws-feed-public.sandbox.exchange.coinbase.com")
    assert cfg.auth_type == "HMAC"


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
        def __init__(self, cfg):
            captured["cfg"] = cfg

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    cfg = captured["cfg"]
    assert cfg.sandbox is False
    assert cfg.api_mode == "advanced"
    assert cfg.base_url == "https://api.coinbase.com"
    assert cfg.ws_url == "wss://advanced-trade-ws.coinbase.com"
    assert cfg.auth_type == "JWT"


def test_unsupported_broker_raises(monkeypatch):
    monkeypatch.setenv("BROKER", "unknown")
    broker_factory = _reload_factory(monkeypatch)
    try:
        broker_factory.create_brokerage()
        assert False, "Expected ValueError for unsupported broker"
    except ValueError as e:
        assert "Unsupported broker" in str(e)
