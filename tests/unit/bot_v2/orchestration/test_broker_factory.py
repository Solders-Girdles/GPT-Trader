"""Unit tests for broker_factory.create_brokerage env-driven selection.

This module tests the BrokerFactory's ability to correctly construct brokerage
adapters based on environment variables and configuration. Tests verify:

- Credential selection priority (PROD-specific vs generic env vars)
- API mode detection (advanced vs exchange)
- Auth type selection (JWT vs HMAC)
- Sandbox vs production URL selection
- Custom URL override behavior
- Error handling for missing/invalid credentials

Production Context:
    The broker factory is the entry point for all trading operations. Incorrect
    configuration here could result in:
    - Production trades executed in sandbox (financial loss)
    - Sandbox credentials used in production (authentication failures)
    - Wrong auth type causing all API calls to fail
    - Missing credentials causing silent failures or startup crashes
"""

import importlib
from types import SimpleNamespace


def _reload_factory(monkeypatch):
    """Ensure a clean import state per scenario by reloading the module."""
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
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
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
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]
    assert api_config.sandbox is False
    assert api_config.api_mode == "advanced"
    assert api_config.base_url == "https://api.coinbase.com"
    assert api_config.ws_url == "wss://advanced-trade-ws.coinbase.com"
    assert api_config.auth_type == "JWT"


def test_unsupported_broker_raises(monkeypatch):
    """Attempting to create unsupported broker raises ValueError.

    Safety check: If a typo or invalid broker name is configured,
    the system should fail fast at startup rather than proceeding
    with undefined behavior.
    """
    monkeypatch.setenv("BROKER", "unknown")
    broker_factory = _reload_factory(monkeypatch)
    try:
        broker_factory.create_brokerage()
        assert False, "Expected ValueError for unsupported broker"
    except ValueError as e:
        assert "Unsupported broker" in str(e)


def test_credential_priority_prod_over_generic(monkeypatch):
    """PROD-specific env vars take priority over generic ones.

    Critical for production safety: When both COINBASE_PROD_API_KEY and
    COINBASE_API_KEY are set, PROD-specific should win. This prevents
    accidentally using sandbox credentials in production.
    """
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "0")
    monkeypatch.setenv("COINBASE_API_KEY", "generic_key")
    monkeypatch.setenv("COINBASE_API_SECRET", "generic_secret")
    monkeypatch.setenv("COINBASE_PROD_API_KEY", "prod_key")
    monkeypatch.setenv("COINBASE_PROD_API_SECRET", "prod_secret")
    monkeypatch.setenv("COINBASE_API_MODE", "advanced")

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.api_key == "prod_key"
    assert api_config.api_secret == "prod_secret"


def test_credential_priority_sandbox_over_generic(monkeypatch):
    """Sandbox-specific env vars take priority over generic ones.

    Ensures sandbox testing uses sandbox credentials when both are present.
    """
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "1")
    monkeypatch.setenv("COINBASE_API_KEY", "generic_key")
    monkeypatch.setenv("COINBASE_API_SECRET", "generic_secret")
    monkeypatch.setenv("COINBASE_SANDBOX_API_KEY", "sandbox_key")
    monkeypatch.setenv("COINBASE_SANDBOX_API_SECRET", "sandbox_secret")

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.api_key == "sandbox_key"
    assert api_config.api_secret == "sandbox_secret"


def test_auth_type_detection_jwt_when_cdp_keys_present(monkeypatch):
    """JWT auth is selected when CDP keys are present in advanced mode.

    Advanced Trade API uses JWT authentication when CDP (Cloud Developer
    Platform) keys are available.
    """
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "0")
    monkeypatch.setenv("COINBASE_CDP_API_KEY", "cdp_key")
    monkeypatch.setenv("COINBASE_CDP_PRIVATE_KEY", "cdp_private")
    monkeypatch.setenv("COINBASE_API_MODE", "advanced")

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.auth_type == "JWT"


def test_auth_type_hmac_for_exchange_mode(monkeypatch):
    """HMAC auth is always used in exchange mode.

    Exchange API (legacy) only supports HMAC authentication, regardless
    of whether CDP keys are present.
    """
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "1")
    monkeypatch.setenv("COINBASE_SANDBOX_API_KEY", "key")
    monkeypatch.setenv("COINBASE_SANDBOX_API_SECRET", "secret")
    monkeypatch.setenv("COINBASE_SANDBOX_API_PASSPHRASE", "pass")
    # Even with CDP keys present, exchange mode should use HMAC
    monkeypatch.setenv("COINBASE_CDP_API_KEY", "cdp_key")
    monkeypatch.setenv("COINBASE_CDP_PRIVATE_KEY", "cdp_private")

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.auth_type == "HMAC"
    assert api_config.api_mode == "exchange"


def test_custom_url_overrides(monkeypatch):
    """Custom COINBASE_API_BASE and COINBASE_WS_URL override defaults.

    Allows advanced users to point to custom endpoints (e.g., proxy,
    mock server) for testing or special network setups.
    """
    custom_base = "https://custom.coinbase.example.com"
    custom_ws = "wss://custom-ws.coinbase.example.com"

    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "0")
    monkeypatch.setenv("COINBASE_API_KEY", "key")
    monkeypatch.setenv("COINBASE_API_SECRET", "secret")
    monkeypatch.setenv("COINBASE_API_BASE", custom_base)
    monkeypatch.setenv("COINBASE_WS_URL", custom_ws)
    monkeypatch.setenv("COINBASE_API_MODE", "advanced")

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.base_url == custom_base
    assert api_config.ws_url == custom_ws


def test_api_mode_auto_detection_defaults_to_advanced(monkeypatch):
    """When COINBASE_API_MODE not set, production defaults to advanced.

    Advanced Trade API is the modern, recommended API for production.
    """
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "0")
    monkeypatch.setenv("COINBASE_API_KEY", "key")
    monkeypatch.setenv("COINBASE_API_SECRET", "secret")
    monkeypatch.delenv("COINBASE_API_MODE", raising=False)

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.api_mode == "advanced"
    assert api_config.base_url == "https://api.coinbase.com"


def test_sandbox_forces_exchange_mode(monkeypatch):
    """Sandbox mode always uses exchange API regardless of COINBASE_API_MODE.

    Critical behavior: Advanced Trade API doesn't have a public sandbox,
    so sandbox requests must use exchange mode even if user sets
    COINBASE_API_MODE=advanced.
    """
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "1")
    monkeypatch.setenv("COINBASE_SANDBOX_API_KEY", "key")
    monkeypatch.setenv("COINBASE_SANDBOX_API_SECRET", "secret")
    monkeypatch.setenv("COINBASE_SANDBOX_API_PASSPHRASE", "pass")
    # Try to set advanced mode - should be ignored
    monkeypatch.setenv("COINBASE_API_MODE", "advanced")

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    # Sandbox must override to exchange mode
    assert api_config.api_mode == "exchange"
    assert "sandbox.exchange" in api_config.base_url


def test_derivatives_flag_disabled_by_default(monkeypatch):
    """Derivatives trading is disabled by default.

    Conservative default: Perpetuals and futures require explicit opt-in
    via COINBASE_ENABLE_DERIVATIVES=1.
    """
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "0")
    monkeypatch.setenv("COINBASE_API_KEY", "key")
    monkeypatch.setenv("COINBASE_API_SECRET", "secret")
    monkeypatch.delenv("COINBASE_ENABLE_DERIVATIVES", raising=False)

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.enable_derivatives is False


def test_derivatives_flag_enabled_explicitly(monkeypatch):
    """Derivatives trading enabled when COINBASE_ENABLE_DERIVATIVES=1.

    Allows bots to trade perpetual futures when explicitly configured.
    """
    monkeypatch.setenv("BROKER", "coinbase")
    monkeypatch.setenv("COINBASE_SANDBOX", "0")
    monkeypatch.setenv("COINBASE_API_KEY", "key")
    monkeypatch.setenv("COINBASE_API_SECRET", "secret")
    monkeypatch.setenv("COINBASE_ENABLE_DERIVATIVES", "1")

    broker_factory = _reload_factory(monkeypatch)

    captured = {}

    class DummyBroker:
        def __init__(self, api_config):
            captured["api_config"] = api_config

    monkeypatch.setattr(broker_factory, "CoinbaseBrokerage", DummyBroker)

    _ = broker_factory.create_brokerage()
    api_config = captured["api_config"]

    assert api_config.enable_derivatives is True
