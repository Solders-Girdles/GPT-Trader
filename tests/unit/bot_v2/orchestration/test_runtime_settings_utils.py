from __future__ import annotations

import logging
from pathlib import Path

import pytest

from bot_v2.orchestration import runtime_settings


@pytest.fixture(autouse=True)
def reset_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure module-level caches do not leak between tests."""
    monkeypatch.setattr(runtime_settings, "_SNAPSHOT_CACHE", None)
    monkeypatch.setattr(runtime_settings, "_OVERRIDE_SNAPSHOT", None)


def test_load_runtime_settings_parses_env_flags(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, runtime_settings.logger.name)
    env = {
        "GPT_TRADER_RUNTIME_ROOT": "/tmp/runtime",
        "EVENT_STORE_ROOT": "/tmp/events",
        "COINBASE_DEFAULT_QUOTE": "eur",
        "COINBASE_ENABLE_DERIVATIVES": "true",
        "PERPS_ENABLE_STREAMING": "yes",
        "PERPS_STREAM_LEVEL": "abc",  # invalid, should trigger fallback warning
        "PERPS_PAPER": "on",
        "PERPS_FORCE_MOCK": "true",
        "PERPS_SKIP_RECONCILE": "TrUe",
        "PERPS_POSITION_FRACTION": "not-a-number",  # invalid, should warn and fall back
        "ORDER_PREVIEW_ENABLED": "0",
        "SPOT_FORCE_LIVE": "1",
        "BROKER": "ADVANCED",
        "COINBASE_SANDBOX": "yes",
        "COINBASE_API_MODE": "Advanced",
        "RISK_CONFIG_PATH": "/etc/risk.json",
        "COINBASE_INTX_PORTFOLIO_UUID": "portfolio-123",
    }

    settings = runtime_settings.load_runtime_settings(env)

    assert settings.runtime_root == Path("/tmp/runtime")
    assert settings.event_store_root_override == Path("/tmp/events")
    assert settings.coinbase_default_quote == "EUR"
    assert settings.coinbase_default_quote_overridden is True
    assert settings.coinbase_enable_derivatives is True
    assert settings.coinbase_enable_derivatives_overridden is True
    assert settings.perps_enable_streaming is True
    assert settings.perps_stream_level == 1  # fallback to default
    assert settings.perps_paper_trading is True
    assert settings.perps_force_mock is True
    assert settings.perps_skip_startup_reconcile is True
    assert settings.perps_position_fraction is None
    assert settings.order_preview_enabled is False
    assert settings.spot_force_live is True
    assert settings.broker_hint == "advanced"
    assert settings.coinbase_sandbox_enabled is True
    assert settings.coinbase_api_mode == "advanced"
    assert settings.risk_config_path == Path("/etc/risk.json")
    assert settings.coinbase_intx_portfolio_uuid == "portfolio-123"

    snapshot = settings.snapshot_env(["COINBASE_DEFAULT_QUOTE", "MISSING"])
    assert snapshot == {"COINBASE_DEFAULT_QUOTE": "eur", "MISSING": None}

    warning_text = caplog.text
    assert "Invalid PERPS_STREAM_LEVEL=abc" in warning_text
    assert "Invalid PERPS_POSITION_FRACTION=not-a-number" in warning_text


def _make_settings(label: str) -> runtime_settings.RuntimeSettings:
    return runtime_settings.RuntimeSettings(
        raw_env={"LABEL": label},
        runtime_root=Path(f"/tmp/{label}"),
        event_store_root_override=None,
        coinbase_default_quote="USD",
        coinbase_default_quote_overridden=False,
        coinbase_enable_derivatives=False,
        coinbase_enable_derivatives_overridden=False,
        perps_enable_streaming=False,
        perps_stream_level=1,
        perps_paper_trading=False,
        perps_force_mock=False,
        perps_skip_startup_reconcile=False,
        perps_position_fraction=None,
        order_preview_enabled=None,
        spot_force_live=False,
        broker_hint=None,
        coinbase_sandbox_enabled=False,
        coinbase_api_mode="advanced",
        risk_config_path=None,
        coinbase_intx_portfolio_uuid=None,
    )


def test_get_runtime_settings_cache_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _make_settings("first")
    second = _make_settings("second")
    third = _make_settings("third")
    env_snapshot = _make_settings("env")
    override = _make_settings("override")

    calls: list[dict[str, str] | None] = []
    sequential = [first, second, third]

    def fake_loader(env: dict[str, str] | None = None) -> runtime_settings.RuntimeSettings:
        calls.append(env)
        if env is None:
            return sequential.pop(0)
        return env_snapshot

    monkeypatch.setattr(runtime_settings, "load_runtime_settings", fake_loader)

    # Initial call populates cache
    initial = runtime_settings.get_runtime_settings()
    assert initial is first

    # Cached call should not invoke loader again
    cached = runtime_settings.get_runtime_settings()
    assert cached is initial

    # Force refresh should fetch a new snapshot
    refreshed = runtime_settings.get_runtime_settings(force_refresh=True)
    assert refreshed is second

    # Override snapshot takes precedence over cache
    runtime_settings.set_runtime_settings_override(override)
    assert runtime_settings.get_runtime_settings() is override

    # Clearing override and cache should allow loader to run again
    runtime_settings.set_runtime_settings_override(None)
    runtime_settings.clear_runtime_settings_cache()
    after_clear = runtime_settings.get_runtime_settings()
    assert after_clear is third

    # Passing explicit env bypasses cache and override
    env_result = runtime_settings.get_runtime_settings(env={"SPECIAL": "1"})
    assert env_result is env_snapshot

    assert calls == [None, None, None, {"SPECIAL": "1"}]


def test_load_runtime_settings_logs_invalid_boolean(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, runtime_settings.logger.name)
    env = {
        "COINBASE_ENABLE_DERIVATIVES": "invalid",
        "PERPS_ENABLE_STREAMING": "not-bool",
        "PERPS_PAPER": "",
    }

    settings = runtime_settings.load_runtime_settings(env)

    # Invalid values should fall back to False
    assert settings.coinbase_enable_derivatives is False
    assert settings.perps_enable_streaming is False
    # Blank string should be treated as unset (no log)
    assert settings.perps_paper_trading is False

    warning_text = caplog.text
    assert "Invalid COINBASE_ENABLE_DERIVATIVES=invalid" in warning_text
    assert "Invalid PERPS_ENABLE_STREAMING=not-bool" in warning_text
    assert "PERPS_PAPER" not in warning_text
