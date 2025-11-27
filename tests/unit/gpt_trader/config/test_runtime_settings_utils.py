from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gpt_trader.config import runtime_settings
from gpt_trader.config.path_registry import RUNTIME_DATA_DIR


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


# ===== Enhanced Tests for Coverage Improvement =====


def test_normalize_bool_edge_cases(caplog: pytest.LogCaptureFixture) -> None:
    """Test _normalize_bool with various input scenarios."""
    caplog.set_level(logging.WARNING, runtime_settings.logger.name)

    # Test None input
    assert runtime_settings._normalize_bool(None) is None

    # Test valid boolean values through interpret_tristate_bool
    assert runtime_settings._normalize_bool("true") is True
    assert runtime_settings._normalize_bool("false") is False
    assert runtime_settings._normalize_bool("1") is True
    assert runtime_settings._normalize_bool("0") is False

    # Test invalid values with field name (should log warning)
    result = runtime_settings._normalize_bool("invalid", field_name="TEST_FIELD")
    assert result is None
    assert "Invalid TEST_FIELD=invalid" in caplog.text

    # Test invalid values without field name (no warning)
    caplog.clear()
    result = runtime_settings._normalize_bool("invalid", field_name=None)
    assert result is None
    assert caplog.text == ""  # No warning without field_name

    # Test empty string with field name - only logs if stripped value is truthy
    caplog.clear()
    result = runtime_settings._normalize_bool("", field_name="TEST_FIELD")
    assert result is None  # Empty string is invalid
    # Empty string doesn't log warning because stripped value is empty


def test_safe_int_and_float_error_handling(caplog: pytest.LogCaptureFixture) -> None:
    """Test _safe_int and _safe_float with invalid inputs."""
    caplog.set_level(logging.WARNING, runtime_settings.logger.name)

    # Test _safe_int with valid inputs
    assert runtime_settings._safe_int("42", fallback=10, field_name="TEST_INT") == 42
    assert runtime_settings._safe_int("-5", fallback=10, field_name="TEST_INT") == -5
    assert runtime_settings._safe_int("0", fallback=10, field_name="TEST_INT") == 0

    # Test _safe_int with invalid inputs
    result = runtime_settings._safe_int("not-a-number", fallback=10, field_name="TEST_INT")
    assert result == 10  # Should return fallback
    assert "Invalid TEST_INT=not-a-number; defaulting to 10" in caplog.text

    # Test _safe_int with None/empty
    assert runtime_settings._safe_int(None, fallback=5, field_name="TEST_INT") == 5
    assert runtime_settings._safe_int("", fallback=5, field_name="TEST_INT") == 5
    assert runtime_settings._safe_int("   ", fallback=5, field_name="TEST_INT") == 5

    # Test _safe_float with valid inputs
    assert runtime_settings._safe_float("3.14", field_name="TEST_FLOAT") == 3.14
    assert runtime_settings._safe_float("-2.5", field_name="TEST_FLOAT") == -2.5
    assert runtime_settings._safe_float("0", field_name="TEST_FLOAT") == 0.0

    # Test _safe_float with invalid inputs
    caplog.clear()
    result = runtime_settings._safe_float("not-a-float", field_name="TEST_FLOAT")
    assert result is None  # Should return None for floats
    assert "Invalid TEST_FLOAT=not-a-float; ignoring override" in caplog.text

    # Test _safe_float with None/empty
    assert runtime_settings._safe_float(None, field_name="TEST_FLOAT") is None
    assert runtime_settings._safe_float("", field_name="TEST_FLOAT") is None
    assert runtime_settings._safe_float("   ", field_name="TEST_FLOAT") is None


def test_runtime_settings_provider_custom_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test RuntimeSettingsProvider with custom loader scenarios."""

    # Create custom loader that returns predictable settings
    def custom_loader(env):
        return runtime_settings.RuntimeSettings(
            raw_env=env or {},
            runtime_root=Path("/custom"),
            event_store_root_override=Path("/custom/events"),
            coinbase_default_quote="USD",
            coinbase_default_quote_overridden=False,
            coinbase_enable_derivatives=True,
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

    provider = runtime_settings.RuntimeSettingsProvider(loader=custom_loader)

    # Test with custom env (bypasses caches and uses custom loader)
    custom_env = {"TEST_VAR": "test_value"}
    settings = provider.get(env=custom_env)
    assert settings.runtime_root == Path("/custom")
    assert settings.coinbase_enable_derivatives is True
    assert settings.raw_env == custom_env

    # Test override
    override_settings = _make_settings("override")
    provider.override(override_settings)
    assert provider.get() is override_settings


def test_load_runtime_settings_edge_cases(caplog: pytest.LogCaptureFixture) -> None:
    """Test edge cases in load_runtime_settings."""
    caplog.set_level(logging.WARNING, runtime_settings.logger.name)

    # Test broker hint None/empty - empty string stays empty string after lower() conversion
    env_empty_broker = {
        "GPT_TRADER_RUNTIME_ROOT": "/tmp/runtime",
        "BROKER": "",
        "COINBASE_DEFAULT_QUOTE": "USD",
    }
    settings = runtime_settings.load_runtime_settings(env_empty_broker)
    assert settings.broker_hint == ""  # Empty string stays empty string after lower() conversion
    assert settings.runtime_root == Path("/tmp/runtime")

    # Test missing runtime root (should use default)
    env_missing_root = {
        "COINBASE_DEFAULT_QUOTE": "USD",
    }
    settings = runtime_settings.load_runtime_settings(env_missing_root)
    assert settings.runtime_root == RUNTIME_DATA_DIR

    # Test intx portfolio uuid with various values
    env_intx = {
        "GPT_TRADER_RUNTIME_ROOT": "/tmp/runtime",
        "COINBASE_DEFAULT_QUOTE": "USD",
        "COINBASE_INTX_PORTFOLIO_UUID": "portfolio-abc-123",
        "BROKER": "intx",
    }
    settings = runtime_settings.load_runtime_settings(env_intx)
    assert settings.coinbase_intx_portfolio_uuid == "portfolio-abc-123"
    assert settings.broker_hint == "intx"

    # Test invalid path values (whitespace creates Path object, empty uses default)
    env_invalid_path = {
        "GPT_TRADER_RUNTIME_ROOT": "",  # Empty should use default
        "EVENT_STORE_ROOT": "   ",  # Whitespace creates Path with whitespace
        "COINBASE_DEFAULT_QUOTE": "USD",
    }
    settings = runtime_settings.load_runtime_settings(env_invalid_path)
    assert settings.runtime_root == RUNTIME_DATA_DIR  # Default fallback for empty
    assert settings.event_store_root_override == Path("   ")  # Whitespace becomes Path


def test_runtime_settings_snapshot_methods() -> None:
    """Test RuntimeSettings.snapshot_env with various inputs."""
    # Create settings with raw_env for snapshot testing
    settings_with_raw = runtime_settings.RuntimeSettings(
        raw_env={"COINBASE_DEFAULT_QUOTE": "USD", "PERPS_ENABLE_STREAMING": "true"},
        runtime_root=Path("/test"),
        event_store_root_override=None,
        coinbase_default_quote="USD",
        coinbase_default_quote_overridden=False,
        coinbase_enable_derivatives=False,
        coinbase_enable_derivatives_overridden=False,
        perps_enable_streaming=True,
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

    # Test snapshot with existing keys
    keys = ["COINBASE_DEFAULT_QUOTE", "PERPS_ENABLE_STREAMING"]
    snapshot = settings_with_raw.snapshot_env(keys)
    assert snapshot == {
        "COINBASE_DEFAULT_QUOTE": "USD",
        "PERPS_ENABLE_STREAMING": "true",
    }

    # Test snapshot with tuple of keys
    snapshot_tuple = settings_with_raw.snapshot_env(("COINBASE_DEFAULT_QUOTE", "NEW_KEY"))
    assert snapshot_tuple == {
        "COINBASE_DEFAULT_QUOTE": "USD",
        "NEW_KEY": None,
    }

    # Test snapshot with mixed types (list containing non-str)
    mixed_keys = ["COINBASE_DEFAULT_QUOTE", 123, None]
    snapshot_mixed = settings_with_raw.snapshot_env(mixed_keys)
    assert snapshot_mixed == {
        "COINBASE_DEFAULT_QUOTE": "USD",
        123: None,
        None: None,
    }

    # Test snapshot with empty list
    empty_snapshot = settings_with_raw.snapshot_env([])
    assert empty_snapshot == {}

    # Test snapshot with mapping input
    mapping_keys = {"COINBASE_DEFAULT_QUOTE": "some_value", "PERPS_ENABLE_STREAMING": "another"}
    mapping_snapshot = settings_with_raw.snapshot_env(mapping_keys)
    assert mapping_snapshot == {
        "COINBASE_DEFAULT_QUOTE": "USD",
        "PERPS_ENABLE_STREAMING": "true",
    }


def test_runtime_settings_provider_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test RuntimeSettingsProvider.clear() method."""
    provider = runtime_settings.RuntimeSettingsProvider()

    # Set up initial state
    initial = _make_settings("initial")
    monkeypatch.setattr(runtime_settings, "_SNAPSHOT_CACHE", initial)

    # Call clear on provider
    provider.clear()

    # Verify cache was cleared
    assert runtime_settings._SNAPSHOT_CACHE is None
