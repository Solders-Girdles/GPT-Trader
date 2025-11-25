"""Enhanced tests for symbols normalization module to improve coverage."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from gpt_trader.config.runtime_settings import RuntimeSettings
from gpt_trader.orchestration import symbols
from gpt_trader.orchestration.configuration import Profile


@pytest.fixture(autouse=True)
def reset_symbols_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset any module-level state between tests."""
    # No module-level cache in symbols.py, but fixture for consistency
    pass


def _make_runtime_settings(
    *,
    coinbase_enable_derivatives: bool = False,
    coinbase_enable_derivatives_overridden: bool = False,
    coinbase_default_quote: str = "USD",
    **kwargs: dict,
) -> RuntimeSettings:
    """Create a RuntimeSettings instance for testing."""
    return RuntimeSettings(
        raw_env={},
        runtime_root=kwargs.get("runtime_root", "/tmp"),
        event_store_root_override=kwargs.get("event_store_root_override"),
        coinbase_default_quote=coinbase_default_quote,
        coinbase_default_quote_overridden=kwargs.get("coinbase_default_quote_overridden", False),
        coinbase_enable_derivatives=coinbase_enable_derivatives,
        coinbase_enable_derivatives_overridden=coinbase_enable_derivatives_overridden,
        perps_enable_streaming=kwargs.get("perps_enable_streaming", False),
        perps_stream_level=kwargs.get("perps_stream_level", 1),
        perps_paper_trading=kwargs.get("perps_paper_trading", False),
        perps_force_mock=kwargs.get("perps_force_mock", False),
        perps_skip_startup_reconcile=kwargs.get("perps_skip_startup_reconcile", False),
        perps_position_fraction=kwargs.get("perps_position_fraction"),
        order_preview_enabled=kwargs.get("order_preview_enabled"),
        spot_force_live=kwargs.get("spot_force_live", False),
        broker_hint=kwargs.get("broker_hint"),
        coinbase_sandbox_enabled=kwargs.get("coinbase_sandbox_enabled", False),
        coinbase_api_mode=kwargs.get("coinbase_api_mode", "advanced"),
        risk_config_path=kwargs.get("risk_config_path"),
        coinbase_intx_portfolio_uuid=kwargs.get("coinbase_intx_portfolio_uuid"),
    )


def test_derivatives_enabled_with_spot_profile() -> None:
    """Test derivatives_enabled returns False for SPOT profile."""
    # Test with actual Profile.SPOT enum
    settings = _make_runtime_settings(coinbase_enable_derivatives=False)
    result = symbols.derivatives_enabled(Profile.SPOT, settings=settings)
    assert result is False

    # Test with string profile value
    string_profile = Mock()
    string_profile.value = "spot"
    result = symbols.derivatives_enabled(string_profile, settings=settings)
    assert result is False

    # Test with plain string
    result = symbols.derivatives_enabled("spot", settings=settings)
    assert result is False


def test_derivatives_enabled_with_runtime_override() -> None:
    """Test derivatives_enabled respects runtime settings override."""
    # Test when runtime settings override derivatives to enabled
    settings = _make_runtime_settings(
        coinbase_enable_derivatives=True, coinbase_enable_derivatives_overridden=True
    )
    result = symbols.derivatives_enabled(Profile.PROD, settings=settings)  # Use PROD, not SPOT
    assert result is True  # Runtime override should take precedence

    # Test when runtime settings override derivatives to disabled
    settings = _make_runtime_settings(
        coinbase_enable_derivatives=False, coinbase_enable_derivatives_overridden=True
    )
    result = symbols.derivatives_enabled(Profile.PROD, settings=settings)
    assert result is False  # Runtime override should take precedence


def test_derivatives_enabled_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test derivatives_enabled with edge cases and import failures."""
    settings = _make_runtime_settings(coinbase_enable_derivatives=False)

    # Test with None profile
    result = symbols.derivatives_enabled(None, settings=settings)
    assert result is True  # None profile doesn't disable derivatives

    # Test with profile object without value attribute
    profile_no_value = Mock()
    del profile_no_value.value
    result = symbols.derivatives_enabled(profile_no_value, settings=settings)
    assert result is True


def test_normalize_symbol_list_with_derivatives_enabled() -> None:
    """Test normalize_symbol_list with derivatives allowed."""
    # Test with allowed perps
    symbols_list = ["BTC-PERP", "ETH-PERP", "INVALID-PERP", "BTC-USD"]
    result, logs = symbols.normalize_symbol_list(
        symbols_list, allow_derivatives=True, quote="USD", allowed_perps=["BTC-PERP", "ETH-PERP"]
    )

    assert result == ["BTC-PERP", "ETH-PERP", "BTC-USD"]
    assert len(logs) == 1
    assert logs[0].level == logging.WARNING
    assert logs[0].message == "Filtering unsupported perpetual symbol %s. Allowed perps: %s"
    assert logs[0].args == ("INVALID-PERP", ["BTC-PERP", "ETH-PERP"])


def test_normalize_symbol_list_with_derivatives_disabled() -> None:
    """Test normalize_symbol_list with derivatives disabled."""
    symbols_list = ["BTC-PERP", "ETH-USD"]
    result, logs = symbols.normalize_symbol_list(symbols_list, allow_derivatives=False, quote="USD")

    assert result == ["BTC-USD", "ETH-USD"]
    assert len(logs) == 1
    assert logs[0].level == logging.WARNING
    assert logs[0].message == "Derivatives disabled. Replacing %s with spot symbol %s"
    assert logs[0].args == ("BTC-PERP", "BTC-USD")


def test_normalize_symbol_list_empty_and_whitespace() -> None:
    """Test normalize_symbol_list with empty and whitespace inputs."""
    # Test with None input
    result, logs = symbols.normalize_symbol_list(None, allow_derivatives=True, quote="USD")

    assert result == [
        "BTC-PERP",
        "ETH-PERP",
        "BTC-FUTURES",
        "ETH-FUTURES",
    ]  # Default fallback for derivatives
    assert len(logs) == 1
    assert logs[0].level == logging.INFO
    assert "No valid symbols provided. Falling back to" in logs[0].message

    # Test with empty strings and whitespace
    symbols_list = ["", "  ", "\t\n", "BTC-PERP"]
    result, logs = symbols.normalize_symbol_list(symbols_list, allow_derivatives=True, quote="USD")

    assert result == ["BTC-PERP"]
    assert logs == []  # No logs when valid symbols exist


def test_normalize_symbol_list_case_and_whitespace_normalization() -> None:
    """Test symbol normalization with various cases and whitespace."""
    symbols_list = ["  btc-perp  ", "\tETH-USD\n", "sol-usd"]
    result, logs = symbols.normalize_symbol_list(symbols_list, allow_derivatives=True, quote="USD")

    assert result == ["BTC-PERP", "ETH-USD", "SOL-USD"]
    assert logs == []


def test_normalize_symbol_list_custom_fallback_bases() -> None:
    """Test normalize_symbol_list with custom fallback bases."""
    # Test with derivatives disabled and custom fallback bases
    result, logs = symbols.normalize_symbol_list(
        [],  # Empty symbols to trigger fallback
        allow_derivatives=False,
        quote="EUR",
        fallback_bases=["SOL", "AVAX"],
    )

    assert result == ["SOL-EUR", "AVAX-EUR"]
    assert len(logs) == 1
    assert logs[0].level == logging.INFO
    assert logs[0].args == (["SOL-EUR", "AVAX-EUR"],)


def test_normalize_symbol_list_duplicate_removal() -> None:
    """Test that duplicate symbols are removed while preserving order."""
    symbols_list = ["BTC-PERP", "ETH-PERP", "BTC-PERP", "BTC-USD", "ETH-PERP"]
    result, logs = symbols.normalize_symbol_list(symbols_list, allow_derivatives=True, quote="USD")

    # Duplicates should be removed, first occurrence preserved
    assert result == ["BTC-PERP", "ETH-PERP", "BTC-USD"]


def test_normalize_symbols_integration() -> None:
    """Test normalize_symbols function with logging integration."""
    settings = _make_runtime_settings(
        coinbase_default_quote="EUR",
        coinbase_enable_derivatives=True,
        coinbase_enable_derivatives_overridden=True,
    )

    # Mock logger to capture log calls
    with pytest.MonkeyPatch().context() as m:
        mock_log = Mock()
        m.setattr(symbols.logger, "log", mock_log)

        symbols_list = ["btc-perp", "invalid-perp"]
        result, derivatives_enabled = symbols.normalize_symbols(
            Profile.PROD, symbols_list, settings=settings
        )

        assert derivatives_enabled is True
        assert "BTC-PERP" in result
        assert "INVALID-PERP" not in result

        # Verify logger was called
        assert mock_log.call_count == 1  # Warning for invalid perp


def test_normalize_symbols_custom_quote_and_settings() -> None:
    """Test normalize_symbols with custom quote and settings."""
    settings = _make_runtime_settings(
        coinbase_default_quote="USD",
        coinbase_enable_derivatives=False,
        coinbase_enable_derivatives_overridden=True,
    )

    result, derivatives_enabled = symbols.normalize_symbols(
        Profile.SPOT, ["BTC-PERP"], quote="JPY", settings=settings  # Custom quote override
    )

    assert derivatives_enabled is False  # SPOT profile
    assert result == ["BTC-JPY"]  # Custom quote used


def test_symbol_normalization_log_dataclass() -> None:
    """Test SymbolNormalizationLog dataclass."""
    log = symbols.SymbolNormalizationLog(
        level=logging.WARNING, message="Test message %s", args=("test_arg",)
    )

    assert log.level == logging.WARNING
    assert log.message == "Test message %s"
    assert log.args == ("test_arg",)

    # Test with default args
    log_default = symbols.SymbolNormalizationLog(level=logging.INFO, message="Test message")

    assert log_default.args == ()


def test_perps_allowlist_constant() -> None:
    """Test PERPS_ALLOWLIST constant contains expected symbols."""
    expected_symbols = {"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"}
    assert symbols.PERPS_ALLOWLIST == expected_symbols
    assert isinstance(symbols.PERPS_ALLOWLIST, frozenset)
