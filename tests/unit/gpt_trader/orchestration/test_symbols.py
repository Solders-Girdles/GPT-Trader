"""Enhanced tests for symbols normalization module to improve coverage."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from gpt_trader.orchestration import symbols
from gpt_trader.orchestration.configuration import BotConfig, Profile


@pytest.fixture(autouse=True)
def reset_symbols_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset any module-level state between tests."""
    # No module-level cache in symbols.py, but fixture for consistency
    pass


def _make_bot_config(
    *,
    derivatives_enabled: bool = False,
    coinbase_default_quote: str = "USD",
    **kwargs: dict,
) -> BotConfig:
    """Create a BotConfig instance for testing."""
    return BotConfig(
        symbols=["BTC-USD"],
        profile=Profile.DEV,
        mock_broker=True,
        dry_run=True,
        runtime_root=kwargs.get("runtime_root", "/tmp"),
        coinbase_default_quote=coinbase_default_quote,
        derivatives_enabled=derivatives_enabled,
        perps_enable_streaming=kwargs.get("perps_enable_streaming", False),
        perps_stream_level=kwargs.get("perps_stream_level", 1),
        perps_paper_trading=kwargs.get("perps_paper_trading", False),
        perps_skip_startup_reconcile=kwargs.get("perps_skip_startup_reconcile", False),
        perps_position_fraction=kwargs.get("perps_position_fraction"),
        enable_order_preview=kwargs.get("enable_order_preview", False),
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
    config = _make_bot_config(derivatives_enabled=False)
    result = symbols.derivatives_enabled(Profile.SPOT, config=config)
    assert result is False

    # Test with string profile value
    string_profile = Mock()
    string_profile.value = "spot"
    result = symbols.derivatives_enabled(string_profile, config=config)
    assert result is False

    # Test with plain string
    result = symbols.derivatives_enabled("spot", config=config)
    assert result is False


def test_derivatives_enabled_with_config_override() -> None:
    """Test derivatives_enabled respects config setting."""
    # Test when config enables derivatives
    config = _make_bot_config(derivatives_enabled=True)
    result = symbols.derivatives_enabled(Profile.PROD, config=config)  # Use PROD, not SPOT
    assert result is True  # Config override should take effect

    # Test when config disables derivatives
    config = _make_bot_config(derivatives_enabled=False)
    result = symbols.derivatives_enabled(Profile.PROD, config=config)
    assert result is False  # Config override should take effect


def test_derivatives_enabled_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test derivatives_enabled with edge cases and import failures."""
    config = _make_bot_config(derivatives_enabled=False)

    # Test with None profile
    result = symbols.derivatives_enabled(None, config=config)
    assert result is False  # Disabled in config

    # Test with profile object without value attribute
    profile_no_value = Mock()
    del profile_no_value.value
    result = symbols.derivatives_enabled(profile_no_value, config=config)
    assert result is False  # Disabled in config


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
    config = _make_bot_config(
        coinbase_default_quote="EUR",
        derivatives_enabled=True,
    )

    # Mock logger to capture log calls
    with pytest.MonkeyPatch().context() as m:
        mock_log = Mock()
        m.setattr(symbols.logger, "log", mock_log)

        symbols_list = ["btc-perp", "invalid-perp"]
        result, derivatives_enabled = symbols.normalize_symbols(
            Profile.PROD, symbols_list, config=config
        )

        assert derivatives_enabled is True
        assert "BTC-PERP" in result
        assert "INVALID-PERP" not in result

        # Verify logger was called
        assert mock_log.call_count == 1  # Warning for invalid perp


def test_normalize_symbols_custom_quote_and_config() -> None:
    """Test normalize_symbols with custom quote and config."""
    config = _make_bot_config(
        coinbase_default_quote="USD",
        derivatives_enabled=False,
    )

    result, derivatives_enabled = symbols.normalize_symbols(
        Profile.SPOT,
        ["BTC-PERP"],
        quote="JPY",
        config=config,  # Custom quote override
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


def test_us_futures_allowlist_constant() -> None:
    """Test US_FUTURES_ALLOWLIST constant contains expected symbols."""
    expected_symbols = {"BTC-FUTURES", "ETH-FUTURES", "SOL-FUTURES", "XRP-FUTURES"}
    assert symbols.US_FUTURES_ALLOWLIST == expected_symbols


def _make_bot_config_extended(
    *,
    derivatives_enabled: bool = False,
    coinbase_default_quote: str = "USD",
    coinbase_us_futures_enabled: bool = False,
    coinbase_intx_perpetuals_enabled: bool = False,
    coinbase_derivatives_type: str = "",
    **kwargs: dict,
) -> BotConfig:
    """Create BotConfig instance with extended options."""
    return BotConfig(
        symbols=["BTC-USD"],
        profile=Profile.DEV,
        mock_broker=True,
        dry_run=True,
        runtime_root=kwargs.get("runtime_root", "/tmp"),
        coinbase_default_quote=coinbase_default_quote,
        derivatives_enabled=derivatives_enabled,
        coinbase_us_futures_enabled=coinbase_us_futures_enabled,
        coinbase_intx_perpetuals_enabled=coinbase_intx_perpetuals_enabled,
        coinbase_derivatives_type=coinbase_derivatives_type,
        perps_enable_streaming=kwargs.get("perps_enable_streaming", False),
        perps_stream_level=kwargs.get("perps_stream_level", 1),
        perps_paper_trading=kwargs.get("perps_paper_trading", False),
        perps_skip_startup_reconcile=kwargs.get("perps_skip_startup_reconcile", False),
        perps_position_fraction=kwargs.get("perps_position_fraction"),
        enable_order_preview=kwargs.get("enable_order_preview", False),
        spot_force_live=kwargs.get("spot_force_live", False),
        broker_hint=kwargs.get("broker_hint"),
        coinbase_sandbox_enabled=kwargs.get("coinbase_sandbox_enabled", False),
        coinbase_api_mode=kwargs.get("coinbase_api_mode", "advanced"),
        risk_config_path=kwargs.get("risk_config_path"),
        coinbase_intx_portfolio_uuid=kwargs.get("coinbase_intx_portfolio_uuid"),
    )


class TestUsFuturesEnabled:
    """Tests for us_futures_enabled function."""

    def test_returns_false_when_derivatives_disabled(self) -> None:
        config = _make_bot_config_extended(derivatives_enabled=False)
        result = symbols.us_futures_enabled(Profile.PROD, config=config)
        assert result is False

    def test_returns_true_when_us_futures_flag_enabled(self) -> None:
        config = _make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_us_futures_enabled=True,
        )
        result = symbols.us_futures_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_true_when_derivatives_type_is_us_futures(self) -> None:
        config = _make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_derivatives_type="us_futures",
        )
        result = symbols.us_futures_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_false_by_default(self) -> None:
        config = _make_bot_config_extended(derivatives_enabled=True)
        result = symbols.us_futures_enabled(Profile.PROD, config=config)
        assert result is False


class TestIntxPerpetualsEnabled:
    """Tests for intx_perpetuals_enabled function."""

    def test_returns_false_when_derivatives_disabled(self) -> None:
        config = _make_bot_config_extended(derivatives_enabled=False)
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is False

    def test_returns_true_when_intx_flag_enabled(self) -> None:
        config = _make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_intx_perpetuals_enabled=True,
        )
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_true_when_derivatives_type_is_intx_perps(self) -> None:
        config = _make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_derivatives_type="intx_perps",
        )
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_true_when_derivatives_type_is_perpetuals(self) -> None:
        config = _make_bot_config_extended(
            derivatives_enabled=True,
            coinbase_derivatives_type="perpetuals",
        )
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is True

    def test_returns_true_by_default(self) -> None:
        config = _make_bot_config_extended(derivatives_enabled=True)
        result = symbols.intx_perpetuals_enabled(Profile.PROD, config=config)
        assert result is True


class TestNormalizeSymbolListUsFutures:
    """Tests for normalize_symbol_list with US futures."""

    def test_allows_valid_us_futures(self) -> None:
        symbols_list = ["BTC-FUTURES", "ETH-FUTURES"]
        result, logs = symbols.normalize_symbol_list(
            symbols_list,
            allow_derivatives=True,
            quote="USD",
            allowed_us_futures=["BTC-FUTURES", "ETH-FUTURES"],
        )
        assert result == ["BTC-FUTURES", "ETH-FUTURES"]
        assert logs == []

    def test_filters_unsupported_us_futures(self) -> None:
        symbols_list = ["BTC-FUTURES", "INVALID-FUTURES"]
        result, logs = symbols.normalize_symbol_list(
            symbols_list,
            allow_derivatives=True,
            quote="USD",
            allowed_us_futures=["BTC-FUTURES"],
        )
        assert result == ["BTC-FUTURES"]
        assert len(logs) == 1
        assert "Filtering unsupported US futures symbol" in logs[0].message


class TestNormalizeSymbolListFallback:
    """Tests for normalize_symbol_list fallback behavior."""

    def test_uses_default_fallback_bases_when_none(self) -> None:
        result, logs = symbols.normalize_symbol_list(
            [],  # Empty to trigger fallback
            allow_derivatives=False,
            quote="USD",
            fallback_bases=None,  # Explicitly None to use default
        )
        # Should use TOP_VOLUME_BASES from configuration
        assert len(result) > 0
        assert all("-USD" in sym for sym in result)
        assert len(logs) == 1
