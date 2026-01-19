"""Tests for normalize_symbol_list in gpt_trader.features.live_trade.symbols."""

from __future__ import annotations

import logging

from gpt_trader.features.live_trade import symbols


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
