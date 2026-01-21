"""Tests for symbol normalization in gpt_trader.features.live_trade.symbols."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from gpt_trader.config.types import Profile
from gpt_trader.features.live_trade import symbols
from tests.unit.gpt_trader.features.live_trade.symbols_test_helpers import make_bot_config


class TestNormalizeSymbolListDerivatives:
    """Tests for normalize_symbol_list with derivatives."""

    def test_with_derivatives_enabled(self) -> None:
        symbols_list = ["BTC-PERP", "ETH-PERP", "INVALID-PERP", "BTC-USD"]
        result, logs = symbols.normalize_symbol_list(
            symbols_list,
            allow_derivatives=True,
            quote="USD",
            allowed_perps=["BTC-PERP", "ETH-PERP"],
        )
        assert result == ["BTC-PERP", "ETH-PERP", "BTC-USD"]
        assert len(logs) == 1
        assert logs[0].level == logging.WARNING
        assert logs[0].message == "Filtering unsupported perpetual symbol %s. Allowed perps: %s"
        assert logs[0].args == ("INVALID-PERP", ["BTC-PERP", "ETH-PERP"])

    def test_with_derivatives_disabled(self) -> None:
        symbols_list = ["BTC-PERP", "ETH-USD"]
        result, logs = symbols.normalize_symbol_list(
            symbols_list, allow_derivatives=False, quote="USD"
        )
        assert result == ["BTC-USD", "ETH-USD"]
        assert len(logs) == 1
        assert logs[0].level == logging.WARNING
        assert logs[0].message == "Derivatives disabled. Replacing %s with spot symbol %s"
        assert logs[0].args == ("BTC-PERP", "BTC-USD")


class TestNormalizeSymbolListEmpty:
    """Tests for normalize_symbol_list with empty inputs."""

    def test_empty_and_whitespace(self) -> None:
        result, logs = symbols.normalize_symbol_list(None, allow_derivatives=True, quote="USD")
        assert result == ["BTC-PERP", "ETH-PERP", "BTC-FUTURES", "ETH-FUTURES"]
        assert len(logs) == 1
        assert logs[0].level == logging.INFO
        assert "No valid symbols provided. Falling back to" in logs[0].message

        symbols_list = ["", "  ", "\t\n", "BTC-PERP"]
        result, logs = symbols.normalize_symbol_list(
            symbols_list, allow_derivatives=True, quote="USD"
        )
        assert result == ["BTC-PERP"]
        assert logs == []


class TestNormalizeSymbolListFormatting:
    """Tests for symbol normalization formatting."""

    def test_case_and_whitespace_normalization(self) -> None:
        symbols_list = ["  btc-perp  ", "\tETH-USD\n", "sol-usd"]
        result, logs = symbols.normalize_symbol_list(
            symbols_list, allow_derivatives=True, quote="USD"
        )
        assert result == ["BTC-PERP", "ETH-USD", "SOL-USD"]
        assert logs == []

    def test_custom_fallback_bases(self) -> None:
        result, logs = symbols.normalize_symbol_list(
            [], allow_derivatives=False, quote="EUR", fallback_bases=["SOL", "AVAX"]
        )
        assert result == ["SOL-EUR", "AVAX-EUR"]
        assert len(logs) == 1
        assert logs[0].level == logging.INFO
        assert logs[0].args == (["SOL-EUR", "AVAX-EUR"],)

    def test_duplicate_removal(self) -> None:
        symbols_list = ["BTC-PERP", "ETH-PERP", "BTC-PERP", "BTC-USD", "ETH-PERP"]
        result, logs = symbols.normalize_symbol_list(
            symbols_list, allow_derivatives=True, quote="USD"
        )
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
            [], allow_derivatives=False, quote="USD", fallback_bases=None
        )
        assert len(result) > 0
        assert all("-USD" in sym for sym in result)
        assert len(logs) == 1


class TestNormalizeSymbolsIntegration:
    """Tests for normalize_symbols function integration."""

    def test_normalize_symbols_integration(self) -> None:
        config = make_bot_config(coinbase_default_quote="EUR", derivatives_enabled=True)
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
            assert mock_log.call_count == 1

    def test_normalize_symbols_custom_quote_and_config(self) -> None:
        config = make_bot_config(coinbase_default_quote="USD", derivatives_enabled=False)
        result, derivatives_enabled = symbols.normalize_symbols(
            Profile.SPOT, ["BTC-PERP"], quote="JPY", config=config
        )
        assert derivatives_enabled is False
        assert result == ["BTC-JPY"]
