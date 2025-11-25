"""Tests for symbol validation in SecurityValidator."""

from __future__ import annotations

from typing import Any

import pytest


class TestSymbolValidation:
    """Test symbol validation scenarios."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC-USD",
            "ETH-USD",
            "AAPL",
            "GOOGL",
            "MSFT",
            "BTC-PERP",
            "ETH-PERP",
            "TSLA",
            "AMZN",
            "META",
        ],
    )
    def test_valid_symbols_pass(self, security_validator: Any, symbol: str) -> None:
        """Test valid trading symbols pass validation."""
        result = security_validator.validate_symbol(symbol)

        assert result.is_valid
        assert result.sanitized_value == symbol.upper()

    @pytest.mark.parametrize(
        "symbol",
        [
            "TEST",
            "DEBUG",
            "HACK",
            "test",  # Case insensitive
            "debug",  # Case insensitive
            "hack",  # Case insensitive
        ],
    )
    def test_blocked_symbols_rejected(self, security_validator: Any, symbol: str) -> None:
        """Test blocked symbols are rejected."""
        result = security_validator.validate_symbol(symbol)

        assert not result.is_valid
        assert any("blocked" in error.lower() for error in result.errors)

    @pytest.mark.parametrize(
        "symbol",
        [
            "",
            "BTC-USD-INVALID",
            "BTCUSD-",  # Trailing dash
            "-BTC-USD",  # Leading dash
            "BTC--USD",  # Double dash
            "VERY-LONG-SYMBOL-NAME-THAT-EXCEEDS-LIMITS",
            "BTC-USD-EXTRA-INVALID",
            "BTC_USD",  # Underscore instead of dash
            "BTC.USD",  # Dot instead of dash
            "BTC USD",  # Space
            "BTC@USD",  # Special character
        ],
    )
    def test_invalid_symbol_formats(self, security_validator: Any, symbol: str) -> None:
        """Test invalid symbol formats are rejected."""
        result = security_validator.validate_symbol(symbol)

        assert not result.is_valid
        assert any("Invalid symbol format" in error for error in result.errors)

    def test_symbol_case_normalisation(self, security_validator: Any) -> None:
        """Test symbol case normalisation."""
        test_cases = [
            ("btc-usd", "BTC-USD"),
            ("eth-usd", "ETH-USD"),
            ("aapl", "AAPL"),
            ("Btc-Usd", "BTC-USD"),
            ("Eth-Usd", "ETH-USD"),
        ]

        for input_symbol, expected in test_cases:
            result = security_validator.validate_symbol(input_symbol)
            assert result.is_valid
            assert result.sanitized_value == expected

    def test_symbol_length_validation(self, security_validator: Any) -> None:
        """Test symbol length validation."""
        # Test very long symbol
        long_symbol = "A" * 20  # Exceeds typical limits

        result = security_validator.validate_symbol(long_symbol)

        assert not result.is_valid
        assert any("Invalid symbol format" in error for error in result.errors)

    def test_symbol_with_numbers(self, security_validator: Any) -> None:
        """Test symbols with numbers."""
        valid_with_numbers = [
            "BTC123-USD",
            "ETH456-USD",
            "AAPL1",
            "GOOGL2",
        ]

        for symbol in valid_with_numbers:
            result = security_validator.validate_symbol(symbol)
            assert result.is_valid
            assert result.sanitized_value == symbol.upper()

    def test_perpetual_symbols(self, security_validator: Any) -> None:
        """Test perpetual futures symbols."""
        perpetual_symbols = [
            "BTC-PERP",
            "ETH-PERP",
            "SOL-PERP",
            "ADA-PERP",
        ]

        for symbol in perpetual_symbols:
            result = security_validator.validate_symbol(symbol)
            assert result.is_valid
            assert result.sanitized_value == symbol.upper()

    def test_symbol_validation_error_details(self, security_validator: Any) -> None:
        """Test symbol validation error details."""
        result = security_validator.validate_symbol("INVALID-SYMBOL-FORMAT")

        assert not result.is_valid
        assert len(result.errors) >= 1
        assert result.sanitized_value is None

    def test_symbol_validation_with_none(self, security_validator: Any) -> None:
        """Test symbol validation with None input."""
        result = security_validator.validate_symbol(None)  # type: ignore

        assert not result.is_valid
        assert any("Invalid symbol format" in error for error in result.errors)

    def test_symbol_validation_with_non_string(self, security_validator: Any) -> None:
        """Test symbol validation with non-string input."""
        non_string_inputs = [123, [], {}, True, 3.14]

        for input_val in non_string_inputs:
            result = security_validator.validate_symbol(input_val)  # type: ignore
            assert not result.is_valid
            assert any("Invalid symbol format" in error for error in result.errors)

    def test_symbol_validation_edge_cases(self, security_validator: Any) -> None:
        """Test symbol validation edge cases."""
        edge_cases = [
            ("A", True, "A"),
            ("AB", True, "AB"),
            ("A-B", False, None),
        ]

        for input_symbol, expected_valid, expected in edge_cases:
            result = security_validator.validate_symbol(input_symbol)
            assert result.is_valid == expected_valid
            assert result.sanitized_value == expected

    def test_symbol_validation_with_whitespace(self, security_validator: Any) -> None:
        """Test symbol validation with whitespace."""
        whitespace_cases = [
            (" BTC-USD", True, "BTC-USD"),
            ("BTC-USD ", True, "BTC-USD"),
            (" BTC-USD ", True, "BTC-USD"),
            ("BTC USD", False, None),
        ]

        for symbol, expected_valid, expected in whitespace_cases:
            result = security_validator.validate_symbol(symbol)
            assert result.is_valid == expected_valid
            assert result.sanitized_value == expected

    def test_symbol_validation_integration_with_rule(self, security_validator: Any) -> None:
        """Test symbol validation integration with underlying rule."""
        # Test that the validation uses the SymbolRule correctly
        result = security_validator.validate_symbol("BTC-USD")

        assert result.is_valid
        assert result.sanitized_value == "BTC-USD"

        # The underlying rule should handle the validation
        from gpt_trader.validation import SymbolRule

        rule = SymbolRule()
        normalized = rule("BTC-USD", "symbol")
        assert normalized == "BTC-USD"

    def test_symbol_validation_consistency(self, security_validator: Any) -> None:
        """Test symbol validation consistency across multiple calls."""
        symbol = "BTC-USD"

        result1 = security_validator.validate_symbol(symbol)
        result2 = security_validator.validate_symbol(symbol)

        assert result1.is_valid == result2.is_valid
        assert result1.sanitized_value == result2.sanitized_value
        assert result1.errors == result2.errors

    def test_symbol_validation_with_special_characters(self, security_validator: Any) -> None:
        """Test symbol validation with special characters."""
        special_char_cases = [
            "BTC@USD",
            "ETH#USD",
            "AAPL$",
            "GOOGL%",
            "MSFT^",
        ]

        for symbol in special_char_cases:
            result = security_validator.validate_symbol(symbol)
            assert not result.is_valid
            assert any("Invalid symbol format" in error for error in result.errors)

    def test_symbol_validation_with_unicode(self, security_validator: Any) -> None:
        """Test symbol validation with Unicode characters."""
        unicode_cases = [
            "BTC-USD测试",  # Chinese characters
            "ETH-USD🚀",  # Emoji
            "AAPL-ñ",  # Accented character
        ]

        for symbol in unicode_cases:
            result = security_validator.validate_symbol(symbol)
            assert not result.is_valid
            assert any("Invalid symbol format" in error for error in result.errors)

    def test_symbol_validation_performance(self, security_validator: Any) -> None:
        """Test symbol validation performance with many symbols."""
        symbols = ["BTC-USD", "ETH-USD", "AAPL", "GOOGL"] * 100

        for symbol in symbols:
            result = security_validator.validate_symbol(symbol)
            assert result.is_valid


class TestSymbolValidationExceptionHandling:
    """Test exception handling in symbol validation."""

    def test_symbol_validation_generic_exception(self) -> None:
        """Test that generic exceptions (non-RuleError) are caught gracefully.

        This covers the generic Exception handler (lines 24-26) in symbol_validator.py.
        """
        from unittest.mock import patch

        from gpt_trader.security.symbol_validator import SymbolValidator

        # Mock the SymbolRule to raise a generic Exception
        with patch.object(SymbolValidator, "_SYMBOL_RULE") as mock_rule:
            mock_rule.side_effect = RuntimeError("Unexpected internal error")

            result = SymbolValidator.validate_symbol("BTC-USD")

            assert not result.is_valid
            assert "Invalid symbol format" in result.errors
            assert result.sanitized_value is None
