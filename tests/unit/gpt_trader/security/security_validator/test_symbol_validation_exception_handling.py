"""Exception handling tests for SymbolValidator."""

from __future__ import annotations


class TestSymbolValidationExceptionHandling:
    """Test exception handling in symbol validation."""

    def test_symbol_validation_generic_exception(self) -> None:
        """Test that generic exceptions (non-RuleError) are caught gracefully."""
        from unittest.mock import patch

        from gpt_trader.security.symbol_validator import SymbolValidator

        # Mock the SymbolRule to raise a generic Exception
        with patch.object(SymbolValidator, "_SYMBOL_RULE") as mock_rule:
            mock_rule.side_effect = RuntimeError("Unexpected internal error")

            result = SymbolValidator.validate_symbol("BTC-USD")

            assert not result.is_valid
            assert "Invalid symbol format" in result.errors
            assert result.sanitized_value is None
