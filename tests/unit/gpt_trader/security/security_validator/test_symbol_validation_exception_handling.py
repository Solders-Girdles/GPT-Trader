"""Exception handling tests for SymbolValidator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestSymbolValidationExceptionHandling:
    """Test exception handling in symbol validation."""

    def test_symbol_validation_generic_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that generic exceptions (non-RuleError) are caught gracefully."""
        from gpt_trader.security.symbol_validator import SymbolValidator

        # Mock the SymbolRule to raise a generic Exception
        rule_mock = MagicMock(side_effect=RuntimeError("Unexpected internal error"))
        monkeypatch.setattr(SymbolValidator, "_SYMBOL_RULE", rule_mock)

        result = SymbolValidator.validate_symbol("BTC-USD")

        assert not result.is_valid
        assert "Invalid symbol format" in result.errors
        assert result.sanitized_value is None
