"""Bulk validation tests for SecurityValidator symbol validation."""

from __future__ import annotations

from typing import Any


def test_symbol_validation_bulk_passes(security_validator: Any) -> None:
    """Validate a larger batch of known-good symbols."""
    symbols = ["BTC-USD", "ETH-USD", "AAPL", "GOOGL"] * 100

    for symbol in symbols:
        result = security_validator.validate_symbol(symbol)
        assert result.is_valid
