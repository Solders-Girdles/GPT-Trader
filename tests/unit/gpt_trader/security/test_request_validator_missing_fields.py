"""Tests for security/request_validator.py missing-field validation."""

from __future__ import annotations

from typing import Any

from gpt_trader.security.request_validator import RequestValidator


class TestValidateRequestMissingFields:
    """Tests for missing required fields."""

    def test_validate_request_missing_action(self) -> None:
        """Test that missing action field returns error."""
        request = {
            "timestamp": "2024-01-01T00:00:00Z",
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False
        assert "Missing required field: action" in result.errors
        assert result.sanitized_value is None

    def test_validate_request_missing_timestamp(self) -> None:
        """Test that missing timestamp field returns error."""
        request = {
            "action": "buy",
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False
        assert "Missing required field: timestamp" in result.errors
        assert result.sanitized_value is None

    def test_validate_request_missing_both_fields(self) -> None:
        """Test that missing both fields returns multiple errors."""
        request = {
            "symbol": "BTC-USD",
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Missing required field: action" in result.errors
        assert "Missing required field: timestamp" in result.errors

    def test_validate_request_empty_dict(self) -> None:
        """Test that empty dict returns errors for both required fields."""
        request: dict[str, Any] = {}

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False
        assert len(result.errors) == 2
