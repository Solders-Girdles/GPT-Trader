"""Tests for security/request_validator.py validation logic."""

from __future__ import annotations

from typing import Any

from gpt_trader.security.request_validator import RequestValidator
from gpt_trader.utilities.datetime_helpers import utc_now


class TestValidateRequestSuccess:
    """Tests for successful request validation."""

    def test_validate_request_with_required_fields(self) -> None:
        """Test valid request with action and timestamp passes."""
        request = {
            "action": "buy",
            "timestamp": utc_now().isoformat(),
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is True
        assert result.errors == []
        assert result.sanitized_value == request

    def test_validate_request_with_extra_fields(self) -> None:
        """Test that extra fields are allowed."""
        request = {
            "action": "sell",
            "timestamp": "2024-01-01T00:00:00Z",
            "symbol": "BTC-USD",
            "quantity": 1.5,
            "price": 50000.0,
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is True
        assert result.errors == []

    def test_validate_request_with_underscore_action(self) -> None:
        """Test action with underscores is valid."""
        request = {
            "action": "place_order",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is True


class TestValidateRequestInvalidAction:
    """Tests for invalid action strings."""

    def test_validate_request_sql_injection_action(self) -> None:
        """Test that SQL injection in action is detected."""
        request = {
            "action": "SELECT * FROM users",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False
        assert any("SQL" in error or "injection" in error.lower() for error in result.errors)

    def test_validate_request_xss_action(self) -> None:
        """Test that XSS in action is detected."""
        request = {
            "action": "<script>alert('xss')</script>",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False

    def test_validate_request_action_too_long(self) -> None:
        """Test that action exceeding max length is detected."""
        request = {
            "action": "a" * 100,  # max_length is 50
            "timestamp": "2024-01-01T00:00:00Z",
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False
        assert any(
            "length" in error.lower() or "exceed" in error.lower() for error in result.errors
        )

    def test_validate_request_path_traversal_action(self) -> None:
        """Test that path traversal in action is detected."""
        request = {
            "action": "../../../etc/passwd",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False


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


class TestValidateRequestEdgeCases:
    """Tests for edge cases."""

    def test_validate_request_action_with_spaces(self) -> None:
        """Test action with spaces."""
        request = {
            "action": "place order",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        result = RequestValidator.validate_request(request)

        # Spaces should be allowed in action
        assert result.is_valid is True

    def test_validate_request_numeric_timestamp(self) -> None:
        """Test that numeric timestamp value is acceptable."""
        request = {
            "action": "buy",
            "timestamp": 1704067200,  # Unix timestamp
        }

        result = RequestValidator.validate_request(request)

        # The validator only checks for field presence, not type
        assert result.is_valid is True

    def test_validate_request_none_values(self) -> None:
        """Test request with None values for required fields."""
        request = {
            "action": None,
            "timestamp": None,
        }

        # Fields are present but contain None
        # The InputSanitizer will handle None action
        result = RequestValidator.validate_request(request)

        # None values should cause validation failure
        assert result.is_valid is False

    def test_validate_request_whitespace_action(self) -> None:
        """Test action that is only whitespace.

        Note: The InputSanitizer allows whitespace-only strings through
        as it focuses on detecting malicious patterns (SQL injection, XSS, etc.)
        rather than enforcing non-empty content. Whitespace validation should
        be handled at the application layer if required.
        """
        request = {
            "action": "   ",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        result = RequestValidator.validate_request(request)

        # Current behavior: whitespace-only passes sanitization
        # The sanitizer focuses on security threats, not content validation
        assert result.is_valid is True
