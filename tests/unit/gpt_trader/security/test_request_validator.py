"""Tests for security/request_validator.py."""

from __future__ import annotations

from typing import Any

from gpt_trader.security.request_validator import RequestValidator
from gpt_trader.utilities.datetime_helpers import utc_now

# ============================================================
# Test: validate_request - Success cases
# ============================================================


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


# ============================================================
# Test: validate_request - Missing fields
# ============================================================


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


# ============================================================
# Test: validate_request - Invalid action string
# ============================================================


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


# ============================================================
# Test: validate_request - Size limits
# ============================================================


class TestValidateRequestSizeLimit:
    """Tests for request size limits."""

    def test_validate_request_within_size_limit(self) -> None:
        """Test that request within size limit passes."""
        request = {
            "action": "buy",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": "x" * 1000,  # Small data
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is True

    def test_validate_request_exceeds_size_limit(self) -> None:
        """Test that request exceeding 1MB limit fails."""
        # Create a request with data exceeding 1MB (1,048,576 bytes)
        # Using a string slightly over 1MB to ensure we exceed the limit
        large_data = "x" * (1024 * 1024 + 100)  # ~1MB + 100 bytes
        request = {
            "action": "upload",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": large_data,
        }

        result = RequestValidator.validate_request(request)

        assert result.is_valid is False
        assert any("1MB" in error for error in result.errors)

    def test_validate_request_unserializable_content(self) -> None:
        """Test that request with unserializable content skips size check gracefully."""

        class UnserializableObject:
            """Object that cannot be JSON serialized even with default=str."""

            def __str__(self) -> str:
                raise ValueError("Cannot convert to string")

            def __repr__(self) -> str:
                raise ValueError("Cannot convert to repr")

        request = {
            "action": "test",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": UnserializableObject(),
        }

        # Should not raise - size check is skipped when serialization fails
        result = RequestValidator.validate_request(request)

        # Request is valid because size check was skipped (serialization failed)
        assert result.is_valid is True


# ============================================================
# Test: validate_request - Edge cases
# ============================================================


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
