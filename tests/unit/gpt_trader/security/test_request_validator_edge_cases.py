"""Tests for security/request_validator.py edge cases."""

from __future__ import annotations

from gpt_trader.security.request_validator import RequestValidator


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
