"""Tests for security/request_validator.py invalid action strings."""

from __future__ import annotations

from gpt_trader.security.request_validator import RequestValidator


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
