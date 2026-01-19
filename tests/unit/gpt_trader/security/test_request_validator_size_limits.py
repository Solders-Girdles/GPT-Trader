"""Tests for security/request_validator.py request-size enforcement."""

from __future__ import annotations

from gpt_trader.security.request_validator import RequestValidator


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
