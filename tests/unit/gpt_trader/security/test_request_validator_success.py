"""Tests for security/request_validator.py validate_request success cases."""

from __future__ import annotations

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
