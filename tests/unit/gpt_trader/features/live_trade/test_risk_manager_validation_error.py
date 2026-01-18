"""Tests for ValidationError."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gpt_trader.features.live_trade.risk.manager import ValidationError


@pytest.fixture(autouse=True)
def mock_load_state():
    """Prevent LiveRiskManager from loading state during tests."""
    with patch("gpt_trader.features.live_trade.risk.manager.LiveRiskManager._load_state"):
        yield


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_is_exception(self) -> None:
        """Test ValidationError inherits from Exception."""
        assert issubclass(ValidationError, Exception)

    def test_validation_error_with_message(self) -> None:
        """Test ValidationError can be raised with a message."""
        with pytest.raises(ValidationError, match="test error"):
            raise ValidationError("test error")

    def test_validation_error_without_message(self) -> None:
        """Test ValidationError can be raised without a message."""
        with pytest.raises(ValidationError):
            raise ValidationError()
