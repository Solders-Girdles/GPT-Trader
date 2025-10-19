from __future__ import annotations

from datetime import datetime

import pytest
from freezegun import freeze_time

from bot_v2.security.security_validator import SecurityValidator


@pytest.fixture
def validator() -> SecurityValidator:
    return SecurityValidator()


def test_sanitize_string_blocks_injection(validator: SecurityValidator) -> None:
    result = validator.sanitize_string("SELECT * FROM users")
    assert not result.is_valid
    assert "SQL injection" in result.errors[0]

    html_input = "<b>alert</b>"
    result = validator.sanitize_string(html_input)
    assert not result.is_valid
    assert result.sanitized_value == "alert"


def test_validate_symbol_enforces_rules(validator: SecurityValidator) -> None:
    blocked = validator.validate_symbol("TEST")
    assert not blocked.is_valid
    assert "blocked" in blocked.errors[0].lower()

    valid = validator.validate_symbol("BTC-USD")
    assert valid.is_valid
    assert valid.sanitized_value == "BTC-USD"


def test_validate_numeric_range(validator: SecurityValidator) -> None:
    too_small = validator.validate_numeric("0.1", min_val=1)
    assert not too_small.is_valid
    assert "at least" in too_small.errors[0]

    valid = validator.validate_numeric("2.5", min_val=1, max_val=5)
    assert valid.is_valid
    assert valid.sanitized_value == 2.5


def test_validate_order_request_checks_limits(validator: SecurityValidator) -> None:
    order = {"symbol": "BTC-USD", "quantity": 0.0001, "order_type": "limit", "price": 0.5}
    result = validator.validate_order_request(order, account_value=10_000)
    assert not result.is_valid
    assert any("Order value below minimum" in err for err in result.errors)

    big_order = {"symbol": "BTC-USD", "quantity": 10, "order_type": "limit", "price": 20_000}
    result = validator.validate_order_request(big_order, account_value=100_000)
    assert not result.is_valid
    assert any("Position size exceeds" in err for err in result.errors)


def test_rate_limit_blocks_after_threshold(validator: SecurityValidator) -> None:
    limit = validator.RATE_LIMITS["api_calls"]
    with freeze_time("2024-01-01 00:00:00") as frozen:
        for _ in range(limit.requests):
            allowed, _ = validator.check_rate_limit("user-1", "api_calls")
            assert allowed

        allowed, message = validator.check_rate_limit("user-1", "api_calls")
        assert not allowed
        assert "Rate limit exceeded" in message

        for _ in range(11):
            frozen.tick(delta=0)  # Same timestamp to increment suspicious count
            validator.check_rate_limit("user-1", "api_calls")

        allowed, message = validator.check_rate_limit("user-1", "api_calls")
        assert not allowed
        assert "temporarily blocked" in message

    validator.clear_rate_limits("user-1")
    allowed, message = validator.check_rate_limit("user-1", "api_calls")
    assert not allowed
    assert "temporarily blocked" in message


def test_trading_hours_validation(validator: SecurityValidator) -> None:
    weekend = datetime(2024, 6, 1, 10, 0)  # Saturday
    result = validator.check_trading_hours("BTC-USD", timestamp=weekend)
    assert not result.is_valid
    assert "Market closed" in result.errors[0]


def test_detect_suspicious_activity(validator: SecurityValidator) -> None:
    activity = {
        "orders_per_minute": 20,
        "average_order_size": 100,
        "current_order_size": 800,
        "pattern_score": 0.9,
    }
    assert validator.detect_suspicious_activity("user-2", activity)


def test_validate_request_missing_fields(validator: SecurityValidator) -> None:
    result = validator.validate_request({"action": "<b>buy</b>"})
    assert not result.is_valid
    assert any("Missing required field" in err for err in result.errors)
