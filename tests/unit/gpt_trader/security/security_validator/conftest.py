"""Shared fixtures for security_validator tests."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import pytest
from freezegun import freeze_time

from gpt_trader.app.config import BotConfig


@pytest.fixture
def validator_bot_config(bot_config_factory) -> BotConfig:
    """Bot config tailored for security_validator tests."""
    return bot_config_factory()


@pytest.fixture
def security_validator() -> Any:
    """SecurityValidator instance for testing."""
    from gpt_trader.security.security_validator import SecurityValidator

    return SecurityValidator()


@pytest.fixture
def frozen_time() -> Any:
    """Freeze time for deterministic rate limiting tests."""
    with freeze_time("2024-01-01 12:00:00") as frozen:
        yield frozen


@pytest.fixture
def rate_limiter_time_control() -> Any:
    """Time control fixture for rate limiter testing."""

    class TimeControl:
        def __init__(self):
            self.current_time = time.time()
            self.selfincrements = 0

        def advance(self, seconds: int) -> None:
            """Advance time by specified seconds."""
            self.current_time += seconds
            self.selfincrements += 1

        def get_time(self) -> float:
            """Get current time."""
            return self.current_time

    return TimeControl()


@pytest.fixture
def sample_order_requests() -> dict[str, dict[str, Any]]:
    """Sample order requests for testing."""
    return {
        "valid_limit_order": {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "limit",
            "price": 50000.0,
        },
        "valid_market_order": {
            "symbol": "ETH-USD",
            "quantity": 0.01,
            "order_type": "market",
        },
        "invalid_small_order": {
            "symbol": "BTC-USD",
            "quantity": 0.0001,
            "order_type": "limit",
            "price": 50000.0,
        },
        "invalid_large_order": {
            "symbol": "BTC-USD",
            "quantity": 10.0,
            "order_type": "limit",
            "price": 50000.0,
        },
        "invalid_symbol": {
            "symbol": "TEST",
            "quantity": 0.001,
            "order_type": "limit",
            "price": 50000.0,
        },
    }


@pytest.fixture
def sample_symbols() -> dict[str, list[str]]:
    """Sample symbols for testing validation."""
    return {
        "valid": ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "BTC-PERP"],
        "blocked": ["TEST", "DEBUG", "HACK"],
        "invalid": ["", "BTC-USD-INVALID", "btc-usd", "BTCUSD-"],
    }


@pytest.fixture
def sample_strings() -> dict[str, list[str]]:
    """Sample strings for input sanitization testing."""
    return {
        "sql_injection": [
            "SELECT * FROM users",
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT password FROM users",
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "<b onclick='alert(1)'>bold</b>",
            "<img src='x' onerror='alert(1)'>",
            "javascript:alert(1)",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "file:///etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
        ],
        "valid": [
            "normal_string",
            "user123",
            "BTC-USD",
            "Valid input string",
        ],
    }


@pytest.fixture
def suspicious_activity_samples() -> dict[str, dict[str, Any]]:
    """Sample suspicious activity patterns for testing."""
    return {
        "rapid_orders": {
            "orders_per_minute": 15,
            "average_order_size": 100,
            "current_order_size": 100,
            "pattern_score": 0.3,
        },
        "unusual_size": {
            "orders_per_minute": 3,
            "average_order_size": 100,
            "current_order_size": 800,
            "pattern_score": 0.3,
        },
        "high_pattern_score": {
            "orders_per_minute": 5,
            "average_order_size": 100,
            "current_order_size": 100,
            "pattern_score": 0.9,
        },
        "multiple_indicators": {
            "orders_per_minute": 12,
            "average_order_size": 100,
            "current_order_size": 600,
            "pattern_score": 0.8,
        },
        "normal": {
            "orders_per_minute": 3,
            "average_order_size": 100,
            "current_order_size": 100,
            "pattern_score": 0.2,
        },
    }


@pytest.fixture
def trading_hours_samples() -> dict[str, datetime]:
    """Sample timestamps for trading hours validation."""
    return {
        "weekend": datetime(2024, 6, 1, 10, 0),  # Saturday
        "pre_market": datetime(2024, 5, 31, 8, 0),  # Friday 8 AM
        "market_open": datetime(2024, 5, 31, 9, 30),  # Friday 9:30 AM
        "market_close": datetime(2024, 5, 31, 16, 0),  # Friday 4 PM
        "after_hours": datetime(2024, 5, 31, 18, 0),  # Friday 6 PM
    }
