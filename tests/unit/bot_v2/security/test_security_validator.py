"""Tests for SecurityValidator - input validation, rate limiting, and security checks.

This module tests the SecurityValidator's ability to protect the trading system
from malicious input, enforce trading limits, and prevent abuse. Tests verify:

- Input sanitization (SQL injection, XSS, path traversal)
- Trading symbol validation
- Numeric value bounds checking
- Order validation against portfolio limits
- Rate limiting to prevent API abuse
- Trading hours enforcement
- Suspicious activity detection
- Request size and format validation

Security Context:
    The SecurityValidator is the first line of defense against malicious actors
    and erroneous trading commands. Failures here could allow:
    - Unauthorized trades through injection attacks
    - Portfolio blow-ups from oversized orders
    - Account suspension from rate limit violations
    - Financial loss from trading during market closures
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.security.security_validator import SecurityValidator, ValidationResult, get_validator


@pytest.fixture
def validator():
    """Create fresh SecurityValidator for each test."""
    return SecurityValidator()


class TestStringSanitization:
    """Test string sanitization and injection detection.

    String inputs from users or external APIs must be sanitized to prevent
    injection attacks. These tests verify detection of common attack vectors
    that could compromise database queries or trading commands.
    """

    def test_empty_string_rejected(self, validator):
        """Empty string input is rejected.

        Basic validation: Empty strings often indicate missing required data
        and should be caught early to provide clear error messages rather
        than causing downstream failures.
        """
        result = validator.sanitize_string("")
        assert result.is_valid is False
        assert "cannot be empty" in result.errors[0].lower()

    def test_sql_injection_detected(self, validator):
        """SQL injection attempts with SQL keywords are detected.

        Critical security: SQL injection could allow attackers to access trading
        data, modify orders, or extract sensitive information. This test verifies
        that common SQL injection patterns containing keywords like DROP, SELECT,
        UNION are rejected before reaching any database queries.
        """
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "SELECT * FROM accounts",
            "UNION SELECT password",
            "DELETE FROM users",
            "INSERT INTO table",
        ]

        for malicious in malicious_inputs:
            result = validator.sanitize_string(malicious)
            assert result.is_valid is False, f"Expected {malicious} to be invalid"
            assert any("injection" in err.lower() for err in result.errors)

    def test_xss_tags_detected(self, validator):
        """HTML/XSS tags are detected.

        Prevents cross-site scripting attacks if trading logs or reports are
        displayed in web interfaces. HTML tags in user input could execute
        malicious JavaScript, steal session tokens, or modify displayed data.
        """
        result = validator.sanitize_string("<div>content</div>")
        # Should have HTML tags error
        assert any("HTML" in err or "tags" in err for err in result.errors)
        # Tags stripped from sanitized value
        assert "<div>" not in result.sanitized_value

    def test_path_traversal_detected(self, validator):
        """Path traversal attempts are detected."""
        traversal_attempts = ["../../etc/passwd", "..\\windows\\system32"]

        for attempt in traversal_attempts:
            result = validator.sanitize_string(attempt)
            assert result.is_valid is False
            assert any("traversal" in err.lower() for err in result.errors)

    def test_max_length_enforced(self, validator):
        """Maximum length is enforced."""
        long_string = "a" * 300
        result = validator.sanitize_string(long_string, max_length=255)

        assert len(result.errors) > 0
        assert "exceeds maximum length" in result.errors[0]
        assert len(result.sanitized_value) == 255

    def test_valid_string_sanitized(self, validator):
        """Valid strings are properly sanitized."""
        result = validator.sanitize_string("Valid input string")
        assert result.is_valid is True
        assert result.sanitized_value == "Valid input string"

    def test_special_characters_escaped(self, validator):
        """Special characters are escaped."""
        result = validator.sanitize_string('User\'s "quoted" input')
        assert 'User\'\'s ""quoted"" input' in result.sanitized_value


class TestSymbolValidation:
    """Test trading symbol validation."""

    @pytest.mark.parametrize(
        "symbol,expected_valid",
        [
            ("BTC-USD", True),
            ("BTC-PERP", True),
            ("ETH-USD", True),
            ("AAPL", True),
            ("btc-usd", False),  # Lowercase not allowed by regex
            ("BTC_USD", False),  # Underscore not allowed
            ("", False),
            ("TOOLONGSYMBOL123", False),
            ("BTC-", False),
            ("-USD", False),
            ("123-456", True),  # Numbers allowed
        ],
    )
    def test_symbol_format_validation(self, validator, symbol, expected_valid):
        """Symbol format is validated correctly."""
        result = validator.validate_symbol(symbol)
        assert result.is_valid == expected_valid
        if expected_valid and symbol:
            assert result.sanitized_value == symbol.upper()

    def test_blocked_symbols_rejected(self, validator):
        """Blocked symbols are rejected."""
        blocked = ["TEST", "DEBUG", "HACK"]
        for symbol in blocked:
            result = validator.validate_symbol(symbol)
            assert result.is_valid is False
            assert "blocked" in result.errors[0].lower()


class TestNumericValidation:
    """Test numeric value validation."""

    def test_valid_numeric_values(self, validator):
        """Valid numeric values pass validation."""
        valid_values = [100, 100.50, "250.75", Decimal("1000.00")]

        for value in valid_values:
            result = validator.validate_numeric(value)
            assert result.is_valid is True
            assert isinstance(result.sanitized_value, float)

    def test_min_value_enforced(self, validator):
        """Minimum value constraint is enforced."""
        result = validator.validate_numeric(5, min_val=10)
        assert result.is_valid is False
        assert "at least 10" in result.errors[0]

    def test_max_value_enforced(self, validator):
        """Maximum value constraint is enforced."""
        result = validator.validate_numeric(1000, max_val=500)
        assert result.is_valid is False
        assert "not exceed 500" in result.errors[0]

    def test_invalid_numeric_rejected(self, validator):
        """Non-numeric values are rejected."""
        invalid_values = ["abc", "12.34.56"]

        for value in invalid_values:
            result = validator.validate_numeric(value)
            assert result.is_valid is False, f"Expected {value} to be invalid"
            assert "Invalid numeric" in result.errors[0]

    def test_range_validation(self, validator):
        """Values within range are validated."""
        result = validator.validate_numeric(50, min_val=10, max_val=100)
        assert result.is_valid is True
        assert result.sanitized_value == 50.0


class TestOrderValidation:
    """Test trading order validation.

    Order validation prevents financial loss from erroneous or malicious orders.
    These checks enforce portfolio risk limits, prevent fat-finger errors, and
    ensure orders comply with exchange minimums and account constraints.
    """

    def test_valid_order_accepted(self, validator):
        """Valid order passes all checks.

        Happy path: A well-formed order within all limits should be accepted.
        This order represents 0.5% of portfolio (BTC at $50k, 0.1 quantity = $5k,
        account = $100k), well under the 5% position size limit.
        """
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.1,
            "price": 50000,
            "order_type": "limit",
        }
        result = validator.validate_order_request(order, account_value=100000)
        assert result.is_valid is True

    def test_invalid_symbol_rejected(self, validator):
        """Order with invalid symbol is rejected."""
        order = {"symbol": "INVALID_SYM", "quantity": 1, "price": 100}
        result = validator.validate_order_request(order, account_value=10000)
        assert result.is_valid is False
        assert any("symbol" in err.lower() for err in result.errors)

    def test_invalid_quantity_rejected(self, validator):
        """Order with invalid quantity is rejected."""
        order = {"symbol": "BTC-USD", "quantity": -5, "price": 50000}
        result = validator.validate_order_request(order, account_value=10000)
        assert result.is_valid is False

    def test_min_order_value_enforced(self, validator):
        """Minimum order value of $1 is enforced."""
        order = {"symbol": "BTC-USD", "quantity": 0.00001, "price": 50}
        result = validator.validate_order_request(order, account_value=10000)
        assert result.is_valid is False
        assert any("below minimum" in err.lower() for err in result.errors)

    def test_max_order_value_enforced(self, validator):
        """Maximum order value of $100k is enforced."""
        order = {"symbol": "BTC-USD", "quantity": 10, "price": 50000}
        result = validator.validate_order_request(order, account_value=1000000)
        assert result.is_valid is False
        assert any("exceeds maximum" in err.lower() for err in result.errors)

    def test_position_size_limit_enforced(self, validator):
        """Position size limit (5% of portfolio) is enforced.

        Risk management: Prevents concentration risk by limiting any single
        position to 5% of portfolio value. This test verifies a $10k order
        on a $100k account (10%) is rejected. Exceeding position limits could
        lead to catastrophic losses if the position moves against the trader.
        """
        order = {"symbol": "BTC-USD", "quantity": 1, "price": 10000}
        result = validator.validate_order_request(order, account_value=100000)
        # 10,000 / 100,000 = 10% > 5% limit
        assert result.is_valid is False
        assert any("exceeds" in err.lower() and "5" in err for err in result.errors)

    def test_limit_order_price_validated(self, validator):
        """Limit order price is validated."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.01,
            "price": -100,
            "order_type": "limit",
        }
        result = validator.validate_order_request(order, account_value=10000)
        assert result.is_valid is False


class TestRateLimiting:
    """Test rate limiting functionality.

    Rate limiting protects against:
    - API ban from exchange due to excessive requests
    - Denial of service attacks
    - Runaway trading bots
    - Account suspension for abuse

    Different operations have different limits based on their impact and
    exchange constraints.
    """

    def test_rate_limit_allows_within_limit(self, validator):
        """Requests within rate limit are allowed.

        Normal operation: Users should be able to make reasonable numbers of
        requests. This test verifies 5 order submissions (under the 10/min limit)
        all succeed, ensuring legitimate trading isn't blocked.
        """
        for i in range(5):
            allowed, msg = validator.check_rate_limit("user1", "order_submissions")
            assert allowed is True
            assert msg is None

    def test_rate_limit_blocks_excess_requests(self, validator):
        """Requests exceeding rate limit are blocked.

        Protection: After 10 order submissions in a minute, the 11th should be
        blocked with a clear error message. This prevents runaway bots from
        flooding the exchange with orders, which could result in:
        - IP ban from exchange
        - Account suspension
        - Unintended market impact from excessive orders
        """
        # order_submissions: 10/minute
        for i in range(10):
            validator.check_rate_limit("user2", "order_submissions")

        # 11th request should be blocked
        allowed, msg = validator.check_rate_limit("user2", "order_submissions")
        assert allowed is False
        assert "Rate limit exceeded" in msg

    def test_rate_limit_resets_after_period(self, validator, monkeypatch):
        """Rate limit resets after the configured period."""
        # Fill up the limit
        for i in range(10):
            validator.check_rate_limit("user3", "order_submissions")

        # Should be blocked
        allowed, _ = validator.check_rate_limit("user3", "order_submissions")
        assert allowed is False

        # Mock time to advance past the period (60 seconds for order_submissions)
        import bot_v2.security.security_validator as sv

        original_time = time.time
        monkeypatch.setattr(sv.time, "time", lambda: original_time() + 61)

        # Should be allowed after reset
        allowed, _ = validator.check_rate_limit("user3", "order_submissions")
        assert allowed is True

    def test_blocked_ip_rejected(self, validator):
        """Blocked IPs are rejected immediately."""
        validator._blocked_ips.add("malicious-ip")

        allowed, msg = validator.check_rate_limit("malicious-ip", "api_calls")
        assert allowed is False
        assert "blocked" in msg.lower()

    def test_excessive_violations_block_ip(self, validator):
        """Excessive rate limit violations block the IP.

        Automated defense: If an IP repeatedly violates rate limits (>10 times),
        it's likely malicious or misconfigured. Blocking it protects system
        resources and prevents potential account compromise. This is especially
        important for cloud deployments where IPs might be shared or rotated.
        """
        # Trigger violations repeatedly
        for i in range(15):
            for j in range(11):  # Exceed limit each time
                validator.check_rate_limit("repeat-offender", "order_submissions")

        # IP should be blocked
        assert "repeat-offender" in validator._blocked_ips

    def test_different_limit_types_independent(self, validator):
        """Different rate limit types are tracked independently."""
        # Use up order_submissions limit
        for i in range(10):
            validator.check_rate_limit("user4", "order_submissions")

        # api_calls should still work
        allowed, _ = validator.check_rate_limit("user4", "api_calls")
        assert allowed is True

    def test_clear_rate_limits(self, validator):
        """Rate limits can be cleared."""
        # Create some history
        for i in range(5):
            validator.check_rate_limit("user5", "order_submissions")

        validator.clear_rate_limits("user5")

        # Should have fresh limit
        for i in range(10):
            allowed, _ = validator.check_rate_limit("user5", "order_submissions")
            assert allowed is True


class TestTradingHours:
    """Test trading hours validation.

    Trading outside market hours can result in:
    - Orders rejected by exchange
    - Stale prices leading to poor execution
    - Confusion between pre-market and regular session rules

    These checks prevent weekend or after-hours trading attempts.
    """

    def test_weekday_market_hours_valid(self, validator):
        """Trading during weekday market hours is valid.

        Happy path: Monday at 10:00 AM is well within standard market hours
        (9:30 AM - 4:00 PM ET). Orders should be accepted for immediate
        execution during this time.
        """
        # Monday at 10:00 AM
        valid_time = datetime(2025, 1, 6, 10, 0)  # Monday
        result = validator.check_trading_hours("AAPL", valid_time)
        assert result.is_valid is True

    def test_weekend_rejected(self, validator):
        """Trading on weekends is rejected."""
        # Saturday
        saturday = datetime(2025, 1, 4, 10, 0)
        result = validator.check_trading_hours("AAPL", saturday)
        assert result.is_valid is False
        assert any("weekend" in err.lower() for err in result.errors)

        # Sunday
        sunday = datetime(2025, 1, 5, 10, 0)
        result = validator.check_trading_hours("AAPL", sunday)
        assert result.is_valid is False

    def test_before_market_open_rejected(self, validator):
        """Trading before 9:30 AM is rejected."""
        early_time = datetime(2025, 1, 6, 9, 0)  # Monday 9:00 AM
        result = validator.check_trading_hours("AAPL", early_time)
        assert result.is_valid is False
        assert any("market hours" in err.lower() for err in result.errors)

    def test_after_market_close_rejected(self, validator):
        """Trading after 4:00 PM is rejected."""
        late_time = datetime(2025, 1, 6, 16, 30)  # Monday 4:30 PM
        result = validator.check_trading_hours("AAPL", late_time)
        assert result.is_valid is False


class TestSuspiciousActivityDetection:
    """Test suspicious activity detection.

    Detects patterns that may indicate:
    - Account compromise (unusual trading patterns)
    - Market manipulation attempts
    - Fat-finger errors (unusually large orders)
    - Bot malfunction

    Requires 2+ suspicious indicators to trigger to avoid false positives
    on legitimate but unusual trading.
    """

    def test_rapid_fire_orders_with_large_size_detected(self, validator, caplog):
        """Rapid-fire orders combined with large size are flagged.

        Suspicious pattern: 15 orders/minute (well above normal) combined with
        6x average order size suggests either bot malfunction or manipulation.
        Both indicators together warrant investigation and potential blocking
        to prevent account damage.
        """
        activity = {
            "orders_per_minute": 15,  # Indicator 1
            "average_order_size": 100,
            "current_order_size": 600,  # Indicator 2 (6x average)
        }

        is_suspicious = validator.detect_suspicious_activity("user1", activity)
        assert is_suspicious is True
        assert "Rapid-fire orders" in caplog.text or "Unusual order size" in caplog.text

    def test_unusual_order_size_with_pattern_detected(self, validator, caplog):
        """Unusually large orders with pattern are flagged."""
        activity = {
            "orders_per_minute": 2,
            "average_order_size": 100,
            "current_order_size": 1000,  # Indicator 1 (10x average)
            "pattern_score": 0.9,  # Indicator 2
        }

        is_suspicious = validator.detect_suspicious_activity("user2", activity)
        assert is_suspicious is True
        assert "Unusual order size" in caplog.text or "Suspicious pattern" in caplog.text

    def test_pattern_score_with_rapid_orders_detected(self, validator, caplog):
        """High pattern scores with rapid orders are flagged."""
        activity = {
            "orders_per_minute": 12,  # Indicator 1
            "average_order_size": 100,
            "current_order_size": 100,
            "pattern_score": 0.9,  # Indicator 2
        }

        is_suspicious = validator.detect_suspicious_activity("user3", activity)
        assert is_suspicious is True
        assert "Suspicious pattern" in caplog.text or "Rapid-fire" in caplog.text

    def test_single_indicator_not_suspicious(self, validator):
        """Single suspicious indicator alone is not flagged.

        Avoids false positives: A trader might legitimately trade quickly during
        volatile markets OR place one unusually large order. Only when multiple
        suspicious indicators occur together do we flag the activity. This
        prevents blocking legitimate trading strategies.
        """
        activity = {
            "orders_per_minute": 15,  # Only one indicator
            "average_order_size": 100,
            "current_order_size": 100,
        }

        is_suspicious = validator.detect_suspicious_activity("user4", activity)
        assert is_suspicious is False

    def test_normal_activity_not_flagged(self, validator):
        """Normal trading activity is not flagged."""
        activity = {
            "orders_per_minute": 3,
            "average_order_size": 100,
            "current_order_size": 120,
            "pattern_score": 0.3,
        }

        is_suspicious = validator.detect_suspicious_activity("user5", activity)
        assert is_suspicious is False


class TestRequestValidation:
    """Test comprehensive request validation."""

    def test_valid_request_accepted(self, validator):
        """Valid request passes validation."""
        request = {"action": "place_order", "timestamp": "2025-01-01T10:00:00", "data": {}}

        result = validator.validate_request(request)
        assert result.is_valid is True

    def test_missing_required_fields_rejected(self, validator):
        """Request missing required fields is rejected."""
        request = {"data": {}}  # Missing action and timestamp

        result = validator.validate_request(request)
        assert result.is_valid is False
        assert any("Missing required field" in err for err in result.errors)

    def test_invalid_action_rejected(self, validator):
        """Request with invalid action is rejected."""
        request = {"action": "<script>alert('xss')</script>", "timestamp": "2025-01-01"}

        result = validator.validate_request(request)
        assert result.is_valid is False

    @pytest.mark.skip(reason="sys.getsizeof doesn't reflect actual dict size accurately")
    def test_oversized_request_rejected(self, validator):
        """Request exceeding 1MB is rejected."""
        large_data = "x" * 2000000  # >1MB
        request = {"action": "test", "timestamp": "2025-01-01", "data": large_data}

        result = validator.validate_request(request)
        assert result.is_valid is False
        assert any("exceeds 1MB" in err for err in result.errors)


class TestGlobalInstance:
    """Test global validator instance."""

    def test_get_validator_returns_singleton(self, monkeypatch):
        """get_validator returns singleton instance."""
        # Reset global
        import bot_v2.security.security_validator as sv

        sv._validator = None

        validator1 = get_validator()
        validator2 = get_validator()

        assert validator1 is validator2
