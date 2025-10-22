"""Tests for convenience wrapper functions in SecurityValidator."""

from __future__ import annotations

from bot_v2.security import security_validator


class TestConvenienceWrappers:
    """Test convenience wrapper functions."""

    def test_validate_order_wrapper(self) -> None:
        """Test validate_order convenience wrapper."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order(order, account_value)

        assert result is not None
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")

    def test_check_rate_limit_wrapper(self) -> None:
        """Test check_rate_limit convenience wrapper."""
        identifier = "test-user"
        limit_type = "api_calls"

        allowed, message = security_validator.check_rate_limit(identifier, limit_type)

        assert isinstance(allowed, bool)
        assert isinstance(message, (str, type(None)))

    def test_sanitize_input_wrapper(self) -> None:
        """Test sanitize_input convenience wrapper."""
        input_str = "normal text"

        result = security_validator.sanitize_input(input_str)

        assert result is not None
        assert hasattr(result, "is_valid")
        assert hasattr(result, "sanitized_value")

    def test_validate_order_with_global_validator(self) -> None:
        """Test validate_order uses global validator instance."""
        order = {
            "symbol": "ETH-USD",
            "quantity": 0.01,
            "order_type": "market",
        }
        account_value = 50000.0

        result = security_validator.validate_order(order, account_value)

        assert result.is_valid

    def test_check_rate_limit_with_global_validator(self) -> None:
        """Test check_rate_limit uses global validator instance."""
        identifier = "wrapper-test"
        limit_type = "order_submissions"

        allowed, message = security_validator.check_rate_limit(identifier, limit_type)

        assert allowed is True
        assert message is None

    def test_sanitize_input_with_global_validator(self) -> None:
        """Test sanitize_input uses global validator instance."""
        input_str = "test string"

        result = security_validator.sanitize_input(input_str)

        assert result.is_valid
        assert result.sanitized_value == "test string"

    def test_validate_order_with_invalid_input(self) -> None:
        """Test validate_order wrapper with invalid input."""
        from bot_v2.security.security_validator import ValidationResult

        invalid_order = {
            "symbol": "INVALID",
            "quantity": "invalid",
            "order_type": "limit",
            "price": -100,
        }
        account_value = 1000.0

        result = security_validator.validate_order(invalid_order, account_value)

        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_check_rate_limit_exceeds_limit(self) -> None:
        """Test check_rate_limit wrapper when limit is exceeded."""
        identifier = "limit-test"
        limit_type = "api_calls"

        # Make requests up to limit
        limit = security_validator.get_validator().RATE_LIMITS[limit_type].requests
        for i in range(limit + 1):
            allowed, message = security_validator.check_rate_limit(identifier, limit_type)

        # Should be blocked
        assert allowed is False
        assert "Rate limit exceeded" in message

    def test_sanitize_input_with_attack_vectors(self) -> None:
        """Test sanitize_input wrapper with attack vectors."""
        attack_inputs = [
            "SELECT * FROM users",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
        ]

        for attack_input in attack_inputs:
            result = security_validator.sanitize_input(attack_input)
            assert not result.is_valid

    def test_wrapper_functions_consistency(self) -> None:
        """Test wrapper functions are consistent with direct validator usage."""
        validator = security_validator.get_validator()

        # Test order validation
        order = {"symbol": "BTC-USD", "quantity": 0.001, "order_type": "limit", "price": 50000.0}
        account_value = 100000.0

        wrapper_result = security_validator.validate_order(order, account_value)
        direct_result = validator.validate_order_request(order, account_value)

        assert wrapper_result.is_valid == direct_result.is_valid

        # Test rate limiting
        identifier = "consistency-test"
        limit_type = "api_calls"

        wrapper_allowed, wrapper_message = security_validator.check_rate_limit(
            identifier, limit_type
        )
        direct_allowed, direct_message = validator.check_rate_limit(identifier, limit_type)

        assert wrapper_allowed == direct_allowed
        assert wrapper_message == direct_message

        # Test sanitization
        input_str = "test string"

        wrapper_result = security_validator.sanitize_input(input_str)
        direct_result = validator.sanitize_string(input_str)

        assert wrapper_result.is_valid == direct_result.is_valid
        assert wrapper_result.sanitized_value == direct_result.sanitized_value

    def test_wrapper_functions_error_handling(self) -> None:
        """Test wrapper functions error handling."""
        # Test with None inputs
        order_result = security_validator.validate_order(None, 1000)  # type: ignore
        assert not order_result.is_valid
        assert any("Order payload must be a mapping" in error for error in order_result.errors)

        rate_result = security_validator.check_rate_limit(None, "invalid")  # type: ignore
        assert isinstance(rate_result, tuple)

        sanitize_result = security_validator.sanitize_input(None)  # type: ignore
        assert not sanitize_result.is_valid

    def test_wrapper_functions_multiple_calls(self) -> None:
        """Test wrapper functions maintain state across multiple calls."""
        identifier = "state-test"
        limit_type = "login_attempts"

        # Make multiple calls to test state persistence
        results = []
        for i in range(3):
            allowed, message = security_validator.check_rate_limit(identifier, limit_type)
            results.append((allowed, message))

        # Should allow first few calls
        assert results[0][0] is True
        assert results[1][0] is True

    def test_wrapper_functions_with_different_parameters(self) -> None:
        """Test wrapper functions with different parameter combinations."""
        # Test order validation with different account values
        order = {"symbol": "BTC-USD", "quantity": 0.01, "order_type": "limit", "price": 50000.0}

        small_account = security_validator.validate_order(order, 1000)
        large_account = security_validator.validate_order(order, 100000)

        # Small account should be rejected, large account accepted
        assert not small_account.is_valid
        assert large_account.is_valid

    def test_wrapper_functions_return_types(self) -> None:
        """Test wrapper functions return correct types."""
        # Test validate_order return type
        order = {"symbol": "BTC-USD", "quantity": 0.001, "order_type": "limit", "price": 50000.0}
        order_result = security_validator.validate_order(order, 100000)

        assert hasattr(order_result, "is_valid")
        assert hasattr(order_result, "errors")
        assert hasattr(order_result, "sanitized_value")

        # Test check_rate_limit return type
        allowed, message = security_validator.check_rate_limit("test", "api_calls")
        assert isinstance(allowed, bool)
        assert isinstance(message, (str, type(None)))

        # Test sanitize_input return type
        sanitize_result = security_validator.sanitize_input("test")
        assert hasattr(sanitize_result, "is_valid")
        assert hasattr(sanitize_result, "errors")
        assert hasattr(sanitize_result, "sanitized_value")

    def test_wrapper_functions_import_path(self) -> None:
        """Test wrapper functions are accessible from correct import path."""
        # Should be importable from bot_v2.security.security_validator
        from bot_v2.security.security_validator import (
            check_rate_limit,
            sanitize_input,
            validate_order,
        )

        assert callable(validate_order)
        assert callable(check_rate_limit)
        assert callable(sanitize_input)
