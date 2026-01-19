"""Tests for suspicious activity detection error handling and robustness."""

from __future__ import annotations

from typing import Any


class TestSuspiciousActivityErrorHandling:
    """Test suspicious activity detection error handling."""

    def test_suspicious_activity_missing_fields(self, security_validator: Any) -> None:
        """Test suspicious activity detection with missing fields."""
        user_id = "test-user"

        # Activity with missing fields
        incomplete_activity = {
            "orders_per_minute": 15,  # High
            # Missing other fields
        }

        result = security_validator.detect_suspicious_activity(user_id, incomplete_activity)

        # Should handle gracefully
        assert isinstance(result, bool)

    def test_suspicious_activity_zero_values(self, security_validator: Any) -> None:
        """Test suspicious activity detection with zero values."""
        user_id = "test-user"

        zero_activity = {
            "orders_per_minute": 0,
            "average_order_size": 0,
            "current_order_size": 0,
            "pattern_score": 0,
        }

        result = security_validator.detect_suspicious_activity(user_id, zero_activity)

        assert result is False

    def test_suspicious_activity_with_zero_average(self, security_validator: Any) -> None:
        """Test suspicious activity detection with zero average order size."""
        user_id = "test-user"

        zero_average_activity = {
            "orders_per_minute": 5,
            "average_order_size": 0,  # Zero average
            "current_order_size": 100,  # Any non-zero is infinite ratio
            "pattern_score": 0.5,
        }

        result = security_validator.detect_suspicious_activity(user_id, zero_average_activity)

        # Should handle division by zero gracefully
        assert isinstance(result, bool)

    def test_suspicious_activity_error_handling(self, security_validator: Any) -> None:
        """Test suspicious activity detection error handling."""
        user_id = "test-user"

        # Test with invalid activity data
        invalid_activities = [
            None,  # type: ignore
            "invalid",  # type: ignore
            [],  # type: ignore
            123,  # type: ignore
        ]

        for invalid_activity in invalid_activities:
            try:
                result = security_validator.detect_suspicious_activity(user_id, invalid_activity)
                # Should handle gracefully or raise appropriate error
                assert isinstance(result, bool)
            except (TypeError, ValueError, AttributeError):
                # Expected for invalid inputs
                pass

    def test_suspicious_activity_with_extreme_pattern_scores(self, security_validator: Any) -> None:
        """Test suspicious activity with extreme pattern scores."""
        user_id = "test-user"

        extreme_activities = [
            {
                "orders_per_minute": 5,
                "average_order_size": 100,
                "current_order_size": 100,
                "pattern_score": -1.0,  # Below 0
            },
            {
                "orders_per_minute": 5,
                "average_order_size": 100,
                "current_order_size": 100,
                "pattern_score": 2.0,  # Above 1
            },
        ]

        for activity in extreme_activities:
            result = security_validator.detect_suspicious_activity(user_id, activity)
            assert isinstance(result, bool)
