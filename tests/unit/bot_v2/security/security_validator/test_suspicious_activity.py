"""Tests for suspicious activity detection in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestSuspiciousActivity:
    """Test suspicious activity detection scenarios."""

    def test_rapid_orders_detection(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test detection of rapid order placement."""
        user_id = "test-user"
        activity = suspicious_activity_samples["rapid_orders"]

        result = security_validator.detect_suspicious_activity(user_id, activity)

        assert result is True

    def test_unusual_order_size_detection(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test detection of unusual order sizes."""
        user_id = "test-user"
        activity = suspicious_activity_samples["unusual_size"]

        result = security_validator.detect_suspicious_activity(user_id, activity)

        assert result is True

    def test_high_pattern_score_detection(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test detection of high pattern scores."""
        user_id = "test-user"
        activity = suspicious_activity_samples["high_pattern_score"]

        result = security_validator.detect_suspicious_activity(user_id, activity)

        assert result is True

    def test_multiple_indicators_detection(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test detection with multiple suspicious indicators."""
        user_id = "test-user"
        activity = suspicious_activity_samples["multiple_indicators"]

        result = security_validator.detect_suspicious_activity(user_id, activity)

        assert result is True

    def test_normal_activity_passes(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test normal activity passes detection."""
        user_id = "test-user"
        activity = suspicious_activity_samples["normal"]

        result = security_validator.detect_suspicious_activity(user_id, activity)

        assert result is False

    def test_suspicious_activity_threshold(self, security_validator: Any) -> None:
        """Test suspicious activity detection threshold."""
        user_id = "test-user"

        # Activity just below threshold
        below_threshold = {
            "orders_per_minute": 9,  # Just below 10
            "average_order_size": 100,
            "current_order_size": 400,  # 4x average, below 5x
            "pattern_score": 0.7,  # Below 0.8
        }

        result = security_validator.detect_suspicious_activity(user_id, below_threshold)
        assert result is False

        # Activity just above threshold
        above_threshold = {
            "orders_per_minute": 11,  # Above 10
            "average_order_size": 100,
            "current_order_size": 400,  # 4x average, below 5x
            "pattern_score": 0.7,  # Below 0.8
        }

        result = security_validator.detect_suspicious_activity(user_id, above_threshold)
        assert result is True

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

    def test_suspicious_activity_edge_cases(self, security_validator: Any) -> None:
        """Test suspicious activity detection edge cases."""
        user_id = "test-user"

        edge_cases = [
            # Exactly at thresholds
            {
                "orders_per_minute": 10,
                "average_order_size": 100,
                "current_order_size": 500,  # Exactly 5x
                "pattern_score": 0.8,  # Exactly threshold
            },
            # Very high values
            {
                "orders_per_minute": 1000,
                "average_order_size": 1000000,
                "current_order_size": 10000000,
                "pattern_score": 1.0,
            },
            # Negative values (should handle gracefully)
            {
                "orders_per_minute": -1,
                "average_order_size": -100,
                "current_order_size": -500,
                "pattern_score": -0.5,
            },
        ]

        for activity in edge_cases:
            result = security_validator.detect_suspicious_activity(user_id, activity)
            assert isinstance(result, bool)

    def test_suspicious_activity_different_users(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test suspicious activity detection for different users."""
        activity = suspicious_activity_samples["rapid_orders"]

        users = ["user1", "user2", "user3"]

        for user_id in users:
            result = security_validator.detect_suspicious_activity(user_id, activity)
            assert result is True

    def test_suspicious_activity_cumulative_detection(self, security_validator: Any) -> None:
        """Test suspicious activity detection is cumulative."""
        user_id = "test-user"

        # First activity with one indicator
        activity1 = {
            "orders_per_minute": 15,  # High
            "average_order_size": 100,
            "current_order_size": 100,  # Normal
            "pattern_score": 0.3,  # Low
        }

        result1 = security_validator.detect_suspicious_activity(user_id, activity1)
        assert result1 is True  # High orders alone should trigger

    def test_suspicious_activity_with_very_large_orders(self, security_validator: Any) -> None:
        """Test suspicious activity with very large orders."""
        user_id = "test-user"

        large_order_activity = {
            "orders_per_minute": 3,  # Normal
            "average_order_size": 100,
            "current_order_size": 10000,  # 100x average
            "pattern_score": 0.5,  # Medium
        }

        result = security_validator.detect_suspicious_activity(user_id, large_order_activity)
        assert result is True

    def test_suspicious_activity_persistence(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test suspicious activity detection persistence."""
        user_id = "test-user"
        activity = suspicious_activity_samples["rapid_orders"]

        # Detect suspicious activity multiple times
        result1 = security_validator.detect_suspicious_activity(user_id, activity)
        result2 = security_validator.detect_suspicious_activity(user_id, activity)

        assert result1 is True
        assert result2 is True

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

    def test_suspicious_activity_with_float_values(self, security_validator: Any) -> None:
        """Test suspicious activity detection with float values."""
        user_id = "test-user"

        float_activity = {
            "orders_per_minute": 12.5,
            "average_order_size": 123.45,
            "current_order_size": 617.25,  # 5x average
            "pattern_score": 0.85,
        }

        result = security_validator.detect_suspicious_activity(user_id, float_activity)
        assert result is True

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

    def test_suspicious_activity_consistency(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test suspicious activity detection consistency."""
        user_id = "test-user"
        activity = suspicious_activity_samples["multiple_indicators"]

        result1 = security_validator.detect_suspicious_activity(user_id, activity)
        result2 = security_validator.detect_suspicious_activity(user_id, activity)

        assert result1 == result2

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
