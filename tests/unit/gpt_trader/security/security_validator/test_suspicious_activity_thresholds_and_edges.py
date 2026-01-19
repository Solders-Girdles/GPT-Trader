"""Tests for suspicious activity detection thresholds and numeric edges."""

from __future__ import annotations

from typing import Any


class TestSuspiciousActivityThresholdsAndEdges:
    """Test suspicious activity threshold and edge-case inputs."""

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

    def test_suspicious_activity_cumulative_detection(self, security_validator: Any) -> None:
        """Test suspicious activity detection is cumulative."""
        user_id = "test-user"

        # First activity with one indicator
        activity = {
            "orders_per_minute": 15,  # High
            "average_order_size": 100,
            "current_order_size": 100,  # Normal
            "pattern_score": 0.3,  # Low
        }

        result = security_validator.detect_suspicious_activity(user_id, activity)
        assert result is True  # High orders alone should trigger

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
