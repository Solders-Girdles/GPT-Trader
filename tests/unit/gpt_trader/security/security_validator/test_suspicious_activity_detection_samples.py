"""Tests for suspicious activity detection with fixture samples."""

from __future__ import annotations

from typing import Any


class TestSuspiciousActivityDetectionSamples:
    """Test suspicious activity detection scenarios using sample fixtures."""

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

    def test_suspicious_activity_different_users(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test suspicious activity detection for different users."""
        activity = suspicious_activity_samples["rapid_orders"]

        users = ["user1", "user2", "user3"]

        for user_id in users:
            result = security_validator.detect_suspicious_activity(user_id, activity)
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

    def test_suspicious_activity_consistency(
        self, security_validator: Any, suspicious_activity_samples: dict[str, dict[str, Any]]
    ) -> None:
        """Test suspicious activity detection consistency."""
        user_id = "test-user"
        activity = suspicious_activity_samples["multiple_indicators"]

        result1 = security_validator.detect_suspicious_activity(user_id, activity)
        result2 = security_validator.detect_suspicious_activity(user_id, activity)

        assert result1 == result2
