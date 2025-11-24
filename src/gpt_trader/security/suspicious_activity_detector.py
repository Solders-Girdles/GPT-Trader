"""Suspicious activity detection for trading operations."""

from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="security")


class SuspiciousActivityDetector:
    """Detect potentially suspicious trading activity."""

    @classmethod
    def detect_suspicious_activity(cls, user_id: str, activity: dict[str, Any]) -> bool:
        """
        Detect potentially suspicious trading activity.

        Args:
            user_id: User identifier
            activity: Activity details

        Returns:
            True if suspicious
        """
        suspicious_indicators = 0

        # Rapid-fire orders
        if activity.get("orders_per_minute", 0) > 10:
            suspicious_indicators += 1
            logger.warning(
                f"Rapid-fire orders detected for {user_id}",
                operation="suspicious_activity",
                status="alert",
            )

        # Unusual order size
        avg_size = activity.get("average_order_size", 0)
        current_size = activity.get("current_order_size", 0)
        if avg_size > 0 and current_size > avg_size * 5:
            suspicious_indicators += 1
            logger.warning(
                f"Unusual order size detected for {user_id}",
                operation="suspicious_activity",
                status="alert",
            )

        # Pattern detection (simplified)
        if activity.get("pattern_score", 0) > 0.8:
            suspicious_indicators += 1
            logger.warning(
                f"Suspicious pattern detected for {user_id}",
                operation="suspicious_activity",
                status="alert",
            )

        return suspicious_indicators >= 1
