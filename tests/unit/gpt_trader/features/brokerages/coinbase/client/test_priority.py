"""Tests for Request Priority System."""

import pytest

from gpt_trader.features.brokerages.coinbase.client.priority import (
    ENDPOINT_PRIORITIES,
    PriorityManager,
    RequestDeferredError,
    RequestPriority,
)


class TestRequestPriority:
    """Tests for RequestPriority enum."""

    def test_priority_ordering(self) -> None:
        """Test that priorities are correctly ordered."""
        assert RequestPriority.CRITICAL < RequestPriority.HIGH
        assert RequestPriority.HIGH < RequestPriority.NORMAL
        assert RequestPriority.NORMAL < RequestPriority.LOW

    def test_critical_is_lowest_value(self) -> None:
        """Test that CRITICAL has the lowest numeric value."""
        assert RequestPriority.CRITICAL.value == 0


class TestEndpointPriorities:
    """Tests for endpoint priority mapping."""

    def test_orders_are_critical(self) -> None:
        """Test that order endpoints are critical priority."""
        assert ENDPOINT_PRIORITIES["orders"] == RequestPriority.CRITICAL
        assert ENDPOINT_PRIORITIES["cancel"] == RequestPriority.CRITICAL

    def test_accounts_are_high(self) -> None:
        """Test that account endpoints are high priority."""
        assert ENDPOINT_PRIORITIES["accounts"] == RequestPriority.HIGH
        assert ENDPOINT_PRIORITIES["positions"] == RequestPriority.HIGH

    def test_products_are_low(self) -> None:
        """Test that product endpoints are low priority."""
        assert ENDPOINT_PRIORITIES["products"] == RequestPriority.LOW
        assert ENDPOINT_PRIORITIES["candles"] == RequestPriority.LOW


class TestPriorityManager:
    """Tests for PriorityManager class."""

    def test_all_allowed_under_threshold(self) -> None:
        """Test that all requests are allowed when under threshold."""
        manager = PriorityManager(threshold_high=0.7)

        # At 50% usage, all priorities should be allowed
        assert manager.should_allow("/api/v3/products", 0.5) is True
        assert manager.should_allow("/api/v3/accounts", 0.5) is True
        assert manager.should_allow("/api/v3/orders", 0.5) is True

    def test_low_blocked_above_high_threshold(self) -> None:
        """Test that LOW priority is blocked above high threshold."""
        manager = PriorityManager(threshold_high=0.7, threshold_critical=0.85)

        # At 75% usage, LOW should be blocked
        assert manager.should_allow("/api/v3/products", 0.75) is False
        assert manager.should_allow("/api/v3/accounts", 0.75) is True
        assert manager.should_allow("/api/v3/orders", 0.75) is True

    def test_normal_blocked_above_critical_threshold(self) -> None:
        """Test that NORMAL priority is blocked above critical threshold."""
        manager = PriorityManager(threshold_high=0.7, threshold_critical=0.85)

        # At 90% usage, NORMAL and LOW should be blocked
        assert manager.should_allow("/api/v3/products", 0.9) is False  # LOW
        assert manager.should_allow("/api/v3/ticker", 0.9) is False  # NORMAL
        assert manager.should_allow("/api/v3/accounts", 0.9) is False  # HIGH
        assert manager.should_allow("/api/v3/orders", 0.9) is True  # CRITICAL

    def test_critical_always_allowed(self) -> None:
        """Test that CRITICAL requests are always allowed."""
        manager = PriorityManager()

        # Even at 99% usage, CRITICAL should be allowed
        assert manager.should_allow("/api/v3/orders", 0.99) is True
        assert manager.should_allow("/api/v3/orders/123/cancel", 0.99) is True

    def test_disabled_manager_allows_all(self) -> None:
        """Test that disabled manager allows all requests."""
        manager = PriorityManager(enabled=False)

        # Even at 100% usage, should allow
        assert manager.should_allow("/api/v3/products", 1.0) is True
        assert manager.should_allow("/api/v3/ticker", 1.0) is True

    def test_get_priority(self) -> None:
        """Test priority detection for endpoints."""
        manager = PriorityManager()

        assert manager.get_priority("/api/v3/orders") == RequestPriority.CRITICAL
        assert manager.get_priority("/api/v3/accounts") == RequestPriority.HIGH
        assert manager.get_priority("/api/v3/ticker") == RequestPriority.NORMAL
        assert manager.get_priority("/api/v3/products") == RequestPriority.LOW
        assert manager.get_priority("/api/v3/unknown") == RequestPriority.NORMAL

    def test_stats_tracking(self) -> None:
        """Test statistics tracking."""
        manager = PriorityManager(threshold_high=0.7)

        # Allow some requests
        manager.should_allow("/api/v3/orders", 0.5)
        manager.should_allow("/api/v3/accounts", 0.5)

        # Block some requests
        manager.should_allow("/api/v3/products", 0.8)  # Should be blocked

        stats = manager.get_stats()
        assert stats["allowed_requests"] == 2
        assert stats["total_blocked"] == 1

    def test_reset_stats(self) -> None:
        """Test resetting statistics."""
        manager = PriorityManager()

        manager.should_allow("/api/v3/orders", 0.5)
        manager.reset_stats()

        stats = manager.get_stats()
        assert stats["allowed_requests"] == 0
        assert stats["total_blocked"] == 0


class TestRequestDeferredError:
    """Tests for RequestDeferredError exception."""

    def test_error_message(self) -> None:
        """Test error message format."""
        error = RequestDeferredError("/api/v3/products", RequestPriority.LOW, 0.9)

        assert error.path == "/api/v3/products"
        assert error.priority == RequestPriority.LOW
        assert error.usage == 0.9
        assert "LOW" in str(error)
        assert "90%" in str(error)

    def test_error_can_be_raised(self) -> None:
        """Test that error can be raised and caught."""
        with pytest.raises(RequestDeferredError) as exc_info:
            raise RequestDeferredError("/api/v3/test", RequestPriority.NORMAL, 0.85)

        assert exc_info.value.priority == RequestPriority.NORMAL
