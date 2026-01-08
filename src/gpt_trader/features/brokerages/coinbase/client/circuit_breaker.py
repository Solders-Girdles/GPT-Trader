"""
Circuit Breaker pattern for API resilience.

Prevents cascade failures by temporarily blocking requests to
failing endpoints, allowing them time to recover.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="circuit_breaker")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation - requests allowed
    OPEN = "open"  # Blocking requests - endpoint is failing
    HALF_OPEN = "half_open"  # Testing recovery - limited requests


@dataclass
class CircuitBreaker:
    """Circuit breaker for a single endpoint category.

    State transitions:
    - CLOSED -> OPEN: After `failure_threshold` consecutive failures
    - OPEN -> HALF_OPEN: After `recovery_timeout` seconds
    - HALF_OPEN -> CLOSED: After `success_threshold` consecutive successes
    - HALF_OPEN -> OPEN: After any failure
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2

    _state: CircuitState = field(default=CircuitState.CLOSED)
    _failure_count: int = 0
    _success_count: int = 0
    _last_failure_time: float = 0.0
    _last_state_change: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for auto-transition to HALF_OPEN."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state (must hold lock)."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._last_state_change = time.time()

            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._success_count = 0

            logger.info(f"Circuit breaker: {old_state.value} -> {new_state.value}")

    def can_proceed(self) -> bool:
        """Check if a request should be allowed to proceed.

        Returns:
            True if the request can proceed, False if blocked.
        """
        current_state = self.state  # This may trigger OPEN -> HALF_OPEN

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return True
        else:  # OPEN
            return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed request.

        Args:
            error: The exception that caused the failure (for logging).
        """
        with self._lock:
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                logger.warning(
                    f"Circuit breaker: failure in half-open state, reopening. Error: {error}"
                )
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        f"Circuit breaker: {self._failure_count} failures, opening circuit. Error: {error}"
                    )
                    self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = 0.0

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dict with state, failure_count, time_until_half_open, etc.
        """
        with self._lock:
            status = {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "last_state_change": self._last_state_change,
            }

            if self._state == CircuitState.OPEN:
                time_in_open = time.time() - self._last_failure_time
                time_until_half_open = max(0, self.recovery_timeout - time_in_open)
                status["time_until_half_open"] = time_until_half_open

            return status


@dataclass
class CircuitBreakerRegistry:
    """Registry of circuit breakers for different endpoint categories.

    Groups endpoints into categories to share circuit breaker state.
    For example, all order-related endpoints share one breaker.
    """

    # Default configuration for new breakers
    default_failure_threshold: int = 5
    default_recovery_timeout: float = 30.0
    default_success_threshold: int = 2
    enabled: bool = True

    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Endpoint path patterns -> category name
    ENDPOINT_CATEGORIES: dict[str, str] = field(
        default_factory=lambda: {
            "orders": "orders",
            "fills": "orders",
            "accounts": "accounts",
            "positions": "positions",
            "cfm": "positions",
            "intx": "positions",
            "products": "products",
            "market": "market",
            "ticker": "market",
            "candles": "market",
        }
    )

    def _categorize_endpoint(self, path: str) -> str:
        """Determine the category for an endpoint path."""
        path_lower = path.lower()
        for keyword, category in self.ENDPOINT_CATEGORIES.items():
            if keyword in path_lower:
                return category
        return "default"

    def get_breaker(self, path: str) -> CircuitBreaker:
        """Get or create a circuit breaker for the given endpoint.

        Args:
            path: The endpoint path

        Returns:
            The circuit breaker for this endpoint's category.
        """
        category = self._categorize_endpoint(path)

        with self._lock:
            if category not in self._breakers:
                self._breakers[category] = CircuitBreaker(
                    failure_threshold=self.default_failure_threshold,
                    recovery_timeout=self.default_recovery_timeout,
                    success_threshold=self.default_success_threshold,
                )
            return self._breakers[category]

    def can_proceed(self, path: str) -> bool:
        """Check if a request to the given endpoint can proceed.

        Args:
            path: The endpoint path

        Returns:
            True if allowed, False if circuit is open.
        """
        if not self.enabled:
            return True

        breaker = self.get_breaker(path)
        return breaker.can_proceed()

    def record_success(self, path: str) -> None:
        """Record a successful request to the given endpoint."""
        if not self.enabled:
            return

        breaker = self.get_breaker(path)
        breaker.record_success()

    def record_failure(self, path: str, error: Exception | None = None) -> None:
        """Record a failed request to the given endpoint."""
        if not self.enabled:
            return

        breaker = self.get_breaker(path)
        breaker.record_failure(error)

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all circuit breakers.

        Returns:
            Dict mapping category -> status dict.
        """
        with self._lock:
            return {category: breaker.get_status() for category, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


class CircuitOpenError(Exception):
    """Raised when a request is blocked by an open circuit breaker."""

    def __init__(self, category: str, time_until_retry: float):
        self.category = category
        self.time_until_retry = time_until_retry
        super().__init__(f"Circuit breaker open for '{category}'. Retry in {time_until_retry:.1f}s")
