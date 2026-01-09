"""
Failure injection harness for resilience testing.

Provides test helpers to simulate broker failures, timeouts, and transient errors
without requiring real network calls or sleeps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar
from unittest.mock import MagicMock

T = TypeVar("T")


@dataclass
class FailureScript:
    """A scripted sequence of failures and successes.

    Attributes:
        sequence: List of outcomes. Each item is either:
            - None: success (call the underlying function)
            - Exception instance: raise this exception
            - callable: call it to get the return value or exception
        loop: If True, repeat the sequence after exhausting it.
    """

    sequence: list[Exception | Callable[[], Any] | None] = field(default_factory=list)
    loop: bool = False
    _index: int = field(default=0, repr=False)

    def next_outcome(self) -> Exception | Callable[[], Any] | None:
        """Get the next scripted outcome.

        Returns:
            None for success, Exception to raise, or callable to invoke.

        Raises:
            StopIteration: If sequence is exhausted and loop=False.
        """
        if self._index >= len(self.sequence):
            if self.loop and self.sequence:
                self._index = 0
            else:
                raise StopIteration("Failure script exhausted")

        outcome = self.sequence[self._index]
        self._index += 1
        return outcome

    def reset(self) -> None:
        """Reset the script to the beginning."""
        self._index = 0

    @property
    def call_count(self) -> int:
        """Return the number of calls made to this script."""
        return self._index

    @classmethod
    def from_outcomes(
        cls,
        *outcomes: Exception | Callable[[], Any] | None,
        loop: bool = False,
    ) -> FailureScript:
        """Create a script from a sequence of outcomes.

        Args:
            *outcomes: Variable number of outcomes:
                - None: success (call underlying)
                - Exception: raise it
                - callable: call it
            loop: Whether to loop the sequence.

        Returns:
            FailureScript with the specified sequence.

        Example:
            >>> script = FailureScript.from_outcomes(
            ...     ConnectionError("fail 1"),
            ...     TimeoutError("fail 2"),
            ...     None,  # success
            ... )
        """
        return cls(sequence=list(outcomes), loop=loop)

    @classmethod
    def fail_then_succeed(
        cls,
        failures: int,
        exception: Exception | None = None,
    ) -> FailureScript:
        """Create a script that fails N times then succeeds.

        Args:
            failures: Number of failures before success.
            exception: Exception to raise (defaults to ConnectionError).

        Returns:
            FailureScript configured for fail-then-succeed pattern.
        """
        exc = exception or ConnectionError("simulated network error")
        return cls(sequence=[exc] * failures + [None])

    @classmethod
    def timeout_then_succeed(cls, timeouts: int) -> FailureScript:
        """Create a script that times out N times then succeeds.

        Args:
            timeouts: Number of timeouts before success.

        Returns:
            FailureScript configured for timeout-then-succeed pattern.
        """
        return cls.fail_then_succeed(
            failures=timeouts,
            exception=TimeoutError("simulated timeout"),
        )

    @classmethod
    def always_fail(cls, exception: Exception | None = None) -> FailureScript:
        """Create a script that always fails.

        Args:
            exception: Exception to raise (defaults to ConnectionError).

        Returns:
            FailureScript that loops a single failure.
        """
        exc = exception or ConnectionError("simulated network error")
        return cls(sequence=[exc], loop=True)

    @classmethod
    def rejection(cls, message: str = "Order rejected") -> FailureScript:
        """Create a script that raises a broker rejection.

        Args:
            message: Rejection message.

        Returns:
            FailureScript with a ValueError (simulating broker rejection).
        """
        return cls(sequence=[ValueError(message)])


class InjectingBroker:
    """A broker wrapper that injects failures based on a script.

    Wraps a broker (or mock) and intercepts method calls to inject
    scripted failures before delegating to the underlying broker.

    Example:
        >>> mock_broker = MagicMock()
        >>> mock_broker.place_order.return_value = Order(...)
        >>> script = FailureScript.fail_then_succeed(2)
        >>> injecting = InjectingBroker(mock_broker, place_order=script)
        >>> injecting.place_order(...)  # raises ConnectionError
        >>> injecting.place_order(...)  # raises ConnectionError
        >>> injecting.place_order(...)  # returns Order
    """

    def __init__(
        self,
        broker: Any,
        **method_scripts: FailureScript,
    ) -> None:
        """
        Initialize the injecting broker.

        Args:
            broker: Underlying broker (or mock) to wrap.
            **method_scripts: Mapping of method name to FailureScript.
                Only methods with scripts will have failures injected.
        """
        self._broker = broker
        self._scripts = method_scripts
        self._call_counts: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access to inject failures."""
        # Get the underlying method/attribute
        underlying = getattr(self._broker, name)

        # If no script for this method, pass through directly
        if name not in self._scripts:
            return underlying

        # Wrap callable methods with failure injection
        if callable(underlying):
            return self._wrap_method(name, underlying)

        return underlying

    def _wrap_method(
        self,
        name: str,
        method: Callable[..., T],
    ) -> Callable[..., T]:
        """Wrap a method to inject scripted failures."""

        def wrapper(*args: Any, **kwargs: Any) -> T:
            self._call_counts[name] = self._call_counts.get(name, 0) + 1
            script = self._scripts[name]

            try:
                outcome = script.next_outcome()
            except StopIteration:
                # Script exhausted - call underlying method
                return method(*args, **kwargs)

            if outcome is None:
                # Success - call underlying method
                return method(*args, **kwargs)
            elif isinstance(outcome, Exception):
                # Failure - raise the exception
                raise outcome
            elif callable(outcome):
                # Custom behavior - call the outcome
                result = outcome()
                if isinstance(result, Exception):
                    raise result
                return result
            else:
                # Treat as return value
                return outcome

        return wrapper

    def get_call_count(self, method: str) -> int:
        """Get the number of times a method was called."""
        return self._call_counts.get(method, 0)

    def reset_scripts(self) -> None:
        """Reset all scripts to their initial state."""
        for script in self._scripts.values():
            script.reset()
        self._call_counts.clear()


def no_op_sleep(seconds: float) -> None:
    """A no-op sleep function for deterministic testing."""
    pass


def counting_sleep() -> tuple[Callable[[float], None], Callable[[], list[float]]]:
    """Create a sleep function that records sleep durations.

    Returns:
        Tuple of (sleep_fn, get_sleeps) where:
            - sleep_fn: Callable to use as sleep function
            - get_sleeps: Callable to retrieve list of sleep durations
    """
    sleeps: list[float] = []

    def sleep_fn(seconds: float) -> None:
        sleeps.append(seconds)

    def get_sleeps() -> list[float]:
        return sleeps.copy()

    return sleep_fn, get_sleeps


@dataclass
class PartialResponse:
    """Simulates a partial/incomplete broker response.

    Use with InjectingBroker to test partial response handling.
    """

    data: dict[str, Any]
    missing_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the partial data."""
        return self.data


def create_failing_broker(
    base_broker: Any | None = None,
    place_order_script: FailureScript | None = None,
    cancel_order_script: FailureScript | None = None,
    get_order_script: FailureScript | None = None,
) -> InjectingBroker:
    """Factory to create an InjectingBroker with common scripts.

    Args:
        base_broker: Underlying broker (creates MagicMock if None).
        place_order_script: Script for place_order failures.
        cancel_order_script: Script for cancel_order failures.
        get_order_script: Script for get_order failures.

    Returns:
        Configured InjectingBroker.
    """
    if base_broker is None:
        base_broker = MagicMock()

    scripts: dict[str, FailureScript] = {}
    if place_order_script:
        scripts["place_order"] = place_order_script
    if cancel_order_script:
        scripts["cancel_order"] = cancel_order_script
    if get_order_script:
        scripts["get_order"] = get_order_script

    return InjectingBroker(base_broker, **scripts)


__all__ = [
    "FailureScript",
    "InjectingBroker",
    "PartialResponse",
    "no_op_sleep",
    "counting_sleep",
    "create_failing_broker",
]
