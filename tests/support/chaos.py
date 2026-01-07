"""
Chaos testing harness for fault injection.

Provides deterministic fault injection for testing graceful degradation
behavior without flaky timing-dependent tests.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FaultAction:
    """
    Describes a fault to inject when a method is called.

    Attributes:
        after_calls: Only trigger after this many calls (0 = immediate).
        times: Number of times to trigger (-1 = forever, default 1).
        raise_exc: Exception to raise (takes precedence over return_value).
        return_value: Value to return instead of delegating.
        delay_seconds: Delay before action (use with time patching in tests).
        predicate: Optional callable(method, *args, **kwargs) -> bool to filter.
    """

    after_calls: int = 0
    times: int = 1
    raise_exc: Exception | None = None
    return_value: Any = None
    delay_seconds: float = 0.0
    predicate: Callable[..., bool] | None = None

    # Internal counter for times triggered
    _triggered: int = field(default=0, repr=False)

    def should_trigger(self, call_count: int, method: str, *args: Any, **kwargs: Any) -> bool:
        """Check if this fault should trigger on this call."""
        # Check call count threshold
        if call_count < self.after_calls:
            return False
        # Check times limit
        if self.times != -1 and self._triggered >= self.times:
            return False
        # Check predicate
        if self.predicate is not None and not self.predicate(method, *args, **kwargs):
            return False
        return True

    def execute(self, sleep_func: Callable[[float], None] = time.sleep) -> Any:
        """Execute the fault action."""
        self._triggered += 1
        if self.delay_seconds > 0:
            sleep_func(self.delay_seconds)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.return_value

    def reset(self) -> None:
        """Reset the trigger counter."""
        self._triggered = 0


@dataclass
class FaultPlan:
    """
    A plan of faults to inject for specific methods.

    Maintains call counts per method and applies matching FaultActions.
    """

    faults: dict[str, list[FaultAction]] = field(default_factory=dict)
    _call_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)

    def add(self, method: str, action: FaultAction) -> FaultPlan:
        """Add a fault action for a method. Returns self for chaining."""
        if method not in self.faults:
            self.faults[method] = []
        self.faults[method].append(action)
        return self

    def apply(self, method: str, *args: Any, **kwargs: Any) -> tuple[bool, FaultAction | None]:
        """
        Check if a fault should be applied for this method call.

        Args:
            method: Name of the method being called.
            *args: Positional arguments to the method.
            **kwargs: Keyword arguments to the method.

        Returns:
            Tuple of (should_fault, FaultAction or None).
        """
        self._call_counts[method] += 1
        call_count = self._call_counts[method]

        actions = self.faults.get(method, [])
        for action in actions:
            if action.should_trigger(call_count, method, *args, **kwargs):
                return True, action
        return False, None

    def get_call_count(self, method: str) -> int:
        """Get the number of times a method has been called."""
        return self._call_counts.get(method, 0)

    def reset(self) -> None:
        """Reset all call counts and fault triggers."""
        self._call_counts.clear()
        for actions in self.faults.values():
            for action in actions:
                action.reset()


class ChaosBroker:
    """
    Wraps a broker and injects faults according to a FaultPlan.

    Usage:
        plan = FaultPlan().add("get_ticker", FaultAction(raise_exc=TimeoutError()))
        chaos_broker = ChaosBroker(real_broker, plan)
        chaos_broker.get_ticker("BTC-USD")  # Raises TimeoutError
    """

    def __init__(
        self,
        wrapped: Any,
        plan: FaultPlan,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        Initialize chaos broker.

        Args:
            wrapped: The real broker to delegate to.
            plan: Fault injection plan.
            sleep_func: Sleep function (patch for tests to avoid real delays).
        """
        self._wrapped = wrapped
        self._plan = plan
        self._sleep_func = sleep_func

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access to inject faults on method calls."""
        attr = getattr(self._wrapped, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            should_fault, action = self._plan.apply(name, *args, **kwargs)
            if should_fault and action is not None:
                return action.execute(self._sleep_func)
            return attr(*args, **kwargs)

        return wrapper

    @property
    def plan(self) -> FaultPlan:
        """Access the fault plan for inspection."""
        return self._plan


# =============================================================================
# Deterministic Helpers
# =============================================================================


def fault_once(
    raise_exc: Exception | None = None,
    return_value: Any = None,
    delay_seconds: float = 0.0,
) -> FaultAction:
    """Create a fault that triggers exactly once on the first call."""
    return FaultAction(
        after_calls=0,
        times=1,
        raise_exc=raise_exc,
        return_value=return_value,
        delay_seconds=delay_seconds,
    )


def fault_after(
    n: int,
    raise_exc: Exception | None = None,
    return_value: Any = None,
    times: int = 1,
) -> FaultAction:
    """Create a fault that triggers after N successful calls."""
    return FaultAction(
        after_calls=n,
        times=times,
        raise_exc=raise_exc,
        return_value=return_value,
    )


def fault_always(
    raise_exc: Exception | None = None,
    return_value: Any = None,
    delay_seconds: float = 0.0,
) -> FaultAction:
    """Create a fault that triggers on every call."""
    return FaultAction(
        after_calls=0,
        times=-1,  # Forever
        raise_exc=raise_exc,
        return_value=return_value,
        delay_seconds=delay_seconds,
    )


def fault_sequence(actions: list[FaultAction]) -> list[FaultAction]:
    """
    Create a sequence of faults that trigger in order.

    Each action triggers once after the previous ones have been exhausted.
    """
    result = []
    cumulative = 0
    for action in actions:
        # Clone with adjusted after_calls
        result.append(
            FaultAction(
                after_calls=cumulative,
                times=action.times if action.times != -1 else 1,
                raise_exc=action.raise_exc,
                return_value=action.return_value,
                delay_seconds=action.delay_seconds,
                predicate=action.predicate,
            )
        )
        cumulative += action.times if action.times != -1 else 1
    return result


# =============================================================================
# Scenario Presets
# =============================================================================


def api_outage_scenario(
    error_rate: float = 0.25,
    rate_limit_usage: float = 0.95,
    open_breakers: list[str] | None = None,
) -> FaultPlan:
    """
    Create a fault plan simulating API health degradation.

    Returns resilience status with high error rate, rate limit usage,
    and optionally open circuit breakers.
    """
    if open_breakers is None:
        open_breakers = ["orders"]

    breakers = {name: {"state": "open"} for name in open_breakers}
    degraded_status = {
        "metrics": {"error_rate": error_rate},
        "circuit_breakers": breakers,
        "rate_limit_usage": rate_limit_usage,
    }

    return FaultPlan().add(
        "get_resilience_status",
        fault_always(return_value=degraded_status),
    )


def rate_limit_burst_scenario(after_calls: int = 5, times: int = 3) -> FaultPlan:
    """
    Create a fault plan simulating rate limit errors after N calls.

    Simulates burst of rate limit errors that eventually recovers.
    """

    class RateLimitError(Exception):
        pass

    return FaultPlan().add(
        "get_ticker",
        fault_after(after_calls, raise_exc=RateLimitError("Rate limit exceeded"), times=times),
    )


def slippage_spike_scenario(expected_bps: int = 200) -> FaultPlan:
    """
    Create a fault plan simulating high slippage market conditions.

    Returns market snapshot with wide spread and shallow depth.
    """
    high_slippage_snapshot = {
        "spread_bps": expected_bps // 2,
        "depth_l1": 100,  # Shallow depth amplifies impact
    }

    return FaultPlan().add(
        "get_market_snapshot",
        fault_always(return_value=high_slippage_snapshot),
    )


def mark_stale_scenario() -> FaultPlan:
    """
    Create a fault plan simulating stale mark price data.

    Returns ticker with zero or missing price.
    """
    return FaultPlan().add(
        "get_ticker",
        fault_always(return_value={"price": "0"}),
    )


def preview_failures_scenario(times: int = 5) -> FaultPlan:
    """
    Create a fault plan simulating repeated order preview failures.

    Args:
        times: Number of consecutive failures before recovery.
    """
    return FaultPlan().add(
        "preview_order",
        FaultAction(
            after_calls=0,
            times=times,
            raise_exc=Exception("Preview service unavailable"),
        ),
    )


def broker_read_failures_scenario(times: int = 3) -> FaultPlan:
    """
    Create a fault plan simulating broker connectivity issues.

    Both list_balances and list_positions fail.
    """
    plan = FaultPlan()
    plan.add(
        "list_balances",
        FaultAction(after_calls=0, times=times, raise_exc=ConnectionError("Broker unavailable")),
    )
    plan.add(
        "list_positions",
        FaultAction(after_calls=0, times=times, raise_exc=ConnectionError("Broker unavailable")),
    )
    return plan


def mixed_failures_scenario() -> FaultPlan:
    """
    Create a fault plan with multiple failure types.

    Useful for testing compound degradation scenarios.
    """
    plan = FaultPlan()
    # First call to get_resilience_status returns degraded
    plan.add(
        "get_resilience_status",
        fault_once(
            return_value={
                "metrics": {"error_rate": 0.3},
                "circuit_breakers": {},
                "rate_limit_usage": 0.85,
            }
        ),
    )
    # First two ticker calls fail
    plan.add("get_ticker", fault_once(raise_exc=TimeoutError("Timeout")))
    plan.add("get_ticker", fault_after(1, raise_exc=TimeoutError("Timeout")))
    return plan


__all__ = [
    "FaultAction",
    "FaultPlan",
    "ChaosBroker",
    "fault_once",
    "fault_after",
    "fault_always",
    "fault_sequence",
    "api_outage_scenario",
    "rate_limit_burst_scenario",
    "slippage_spike_scenario",
    "mark_stale_scenario",
    "preview_failures_scenario",
    "broker_read_failures_scenario",
    "mixed_failures_scenario",
]
