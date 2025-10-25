"""Circuit breaker adapter utilities."""

from __future__ import annotations

from typing import Any

from bot_v2.features.live_trade.risk.state_management import StateManager
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerState


class CircuitBreakerStateAdapter:
    """Backwards-compatible adapter exposing simple access helpers for tests."""

    def __init__(self, state: CircuitBreakerState, state_manager: StateManager) -> None:
        self._state = state
        self._state_manager = state_manager

    def update_state(self, state: CircuitBreakerState) -> None:
        self._state = state

    def get(self, *args: Any) -> Any:
        if not args:
            raise TypeError("get expected at least 1 argument")
        if len(args) == 1:
            key = args[0]
            if key == "active":
                return self._state_manager.is_reduce_only_mode()
            raise TypeError("Rule name and symbol required for circuit breaker snapshot lookups")
        if len(args) == 2:
            key, second = args
            if key == "active" and not isinstance(second, str):
                return self._state_manager.is_reduce_only_mode()
            return self._state.get(key, second)
        if len(args) == 3:
            key, second, default = args
            if key == "active" and not isinstance(second, str):
                return self._state_manager.is_reduce_only_mode()
            try:
                return self._state.get(key, second)
            except Exception:
                return default
        raise TypeError("get expected at most 3 arguments")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, name)


__all__ = ["CircuitBreakerStateAdapter"]
