from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TradingBotState(str, Enum):
    INIT = "init"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class EngineState(str, Enum):
    INIT = "init"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


TRADING_BOT_TRANSITIONS: dict[TradingBotState, set[TradingBotState]] = {
    TradingBotState.INIT: {
        TradingBotState.STARTING,
        TradingBotState.STOPPING,
        TradingBotState.STOPPED,
    },
    TradingBotState.STARTING: {
        TradingBotState.RUNNING,
        TradingBotState.STOPPING,
        TradingBotState.ERROR,
    },
    TradingBotState.RUNNING: {
        TradingBotState.STOPPING,
        TradingBotState.ERROR,
    },
    TradingBotState.STOPPING: {
        TradingBotState.STOPPED,
        TradingBotState.ERROR,
    },
    TradingBotState.STOPPED: {
        TradingBotState.STARTING,
        TradingBotState.STOPPING,
    },
    TradingBotState.ERROR: {
        TradingBotState.STOPPING,
        TradingBotState.STOPPED,
    },
}


ENGINE_TRANSITIONS: dict[EngineState, set[EngineState]] = {
    EngineState.INIT: {
        EngineState.STARTING,
        EngineState.STOPPING,
        EngineState.STOPPED,
    },
    EngineState.STARTING: {
        EngineState.RUNNING,
        EngineState.STOPPING,
        EngineState.ERROR,
    },
    EngineState.RUNNING: {
        EngineState.STOPPING,
        EngineState.ERROR,
    },
    EngineState.STOPPING: {
        EngineState.STOPPED,
        EngineState.ERROR,
    },
    EngineState.STOPPED: {
        EngineState.STARTING,
        EngineState.STOPPING,
    },
    EngineState.ERROR: {
        EngineState.STOPPING,
        EngineState.STOPPED,
    },
}


@dataclass(frozen=True)
class StateTransition:
    entity: str
    from_state: str
    to_state: str
    reason: str
    timestamp: float
    details: Mapping[str, Any] = field(default_factory=dict)
    forced: bool = False


class LifecycleStateMachine:
    def __init__(
        self,
        *,
        initial_state: Enum,
        entity: str,
        transitions: Mapping[Enum, set[Enum]],
        logger: Any,
    ) -> None:
        self._state = initial_state
        self._entity = entity
        self._transitions = transitions
        self._logger = logger
        self._last_transition: StateTransition | None = None

    @property
    def state(self) -> Enum:
        return self._state

    @property
    def last_transition(self) -> StateTransition | None:
        return self._last_transition

    def transition(
        self,
        target_state: Enum,
        *,
        reason: str,
        details: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        if target_state == self._state:
            return True

        allowed = self._transitions.get(self._state, set())
        if not force and target_state not in allowed:
            self._logger.warning(
                "Invalid lifecycle transition",
                operation="lifecycle_transition",
                entity=self._entity,
                from_state=getattr(self._state, "value", str(self._state)),
                to_state=getattr(target_state, "value", str(target_state)),
                reason=reason,
            )
            return False

        transition = StateTransition(
            entity=self._entity,
            from_state=getattr(self._state, "value", str(self._state)),
            to_state=getattr(target_state, "value", str(target_state)),
            reason=reason,
            timestamp=time.time(),
            details=details or {},
            forced=force,
        )
        self._last_transition = transition
        self._state = target_state
        self._logger.info(
            "Lifecycle transition",
            operation="lifecycle_transition",
            entity=transition.entity,
            from_state=transition.from_state,
            to_state=transition.to_state,
            reason=transition.reason,
            forced=transition.forced,
            details=dict(transition.details),
        )
        return True


__all__ = [
    "EngineState",
    "ENGINE_TRANSITIONS",
    "LifecycleStateMachine",
    "StateTransition",
    "TradingBotState",
    "TRADING_BOT_TRANSITIONS",
]
