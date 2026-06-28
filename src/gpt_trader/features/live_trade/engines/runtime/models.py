"""Data models for runtime coordinator bootstrap and lifecycle control."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class BrokerBootstrapError(RuntimeError):
    """Raised when broker initialization fails."""


class RuntimeLifecycleError(RuntimeError):
    """Raised when runtime lifecycle coordination fails."""


class RuntimeContextValidationError(RuntimeLifecycleError):
    """Raised when required runtime context dependencies are missing."""


class RuntimeStopRequested(RuntimeLifecycleError):
    """Raised when a stop condition prevents runtime startup."""


class RuntimeEngineState(str, Enum):
    """Runtime coordinator lifecycle states."""

    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    DRAINING = "draining"
    TERMINATED = "terminated"
    FAILED = "failed"


class RuntimeStepKind(str, Enum):
    """Lifecycle step categories used for sequencing and diagnostics."""

    STARTUP_HOOK = "startup_hook"
    BACKGROUND_TASK = "background_task"
    HEARTBEAT = "heartbeat"
    HEALTH = "health"
    POLICY_CHECKPOINT = "policy_checkpoint"
    STREAMING = "streaming"
    SHUTDOWN_HOOK = "shutdown_hook"


RuntimeStepCallback = Callable[
    [],
    Awaitable[Any] | Any,
]
RuntimeStopConditionCallback = Callable[[], bool]


@dataclass
class BrokerBootstrapArtifacts:
    """Artifacts returned from broker bootstrap routines."""

    broker: object
    registry_updates: dict[str, Any]
    event_store: object | None = None
    products: Sequence[object] = ()
    market_data: Any | None = None
    product_catalog: Any | None = None
    account_manager: Any | None = None


@dataclass(frozen=True)
class RuntimeDependency:
    """Context dependency validated before runtime startup."""

    name: str
    value: Any
    required: bool = True


@dataclass(frozen=True)
class RuntimeStopCondition:
    """Stop predicate checked before runtime tasks are started."""

    name: str
    is_met: RuntimeStopConditionCallback
    reason: str


@dataclass(frozen=True)
class RuntimeLifecycleStep:
    """A named lifecycle callback executed by the runtime coordinator."""

    name: str
    kind: RuntimeStepKind
    callback: RuntimeStepCallback
    timeout_seconds: float | None = None
    required: bool = True
    register_task: bool = True


@dataclass(frozen=True)
class RuntimeLifecyclePlan:
    """Explicit startup and shutdown sequencing for a runtime owner."""

    dependencies: Sequence[RuntimeDependency] = ()
    stop_conditions: Sequence[RuntimeStopCondition] = ()
    startup_steps: Sequence[RuntimeLifecycleStep] = ()
    shutdown_steps: Sequence[RuntimeLifecycleStep] = ()
    task_cleanup_timeout_seconds: float = 5.0
    shutdown_step_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class RuntimeLifecycleEvent:
    """Recorded runtime lifecycle event for tests and diagnostics."""

    name: str
    kind: RuntimeStepKind | str
    state: RuntimeEngineState
    success: bool
    error: str | None = None


RuntimeTaskResult = asyncio.Task[Any] | Sequence[asyncio.Task[Any]] | None


__all__ = [
    "BrokerBootstrapArtifacts",
    "BrokerBootstrapError",
    "RuntimeContextValidationError",
    "RuntimeDependency",
    "RuntimeEngineState",
    "RuntimeLifecycleError",
    "RuntimeLifecycleEvent",
    "RuntimeLifecyclePlan",
    "RuntimeLifecycleStep",
    "RuntimeStepKind",
    "RuntimeStopCondition",
    "RuntimeStopRequested",
    "RuntimeTaskResult",
]
