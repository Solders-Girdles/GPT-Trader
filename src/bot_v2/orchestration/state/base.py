"""
Base interfaces for manager pattern standardization.

Provides standardized interfaces and base implementations for all manager types
in the GPT-Trader orchestration system. This ensures consistent behavior,
error handling, logging, and lifecycle management across all managers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="manager_base")


class ManagerInterface(Protocol):
    """Protocol defining the standard interface for all manager classes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the manager with its dependencies."""
        ...

    def initialize(self) -> None:
        """Initialize the manager and prepare it for operation."""
        ...

    def shutdown(self) -> None:
        """Shutdown the manager and clean up resources."""
        ...

    def get_status(self) -> ManagerStatus:
        """Get the current status of the manager."""
        ...


@dataclass
class ManagerStatus:
    """Standardized status information for managers."""

    state: str  # e.g., "initialized", "running", "error", "shutdown"
    last_updated: datetime = field(default_factory=datetime.utcnow)
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert status to dictionary for logging."""
        return {
            "state": self.state,
            "last_updated": self.last_updated.isoformat(),
            "message": self.message,
            "details": self.details,
        }


class BaseManager(ABC):
    """Abstract base class for all managers providing consistent behavior.

    This class establishes:
    - Standard initialization pattern
    - Consistent error handling and logging
    - Status tracking and reporting
    - Lifecycle management (initialize/shutdown)
    - State validation
    """

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the base manager.

        Args:
            name: Human-readable name for the manager (used in logging)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.name = name
        self._status = ManagerStatus(state="uninitialized")
        self._initialized = False
        self._shutdown = False

        # Log initialization
        logger.info(
            f"Created {self.name} manager",
            operation="manager_init",
            manager_name=self.name,
        )

        # Call subclass initialization
        self._initialize_manager(*args, **kwargs)

    @abstractmethod
    def _initialize_manager(self, *args: Any, **kwargs: Any) -> None:
        """Subclass-specific initialization logic.

        This method must be implemented by concrete managers to handle their
        specific initialization requirements.
        """
        ...

    def initialize(self) -> None:
        """Initialize the manager and prepare it for operation.

        Performs common initialization steps and delegates to subclass-specific
        initialization logic.
        """
        if self._initialized:
            logger.warning(
                f"{self.name} manager already initialized",
                operation="manager_initialize",
                manager_name=self.name,
                status="already_initialized",
            )
            return

        try:
            logger.info(
                f"Initializing {self.name} manager",
                operation="manager_initialize",
                manager_name=self.name,
            )

            # Call subclass-specific initialization
            self._on_initialize()

            self._initialized = True
            self._update_status("initialized", "Manager successfully initialized")

            logger.info(
                f"{self.name} manager initialized successfully",
                operation="manager_initialize",
                manager_name=self.name,
                status="success",
            )

        except Exception as exc:
            self._update_status("error", f"Initialization failed: {exc}", {"error": str(exc)})
            logger.error(
                f"{self.name} manager initialization failed",
                operation="manager_initialize",
                manager_name=self.name,
                error=str(exc),
                exc_info=True,
            )
            raise

    def _on_initialize(self) -> None:
        """Hook for subclasses to perform initialization logic.

        Default implementation does nothing. Subclasses can override this
        to perform their specific initialization steps.
        """
        pass

    def shutdown(self) -> None:
        """Shutdown the manager and clean up resources.

        Performs common shutdown steps and delegates to subclass-specific
        cleanup logic.
        """
        if self._shutdown:
            logger.warning(
                f"{self.name} manager already shutdown",
                operation="manager_shutdown",
                manager_name=self.name,
                status="already_shutdown",
            )
            return

        if not self._initialized:
            logger.warning(
                f"{self.name} manager not initialized, shutdown not needed",
                operation="manager_shutdown",
                manager_name=self.name,
                status="not_initialized",
            )
            return

        try:
            logger.info(
                f"Shutting down {self.name} manager",
                operation="manager_shutdown",
                manager_name=self.name,
            )

            # Call subclass-specific cleanup
            self._on_shutdown()

            self._shutdown = True
            self._initialized = False
            self._update_status("shutdown", "Manager successfully shutdown")

            logger.info(
                f"{self.name} manager shutdown successfully",
                operation="manager_shutdown",
                manager_name=self.name,
                status="success",
            )

        except Exception as exc:
            self._update_status("error", f"Shutdown failed: {exc}", {"error": str(exc)})
            logger.error(
                f"{self.name} manager shutdown failed",
                operation="manager_shutdown",
                manager_name=self.name,
                error=str(exc),
                exc_info=True,
            )
            raise

    def _on_shutdown(self) -> None:
        """Hook for subclasses to perform cleanup logic.

        Default implementation does nothing. Subclasses can override this
        to perform their specific cleanup steps.
        """
        pass

    def get_status(self) -> ManagerStatus:
        """Get the current status of the manager."""
        return self._status

    def _update_status(
        self, state: str, message: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        """Update the manager status with new information."""
        self._status = ManagerStatus(
            state=state,
            last_updated=datetime.utcnow(),
            message=message,
            details=details or {},
        )

    def _ensure_initialized(self) -> None:
        """Ensure the manager is initialized before performing operations."""
        if not self._initialized:
            state = self.get_status().state
            raise RuntimeError(
                f"{self.name} manager not initialized (current state: {state}). "
                f"Call initialize() first."
            )

    def _ensure_not_shutdown(self) -> None:
        """Ensure the manager has not been shutdown."""
        if self._shutdown:
            raise RuntimeError(f"{self.name} manager has been shutdown and cannot be used.")


class StatefulManager(BaseManager):
    """Base class for managers that maintain mutable state.

    Extends BaseManager with state management capabilities including:
    - State validation
    - State change tracking
    - Audit logging for state changes
    - Snapshot capabilities
    """

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(name, *args, **kwargs)
        self._state_history: list[dict[str, Any]] = []
        self._max_history_size = 100  # Keep last 100 state changes

    def _record_state_change(
        self,
        change_type: str,
        old_value: Any,
        new_value: Any,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a state change for auditing."""
        change_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "change_type": change_type,
            "old_value": old_value,
            "new_value": new_value,
            "context": context or {},
        }

        self._state_history.append(change_record)

        # Trim history if it gets too large
        if len(self._state_history) > self._max_history_size:
            self._state_history = self._state_history[-self._max_history_size :]

        logger.debug(
            f"{self.name} state change: {change_type}",
            operation="manager_state_change",
            manager_name=self.name,
            change_type=change_type,
            old_value=old_value,
            new_value=new_value,
            context=context,
        )

    def get_state_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get the history of state changes."""
        if limit is None:
            return self._state_history.copy()
        return self._state_history[-limit:]

    def clear_state_history(self) -> None:
        """Clear the state change history."""
        self._state_history.clear()
        logger.debug(
            f"{self.name} state history cleared",
            operation="manager_state_history_clear",
            manager_name=self.name,
        )


class ConfigurableManager(BaseManager):
    """Base class for managers that need configuration management.

    Extends BaseManager with configuration-specific capabilities:
    - Configuration validation
    - Hot reload support
    - Configuration change notifications
    - Environment integration
    """

    def __init__(
        self, name: str, config_path: str | None = None, *args: Any, **kwargs: Any
    ) -> None:
        self._config_path = config_path
        self._config_version: str | None = None
        super().__init__(name, *args, **kwargs)

    def _on_initialize(self) -> None:
        """Load and validate configuration during initialization."""
        self._load_configuration()
        self._validate_configuration()

    def _load_configuration(self) -> None:
        """Load configuration for the manager."""
        # This should be implemented by subclasses to load their specific config
        pass

    def _validate_configuration(self) -> None:
        """Validate the loaded configuration."""
        # This should be implemented by subclasses to validate their specific config
        pass

    def reload_configuration(self) -> None:
        """Reload the manager configuration."""
        if not self._initialized:
            logger.warning(
                f"{self.name} manager not initialized, cannot reload configuration",
                operation="manager_config_reload",
                manager_name=self.name,
                status="not_initialized",
            )
            return

        try:
            logger.info(
                f"Reloading {self.name} manager configuration",
                operation="manager_config_reload",
                manager_name=self.name,
            )

            self._load_configuration()
            self._validate_configuration()

            self._update_status("configured", "Configuration reloaded successfully")

            logger.info(
                f"{self.name} manager configuration reloaded successfully",
                operation="manager_config_reload",
                manager_name=self.name,
                status="success",
            )

        except Exception as exc:
            self._update_status("error", f"Configuration reload failed: {exc}")
            logger.error(
                f"{self.name} manager configuration reload failed",
                operation="manager_config_reload",
                manager_name=self.name,
                error=str(exc),
                exc_info=True,
            )
            raise


# Factory function for creating managers with consistent patterns
def create_manager(
    manager_class: type[BaseManager],
    name: str,
    *args: Any,
    **kwargs: Any,
) -> BaseManager:
    """Create a manager instance with consistent error handling."""
    try:
        return manager_class(name, *args, **kwargs)
    except Exception as exc:
        logger.error(
            f"Failed to create {name} manager",
            operation="manager_create",
            manager_name=name,
            manager_class=manager_class.__name__,
            error=str(exc),
            exc_info=True,
        )
        raise


__all__ = [
    "ManagerInterface",
    "ManagerStatus",
    "BaseManager",
    "StatefulManager",
    "ConfigurableManager",
    "create_manager",
]
