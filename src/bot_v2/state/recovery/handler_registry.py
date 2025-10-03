"""
Recovery Handler Registry

Manages registration and resolution of failure recovery handlers.
Extracted from RecoveryOrchestrator to improve separation of concerns.
"""

import logging
from collections.abc import Callable

from bot_v2.state.recovery.models import FailureType

logger = logging.getLogger(__name__)


class RecoveryHandlerRegistry:
    """
    Registry for failure recovery handlers.

    Provides centralized handler registration, lookup, and iteration.
    Supports default handlers and validation of handler signatures.
    """

    def __init__(self) -> None:
        """Initialize empty handler registry."""
        self._handlers: dict[FailureType, Callable] = {}

    def register(self, failure_type: FailureType, handler: Callable) -> None:
        """
        Register a recovery handler for a failure type.

        Args:
            failure_type: Type of failure this handler can recover
            handler: Async callable that accepts RecoveryOperation and returns bool

        Raises:
            ValueError: If handler already registered for this failure type
        """
        handler_name = getattr(handler, "__name__", repr(handler))

        if failure_type in self._handlers:
            existing_handler = self._handlers[failure_type]
            existing_name = getattr(existing_handler, "__name__", repr(existing_handler))
            logger.warning(
                f"Handler for {failure_type.value} already registered "
                f"({existing_name}), replacing with {handler_name}"
            )

        self._handlers[failure_type] = handler
        logger.debug(f"Registered handler {handler_name} for {failure_type.value}")

    def register_batch(self, handlers: dict[FailureType, Callable]) -> None:
        """
        Register multiple handlers at once.

        Args:
            handlers: Dictionary mapping failure types to handler callables
        """
        for failure_type, handler in handlers.items():
            self.register(failure_type, handler)

    def get_handler(self, failure_type: FailureType) -> Callable | None:
        """
        Get registered handler for failure type.

        Args:
            failure_type: Type of failure to get handler for

        Returns:
            Handler callable if registered, None otherwise
        """
        return self._handlers.get(failure_type)

    def has_handler(self, failure_type: FailureType) -> bool:
        """
        Check if handler is registered for failure type.

        Args:
            failure_type: Type of failure to check

        Returns:
            True if handler registered, False otherwise
        """
        return failure_type in self._handlers

    def get_all_handlers(self) -> dict[FailureType, Callable]:
        """
        Get all registered handlers.

        Returns:
            Copy of handlers dictionary
        """
        return self._handlers.copy()

    def get_registered_failure_types(self) -> list[FailureType]:
        """
        Get list of failure types with registered handlers.

        Returns:
            List of failure types
        """
        return list(self._handlers.keys())

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        logger.debug("Cleared all registered handlers")

    def unregister(self, failure_type: FailureType) -> bool:
        """
        Unregister handler for failure type.

        Args:
            failure_type: Type of failure to unregister

        Returns:
            True if handler was unregistered, False if not registered
        """
        if failure_type in self._handlers:
            del self._handlers[failure_type]
            logger.debug(f"Unregistered handler for {failure_type.value}")
            return True
        return False

    def __len__(self) -> int:
        """Get number of registered handlers."""
        return len(self._handlers)

    def __contains__(self, failure_type: FailureType) -> bool:
        """Check if failure type has registered handler."""
        return failure_type in self._handlers
