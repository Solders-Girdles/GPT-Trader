"""Base protocol for recovery strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation


class RecoveryStrategy(ABC):
    """
    Base protocol for failure recovery strategies.

    Each strategy implements recovery logic for a specific failure type,
    maintaining async compatibility and updating operation state.
    """

    def __init__(
        self,
        state_manager: Any,
        checkpoint_handler: Any,
        backup_manager: Any | None = None,
    ) -> None:
        """
        Initialize recovery strategy with system dependencies.

        Args:
            state_manager: State management system
            checkpoint_handler: Checkpoint/snapshot handler
            backup_manager: Optional backup system
        """
        self.state_manager = state_manager
        self.checkpoint_handler = checkpoint_handler
        self.backup_manager = backup_manager

    @abstractmethod
    async def recover(self, operation: RecoveryOperation) -> bool:
        """
        Execute recovery for this failure type.

        Args:
            operation: Recovery operation context with metadata and tracking

        Returns:
            bool: True if recovery succeeded, False otherwise
        """
        ...

    @property
    @abstractmethod
    def failure_type_name(self) -> str:
        """Return human-readable name of the failure type this strategy handles."""
        ...
