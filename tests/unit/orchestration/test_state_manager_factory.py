"""Unit tests for StateManager factory functions.

These tests verify that the StateManager creation functions work correctly.
"""

from unittest.mock import Mock

import pytest

from bot_v2.orchestration.state_manager import (
    ReduceOnlyModeStateManager,
    create_reduce_only_state_manager,
)
from bot_v2.persistence.event_store import EventStore


class TestReduceOnlyModeStateManagerFactory:
    """Test cases for StateManager factory functions."""

    def test_create_reduce_only_state_manager(self):
        """Test the convenience factory function."""
        event_store = Mock(spec=EventStore)

        # Test default creation
        manager = create_reduce_only_state_manager(event_store)
        assert isinstance(manager, ReduceOnlyModeStateManager)
        assert manager.is_reduce_only_mode is False
        assert manager._validation_enabled is True

        # Test with custom parameters
        manager_custom = create_reduce_only_state_manager(
            event_store,
            initial_state=True,
            validation_enabled=False
        )
        assert manager_custom.is_reduce_only_mode is True
        assert manager_custom._validation_enabled is False


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])