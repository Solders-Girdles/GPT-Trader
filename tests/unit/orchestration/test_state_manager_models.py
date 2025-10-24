"""Unit tests for StateManager data models.

These tests verify that the StateChangeRequest and ReduceOnlyModeState
data models work correctly.
"""

from datetime import datetime

import pytest

from bot_v2.orchestration.state_manager import (
    ReduceOnlyModeSource,
    ReduceOnlyModeState,
    StateChangeRequest,
)


class TestStateChangeRequest:
    """Test cases for StateChangeRequest."""

    def test_state_change_request_creation(self):
        """Test StateChangeRequest creation and defaults."""
        request = StateChangeRequest(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.USER_REQUEST
        )

        assert request.enabled is True
        assert request.reason == "test"
        assert request.source == ReduceOnlyModeSource.USER_REQUEST
        assert isinstance(request.timestamp, datetime)
        assert request.metadata == {}

    def test_state_change_request_with_metadata(self):
        """Test StateChangeRequest with metadata."""
        metadata = {"loss_amount": 1000.0, "trigger": "automatic"}
        request = StateChangeRequest(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.DAILY_LOSS_LIMIT,
            metadata=metadata
        )

        assert request.metadata == metadata


class TestReduceOnlyModeState:
    """Test cases for ReduceOnlyModeState."""

    def test_state_to_dict(self):
        """Test state serialization to dictionary."""
        timestamp = datetime.utcnow()
        state = ReduceOnlyModeState(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.USER_REQUEST,
            timestamp=timestamp,
            previous_state=False
        )

        state_dict = state.to_dict()

        assert state_dict["enabled"] is True
        assert state_dict["reason"] == "test"
        assert state_dict["source"] == "user_request"  # Enum values are lowercase
        assert state_dict["timestamp"] == timestamp.isoformat()
        assert state_dict["previous_state"] is False

    def test_state_to_dict_with_none_values(self):
        """Test state serialization with None values."""
        state = ReduceOnlyModeState(enabled=False)

        state_dict = state.to_dict()

        assert state_dict["enabled"] is False
        assert state_dict["reason"] is None
        assert state_dict["source"] is None
        assert state_dict["timestamp"] is None
        assert state_dict["previous_state"] is False


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])