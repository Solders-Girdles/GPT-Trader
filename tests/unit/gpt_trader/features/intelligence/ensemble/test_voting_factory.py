"""Tests for voting mechanism factory."""

import pytest

from gpt_trader.features.intelligence.ensemble.voting import (
    ConfidenceLeaderVoting,
    VotingMechanism,
    WeightedMajorityVoting,
)


class TestVotingMechanismFactory:
    """Test VotingMechanism factory method."""

    def test_create_weighted_majority(self):
        """Test creating weighted majority voter."""
        voter = VotingMechanism.create("weighted_majority")
        assert isinstance(voter, WeightedMajorityVoting)

    def test_create_confidence_leader(self):
        """Test creating confidence leader voter."""
        voter = VotingMechanism.create("confidence_leader")
        assert isinstance(voter, ConfidenceLeaderVoting)

    def test_create_unknown_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown voting method"):
            VotingMechanism.create("unknown_method")
