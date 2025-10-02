"""
Unit tests for TierPromotionPolicy

Tests tier promotion/demotion decision logic in isolation.
"""

import pytest

from bot_v2.state.state_manager import StateCategory
from bot_v2.state.tier_promotion_policy import TierPromotionPolicy


class MockRepository:
    """Mock repository for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.stored_values = {}
        self.deleted_keys = set()

    async def store(self, key: str, value: str, metadata: dict) -> bool:
        if self.should_fail:
            return False
        self.stored_values[key] = (value, metadata)
        return True

    async def delete(self, key: str) -> bool:
        if self.should_fail:
            return False
        self.deleted_keys.add(key)
        return True


@pytest.fixture
def mock_repos():
    """Create mock repositories for all tiers."""
    return {
        "redis": MockRepository(),
        "postgres": MockRepository(),
        "s3": MockRepository(),
    }


@pytest.fixture
def policy(mock_repos):
    """Create TierPromotionPolicy with mock repositories."""
    return TierPromotionPolicy(
        redis_repo=mock_repos["redis"],
        postgres_repo=mock_repos["postgres"],
        s3_repo=mock_repos["s3"],
    )


class TestShouldAutoPromote:
    """Test auto-promotion decision logic."""

    def test_returns_false_when_auto_promote_disabled(self, policy):
        """Should not auto-promote when flag is False."""
        assert not policy.should_auto_promote(StateCategory.COLD, auto_promote=False)
        assert not policy.should_auto_promote(StateCategory.WARM, auto_promote=False)
        assert not policy.should_auto_promote(StateCategory.HOT, auto_promote=False)

    def test_returns_true_for_warm_tier_when_enabled(self, policy):
        """Should auto-promote from WARM when enabled."""
        assert policy.should_auto_promote(StateCategory.WARM, auto_promote=True)

    def test_returns_true_for_cold_tier_when_enabled(self, policy):
        """Should auto-promote from COLD when enabled."""
        assert policy.should_auto_promote(StateCategory.COLD, auto_promote=True)

    def test_returns_false_for_hot_tier_even_when_enabled(self, policy):
        """Should not auto-promote from HOT (already at highest tier)."""
        assert not policy.should_auto_promote(StateCategory.HOT, auto_promote=True)


class TestGetPromotionTarget:
    """Test promotion target tier calculation."""

    def test_cold_promotes_to_warm(self, policy):
        """COLD tier should promote to WARM."""
        assert policy.get_promotion_target(StateCategory.COLD) == StateCategory.WARM

    def test_warm_promotes_to_hot(self, policy):
        """WARM tier should promote to HOT."""
        assert policy.get_promotion_target(StateCategory.WARM) == StateCategory.HOT

    def test_hot_stays_at_hot(self, policy):
        """HOT tier should stay at HOT (already highest)."""
        assert policy.get_promotion_target(StateCategory.HOT) == StateCategory.HOT


class TestPromoteValue:
    """Test value promotion execution."""

    @pytest.mark.asyncio
    async def test_promotes_cold_to_warm_successfully(self, policy, mock_repos):
        """Should promote COLD → WARM using PostgreSQL."""
        result = await policy.promote_value(
            "test_key", "test_value", StateCategory.COLD, {"checksum": "abc123"}
        )

        assert result is True
        assert "test_key" in mock_repos["postgres"].stored_values
        assert mock_repos["postgres"].stored_values["test_key"] == (
            "test_value",
            {"checksum": "abc123"},
        )

    @pytest.mark.asyncio
    async def test_promotes_warm_to_hot_successfully(self, policy, mock_repos):
        """Should promote WARM → HOT using Redis."""
        result = await policy.promote_value(
            "test_key", "test_value", StateCategory.WARM, {"ttl_seconds": 3600}
        )

        assert result is True
        assert "test_key" in mock_repos["redis"].stored_values
        assert mock_repos["redis"].stored_values["test_key"] == (
            "test_value",
            {"ttl_seconds": 3600},
        )

    @pytest.mark.asyncio
    async def test_returns_false_when_target_repo_unavailable(self):
        """Should return False when target repository is None."""
        policy = TierPromotionPolicy(
            redis_repo=None, postgres_repo=None, s3_repo=None
        )

        result = await policy.promote_value(
            "test_key", "test_value", StateCategory.COLD, {}
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_promotion_fails(self):
        """Should return False when repository store fails."""
        failing_redis = MockRepository(should_fail=True)
        policy = TierPromotionPolicy(
            redis_repo=failing_redis, postgres_repo=None, s3_repo=None
        )

        result = await policy.promote_value(
            "test_key", "test_value", StateCategory.WARM, {}
        )

        assert result is False


class TestPromoteToHot:
    """Test manual promotion to HOT tier."""

    @pytest.mark.asyncio
    async def test_promotes_to_hot_successfully(self, policy, mock_repos):
        """Should store value in Redis (HOT tier)."""
        result = await policy.promote_to_hot(
            "test_key", "test_value", {"ttl_seconds": 3600}
        )

        assert result is True
        assert "test_key" in mock_repos["redis"].stored_values

    @pytest.mark.asyncio
    async def test_returns_false_when_redis_unavailable(self):
        """Should return False when Redis repository is None."""
        policy = TierPromotionPolicy(
            redis_repo=None, postgres_repo=None, s3_repo=None
        )

        result = await policy.promote_to_hot("test_key", "test_value", {})

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_redis_store_fails(self):
        """Should return False when Redis store operation fails."""
        failing_redis = MockRepository(should_fail=True)
        policy = TierPromotionPolicy(
            redis_repo=failing_redis, postgres_repo=None, s3_repo=None
        )

        result = await policy.promote_to_hot("test_key", "test_value", {})

        assert result is False


class TestDemoteToCold:
    """Test demotion to COLD tier."""

    @pytest.mark.asyncio
    async def test_demotes_to_cold_successfully(self, policy, mock_repos):
        """Should delete from HOT/WARM and store in COLD."""
        result = await policy.demote_to_cold(
            "test_key", "test_value", {"checksum": "abc123"}
        )

        assert result is True
        # Verify deletions
        assert "test_key" in mock_repos["redis"].deleted_keys
        assert "test_key" in mock_repos["postgres"].deleted_keys
        # Verify storage in COLD
        assert "test_key" in mock_repos["s3"].stored_values

    @pytest.mark.asyncio
    async def test_deletes_from_all_higher_tiers(self, policy, mock_repos):
        """Should delete key from both HOT and WARM tiers."""
        await policy.demote_to_cold("test_key", "test_value", {})

        assert "test_key" in mock_repos["redis"].deleted_keys
        assert "test_key" in mock_repos["postgres"].deleted_keys

    @pytest.mark.asyncio
    async def test_returns_false_when_s3_unavailable(self):
        """Should return False when S3 repository is None."""
        policy = TierPromotionPolicy(
            redis_repo=MockRepository(),
            postgres_repo=MockRepository(),
            s3_repo=None,
        )

        result = await policy.demote_to_cold("test_key", "test_value", {})

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_s3_store_fails(self):
        """Should return False when S3 store operation fails."""
        policy = TierPromotionPolicy(
            redis_repo=MockRepository(),
            postgres_repo=MockRepository(),
            s3_repo=MockRepository(should_fail=True),
        )

        result = await policy.demote_to_cold("test_key", "test_value", {})

        assert result is False

    @pytest.mark.asyncio
    async def test_continues_despite_deletion_failures(self, policy, mock_repos):
        """Should attempt S3 storage even if deletions fail."""
        # Make deletions fail but S3 succeed
        mock_repos["redis"].should_fail = True
        mock_repos["postgres"].should_fail = True

        result = await policy.demote_to_cold("test_key", "test_value", {})

        # Should still succeed because S3 storage worked
        assert result is True
        assert "test_key" in mock_repos["s3"].stored_values


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_handles_all_repos_none(self):
        """Should handle gracefully when all repositories are None."""
        policy = TierPromotionPolicy(
            redis_repo=None, postgres_repo=None, s3_repo=None
        )

        # All operations should return False
        assert not await policy.promote_value("k", "v", StateCategory.COLD, {})
        assert not await policy.promote_to_hot("k", "v", {})
        assert not await policy.demote_to_cold("k", "v", {})

    @pytest.mark.asyncio
    async def test_handles_empty_metadata(self, policy, mock_repos):
        """Should handle empty metadata dict."""
        result = await policy.promote_to_hot("test_key", "test_value", {})

        assert result is True
        assert mock_repos["redis"].stored_values["test_key"][1] == {}

    @pytest.mark.asyncio
    async def test_handles_none_metadata_values(self, policy, mock_repos):
        """Should handle metadata with None values."""
        metadata = {"ttl_seconds": None, "checksum": None}
        result = await policy.promote_to_hot("test_key", "test_value", metadata)

        assert result is True
        assert mock_repos["redis"].stored_values["test_key"][1] == metadata
