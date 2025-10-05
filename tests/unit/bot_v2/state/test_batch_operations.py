"""
Unit Tests for StateBatchOperations

Tests batch operations for state management including batch get, set, and delete
operations with proper cache and metadata synchronization.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, call
from bot_v2.state.batch_operations import StateBatchOperations
from bot_v2.state.state_manager import StateCategory, StateConfig


@pytest.fixture
def mock_repos():
    """Mock repositories for testing."""
    redis_repo = Mock()
    redis_repo.delete_many = AsyncMock()
    redis_repo.store_many = AsyncMock(return_value=["key1", "key2"])

    postgres_repo = Mock()
    postgres_repo.delete_many = AsyncMock()
    postgres_repo.store_many = AsyncMock(return_value=["key3"])

    s3_repo = Mock()
    s3_repo.delete_many = AsyncMock()
    s3_repo.store_many = AsyncMock(return_value=["key4"])

    return redis_repo, postgres_repo, s3_repo


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager."""
    cache = Mock()
    cache.delete = Mock()
    cache.set = Mock()
    cache.calculate_checksum = Mock(return_value="abc123")
    cache.update_metadata = Mock()
    return cache


@pytest.fixture
def mock_config():
    """Mock state config."""
    return StateConfig(redis_ttl_seconds=3600)


@pytest.fixture
def mock_metrics():
    """Mock performance metrics."""
    metrics = Mock()
    metrics.time_operation = Mock()
    metrics.time_operation.return_value.__enter__ = Mock()
    metrics.time_operation.return_value.__exit__ = Mock(return_value=False)
    return metrics


@pytest.fixture
def batch_ops(mock_repos, mock_cache_manager, mock_config, mock_metrics):
    """StateBatchOperations instance with mocks."""
    redis, postgres, s3 = mock_repos
    return StateBatchOperations(
        redis_repo=redis,
        postgres_repo=postgres,
        s3_repo=s3,
        cache_manager=mock_cache_manager,
        config=mock_config,
        metrics=mock_metrics,
    )


class TestBatchDelete:
    """Test batch delete operations."""

    @pytest.mark.asyncio
    async def test_batch_delete_empty_list(self, batch_ops):
        """Verify batch delete with empty list returns 0."""
        result = await batch_ops.batch_delete([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_batch_delete_calls_all_repos(self, batch_ops, mock_repos):
        """Verify batch delete calls all repository delete_many methods."""
        redis, postgres, s3 = mock_repos
        keys = ["key1", "key2", "key3"]

        result = await batch_ops.batch_delete(keys)

        # Verify all repos called
        redis.delete_many.assert_awaited_once_with(keys)
        postgres.delete_many.assert_awaited_once_with(keys)
        s3.delete_many.assert_awaited_once_with(keys)

        # Verify count returned
        assert result == 3

    @pytest.mark.asyncio
    async def test_batch_delete_invalidates_cache(self, batch_ops, mock_cache_manager):
        """Verify batch delete invalidates cache for all keys."""
        keys = ["key1", "key2", "key3"]

        await batch_ops.batch_delete(keys)

        # Verify cache.delete called for each key
        assert mock_cache_manager.delete.call_count == 3
        mock_cache_manager.delete.assert_any_call("key1")
        mock_cache_manager.delete.assert_any_call("key2")
        mock_cache_manager.delete.assert_any_call("key3")

    @pytest.mark.asyncio
    async def test_batch_delete_handles_repo_failures(self, batch_ops, mock_repos):
        """Verify batch delete continues despite individual repo failures."""
        redis, postgres, s3 = mock_repos
        redis.delete_many.side_effect = Exception("Redis error")
        postgres.delete_many.side_effect = Exception("Postgres error")

        keys = ["key1", "key2"]
        result = await batch_ops.batch_delete(keys)

        # Should still return count (best effort)
        assert result == 2

        # S3 should still be called
        s3.delete_many.assert_awaited_once()


class TestBatchSet:
    """Test batch set operations."""

    @pytest.mark.asyncio
    async def test_batch_set_empty_items(self, batch_ops):
        """Verify batch set with empty dict returns 0."""
        result = await batch_ops.batch_set({})
        assert result == 0

    @pytest.mark.asyncio
    async def test_batch_set_groups_by_tier(self, batch_ops, mock_repos):
        """Verify batch set groups items by storage tier."""
        redis, postgres, s3 = mock_repos

        items = {
            "key1": ({"value": 1}, StateCategory.HOT),
            "key2": ({"value": 2}, StateCategory.HOT),
            "key3": ({"value": 3}, StateCategory.WARM),
            "key4": ({"value": 4}, StateCategory.COLD),
        }

        await batch_ops.batch_set(items)

        # Verify redis called with HOT items
        redis.store_many.assert_awaited_once()
        hot_items = redis.store_many.call_args[0][0]
        assert "key1" in hot_items
        assert "key2" in hot_items

        # Verify postgres called with WARM items
        postgres.store_many.assert_awaited_once()
        warm_items = postgres.store_many.call_args[0][0]
        assert "key3" in warm_items

        # Verify s3 called with COLD items
        s3.store_many.assert_awaited_once()
        cold_items = s3.store_many.call_args[0][0]
        assert "key4" in cold_items

    @pytest.mark.asyncio
    async def test_batch_set_updates_cache_only_on_success(
        self, batch_ops, mock_repos, mock_cache_manager
    ):
        """Verify cache updated only for successfully stored keys."""
        redis, postgres, s3 = mock_repos
        redis.store_many.return_value = ["key1"]  # Only key1 succeeds

        items = {
            "key1": ({"value": 1}, StateCategory.HOT),
            "key2": ({"value": 2}, StateCategory.HOT),
        }

        await batch_ops.batch_set(items)

        # Verify cache.set called only once (for successful key1)
        mock_cache_manager.set.assert_called_once()
        # Verify it was called with key1's value
        call_args = mock_cache_manager.set.call_args[0]
        assert call_args[0] == "key1"
        assert call_args[1] == {"value": 1}

    @pytest.mark.asyncio
    async def test_batch_set_updates_metadata_on_success(
        self, batch_ops, mock_repos, mock_cache_manager
    ):
        """Verify metadata updated only for successfully stored keys."""
        redis, postgres, s3 = mock_repos
        redis.store_many.return_value = ["key1"]

        items = {
            "key1": ({"value": 1}, StateCategory.HOT),
        }

        await batch_ops.batch_set(items)

        # Verify update_metadata called
        mock_cache_manager.update_metadata.assert_called_once()
        call_kwargs = mock_cache_manager.update_metadata.call_args[1]
        assert call_kwargs["key"] == "key1"
        assert call_kwargs["category"] == StateCategory.HOT
        assert call_kwargs["checksum"] == "abc123"

    @pytest.mark.asyncio
    async def test_batch_set_applies_ttl_to_hot_items(self, batch_ops, mock_repos, mock_config):
        """Verify TTL applied to HOT tier items."""
        redis, postgres, s3 = mock_repos

        items = {
            "key1": ({"value": 1}, StateCategory.HOT),
        }

        await batch_ops.batch_set(items, ttl_seconds=7200)

        # Verify redis store_many called with TTL in metadata
        hot_items = redis.store_many.call_args[0][0]
        assert "key1" in hot_items
        serialized, metadata = hot_items["key1"]
        assert metadata["ttl_seconds"] == 7200

    @pytest.mark.asyncio
    async def test_batch_set_uses_default_ttl_when_not_specified(
        self, batch_ops, mock_repos, mock_config
    ):
        """Verify default TTL used when not specified for HOT items."""
        redis, postgres, s3 = mock_repos

        items = {
            "key1": ({"value": 1}, StateCategory.HOT),
        }

        await batch_ops.batch_set(items)  # No TTL specified

        # Verify redis store_many called with default TTL (3600)
        hot_items = redis.store_many.call_args[0][0]
        serialized, metadata = hot_items["key1"]
        assert metadata["ttl_seconds"] == 3600  # From mock_config

    @pytest.mark.asyncio
    async def test_batch_set_returns_total_stored_count(self, batch_ops, mock_repos):
        """Verify batch set returns total count of successfully stored items."""
        redis, postgres, s3 = mock_repos
        redis.store_many.return_value = ["key1", "key2"]
        postgres.store_many.return_value = ["key3"]
        s3.store_many.return_value = ["key4"]

        items = {
            "key1": ({"value": 1}, StateCategory.HOT),
            "key2": ({"value": 2}, StateCategory.HOT),
            "key3": ({"value": 3}, StateCategory.WARM),
            "key4": ({"value": 4}, StateCategory.COLD),
        }

        result = await batch_ops.batch_set(items)

        # Total: 2 (redis) + 1 (postgres) + 1 (s3) = 4
        assert result == 4

    @pytest.mark.asyncio
    async def test_batch_set_handles_repo_failures(self, batch_ops, mock_repos):
        """Verify batch set handles repository failures gracefully."""
        redis, postgres, s3 = mock_repos
        redis.store_many.side_effect = Exception("Redis connection failed")

        items = {
            "key1": ({"value": 1}, StateCategory.HOT),
            "key3": ({"value": 3}, StateCategory.WARM),
        }

        # Should not raise, just log error
        result = await batch_ops.batch_set(items)

        # WARM items should still succeed
        postgres.store_many.assert_awaited_once()

        # Result should be 1 (only WARM succeeded)
        assert result == 1
