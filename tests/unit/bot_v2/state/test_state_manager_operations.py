"""Tests for StateManager storage operations and tier management."""

import json
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from bot_v2.state.state_manager import (
    StateCategory,
    StateConfig,
    StateManager,
    get_state_manager,
    get_state,
    set_state,
    delete_state,
)


@pytest.fixture
def mock_redis():
    """Create mock Redis adapter"""
    adapter = Mock()
    adapter.ping = Mock(return_value=True)
    adapter.get = Mock(return_value=None)
    adapter.setex = Mock(return_value=True)
    adapter.delete = Mock()
    adapter.keys = Mock(return_value=[])
    adapter.dbsize = Mock(return_value=0)
    adapter.close = Mock()
    return adapter


@pytest.fixture
def mock_postgres():
    """Create mock PostgreSQL adapter"""
    adapter = Mock()
    adapter.execute = Mock(return_value=[])
    adapter.commit = Mock()
    adapter.rollback = Mock()
    adapter.close = Mock()
    return adapter


@pytest.fixture
def mock_s3():
    """Create mock S3 adapter"""
    adapter = Mock()
    adapter.head_bucket = Mock()
    adapter.get_object = Mock()
    adapter.put_object = Mock()
    adapter.delete_object = Mock()
    adapter.list_objects_v2 = Mock(return_value={})
    return adapter


@pytest.fixture
def state_manager_with_mocks(mock_redis, mock_postgres, mock_s3):
    """Create StateManager with mock adapters"""
    config = StateConfig(
        redis_host="localhost",
        redis_port=6379,
        postgres_database="test_db",
        s3_bucket="test-bucket",
        cache_size_mb=1,  # Small cache for testing
    )
    return StateManager(
        config=config,
        redis_adapter=mock_redis,
        postgres_adapter=mock_postgres,
        s3_adapter=mock_s3,
    )


class TestStateManagerOperations:
    """Test StateManager storage operations"""

    @pytest.mark.asyncio
    async def test_get_state_from_local_cache(self, state_manager_with_mocks):
        """Test retrieving state from local cache"""
        manager = state_manager_with_mocks
        manager._local_cache["test_key"] = {"value": 123}

        result = await manager.get_state("test_key")

        assert result == {"value": 123}
        assert "test_key" in manager._access_history

    @pytest.mark.asyncio
    async def test_get_state_from_redis(self, state_manager_with_mocks, mock_redis):
        """Test retrieving state from Redis"""
        manager = state_manager_with_mocks
        mock_redis.get.return_value = json.dumps({"value": 456})

        result = await manager.get_state("test_key")

        assert result == {"value": 456}
        assert "test_key" in manager._local_cache
        mock_redis.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_state_from_postgres(self, state_manager_with_mocks, mock_postgres):
        """Test retrieving state from PostgreSQL with auto-promotion"""
        manager = state_manager_with_mocks
        mock_postgres.execute.return_value = [{"data": {"value": 789}}]

        result = await manager.get_state("test_key", auto_promote=True)

        assert result == {"value": 789}
        assert "test_key" in manager._local_cache
        # Should be promoted to Redis
        assert manager.redis_adapter.setex.called

    @pytest.mark.asyncio
    async def test_get_state_from_s3(self, state_manager_with_mocks, mock_s3):
        """Test retrieving state from S3 with auto-promotion"""
        manager = state_manager_with_mocks
        mock_body = Mock()
        mock_body.read = Mock(return_value=json.dumps({"value": 999}).encode())
        mock_s3.get_object.return_value = {"Body": mock_body}

        result = await manager.get_state("test_key", auto_promote=True)

        assert result == {"value": 999}
        # Should be promoted to PostgreSQL
        assert manager.postgres_adapter.execute.called

    @pytest.mark.asyncio
    async def test_set_state_hot_tier(self, state_manager_with_mocks, mock_redis):
        """Test storing state in hot tier (Redis)"""
        manager = state_manager_with_mocks

        success = await manager.set_state("test_key", {"value": 123}, StateCategory.HOT)

        assert success is True
        mock_redis.setex.assert_called_once()
        assert "test_key" in manager._local_cache
        assert "test_key" in manager._metadata_cache

    @pytest.mark.asyncio
    async def test_set_state_warm_tier(self, state_manager_with_mocks, mock_postgres):
        """Test storing state in warm tier (PostgreSQL)"""
        manager = state_manager_with_mocks

        success = await manager.set_state("test_key", {"value": 456}, StateCategory.WARM)

        assert success is True
        mock_postgres.execute.assert_called()
        mock_postgres.commit.assert_called()

    @pytest.mark.asyncio
    async def test_set_state_cold_tier(self, state_manager_with_mocks, mock_s3):
        """Test storing state in cold tier (S3)"""
        manager = state_manager_with_mocks

        success = await manager.set_state("test_key", {"value": 789}, StateCategory.COLD)

        assert success is True
        mock_s3.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_state_with_custom_ttl(self, state_manager_with_mocks, mock_redis):
        """Test setting state with custom TTL"""
        manager = state_manager_with_mocks

        await manager.set_state("test_key", {"value": 123}, StateCategory.HOT, ttl_seconds=300)

        # Verify TTL was passed to Redis
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300  # TTL argument

    @pytest.mark.asyncio
    async def test_set_state_serialization_error(self, state_manager_with_mocks):
        """Test set_state handles serialization errors"""
        manager = state_manager_with_mocks

        # Create non-serializable object
        class NonSerializable:
            pass

        # Should handle gracefully, default=str will convert it
        success = await manager.set_state("test_key", NonSerializable())

        # Will succeed because default=str handles it
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_state_all_tiers(
        self, state_manager_with_mocks, mock_redis, mock_postgres, mock_s3
    ):
        """Test deleting state from all tiers"""
        manager = state_manager_with_mocks
        manager._local_cache["test_key"] = {"value": 123}
        manager._metadata_cache["test_key"] = Mock()
        manager._access_history["test_key"] = [datetime.now()]

        success = await manager.delete_state("test_key")

        assert success is True
        mock_redis.delete.assert_called_with("test_key")
        mock_postgres.execute.assert_called()
        mock_postgres.commit.assert_called()
        mock_s3.delete_object.assert_called()
        assert "test_key" not in manager._local_cache
        assert "test_key" not in manager._metadata_cache

    @pytest.mark.asyncio
    async def test_delete_state_postgres_error_with_rollback(
        self, state_manager_with_mocks, mock_postgres
    ):
        """Test delete handles PostgreSQL errors with rollback"""
        manager = state_manager_with_mocks
        mock_postgres.execute.side_effect = Exception("DB error")

        success = await manager.delete_state("test_key")

        # Should continue despite error
        assert success is False
        mock_postgres.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_get_from_redis_with_exception(self, state_manager_with_mocks, mock_redis):
        """Test Redis get with exception handling"""
        manager = state_manager_with_mocks
        mock_redis.get.side_effect = Exception("Redis error")

        result = await manager._get_from_redis("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_from_postgres_with_exception(
        self, state_manager_with_mocks, mock_postgres
    ):
        """Test PostgreSQL get with exception handling"""
        manager = state_manager_with_mocks
        mock_postgres.execute.side_effect = Exception("DB error")

        result = await manager._get_from_postgres("test_key")

        assert result is None
        mock_postgres.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_get_from_s3_with_exception(self, state_manager_with_mocks, mock_s3):
        """Test S3 get with exception handling"""
        manager = state_manager_with_mocks
        mock_s3.get_object.side_effect = Exception("S3 error")

        result = await manager._get_from_s3("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_in_redis_with_exception(self, state_manager_with_mocks, mock_redis):
        """Test Redis set with exception handling"""
        manager = state_manager_with_mocks
        mock_redis.setex.side_effect = Exception("Redis error")

        result = await manager._set_in_redis("test_key", "value", None)

        assert result is False

    @pytest.mark.asyncio
    async def test_set_in_postgres_with_exception(
        self, state_manager_with_mocks, mock_postgres
    ):
        """Test PostgreSQL set with exception handling"""
        manager = state_manager_with_mocks
        mock_postgres.execute.side_effect = Exception("DB error")

        result = await manager._set_in_postgres("test_key", "value", "checksum")

        assert result is False
        mock_postgres.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_set_in_s3_with_exception(self, state_manager_with_mocks, mock_s3):
        """Test S3 set with exception handling"""
        manager = state_manager_with_mocks
        mock_s3.put_object.side_effect = Exception("S3 error")

        result = await manager._set_in_s3("test_key", "value", "checksum")

        assert result is False

    def test_calculate_checksum(self, state_manager_with_mocks):
        """Test checksum calculation"""
        manager = state_manager_with_mocks

        checksum = manager._calculate_checksum("test data")

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest

    def test_update_access_history(self, state_manager_with_mocks):
        """Test access history tracking"""
        manager = state_manager_with_mocks

        manager._update_access_history("test_key")
        manager._update_access_history("test_key")

        assert "test_key" in manager._access_history
        assert len(manager._access_history["test_key"]) == 2

    def test_update_access_history_max_length(self, state_manager_with_mocks):
        """Test access history maintains max length"""
        manager = state_manager_with_mocks

        # Add more than 100 accesses
        for _ in range(150):
            manager._update_access_history("test_key")

        assert len(manager._access_history["test_key"]) == 100

    def test_manage_cache_size(self, state_manager_with_mocks):
        """Test cache size management"""
        manager = state_manager_with_mocks
        manager.config.cache_size_mb = 0.001  # Very small cache

        # Fill cache with data
        for i in range(50):
            manager._local_cache[f"key_{i}"] = {"data": "x" * 1000}
            manager._access_history[f"key_{i}"] = [datetime.now()]

        manager._manage_cache_size()

        # Cache should be reduced
        assert len(manager._local_cache) < 50

    @pytest.mark.asyncio
    async def test_get_keys_by_pattern_redis(self, state_manager_with_mocks, mock_redis):
        """Test getting keys by pattern from Redis"""
        manager = state_manager_with_mocks
        mock_redis.keys.return_value = ["key_1", "key_2"]

        keys = await manager.get_keys_by_pattern("key_*")

        assert "key_1" in keys
        assert "key_2" in keys

    @pytest.mark.asyncio
    async def test_get_keys_by_pattern_postgres(
        self, state_manager_with_mocks, mock_postgres
    ):
        """Test getting keys by pattern from PostgreSQL"""
        manager = state_manager_with_mocks
        mock_postgres.execute.return_value = [{"key": "key_3"}, {"key": "key_4"}]

        keys = await manager.get_keys_by_pattern("key_*")

        assert "key_3" in keys
        assert "key_4" in keys

    @pytest.mark.asyncio
    async def test_get_keys_by_pattern_s3(self, state_manager_with_mocks, mock_s3):
        """Test getting keys by pattern from S3"""
        manager = state_manager_with_mocks
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "cold/key_5"}, {"Key": "cold/key_6"}]
        }

        keys = await manager.get_keys_by_pattern("key_*")

        assert "key_5" in keys
        assert "key_6" in keys

    @pytest.mark.asyncio
    async def test_get_keys_by_pattern_with_errors(
        self, state_manager_with_mocks, mock_redis, mock_postgres, mock_s3
    ):
        """Test getting keys handles errors gracefully"""
        manager = state_manager_with_mocks
        mock_redis.keys.side_effect = Exception("Redis error")
        mock_postgres.execute.side_effect = Exception("DB error")
        mock_s3.list_objects_v2.side_effect = Exception("S3 error")

        keys = await manager.get_keys_by_pattern("key_*")

        # Should return empty list without raising
        assert keys == []

    @pytest.mark.asyncio
    async def test_promote_to_hot(self, state_manager_with_mocks):
        """Test manual promotion to hot tier"""
        manager = state_manager_with_mocks
        manager._local_cache["test_key"] = {"value": 123}

        success = await manager.promote_to_hot("test_key")

        assert success is True
        assert manager.redis_adapter.setex.called

    @pytest.mark.asyncio
    async def test_promote_to_hot_missing_key(self, state_manager_with_mocks):
        """Test promotion fails for missing key"""
        manager = state_manager_with_mocks

        success = await manager.promote_to_hot("missing_key")

        assert success is False

    @pytest.mark.asyncio
    async def test_demote_to_cold(self, state_manager_with_mocks):
        """Test manual demotion to cold tier"""
        manager = state_manager_with_mocks
        manager._local_cache["test_key"] = {"value": 123}

        success = await manager.demote_to_cold("test_key")

        assert success is True
        assert manager.s3_adapter.put_object.called
        # Should delete from hot/warm tiers
        assert manager.redis_adapter.delete.called
        assert manager.postgres_adapter.execute.called

    @pytest.mark.asyncio
    async def test_demote_to_cold_missing_key(self, state_manager_with_mocks):
        """Test demotion fails for missing key"""
        manager = state_manager_with_mocks

        success = await manager.demote_to_cold("missing_key")

        assert success is False

    @pytest.mark.asyncio
    async def test_get_storage_stats(
        self, state_manager_with_mocks, mock_redis, mock_postgres, mock_s3
    ):
        """Test getting storage statistics"""
        manager = state_manager_with_mocks
        mock_redis.dbsize.return_value = 10
        mock_postgres.execute.return_value = [{"count": 20}]
        mock_s3.list_objects_v2.return_value = {"KeyCount": 30}
        manager._local_cache["key1"] = {"value": 123}

        stats = await manager.get_storage_stats()

        assert stats["hot_keys"] == 10
        assert stats["warm_keys"] == 20
        assert stats["cold_keys"] == 30
        assert stats["cache_keys"] == 1
        assert stats["total_keys"] == 60

    @pytest.mark.asyncio
    async def test_get_storage_stats_with_errors(
        self, state_manager_with_mocks, mock_redis, mock_postgres, mock_s3
    ):
        """Test storage stats handles errors gracefully"""
        manager = state_manager_with_mocks
        mock_redis.dbsize.side_effect = Exception("Redis error")
        mock_postgres.execute.side_effect = Exception("DB error")
        mock_s3.list_objects_v2.side_effect = Exception("S3 error")

        stats = await manager.get_storage_stats()

        # Should return zeros for failed backends
        assert stats["hot_keys"] == 0
        assert stats["warm_keys"] == 0
        assert stats["cold_keys"] == 0

    def test_close_connections(self, state_manager_with_mocks, mock_redis, mock_postgres):
        """Test closing all connections"""
        manager = state_manager_with_mocks

        manager.close()

        mock_redis.close.assert_called_once()
        mock_postgres.close.assert_called_once()


class TestStateManagerInitialization:
    """Test StateManager initialization paths"""

    def test_init_with_failing_postgres_adapter_creation(self, mock_redis, mock_s3):
        """Test initialization with failing PostgreSQL adapter but provided adapter"""
        config = StateConfig()
        mock_postgres = Mock()
        mock_postgres.execute.side_effect = Exception("Table creation failed")

        manager = StateManager(
            config=config,
            redis_adapter=mock_redis,
            postgres_adapter=mock_postgres,
            s3_adapter=mock_s3,
        )

        # Should set postgres_adapter to None after table creation failure
        assert manager.postgres_adapter is None

    def test_init_with_failing_s3_bucket_verification(self, mock_redis, mock_postgres):
        """Test initialization with failing S3 bucket verification"""
        config = StateConfig()
        mock_s3 = Mock()
        mock_s3.head_bucket.side_effect = Exception("Bucket not found")

        manager = StateManager(
            config=config,
            redis_adapter=mock_redis,
            postgres_adapter=mock_postgres,
            s3_adapter=mock_s3,
        )

        # Should set s3_adapter to None after verification failure
        assert manager.s3_adapter is None


class TestConvenienceFunctions:
    """Test global convenience functions"""

    @pytest.mark.asyncio
    async def test_get_state_convenience_function(self):
        """Test get_state convenience function"""
        with patch("bot_v2.state.state_manager._state_manager") as mock_manager_instance:
            mock_instance = Mock()
            mock_instance.get_state = AsyncMock(return_value={"value": 123})
            mock_manager_instance.return_value = mock_instance

            # Reset global instance
            import bot_v2.state.state_manager as sm

            sm._state_manager = mock_instance

            result = await get_state("test_key")

            assert result == {"value": 123}
            mock_instance.get_state.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_set_state_convenience_function(self):
        """Test set_state convenience function"""
        with patch("bot_v2.state.state_manager._state_manager") as mock_manager_instance:
            mock_instance = Mock()
            mock_instance.set_state = AsyncMock(return_value=True)
            mock_manager_instance.return_value = mock_instance

            import bot_v2.state.state_manager as sm

            sm._state_manager = mock_instance

            result = await set_state("test_key", {"value": 123})

            assert result is True
            mock_instance.set_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_state_convenience_function(self):
        """Test delete_state convenience function"""
        with patch("bot_v2.state.state_manager._state_manager") as mock_manager_instance:
            mock_instance = Mock()
            mock_instance.delete_state = AsyncMock(return_value=True)
            mock_manager_instance.return_value = mock_instance

            import bot_v2.state.state_manager as sm

            sm._state_manager = mock_instance

            result = await delete_state("test_key")

            assert result is True
            mock_instance.delete_state.assert_called_once_with("test_key")

    def test_get_state_manager_creates_instance(self):
        """Test get_state_manager creates global instance"""
        # Reset global instance
        import bot_v2.state.state_manager as sm

        sm._state_manager = None

        manager = get_state_manager()

        assert manager is not None
        assert isinstance(manager, StateManager)

    def test_get_state_manager_returns_existing_instance(self):
        """Test get_state_manager returns existing instance"""
        import bot_v2.state.state_manager as sm

        mock_manager = Mock(spec=StateManager)
        sm._state_manager = mock_manager

        manager = get_state_manager()

        assert manager is mock_manager
