"""Unit tests for RedisStateRepository - HOT tier storage."""

import json
import pytest
from unittest.mock import Mock, patch

from bot_v2.state.repositories.redis_repository import RedisStateRepository
from bot_v2.state.utils.adapters import RedisAdapter


@pytest.fixture
def mock_redis_adapter():
    """Create a mock RedisAdapter for testing."""
    return Mock(spec=RedisAdapter)


@pytest.fixture
def redis_repo(mock_redis_adapter):
    """Create a RedisStateRepository with mock adapter."""
    return RedisStateRepository(adapter=mock_redis_adapter, default_ttl=3600)


class TestRedisRepositoryStore:
    """Test store operations."""

    @pytest.mark.asyncio
    async def test_store_with_default_ttl(self, redis_repo, mock_redis_adapter):
        """Verify store uses default TTL when not specified in metadata."""
        mock_redis_adapter.setex.return_value = True

        result = await redis_repo.store("test_key", '{"data": "value"}', {})

        assert result is True
        mock_redis_adapter.setex.assert_called_once_with("test_key", 3600, '{"data": "value"}')

    @pytest.mark.asyncio
    async def test_store_with_custom_ttl(self, redis_repo, mock_redis_adapter):
        """Verify store respects custom TTL from metadata."""
        mock_redis_adapter.setex.return_value = True

        result = await redis_repo.store("test_key", '{"data": "value"}', {"ttl_seconds": 7200})

        assert result is True
        mock_redis_adapter.setex.assert_called_once_with("test_key", 7200, '{"data": "value"}')

    @pytest.mark.asyncio
    async def test_store_failure_returns_false(self, redis_repo, mock_redis_adapter):
        """Verify store returns False on adapter exception."""
        mock_redis_adapter.setex.side_effect = Exception("Redis connection failed")

        result = await redis_repo.store("test_key", '{"data": "value"}', {})

        assert result is False

    @pytest.mark.asyncio
    async def test_store_logs_error_on_failure(self, redis_repo, mock_redis_adapter):
        """Verify store logs error when adapter raises exception."""
        mock_redis_adapter.setex.side_effect = Exception("Redis error")

        with patch("bot_v2.state.repositories.redis_repository.logger") as mock_logger:
            await redis_repo.store("test_key", '{"data": "value"}', {})
            mock_logger.error.assert_called_once()


class TestRedisRepositoryFetch:
    """Test fetch operations."""

    @pytest.mark.asyncio
    async def test_fetch_deserializes_json(self, redis_repo, mock_redis_adapter):
        """Verify fetch deserializes JSON value from Redis."""
        mock_redis_adapter.get.return_value = '{"data": "value", "count": 42}'

        result = await redis_repo.fetch("test_key")

        assert result == {"data": "value", "count": 42}
        mock_redis_adapter.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_missing_key(self, redis_repo, mock_redis_adapter):
        """Verify fetch returns None when key doesn't exist."""
        mock_redis_adapter.get.return_value = None

        result = await redis_repo.fetch("missing_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_exception(self, redis_repo, mock_redis_adapter):
        """Verify fetch returns None when adapter raises exception."""
        mock_redis_adapter.get.side_effect = Exception("Redis error")

        result = await redis_repo.fetch("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_logs_debug_on_failure(self, redis_repo, mock_redis_adapter):
        """Verify fetch logs debug message on error."""
        mock_redis_adapter.get.side_effect = Exception("Connection timeout")

        with patch("bot_v2.state.repositories.redis_repository.logger") as mock_logger:
            await redis_repo.fetch("test_key")
            mock_logger.debug.assert_called_once()


class TestRedisRepositoryDelete:
    """Test delete operations."""

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, redis_repo, mock_redis_adapter):
        """Verify delete returns True when successful."""
        mock_redis_adapter.delete.return_value = 1

        result = await redis_repo.delete("test_key")

        assert result is True
        mock_redis_adapter.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_exception(self, redis_repo, mock_redis_adapter):
        """Verify delete returns False when adapter raises exception."""
        mock_redis_adapter.delete.side_effect = Exception("Redis error")

        result = await redis_repo.delete("test_key")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_logs_warning_on_failure(self, redis_repo, mock_redis_adapter):
        """Verify delete logs warning on error."""
        mock_redis_adapter.delete.side_effect = Exception("Connection lost")

        with patch("bot_v2.state.repositories.redis_repository.logger") as mock_logger:
            await redis_repo.delete("test_key")
            mock_logger.warning.assert_called_once()


class TestRedisRepositoryKeys:
    """Test key pattern matching."""

    @pytest.mark.asyncio
    async def test_keys_respects_pattern(self, redis_repo, mock_redis_adapter):
        """Verify keys returns matching keys from adapter."""
        mock_redis_adapter.keys.return_value = ["position:BTC", "position:ETH", "position:SOL"]

        result = await redis_repo.keys("position:*")

        assert result == ["position:BTC", "position:ETH", "position:SOL"]
        mock_redis_adapter.keys.assert_called_once_with("position:*")

    @pytest.mark.asyncio
    async def test_keys_returns_empty_list_on_no_match(self, redis_repo, mock_redis_adapter):
        """Verify keys returns empty list when no keys match."""
        mock_redis_adapter.keys.return_value = []

        result = await redis_repo.keys("nonexistent:*")

        assert result == []

    @pytest.mark.asyncio
    async def test_keys_returns_empty_list_on_exception(self, redis_repo, mock_redis_adapter):
        """Verify keys returns empty list on adapter exception."""
        mock_redis_adapter.keys.side_effect = Exception("Redis error")

        result = await redis_repo.keys("test:*")

        assert result == []


class TestRedisRepositoryStats:
    """Test statistics collection."""

    @pytest.mark.asyncio
    async def test_stats_returns_key_count(self, redis_repo, mock_redis_adapter):
        """Verify stats returns key count from adapter."""
        mock_redis_adapter.dbsize.return_value = 42

        result = await redis_repo.stats()

        assert result == {"key_count": 42}
        mock_redis_adapter.dbsize.assert_called_once()

    @pytest.mark.asyncio
    async def test_stats_returns_zero_on_exception(self, redis_repo, mock_redis_adapter):
        """Verify stats returns zero count on adapter exception."""
        mock_redis_adapter.dbsize.side_effect = Exception("Redis error")

        result = await redis_repo.stats()

        assert result == {"key_count": 0}


class TestRedisRepositoryBatchOperations:
    """Test batch store/delete operations."""

    @pytest.mark.asyncio
    async def test_store_many_with_single_ttl(self, redis_repo, mock_redis_adapter):
        """Verify store_many batches items with same TTL efficiently."""
        items = {
            "key1": ('{"value": 1}', {"ttl_seconds": 3600}),
            "key2": ('{"value": 2}', {"ttl_seconds": 3600}),
            "key3": ('{"value": 3}', {"ttl_seconds": 3600}),
        }

        result = await redis_repo.store_many(items)

        assert result == {"key1", "key2", "key3"}
        # Verify msetex called once with all items grouped by TTL
        mock_redis_adapter.msetex.assert_called_once()
        call_args = mock_redis_adapter.msetex.call_args
        assert call_args[0][0] == {
            "key1": '{"value": 1}',
            "key2": '{"value": 2}',
            "key3": '{"value": 3}',
        }
        assert call_args[0][1] == 3600

    @pytest.mark.asyncio
    async def test_store_many_groups_by_ttl(self, redis_repo, mock_redis_adapter):
        """Verify store_many groups items by different TTLs."""
        items = {
            "key1": ('{"value": 1}', {"ttl_seconds": 3600}),
            "key2": ('{"value": 2}', {"ttl_seconds": 7200}),
            "key3": ('{"value": 3}', {"ttl_seconds": 3600}),
        }

        result = await redis_repo.store_many(items)

        assert result == {"key1", "key2", "key3"}
        # Should be called twice - once per TTL group
        assert mock_redis_adapter.msetex.call_count == 2

    @pytest.mark.asyncio
    async def test_store_many_uses_default_ttl(self, redis_repo, mock_redis_adapter):
        """Verify store_many uses default TTL when not specified."""
        items = {
            "key1": ('{"value": 1}', {}),
            "key2": ('{"value": 2}', {}),
        }

        result = await redis_repo.store_many(items)

        assert result == {"key1", "key2"}
        call_args = mock_redis_adapter.msetex.call_args
        assert call_args[0][1] == 3600  # default_ttl

    @pytest.mark.asyncio
    async def test_store_many_returns_empty_set_on_empty_input(
        self, redis_repo, mock_redis_adapter
    ):
        """Verify store_many returns empty set for empty input."""
        result = await redis_repo.store_many({})

        assert result == set()
        mock_redis_adapter.msetex.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_many_returns_empty_set_on_exception(self, redis_repo, mock_redis_adapter):
        """Verify store_many returns empty set on adapter exception."""
        mock_redis_adapter.msetex.side_effect = Exception("Pipeline failed")
        items = {"key1": ('{"value": 1}', {})}

        result = await redis_repo.store_many(items)

        assert result == set()

    @pytest.mark.asyncio
    async def test_delete_many_uses_adapter_batch_delete(self, redis_repo, mock_redis_adapter):
        """Verify delete_many uses adapter's batch delete."""
        mock_redis_adapter.delete_many.return_value = 3
        keys = ["key1", "key2", "key3"]

        result = await redis_repo.delete_many(keys)

        assert result == 3
        mock_redis_adapter.delete_many.assert_called_once_with(["key1", "key2", "key3"])

    @pytest.mark.asyncio
    async def test_delete_many_returns_zero_on_empty_input(self, redis_repo, mock_redis_adapter):
        """Verify delete_many returns 0 for empty input."""
        result = await redis_repo.delete_many([])

        assert result == 0
        mock_redis_adapter.delete_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_many_returns_zero_on_exception(self, redis_repo, mock_redis_adapter):
        """Verify delete_many returns 0 on adapter exception."""
        mock_redis_adapter.delete_many.side_effect = Exception("Redis error")

        result = await redis_repo.delete_many(["key1", "key2"])

        assert result == 0


class TestRedisRepositoryInitialization:
    """Test repository initialization."""

    def test_init_sets_default_ttl(self, mock_redis_adapter):
        """Verify constructor sets default TTL."""
        repo = RedisStateRepository(adapter=mock_redis_adapter, default_ttl=7200)

        assert repo.default_ttl == 7200
        assert repo.adapter is mock_redis_adapter

    def test_init_uses_default_ttl_parameter(self, mock_redis_adapter):
        """Verify default TTL parameter defaults to 3600."""
        repo = RedisStateRepository(adapter=mock_redis_adapter)

        assert repo.default_ttl == 3600


class TestRedisRepositoryMetrics:
    """Test metrics collection integration."""

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        collector.record_counter = Mock()
        collector.record_histogram = Mock()
        return collector

    @pytest.fixture
    def redis_repo_with_metrics(self, mock_redis_adapter, mock_metrics_collector):
        """Create Redis repository with metrics collector."""
        return RedisStateRepository(
            adapter=mock_redis_adapter,
            default_ttl=3600,
            metrics_collector=mock_metrics_collector,
        )

    @pytest.mark.asyncio
    async def test_fetch_records_counter(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify fetch records counter metric."""
        mock_redis_adapter.get.return_value = '{"data": "value"}'

        await redis_repo_with_metrics.fetch("test_key")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.redis.operations.fetch_total"
        )

    @pytest.mark.asyncio
    async def test_fetch_records_error_on_exception(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify fetch records error counter on exception."""
        mock_redis_adapter.get.side_effect = Exception("Redis error")

        await redis_repo_with_metrics.fetch("test_key")

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.redis.operations.fetch_total" in counter_calls
        assert "state.repository.redis.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_store_records_counter(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify store records counter metric."""
        mock_redis_adapter.setex.return_value = True

        await redis_repo_with_metrics.store("test_key", "test_value", {})

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.redis.operations.store_total"
        )

    @pytest.mark.asyncio
    async def test_store_records_error_on_exception(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify store records error counter on exception."""
        mock_redis_adapter.setex.side_effect = Exception("Redis error")

        await redis_repo_with_metrics.store("test_key", "test_value", {})

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.redis.operations.store_total" in counter_calls
        assert "state.repository.redis.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_delete_records_counter(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify delete records counter metric."""
        await redis_repo_with_metrics.delete("test_key")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.redis.operations.delete_total"
        )

    @pytest.mark.asyncio
    async def test_delete_records_error_on_exception(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify delete records error counter on exception."""
        mock_redis_adapter.delete.side_effect = Exception("Redis error")

        await redis_repo_with_metrics.delete("test_key")

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.redis.operations.delete_total" in counter_calls
        assert "state.repository.redis.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_keys_records_error_on_exception(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify keys records error counter on exception."""
        mock_redis_adapter.keys.side_effect = Exception("Redis error")

        await redis_repo_with_metrics.keys("test_*")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.redis.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_stats_records_error_on_exception(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify stats records error counter on exception."""
        mock_redis_adapter.dbsize.side_effect = Exception("Redis error")

        await redis_repo_with_metrics.stats()

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.redis.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_store_many_records_counter_and_batch_size(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify store_many records counter and batch size histogram."""
        items = {
            "key1": ("value1", {"ttl_seconds": 3600}),
            "key2": ("value2", {"ttl_seconds": 3600}),
            "key3": ("value3", {"ttl_seconds": 7200}),
        }

        await redis_repo_with_metrics.store_many(items)

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.redis.operations.store_many_total" in counter_calls

        histogram_calls = mock_metrics_collector.record_histogram.call_args_list
        batch_size_call = next(
            call
            for call in histogram_calls
            if call[0][0] == "state.repository.redis.operations.batch_size"
        )
        assert batch_size_call[0][1] == 3.0

    @pytest.mark.asyncio
    async def test_store_many_records_error_on_exception(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify store_many records error counter on exception."""
        mock_redis_adapter.msetex.side_effect = Exception("Redis error")
        items = {"key1": ("value1", {})}

        await redis_repo_with_metrics.store_many(items)

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.redis.operations.store_many_total" in counter_calls
        assert "state.repository.redis.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_delete_many_records_counter_and_batch_size(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify delete_many records counter and batch size histogram."""
        mock_redis_adapter.delete_many.return_value = 5
        keys = ["key1", "key2", "key3", "key4", "key5"]

        await redis_repo_with_metrics.delete_many(keys)

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.redis.operations.delete_many_total" in counter_calls

        histogram_calls = mock_metrics_collector.record_histogram.call_args_list
        batch_size_call = next(
            call
            for call in histogram_calls
            if call[0][0] == "state.repository.redis.operations.batch_size"
        )
        assert batch_size_call[0][1] == 5.0

    @pytest.mark.asyncio
    async def test_delete_many_records_error_on_exception(
        self, redis_repo_with_metrics, mock_redis_adapter, mock_metrics_collector
    ):
        """Verify delete_many records error counter on exception."""
        mock_redis_adapter.delete_many.side_effect = Exception("Redis error")

        await redis_repo_with_metrics.delete_many(["key1", "key2"])

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.redis.operations.delete_many_total" in counter_calls
        assert "state.repository.redis.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_metrics_not_recorded_when_collector_none(self, mock_redis_adapter):
        """Verify no metrics recorded when collector is None."""
        repo = RedisStateRepository(adapter=mock_redis_adapter)
        assert repo.metrics_collector is None

        # Execute operations (should not crash)
        mock_redis_adapter.get.return_value = '{"data": "value"}'
        await repo.fetch("test_key")
        await repo.store("test_key", "value", {})
        await repo.delete("test_key")
        await repo.store_many({"key1": ("value1", {})})
        await repo.delete_many(["key1"])

        # No assertions on mock calls - just verify no crashes
