"""Unit tests for PostgresStateRepository - WARM tier storage."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, call

from bot_v2.state.repositories.postgres_repository import PostgresStateRepository
from bot_v2.state.utils.adapters import PostgresAdapter


@pytest.fixture
def mock_postgres_adapter():
    """Create a mock PostgresAdapter for testing."""
    return Mock(spec=PostgresAdapter)


@pytest.fixture
def postgres_repo(mock_postgres_adapter):
    """Create a PostgresStateRepository with mock adapter."""
    return PostgresStateRepository(adapter=mock_postgres_adapter)


class TestPostgresRepositoryStore:
    """Test store operations."""

    @pytest.mark.asyncio
    async def test_store_executes_upsert_query(self, postgres_repo, mock_postgres_adapter):
        """Verify store executes INSERT ... ON CONFLICT UPDATE."""
        mock_postgres_adapter.execute.return_value = None

        result = await postgres_repo.store("test_key", '{"data": "value"}', {"checksum": "abc123"})

        assert result is True
        # Verify the UPSERT query was executed
        assert mock_postgres_adapter.execute.called
        call_args = mock_postgres_adapter.execute.call_args[0]
        assert "INSERT INTO state_warm" in call_args[0]
        assert "ON CONFLICT (key) DO UPDATE" in call_args[0]
        assert call_args[1] == ("test_key", '{"data": "value"}', "abc123", 17)
        mock_postgres_adapter.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_uses_metadata_checksum_and_size(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify store uses checksum and size_bytes from metadata."""
        metadata = {"checksum": "xyz789", "size_bytes": 42}

        await postgres_repo.store("key1", "test_value", metadata)

        call_args = mock_postgres_adapter.execute.call_args[0]
        assert call_args[1] == ("key1", "test_value", "xyz789", 42)

    @pytest.mark.asyncio
    async def test_store_calculates_size_when_not_provided(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify store calculates size_bytes when not in metadata."""
        await postgres_repo.store("key1", "test", {})

        call_args = mock_postgres_adapter.execute.call_args[0]
        # "test" encoded is 4 bytes
        assert call_args[1][3] == 4

    @pytest.mark.asyncio
    async def test_store_commits_transaction_on_success(self, postgres_repo, mock_postgres_adapter):
        """Verify store commits transaction on successful insert."""
        await postgres_repo.store("key1", "value", {})

        mock_postgres_adapter.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_rolls_back_on_exception(self, postgres_repo, mock_postgres_adapter):
        """Verify store rolls back transaction on error."""
        mock_postgres_adapter.execute.side_effect = Exception("DB error")

        result = await postgres_repo.store("key1", "value", {})

        assert result is False
        mock_postgres_adapter.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_logs_error_on_failure(self, postgres_repo, mock_postgres_adapter):
        """Verify store logs error when execution fails."""
        mock_postgres_adapter.execute.side_effect = Exception("Connection lost")

        with patch("bot_v2.state.repositories.postgres_repository.logger") as mock_logger:
            await postgres_repo.store("key1", "value", {})
            mock_logger.error.assert_called_once()


class TestPostgresRepositoryFetch:
    """Test fetch operations."""

    @pytest.mark.asyncio
    async def test_fetch_returns_data_from_query(self, postgres_repo, mock_postgres_adapter):
        """Verify fetch returns data field from query result."""
        mock_postgres_adapter.execute.side_effect = [
            [{"data": {"count": 42, "status": "active"}}],  # SELECT query
            None,  # UPDATE last_accessed
        ]

        result = await postgres_repo.fetch("test_key")

        assert result == {"count": 42, "status": "active"}

    @pytest.mark.asyncio
    async def test_fetch_updates_last_accessed_timestamp(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify fetch updates last_accessed column."""
        mock_postgres_adapter.execute.side_effect = [
            [{"data": "value"}],  # SELECT
            None,  # UPDATE
        ]

        with patch("bot_v2.state.repositories.postgres_repository.datetime") as mock_datetime:
            mock_now = datetime(2025, 10, 3, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            await postgres_repo.fetch("test_key")

            # Verify UPDATE was called with timestamp
            update_call = mock_postgres_adapter.execute.call_args_list[1]
            assert "UPDATE state_warm SET last_accessed" in update_call[0][0]
            assert update_call[0][1] == (mock_now, "test_key")

    @pytest.mark.asyncio
    async def test_fetch_commits_after_updating_timestamp(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify fetch commits after updating last_accessed."""
        mock_postgres_adapter.execute.side_effect = [
            [{"data": "value"}],
            None,
        ]

        await postgres_repo.fetch("test_key")

        mock_postgres_adapter.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_returns_none_when_key_not_found(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify fetch returns None when key doesn't exist."""
        mock_postgres_adapter.execute.return_value = []

        result = await postgres_repo.fetch("missing_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rolls_back_on_exception(self, postgres_repo, mock_postgres_adapter):
        """Verify fetch rolls back transaction on error."""
        mock_postgres_adapter.execute.side_effect = Exception("DB error")

        result = await postgres_repo.fetch("key1")

        assert result is None
        mock_postgres_adapter.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_logs_debug_on_failure(self, postgres_repo, mock_postgres_adapter):
        """Verify fetch logs debug message on error."""
        mock_postgres_adapter.execute.side_effect = Exception("Connection timeout")

        with patch("bot_v2.state.repositories.postgres_repository.logger") as mock_logger:
            await postgres_repo.fetch("key1")
            mock_logger.debug.assert_called_once()


class TestPostgresRepositoryDelete:
    """Test delete operations."""

    @pytest.mark.asyncio
    async def test_delete_executes_delete_query(self, postgres_repo, mock_postgres_adapter):
        """Verify delete executes DELETE statement."""
        await postgres_repo.delete("test_key")

        mock_postgres_adapter.execute.assert_called_once_with(
            "DELETE FROM state_warm WHERE key = %s", ("test_key",)
        )

    @pytest.mark.asyncio
    async def test_delete_commits_transaction(self, postgres_repo, mock_postgres_adapter):
        """Verify delete commits transaction after execution."""
        await postgres_repo.delete("test_key")

        mock_postgres_adapter.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, postgres_repo, mock_postgres_adapter):
        """Verify delete returns True when successful."""
        result = await postgres_repo.delete("test_key")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_rolls_back_on_exception(self, postgres_repo, mock_postgres_adapter):
        """Verify delete rolls back transaction on error."""
        mock_postgres_adapter.execute.side_effect = Exception("DB error")

        result = await postgres_repo.delete("key1")

        assert result is False
        mock_postgres_adapter.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_handles_rollback_failure(self, postgres_repo, mock_postgres_adapter):
        """Verify delete handles rollback failure gracefully."""
        mock_postgres_adapter.execute.side_effect = Exception("Delete failed")
        mock_postgres_adapter.rollback.side_effect = Exception("Rollback failed")

        with patch("bot_v2.state.repositories.postgres_repository.logger") as mock_logger:
            result = await postgres_repo.delete("key1")

            assert result is False
            # Should log both warning and debug
            assert mock_logger.warning.called
            assert mock_logger.debug.called

    @pytest.mark.asyncio
    async def test_delete_logs_warning_on_failure(self, postgres_repo, mock_postgres_adapter):
        """Verify delete logs warning on error."""
        mock_postgres_adapter.execute.side_effect = Exception("Connection lost")

        with patch("bot_v2.state.repositories.postgres_repository.logger") as mock_logger:
            await postgres_repo.delete("key1")
            mock_logger.warning.assert_called_once()


class TestPostgresRepositoryKeys:
    """Test key pattern matching."""

    @pytest.mark.asyncio
    async def test_keys_converts_wildcard_to_sql_pattern(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify keys converts * wildcard to SQL % pattern."""
        mock_postgres_adapter.execute.return_value = []

        await postgres_repo.keys("position:*")

        call_args = mock_postgres_adapter.execute.call_args[0]
        assert "WHERE key LIKE %s" in call_args[0]
        assert call_args[1] == ("position:%",)

    @pytest.mark.asyncio
    async def test_keys_returns_matching_keys(self, postgres_repo, mock_postgres_adapter):
        """Verify keys returns list of matching keys."""
        mock_postgres_adapter.execute.return_value = [
            {"key": "position:BTC"},
            {"key": "position:ETH"},
            {"key": "position:SOL"},
        ]

        result = await postgres_repo.keys("position:*")

        assert result == ["position:BTC", "position:ETH", "position:SOL"]

    @pytest.mark.asyncio
    async def test_keys_returns_empty_list_on_no_match(self, postgres_repo, mock_postgres_adapter):
        """Verify keys returns empty list when no keys match."""
        mock_postgres_adapter.execute.return_value = []

        result = await postgres_repo.keys("nonexistent:*")

        assert result == []

    @pytest.mark.asyncio
    async def test_keys_returns_empty_list_on_exception(self, postgres_repo, mock_postgres_adapter):
        """Verify keys returns empty list on database error."""
        mock_postgres_adapter.execute.side_effect = Exception("DB error")

        result = await postgres_repo.keys("test:*")

        assert result == []


class TestPostgresRepositoryStats:
    """Test statistics collection."""

    @pytest.mark.asyncio
    async def test_stats_returns_key_count(self, postgres_repo, mock_postgres_adapter):
        """Verify stats returns key count from COUNT query."""
        mock_postgres_adapter.execute.return_value = [{"count": 42}]

        result = await postgres_repo.stats()

        assert result == {"key_count": 42}
        mock_postgres_adapter.execute.assert_called_once_with(
            "SELECT COUNT(*) as count FROM state_warm"
        )

    @pytest.mark.asyncio
    async def test_stats_returns_zero_when_empty(self, postgres_repo, mock_postgres_adapter):
        """Verify stats returns zero when table is empty."""
        mock_postgres_adapter.execute.return_value = [{"count": 0}]

        result = await postgres_repo.stats()

        assert result == {"key_count": 0}

    @pytest.mark.asyncio
    async def test_stats_returns_zero_on_exception(self, postgres_repo, mock_postgres_adapter):
        """Verify stats returns zero count on database error."""
        mock_postgres_adapter.execute.side_effect = Exception("DB error")

        result = await postgres_repo.stats()

        assert result == {"key_count": 0}


class TestPostgresRepositoryBatchOperations:
    """Test batch store/delete operations."""

    @pytest.mark.asyncio
    async def test_store_many_uses_batch_upsert(self, postgres_repo, mock_postgres_adapter):
        """Verify store_many uses adapter batch_upsert."""
        mock_postgres_adapter.batch_upsert.return_value = 3
        items = {
            "key1": ('{"value": 1}', {"checksum": "abc"}),
            "key2": ('{"value": 2}', {"checksum": "def"}),
            "key3": ('{"value": 3}', {"checksum": "ghi"}),
        }

        result = await postgres_repo.store_many(items)

        assert result == {"key1", "key2", "key3"}
        # Verify batch_upsert called with correct table and records
        mock_postgres_adapter.batch_upsert.assert_called_once()
        call_args = mock_postgres_adapter.batch_upsert.call_args[0]
        assert call_args[0] == "state_warm"
        assert call_args[1] == "key"
        assert len(call_args[2]) == 3
        mock_postgres_adapter.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_many_builds_correct_records(self, postgres_repo, mock_postgres_adapter):
        """Verify store_many builds record dicts with all fields."""
        mock_postgres_adapter.batch_upsert.return_value = 2
        items = {
            "key1": ('{"v": 1}', {"checksum": "abc", "size_bytes": 7}),
            "key2": ('{"v": 2}', {"checksum": "def"}),
        }

        await postgres_repo.store_many(items)

        records = mock_postgres_adapter.batch_upsert.call_args[0][2]
        assert records[0] == {"key": "key1", "data": '{"v": 1}', "checksum": "abc", "size_bytes": 7}
        assert records[1]["key"] == "key2"
        assert records[1]["checksum"] == "def"
        assert records[1]["size_bytes"] == 8  # calculated from encoded string

    @pytest.mark.asyncio
    async def test_store_many_returns_empty_set_on_empty_input(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify store_many returns empty set for empty input."""
        result = await postgres_repo.store_many({})

        assert result == set()
        mock_postgres_adapter.batch_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_many_returns_empty_set_when_count_zero(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify store_many returns empty set when batch_upsert returns 0."""
        mock_postgres_adapter.batch_upsert.return_value = 0
        items = {"key1": ("value", {})}

        result = await postgres_repo.store_many(items)

        assert result == set()

    @pytest.mark.asyncio
    async def test_store_many_rolls_back_on_exception(self, postgres_repo, mock_postgres_adapter):
        """Verify store_many rolls back on batch_upsert error."""
        mock_postgres_adapter.batch_upsert.side_effect = Exception("Batch failed")
        items = {"key1": ("value", {})}

        result = await postgres_repo.store_many(items)

        assert result == set()
        mock_postgres_adapter.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_many_uses_batch_delete(self, postgres_repo, mock_postgres_adapter):
        """Verify delete_many uses adapter batch_delete."""
        mock_postgres_adapter.batch_delete.return_value = 3
        keys = ["key1", "key2", "key3"]

        result = await postgres_repo.delete_many(keys)

        assert result == 3
        mock_postgres_adapter.batch_delete.assert_called_once_with("state_warm", "key", keys)
        mock_postgres_adapter.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_many_returns_zero_on_empty_input(
        self, postgres_repo, mock_postgres_adapter
    ):
        """Verify delete_many returns 0 for empty input."""
        result = await postgres_repo.delete_many([])

        assert result == 0
        mock_postgres_adapter.batch_delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_many_rolls_back_on_exception(self, postgres_repo, mock_postgres_adapter):
        """Verify delete_many rolls back on batch_delete error."""
        mock_postgres_adapter.batch_delete.side_effect = Exception("Batch delete failed")

        result = await postgres_repo.delete_many(["key1", "key2"])

        assert result == 0
        mock_postgres_adapter.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_many_handles_rollback_failure(self, postgres_repo, mock_postgres_adapter):
        """Verify delete_many handles rollback failure gracefully."""
        mock_postgres_adapter.batch_delete.side_effect = Exception("Delete failed")
        mock_postgres_adapter.rollback.side_effect = Exception("Rollback failed")

        with patch("bot_v2.state.repositories.postgres_repository.logger") as mock_logger:
            result = await postgres_repo.delete_many(["key1"])

            assert result == 0
            assert mock_logger.error.called
            assert mock_logger.debug.called


class TestPostgresRepositoryInitialization:
    """Test repository initialization."""

    def test_init_sets_adapter(self, mock_postgres_adapter):
        """Verify constructor sets adapter."""
        repo = PostgresStateRepository(adapter=mock_postgres_adapter)

        assert repo.adapter is mock_postgres_adapter


class TestPostgresRepositoryMetrics:
    """Test metrics collection integration."""

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        collector.record_counter = Mock()
        collector.record_histogram = Mock()
        return collector

    @pytest.fixture
    def postgres_repo_with_metrics(self, mock_postgres_adapter, mock_metrics_collector):
        """Create Postgres repository with metrics collector."""
        return PostgresStateRepository(
            adapter=mock_postgres_adapter,
            metrics_collector=mock_metrics_collector,
        )

    @pytest.mark.asyncio
    async def test_fetch_records_counter(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify fetch records counter metric."""
        mock_postgres_adapter.execute.return_value = [{"data": {"value": "test"}}]

        await postgres_repo_with_metrics.fetch("test_key")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.postgres.operations.fetch_total"
        )

    @pytest.mark.asyncio
    async def test_fetch_records_error_on_exception(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify fetch records error counter on exception."""
        mock_postgres_adapter.execute.side_effect = Exception("Postgres error")

        await postgres_repo_with_metrics.fetch("test_key")

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.postgres.operations.fetch_total" in counter_calls
        assert "state.repository.postgres.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_store_records_counter(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify store records counter metric."""
        await postgres_repo_with_metrics.store("test_key", "test_value", {})

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.postgres.operations.store_total"
        )

    @pytest.mark.asyncio
    async def test_store_records_error_on_exception(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify store records error counter on exception."""
        mock_postgres_adapter.execute.side_effect = Exception("Postgres error")

        await postgres_repo_with_metrics.store("test_key", "test_value", {})

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.postgres.operations.store_total" in counter_calls
        assert "state.repository.postgres.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_delete_records_counter(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify delete records counter metric."""
        await postgres_repo_with_metrics.delete("test_key")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.postgres.operations.delete_total"
        )

    @pytest.mark.asyncio
    async def test_delete_records_error_on_exception(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify delete records error counter on exception."""
        mock_postgres_adapter.execute.side_effect = Exception("Postgres error")

        await postgres_repo_with_metrics.delete("test_key")

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.postgres.operations.delete_total" in counter_calls
        assert "state.repository.postgres.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_keys_records_error_on_exception(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify keys records error counter on exception."""
        mock_postgres_adapter.execute.side_effect = Exception("Postgres error")

        await postgres_repo_with_metrics.keys("test_*")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.postgres.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_stats_records_error_on_exception(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify stats records error counter on exception."""
        mock_postgres_adapter.execute.side_effect = Exception("Postgres error")

        await postgres_repo_with_metrics.stats()

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.postgres.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_store_many_records_counter_and_batch_size(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify store_many records counter and batch size histogram."""
        mock_postgres_adapter.batch_upsert.return_value = 3
        items = {
            "key1": ("value1", {}),
            "key2": ("value2", {}),
            "key3": ("value3", {}),
        }

        await postgres_repo_with_metrics.store_many(items)

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.postgres.operations.store_many_total" in counter_calls

        histogram_calls = mock_metrics_collector.record_histogram.call_args_list
        batch_size_call = next(
            call
            for call in histogram_calls
            if call[0][0] == "state.repository.postgres.operations.batch_size"
        )
        assert batch_size_call[0][1] == 3.0

    @pytest.mark.asyncio
    async def test_store_many_records_error_on_exception(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify store_many records error counter on exception."""
        mock_postgres_adapter.batch_upsert.side_effect = Exception("Postgres error")
        items = {"key1": ("value1", {})}

        await postgres_repo_with_metrics.store_many(items)

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.postgres.operations.store_many_total" in counter_calls
        assert "state.repository.postgres.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_delete_many_records_counter_and_batch_size(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify delete_many records counter and batch size histogram."""
        mock_postgres_adapter.batch_delete.return_value = 5
        keys = ["key1", "key2", "key3", "key4", "key5"]

        await postgres_repo_with_metrics.delete_many(keys)

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.postgres.operations.delete_many_total" in counter_calls

        histogram_calls = mock_metrics_collector.record_histogram.call_args_list
        batch_size_call = next(
            call
            for call in histogram_calls
            if call[0][0] == "state.repository.postgres.operations.batch_size"
        )
        assert batch_size_call[0][1] == 5.0

    @pytest.mark.asyncio
    async def test_delete_many_records_error_on_exception(
        self, postgres_repo_with_metrics, mock_postgres_adapter, mock_metrics_collector
    ):
        """Verify delete_many records error counter on exception."""
        mock_postgres_adapter.batch_delete.side_effect = Exception("Postgres error")

        await postgres_repo_with_metrics.delete_many(["key1", "key2"])

        counter_calls = [
            call[0][0] for call in mock_metrics_collector.record_counter.call_args_list
        ]
        assert "state.repository.postgres.operations.delete_many_total" in counter_calls
        assert "state.repository.postgres.operations.errors_total" in counter_calls

    @pytest.mark.asyncio
    async def test_metrics_not_recorded_when_collector_none(self, mock_postgres_adapter):
        """Verify no metrics recorded when collector is None."""
        repo = PostgresStateRepository(adapter=mock_postgres_adapter)
        assert repo.metrics_collector is None

        # Execute operations (should not crash)
        mock_postgres_adapter.execute.return_value = [{"data": {"value": "test"}}]
        await repo.fetch("test_key")
        await repo.store("test_key", "value", {})
        await repo.delete("test_key")
        mock_postgres_adapter.batch_upsert.return_value = 1
        await repo.store_many({"key1": ("value1", {})})
        mock_postgres_adapter.batch_delete.return_value = 1
        await repo.delete_many(["key1"])

        # No assertions on mock calls - just verify no crashes
