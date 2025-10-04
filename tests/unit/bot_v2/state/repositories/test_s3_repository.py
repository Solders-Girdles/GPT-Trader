"""Unit tests for S3StateRepository - COLD tier storage."""

import json
import pytest
from io import BytesIO
from unittest.mock import Mock, patch

from bot_v2.state.repositories.s3_repository import S3StateRepository
from bot_v2.state.utils.adapters import S3Adapter


@pytest.fixture
def mock_s3_adapter():
    """Create a mock S3Adapter for testing."""
    return Mock(spec=S3Adapter)


@pytest.fixture
def s3_repo(mock_s3_adapter):
    """Create an S3StateRepository with mock adapter."""
    return S3StateRepository(adapter=mock_s3_adapter, bucket="test-bucket", prefix="cold/")


class TestS3RepositoryStore:
    """Test store operations."""

    @pytest.mark.asyncio
    async def test_store_uploads_with_prefix(self, s3_repo, mock_s3_adapter):
        """Verify store uses bucket and prefix for S3 key."""
        await s3_repo.store("test_key", '{"data": "value"}', {"checksum": "abc123"})

        mock_s3_adapter.put_object.assert_called_once()
        call_kwargs = mock_s3_adapter.put_object.call_args[1]
        assert call_kwargs["bucket"] == "test-bucket"
        assert call_kwargs["key"] == "cold/test_key"

    @pytest.mark.asyncio
    async def test_store_uses_standard_ia_storage_class(self, s3_repo, mock_s3_adapter):
        """Verify store uses STANDARD_IA storage class."""
        await s3_repo.store("test_key", "value", {})

        call_kwargs = mock_s3_adapter.put_object.call_args[1]
        assert call_kwargs["storage_class"] == "STANDARD_IA"

    @pytest.mark.asyncio
    async def test_store_includes_checksum_metadata(self, s3_repo, mock_s3_adapter):
        """Verify store includes checksum in S3 metadata."""
        await s3_repo.store("test_key", "value", {"checksum": "xyz789"})

        call_kwargs = mock_s3_adapter.put_object.call_args[1]
        assert call_kwargs["metadata"] == {"checksum": "xyz789"}

    @pytest.mark.asyncio
    async def test_store_encodes_value_as_bytes(self, s3_repo, mock_s3_adapter):
        """Verify store encodes value to bytes."""
        test_value = '{"count": 42}'
        await s3_repo.store("test_key", test_value, {})

        call_kwargs = mock_s3_adapter.put_object.call_args[1]
        assert call_kwargs["body"] == test_value.encode()

    @pytest.mark.asyncio
    async def test_store_returns_true_on_success(self, s3_repo, mock_s3_adapter):
        """Verify store returns True when upload succeeds."""
        result = await s3_repo.store("test_key", "value", {})

        assert result is True

    @pytest.mark.asyncio
    async def test_store_returns_false_on_exception(self, s3_repo, mock_s3_adapter):
        """Verify store returns False on S3 error."""
        mock_s3_adapter.put_object.side_effect = Exception("S3 error")

        result = await s3_repo.store("test_key", "value", {})

        assert result is False

    @pytest.mark.asyncio
    async def test_store_logs_error_on_failure(self, s3_repo, mock_s3_adapter):
        """Verify store logs error on upload failure."""
        mock_s3_adapter.put_object.side_effect = Exception("Upload failed")

        with patch("bot_v2.state.repositories.s3_repository.logger") as mock_logger:
            await s3_repo.store("test_key", "value", {})
            mock_logger.error.assert_called_once()


class TestS3RepositoryFetch:
    """Test fetch operations."""

    @pytest.mark.asyncio
    async def test_fetch_uses_prefixed_key(self, s3_repo, mock_s3_adapter):
        """Verify fetch uses bucket and prefixed key."""
        mock_body = Mock()
        mock_body.read.return_value = b'{"data": "value"}'
        mock_s3_adapter.get_object.return_value = {"Body": mock_body}

        await s3_repo.fetch("test_key")

        mock_s3_adapter.get_object.assert_called_once_with(
            bucket="test-bucket", key="cold/test_key"
        )

    @pytest.mark.asyncio
    async def test_fetch_deserializes_json(self, s3_repo, mock_s3_adapter):
        """Verify fetch deserializes JSON from S3 body."""
        mock_body = Mock()
        mock_body.read.return_value = b'{"count": 42, "status": "active"}'
        mock_s3_adapter.get_object.return_value = {"Body": mock_body}

        result = await s3_repo.fetch("test_key")

        assert result == {"count": 42, "status": "active"}

    @pytest.mark.asyncio
    async def test_fetch_returns_none_on_missing_object(self, s3_repo, mock_s3_adapter):
        """Verify fetch returns None when object doesn't exist."""
        mock_s3_adapter.get_object.side_effect = Exception("NoSuchKey")

        result = await s3_repo.fetch("missing_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_logs_debug_on_failure(self, s3_repo, mock_s3_adapter):
        """Verify fetch logs debug message on error."""
        mock_s3_adapter.get_object.side_effect = Exception("Access denied")

        with patch("bot_v2.state.repositories.s3_repository.logger") as mock_logger:
            await s3_repo.fetch("test_key")
            mock_logger.debug.assert_called_once()


class TestS3RepositoryDelete:
    """Test delete operations."""

    @pytest.mark.asyncio
    async def test_delete_uses_prefixed_key(self, s3_repo, mock_s3_adapter):
        """Verify delete uses bucket and prefixed key."""
        await s3_repo.delete("test_key")

        mock_s3_adapter.delete_object.assert_called_once_with(
            bucket="test-bucket", key="cold/test_key"
        )

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_success(self, s3_repo, mock_s3_adapter):
        """Verify delete returns True when successful."""
        result = await s3_repo.delete("test_key")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_exception(self, s3_repo, mock_s3_adapter):
        """Verify delete returns False on S3 error."""
        mock_s3_adapter.delete_object.side_effect = Exception("S3 error")

        result = await s3_repo.delete("test_key")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_logs_warning_on_failure(self, s3_repo, mock_s3_adapter):
        """Verify delete logs warning on error."""
        mock_s3_adapter.delete_object.side_effect = Exception("Delete failed")

        with patch("bot_v2.state.repositories.s3_repository.logger") as mock_logger:
            await s3_repo.delete("test_key")
            mock_logger.warning.assert_called_once()


class TestS3RepositoryKeys:
    """Test key listing and pattern matching."""

    @pytest.mark.asyncio
    async def test_keys_converts_wildcard_to_prefix(self, s3_repo, mock_s3_adapter):
        """Verify keys extracts prefix from wildcard pattern."""
        mock_s3_adapter.list_objects_v2.return_value = {}

        await s3_repo.keys("position:*")

        mock_s3_adapter.list_objects_v2.assert_called_once_with(
            bucket="test-bucket", prefix="cold/position:"
        )

    @pytest.mark.asyncio
    async def test_keys_returns_stripped_keys(self, s3_repo, mock_s3_adapter):
        """Verify keys strips prefix from returned S3 keys."""
        mock_s3_adapter.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "cold/position:BTC"},
                {"Key": "cold/position:ETH"},
                {"Key": "cold/position:SOL"},
            ]
        }

        result = await s3_repo.keys("position:*")

        assert result == ["position:BTC", "position:ETH", "position:SOL"]

    @pytest.mark.asyncio
    async def test_keys_returns_empty_list_when_no_contents(self, s3_repo, mock_s3_adapter):
        """Verify keys returns empty list when no objects match."""
        mock_s3_adapter.list_objects_v2.return_value = {}

        result = await s3_repo.keys("nonexistent:*")

        assert result == []

    @pytest.mark.asyncio
    async def test_keys_returns_empty_list_on_exception(self, s3_repo, mock_s3_adapter):
        """Verify keys returns empty list on S3 error."""
        mock_s3_adapter.list_objects_v2.side_effect = Exception("S3 error")

        result = await s3_repo.keys("test:*")

        assert result == []

    @pytest.mark.asyncio
    async def test_keys_handles_exact_pattern_without_wildcard(self, s3_repo, mock_s3_adapter):
        """Verify keys handles pattern without wildcard."""
        mock_s3_adapter.list_objects_v2.return_value = {"Contents": [{"Key": "cold/exact_key"}]}

        result = await s3_repo.keys("exact_key")

        # Should use the pattern as-is as prefix
        mock_s3_adapter.list_objects_v2.assert_called_once_with(
            bucket="test-bucket", prefix="cold/exact_key"
        )
        assert result == ["exact_key"]


class TestS3RepositoryStats:
    """Test statistics collection."""

    @pytest.mark.asyncio
    async def test_stats_returns_key_count(self, s3_repo, mock_s3_adapter):
        """Verify stats returns KeyCount from S3 response."""
        mock_s3_adapter.list_objects_v2.return_value = {"KeyCount": 42}

        result = await s3_repo.stats()

        assert result == {"key_count": 42}
        mock_s3_adapter.list_objects_v2.assert_called_once_with(
            bucket="test-bucket", prefix="cold/"
        )

    @pytest.mark.asyncio
    async def test_stats_returns_zero_when_key_count_missing(self, s3_repo, mock_s3_adapter):
        """Verify stats returns 0 when KeyCount not in response."""
        mock_s3_adapter.list_objects_v2.return_value = {}

        result = await s3_repo.stats()

        assert result == {"key_count": 0}

    @pytest.mark.asyncio
    async def test_stats_returns_zero_on_exception(self, s3_repo, mock_s3_adapter):
        """Verify stats returns zero count on S3 error."""
        mock_s3_adapter.list_objects_v2.side_effect = Exception("S3 error")

        result = await s3_repo.stats()

        assert result == {"key_count": 0}


class TestS3RepositoryBatchOperations:
    """Test batch store/delete operations."""

    @pytest.mark.asyncio
    async def test_store_many_iterates_sequentially(self, s3_repo, mock_s3_adapter):
        """Verify store_many uploads each item individually."""
        items = {
            "key1": ('{"value": 1}', {"checksum": "abc"}),
            "key2": ('{"value": 2}', {"checksum": "def"}),
            "key3": ('{"value": 3}', {"checksum": "ghi"}),
        }

        result = await s3_repo.store_many(items)

        assert result == {"key1", "key2", "key3"}
        assert mock_s3_adapter.put_object.call_count == 3

    @pytest.mark.asyncio
    async def test_store_many_continues_on_partial_failure(self, s3_repo, mock_s3_adapter):
        """Verify store_many continues uploading after individual failures."""
        # Fail on second item only
        mock_s3_adapter.put_object.side_effect = [None, Exception("Upload failed"), None]
        items = {
            "key1": ("value1", {}),
            "key2": ("value2", {}),
            "key3": ("value3", {}),
        }

        result = await s3_repo.store_many(items)

        # Should succeed for key1 and key3, fail for key2
        assert "key1" in result
        assert "key2" not in result
        assert "key3" in result
        assert mock_s3_adapter.put_object.call_count == 3

    @pytest.mark.asyncio
    async def test_store_many_returns_empty_set_on_empty_input(self, s3_repo, mock_s3_adapter):
        """Verify store_many returns empty set for empty input."""
        result = await s3_repo.store_many({})

        assert result == set()
        mock_s3_adapter.put_object.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_many_uses_batch_delete(self, s3_repo, mock_s3_adapter):
        """Verify delete_many uses S3 batch delete API."""
        mock_s3_adapter.delete_objects.return_value = {
            "Deleted": [{"Key": "cold/key1"}, {"Key": "cold/key2"}, {"Key": "cold/key3"}]
        }
        keys = ["key1", "key2", "key3"]

        result = await s3_repo.delete_many(keys)

        assert result == 3
        mock_s3_adapter.delete_objects.assert_called_once_with(
            bucket="test-bucket", keys=["cold/key1", "cold/key2", "cold/key3"]
        )

    @pytest.mark.asyncio
    async def test_delete_many_handles_partial_errors(self, s3_repo, mock_s3_adapter):
        """Verify delete_many handles partial deletion errors."""
        mock_s3_adapter.delete_objects.return_value = {
            "Deleted": [{"Key": "cold/key1"}, {"Key": "cold/key2"}],
            "Errors": [{"Key": "cold/key3", "Code": "AccessDenied"}],
        }

        with patch("bot_v2.state.repositories.s3_repository.logger") as mock_logger:
            result = await s3_repo.delete_many(["key1", "key2", "key3"])

            assert result == 2
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_many_returns_zero_on_empty_input(self, s3_repo, mock_s3_adapter):
        """Verify delete_many returns 0 for empty input."""
        result = await s3_repo.delete_many([])

        assert result == 0
        mock_s3_adapter.delete_objects.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_many_returns_zero_on_exception(self, s3_repo, mock_s3_adapter):
        """Verify delete_many returns 0 on S3 error."""
        mock_s3_adapter.delete_objects.side_effect = Exception("Batch delete failed")

        result = await s3_repo.delete_many(["key1", "key2"])

        assert result == 0


class TestS3RepositoryInitialization:
    """Test repository initialization."""

    def test_init_sets_bucket_and_prefix(self, mock_s3_adapter):
        """Verify constructor sets bucket and prefix."""
        repo = S3StateRepository(adapter=mock_s3_adapter, bucket="my-bucket", prefix="archive/")

        assert repo.adapter is mock_s3_adapter
        assert repo.bucket == "my-bucket"
        assert repo.prefix == "archive/"

    def test_init_uses_default_prefix(self, mock_s3_adapter):
        """Verify constructor uses default prefix 'cold/'."""
        repo = S3StateRepository(adapter=mock_s3_adapter, bucket="my-bucket")

        assert repo.prefix == "cold/"


class TestS3RepositoryHelperMethods:
    """Test helper methods for key manipulation."""

    def test_build_key_adds_prefix(self, s3_repo):
        """Verify _build_key prepends prefix to key."""
        result = s3_repo._build_key("test_key")

        assert result == "cold/test_key"

    def test_strip_prefix_removes_prefix(self, s3_repo):
        """Verify _strip_prefix removes prefix from key."""
        result = s3_repo._strip_prefix("cold/test_key")

        assert result == "test_key"


class TestS3RepositoryMetrics:
    """Test metrics collection integration."""

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        collector.record_counter = Mock()
        collector.record_histogram = Mock()
        return collector

    @pytest.fixture
    def s3_repo_with_metrics(self, mock_s3_adapter, mock_metrics_collector):
        """Create S3 repository with metrics collector."""
        return S3StateRepository(
            adapter=mock_s3_adapter,
            bucket="test-bucket",
            prefix="cold/",
            metrics_collector=mock_metrics_collector,
        )

    @pytest.mark.asyncio
    async def test_fetch_records_counter(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify fetch records counter metric."""
        mock_body = Mock()
        mock_body.read.return_value = b'{"data": "value"}'
        mock_s3_adapter.get_object.return_value = {"Body": mock_body}

        await s3_repo_with_metrics.fetch("test_key")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.s3.operations.fetch_total"
        )

    @pytest.mark.asyncio
    async def test_fetch_records_error_on_exception(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify fetch records error metric on exception."""
        mock_s3_adapter.get_object.side_effect = Exception("S3 error")

        await s3_repo_with_metrics.fetch("test_key")

        assert mock_metrics_collector.record_counter.call_count == 2
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.fetch_total"
        )
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_store_records_counter(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify store records counter metric."""
        await s3_repo_with_metrics.store("test_key", '{"data": "value"}', {"checksum": "abc"})

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.s3.operations.store_total"
        )

    @pytest.mark.asyncio
    async def test_store_records_error_on_exception(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify store records error metric on exception."""
        mock_s3_adapter.put_object.side_effect = Exception("S3 error")

        await s3_repo_with_metrics.store("test_key", "value", {})

        assert mock_metrics_collector.record_counter.call_count == 2
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.store_total"
        )
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_delete_records_counter(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify delete records counter metric."""
        await s3_repo_with_metrics.delete("test_key")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.s3.operations.delete_total"
        )

    @pytest.mark.asyncio
    async def test_delete_records_error_on_exception(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify delete records error metric on exception."""
        mock_s3_adapter.delete_object.side_effect = Exception("S3 error")

        await s3_repo_with_metrics.delete("test_key")

        assert mock_metrics_collector.record_counter.call_count == 2
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.delete_total"
        )
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_keys_records_error_on_exception(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify keys records error metric on exception."""
        mock_s3_adapter.list_objects_v2.side_effect = Exception("S3 error")

        await s3_repo_with_metrics.keys("pattern*")

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.s3.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_stats_records_error_on_exception(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify stats records error metric on exception."""
        mock_s3_adapter.list_objects_v2.side_effect = Exception("S3 error")

        await s3_repo_with_metrics.stats()

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.s3.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_store_many_records_counter_and_batch_size(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify store_many records counter and batch size histogram."""
        items = {
            "key1": ('{"value": 1}', {"checksum": "abc"}),
            "key2": ('{"value": 2}', {"checksum": "def"}),
            "key3": ('{"value": 3}', {"checksum": "ghi"}),
        }

        await s3_repo_with_metrics.store_many(items)

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.s3.operations.store_many_total"
        )
        mock_metrics_collector.record_histogram.assert_called_with(
            "state.repository.s3.operations.batch_size", 3.0
        )

    @pytest.mark.asyncio
    async def test_store_many_records_error_on_exception(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify store_many records error metric on per-item exception."""
        mock_s3_adapter.put_object.side_effect = Exception("S3 error")
        items = {"key1": ("value1", {})}

        await s3_repo_with_metrics.store_many(items)

        # Should record store_many_total and errors_total
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.store_many_total"
        )
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_delete_many_records_counter_and_batch_size(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify delete_many records counter and batch size histogram."""
        mock_s3_adapter.delete_objects.return_value = {
            "Deleted": [{"Key": "cold/key1"}, {"Key": "cold/key2"}]
        }
        keys = ["key1", "key2"]

        await s3_repo_with_metrics.delete_many(keys)

        mock_metrics_collector.record_counter.assert_called_with(
            "state.repository.s3.operations.delete_many_total"
        )
        mock_metrics_collector.record_histogram.assert_called_with(
            "state.repository.s3.operations.batch_size", 2.0
        )

    @pytest.mark.asyncio
    async def test_delete_many_records_error_on_exception(
        self, s3_repo_with_metrics, mock_s3_adapter, mock_metrics_collector
    ):
        """Verify delete_many records error metric on exception."""
        mock_s3_adapter.delete_objects.side_effect = Exception("Batch delete failed")

        await s3_repo_with_metrics.delete_many(["key1", "key2"])

        # Should record delete_many_total and errors_total
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.delete_many_total"
        )
        mock_metrics_collector.record_counter.assert_any_call(
            "state.repository.s3.operations.errors_total"
        )

    @pytest.mark.asyncio
    async def test_metrics_not_recorded_when_collector_none(self, mock_s3_adapter):
        """Verify no metrics recorded when collector is None."""
        repo = S3StateRepository(adapter=mock_s3_adapter, bucket="test-bucket")

        mock_body = Mock()
        mock_body.read.return_value = b'{"data": "value"}'
        mock_s3_adapter.get_object.return_value = {"Body": mock_body}

        await repo.fetch("test_key")

        # No errors should occur when collector is None
