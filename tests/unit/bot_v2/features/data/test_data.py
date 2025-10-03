"""Tests for data management service."""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from bot_v2.features.data.cache import DataCache
from bot_v2.features.data.data import DataService
from bot_v2.features.data.quality import DataQualityChecker
from bot_v2.features.data.storage import DataStorage
from bot_v2.features.data.types import DataQuery, DataSource, DataType, StorageStats


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + i for i in range(30)],
            "high": [101 + i for i in range(30)],
            "low": [99 + i for i in range(30)],
            "close": [100.5 + i for i in range(30)],
            "volume": [1000000 for _ in range(30)],
        },
        index=dates,
    )


@pytest.fixture
def mock_storage():
    """Create a mock DataStorage instance."""
    storage = Mock(spec=DataStorage)
    storage.store.return_value = True
    storage.fetch.return_value = None
    storage.delete_before.return_value = 0
    storage.get_stats.return_value = {
        "total_records": 100,
        "total_size_mb": 1.5,
        "oldest_record": datetime.now() - timedelta(days=365),
        "newest_record": datetime.now(),
        "symbols_count": 5,
    }
    return storage


@pytest.fixture
def mock_cache():
    """Create a mock DataCache instance."""
    cache = Mock(spec=DataCache)
    cache.get.return_value = None
    cache.put.return_value = True
    cache.clear_expired.return_value = None
    cache.get_stats.return_value = {
        "entries": 10,
        "size_mb": 0.5,
        "hit_rate": 0.75,
    }
    return cache


@pytest.fixture
def mock_quality_checker():
    """Create a mock DataQualityChecker instance."""
    checker = Mock(spec=DataQualityChecker)
    quality_mock = Mock()
    quality_mock.is_acceptable.return_value = True
    quality_mock.overall_score.return_value = 0.9
    checker.check_quality.return_value = quality_mock
    return checker


@pytest.fixture
def data_service(mock_storage, mock_cache, mock_quality_checker):
    """Create a DataService instance with mocked dependencies."""
    return DataService(
        storage=mock_storage,
        cache=mock_cache,
        quality_checker=mock_quality_checker,
    )


class TestDataServiceInit:
    """Tests for DataService initialization."""

    def test_init_with_dependencies(self, mock_storage, mock_cache, mock_quality_checker):
        """Test initialization with injected dependencies."""
        service = DataService(
            storage=mock_storage,
            cache=mock_cache,
            quality_checker=mock_quality_checker,
        )

        assert service.storage is mock_storage
        assert service.cache is mock_cache
        assert service.quality_checker is mock_quality_checker

    def test_init_with_defaults(self):
        """Test initialization with default dependencies."""
        service = DataService()

        assert isinstance(service.storage, DataStorage)
        assert isinstance(service.cache, DataCache)
        assert isinstance(service.quality_checker, DataQualityChecker)


class TestStoreData:
    """Tests for store_data method."""

    def test_store_data_success(
        self, data_service, sample_data, mock_storage, mock_quality_checker
    ):
        """Test successful data storage."""
        result = data_service.store_data(
            symbol="AAPL",
            data=sample_data,
            data_type=DataType.OHLCV,
            source=DataSource.YAHOO,
        )

        assert result is True
        mock_quality_checker.check_quality.assert_called_once_with(sample_data)
        mock_storage.store.assert_called_once()

    def test_store_data_with_poor_quality(self, data_service, sample_data, mock_quality_checker):
        """Test storing data with poor quality."""
        # Make quality check return poor quality
        quality_mock = Mock()
        quality_mock.is_acceptable.return_value = False
        quality_mock.overall_score.return_value = 0.5
        mock_quality_checker.check_quality.return_value = quality_mock

        result = data_service.store_data(symbol="AAPL", data=sample_data)

        # Should still store but log warning
        assert result is True

    def test_store_data_updates_cache(self, data_service, sample_data, mock_cache):
        """Test that storing data updates cache."""
        data_service.store_data(symbol="AAPL", data=sample_data)

        # Should update cache
        mock_cache.put.assert_called()

    def test_store_data_failure(self, data_service, sample_data, mock_storage):
        """Test data storage failure."""
        mock_storage.store.return_value = False

        result = data_service.store_data(symbol="AAPL", data=sample_data)

        assert result is False

    def test_store_data_with_exception(self, data_service, sample_data, mock_storage):
        """Test data storage with exception."""
        mock_storage.store.side_effect = Exception("Storage error")

        result = data_service.store_data(symbol="AAPL", data=sample_data)

        assert result is False


class TestFetchData:
    """Tests for fetch_data method."""

    def test_fetch_from_cache(self, data_service, sample_data, mock_cache):
        """Test fetching data from cache."""
        mock_cache.get.return_value = sample_data

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
        )

        result = data_service.fetch_data(query, use_cache=True)

        assert result is not None
        mock_cache.get.assert_called_once()
        pd.testing.assert_frame_equal(result, sample_data)

    def test_fetch_from_storage(self, data_service, sample_data, mock_cache, mock_storage):
        """Test fetching data from storage when not in cache."""
        mock_cache.get.return_value = None
        mock_storage.fetch.return_value = sample_data

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
        )

        result = data_service.fetch_data(query, use_cache=True)

        assert result is not None
        mock_storage.fetch.assert_called_once_with(query)
        # Should update cache after fetching from storage
        assert mock_cache.put.call_count >= 1

    def test_fetch_without_cache(self, data_service, sample_data, mock_cache, mock_storage):
        """Test fetching data without using cache."""
        mock_storage.fetch.return_value = sample_data

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
        )

        result = data_service.fetch_data(query, use_cache=False)

        assert result is not None
        mock_cache.get.assert_not_called()

    @patch("bot_v2.features.data.data.get_data_provider")
    def test_fetch_downloads_if_not_found(
        self, mock_get_provider, data_service, sample_data, mock_storage
    ):
        """Test downloading data if not in storage."""
        mock_storage.fetch.return_value = None

        # Mock the provider
        mock_provider = Mock()
        mock_provider.get_historical_data.return_value = sample_data
        mock_get_provider.return_value = mock_provider

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            source=DataSource.YAHOO,
        )

        result = data_service.fetch_data(query, use_cache=False)

        assert result is not None
        mock_provider.get_historical_data.assert_called()

    def test_fetch_downloads_with_date_range(
        self,
        data_service,
        mock_storage,
        mock_cache,
        mock_quality_checker,
        monkeypatch,
    ):
        """Ensure provider receives explicit start/end parameters."""

        class DummyProvider:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def get_historical_data(
                self,
                symbol: str,
                period: str = "60d",
                interval: str = "1d",
                *,
                start=None,
                end=None,
            ) -> pd.DataFrame:
                self.calls.append(
                    {
                        "symbol": symbol,
                        "period": period,
                        "interval": interval,
                        "start": start,
                        "end": end,
                    }
                )
                index = pd.date_range(start=start, end=end, freq="D")
                return pd.DataFrame(
                    {
                        "open": [100 + i for i in range(len(index))],
                        "high": [101 + i for i in range(len(index))],
                        "low": [99 + i for i in range(len(index))],
                        "close": [100.5 + i for i in range(len(index))],
                        "volume": [1_000_000 for _ in index],
                    },
                    index=index,
                )

        provider = DummyProvider()
        monkeypatch.setattr("bot_v2.features.data.data.get_data_provider", lambda: provider)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        query = DataQuery(
            symbols=["AAPL"],
            start_date=start_date,
            end_date=end_date,
            data_type=DataType.OHLCV,
            source=DataSource.YAHOO,
        )

        mock_storage.fetch.return_value = None

        result = data_service.fetch_data(query, use_cache=True)

        assert result is not None
        assert "AAPL" in result
        assert len(provider.calls) == 1
        call = provider.calls[0]
        assert call["symbol"] == "AAPL"
        assert call["interval"] == "1d"
        assert call["start"] == start_date
        assert call["end"] == end_date
        mock_storage.store.assert_called()


class TestCacheOperations:
    """Tests for cache operations."""

    def test_cache_data(self, data_service, sample_data, mock_cache):
        """Test caching data."""
        result = data_service.cache_data("test_key", sample_data, ttl_seconds=3600)

        assert result is True
        mock_cache.put.assert_called_with("test_key", sample_data, 3600)

    def test_get_cache(self, data_service, sample_data, mock_cache):
        """Test getting data from cache."""
        mock_cache.get.return_value = sample_data

        result = data_service.get_cache("test_key")

        assert result is not None
        mock_cache.get.assert_called_with("test_key")
        pd.testing.assert_frame_equal(result, sample_data)


class TestDataCacheWarmUp:
    """Tests for DataCache.warm_up."""

    def test_warm_up_without_loader(self):
        """Verify warm_up returns zero when no loader provided."""
        cache = DataCache()
        warmed = cache.warm_up(["alpha", "beta"], loader=None)

        assert warmed == 0
        assert cache.get("alpha") is None

    def test_warm_up_with_loader(self):
        """Warm-up should populate cache for keys with data."""
        cache = DataCache()

        def loader(key: str) -> pd.DataFrame | None:
            if key == "alpha":
                index = pd.date_range(start="2024-01-01", periods=3, freq="D")
                return pd.DataFrame({"close": [1, 2, 3]}, index=index)
            return None

        warmed = cache.warm_up(["alpha", "beta"], loader=loader, ttl_seconds=120)

        assert warmed == 1
        warmed_frame = cache.get("alpha")
        assert warmed_frame is not None
        assert cache.get("beta") is None


class TestDownloadHistorical:
    """Tests for download_historical method."""

    @patch("bot_v2.features.data.data.get_data_provider")
    def test_download_historical_yahoo(self, mock_get_provider, data_service, sample_data):
        """Test downloading historical data from Yahoo."""
        mock_provider = Mock()
        mock_provider.get_historical_data.return_value = sample_data
        mock_get_provider.return_value = mock_provider

        results = data_service.download_historical(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            source=DataSource.YAHOO,
        )

        assert "AAPL" in results
        mock_provider.get_historical_data.assert_called()

    def test_download_historical_unsupported_source(self, data_service):
        """Test downloading from unsupported source."""
        results = data_service.download_historical(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            source=DataSource.CSV,
        )

        # Should return empty dict for unsupported sources
        assert results == {}


class TestCleanOldData:
    """Tests for clean_old_data method."""

    def test_clean_old_data(self, data_service, mock_storage, mock_cache):
        """Test cleaning old data."""
        mock_storage.delete_before.return_value = 50

        deleted = data_service.clean_old_data(days_to_keep=365)

        assert deleted == 50
        mock_storage.delete_before.assert_called_once()
        mock_cache.clear_expired.assert_called_once()

    def test_clean_old_data_custom_days(self, data_service, mock_storage):
        """Test cleaning old data with custom days."""
        data_service.clean_old_data(days_to_keep=180)

        # Verify cutoff date calculation
        args = mock_storage.delete_before.call_args[0]
        cutoff_date = args[0]

        expected_cutoff = datetime.now() - timedelta(days=180)
        assert abs((cutoff_date - expected_cutoff).total_seconds()) < 60


class TestGetStorageStats:
    """Tests for get_storage_stats method."""

    def test_get_storage_stats(self, data_service, mock_storage, mock_cache):
        """Test getting storage statistics."""
        stats = data_service.get_storage_stats()

        assert isinstance(stats, StorageStats)
        assert stats.total_records == 100
        assert stats.total_size_mb == 1.5
        assert stats.symbols_count == 5
        assert stats.cache_entries == 10
        assert stats.cache_size_mb == 0.5
        assert stats.cache_hit_rate == 0.75


class TestExportData:
    """Tests for export_data method."""

    def test_export_csv(self, data_service, sample_data, mock_storage):
        """Test exporting data to CSV."""
        mock_storage.fetch.return_value = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            query = DataQuery(
                symbols=["AAPL"],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 31),
            )

            result = data_service.export_data(query, format="csv", path=tmpdir)

            assert result is True
            # Check that file was created
            files = os.listdir(tmpdir)
            assert len(files) > 0
            assert any(f.endswith(".csv") for f in files)

    def test_export_json(self, data_service, sample_data, mock_storage):
        """Test exporting data to JSON."""
        mock_storage.fetch.return_value = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            query = DataQuery(
                symbols=["AAPL"],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 31),
            )

            result = data_service.export_data(query, format="json", path=tmpdir)

            assert result is True
            files = os.listdir(tmpdir)
            assert any(f.endswith(".json") for f in files)

    @patch("bot_v2.features.data.data.get_data_provider")
    def test_export_no_data(self, mock_get_provider, data_service, mock_storage):
        """Test exporting when no data available."""
        mock_storage.fetch.return_value = None

        # Mock provider to return empty DataFrame
        mock_provider = Mock()
        mock_provider.get_historical_data.return_value = pd.DataFrame()
        mock_get_provider.return_value = mock_provider

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            source=DataSource.CSV,  # Use CSV to avoid Yahoo download
        )

        result = data_service.export_data(query, format="csv")

        assert result is False

    def test_export_unsupported_format(self, data_service, sample_data, mock_storage):
        """Test exporting with unsupported format."""
        mock_storage.fetch.return_value = sample_data

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
        )

        result = data_service.export_data(query, format="xml")

        assert result is False


class TestImportData:
    """Tests for import_data method."""

    def test_import_csv(self, data_service, sample_data):
        """Test importing data from CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_data.to_csv(f.name)
            temp_path = f.name

        try:
            result = data_service.import_data(
                filepath=temp_path,
                symbol="AAPL",
                data_type=DataType.OHLCV,
            )

            assert result is True
        finally:
            os.unlink(temp_path)

    def test_import_json(self, data_service, sample_data):
        """Test importing data from JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            sample_data.to_json(f.name)
            temp_path = f.name

        try:
            result = data_service.import_data(
                filepath=temp_path,
                symbol="AAPL",
                data_type=DataType.OHLCV,
            )

            assert result is True
        finally:
            os.unlink(temp_path)

    def test_import_unsupported_format(self, data_service):
        """Test importing from unsupported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            temp_path = f.name

        try:
            result = data_service.import_data(
                filepath=temp_path,
                symbol="AAPL",
            )

            assert result is False
        finally:
            os.unlink(temp_path)

    def test_import_nonexistent_file(self, data_service):
        """Test importing from nonexistent file."""
        result = data_service.import_data(
            filepath="/nonexistent/path/file.csv",
            symbol="AAPL",
        )

        assert result is False


class TestDownloadFromYahoo:
    """Tests for _download_from_yahoo internal method."""

    @patch("bot_v2.features.data.data.get_data_provider")
    def test_download_single_symbol(self, mock_get_provider, data_service, sample_data):
        """Test downloading single symbol from Yahoo."""
        mock_provider = Mock()
        mock_provider.get_historical_data.return_value = sample_data
        mock_get_provider.return_value = mock_provider

        results = data_service._download_from_yahoo(
            symbols=["AAPL"],
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 31),
        )

        assert "AAPL" in results
        assert not results["AAPL"].empty

    @patch("bot_v2.features.data.data.get_data_provider")
    def test_download_multiple_symbols(self, mock_get_provider, data_service, sample_data):
        """Test downloading multiple symbols from Yahoo."""
        mock_provider = Mock()
        mock_provider.get_historical_data.return_value = sample_data
        mock_get_provider.return_value = mock_provider

        results = data_service._download_from_yahoo(
            symbols=["AAPL", "GOOGL"],
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 31),
        )

        assert "AAPL" in results
        assert "GOOGL" in results

    @patch("bot_v2.features.data.data.get_data_provider")
    def test_download_with_empty_result(self, mock_get_provider, data_service):
        """Test downloading when provider returns empty data."""
        mock_provider = Mock()
        mock_provider.get_historical_data.return_value = pd.DataFrame()
        mock_get_provider.return_value = mock_provider

        results = data_service._download_from_yahoo(
            symbols=["INVALID"],
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 31),
        )

        assert "INVALID" not in results

    @patch("bot_v2.features.data.data.get_data_provider")
    def test_download_with_exception(self, mock_get_provider, data_service):
        """Test downloading with provider exception."""
        mock_provider = Mock()
        mock_provider.get_historical_data.side_effect = Exception("Provider error")
        mock_get_provider.return_value = mock_provider

        results = data_service._download_from_yahoo(
            symbols=["AAPL"],
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 31),
        )

        # Should handle exception gracefully
        assert results == {}
