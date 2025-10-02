"""
Comprehensive tests for DataStorage.

Covers persistence, deduplication, index management, and pruning.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from bot_v2.features.data.storage import DataStorage
from bot_v2.features.data.types import DataQuery, DataSource, DataType


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage."""
    return DataStorage(base_path=str(tmp_path / "test_storage"))


@pytest.fixture
def sample_ohlcv():
    """Sample OHLCV data with datetime index."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "open": range(100, 110),
        "high": range(101, 111),
        "low": range(99, 109),
        "close": range(100, 110),
        "volume": [1000000] * 10,
    }, index=dates)


@pytest.fixture
def additional_ohlcv():
    """Additional OHLCV data for merge tests."""
    dates = pd.date_range(start="2023-01-08", periods=10, freq="D")
    return pd.DataFrame({
        "open": range(107, 117),
        "high": range(108, 118),
        "low": range(106, 116),
        "close": range(107, 117),
        "volume": [1100000] * 10,
    }, index=dates)


class TestStorageInitialization:
    """Test storage initialization."""

    def test_creates_directories(self, tmp_path):
        """Should create base and subdirectories."""
        storage_path = tmp_path / "test_storage"
        storage = DataStorage(base_path=str(storage_path))

        assert storage_path.exists()
        assert (storage_path / "ohlcv").exists()
        assert (storage_path / "metadata").exists()

    def test_loads_empty_index(self, temp_storage):
        """Should start with empty index."""
        assert temp_storage.index == {}

    def test_loads_existing_index(self, tmp_path):
        """Should load existing index file."""
        storage_path = tmp_path / "test_storage"
        storage_path.mkdir()

        # Create index file
        index_data = {"AAPL_ohlcv_yahoo": str(storage_path / "ohlcv" / "AAPL_ohlcv_yahoo.pkl")}
        with open(storage_path / "index.json", "w") as f:
            json.dump(index_data, f)

        storage = DataStorage(base_path=str(storage_path))
        assert storage.index == index_data

    def test_rebuilds_index_from_files(self, tmp_path, sample_ohlcv):
        """Should rebuild index from existing pickle files."""
        storage_path = tmp_path / "test_storage"
        ohlcv_path = storage_path / "ohlcv"
        ohlcv_path.mkdir(parents=True)

        # Create pickle file without index
        pkl_file = ohlcv_path / "AAPL_ohlcv_yahoo.pkl"
        sample_ohlcv.to_pickle(pkl_file)

        storage = DataStorage(base_path=str(storage_path))

        assert "AAPL_ohlcv_yahoo" in storage.index


class TestStoreData:
    """Test data storage."""

    def test_store_new_data(self, temp_storage, sample_ohlcv):
        """Should store new data successfully."""
        result = temp_storage.store(
            symbol="AAPL",
            data=sample_ohlcv,
            data_type=DataType.OHLCV,
            source=DataSource.YAHOO,
        )

        assert result is True
        # Check file was created
        expected_file = Path(temp_storage.ohlcv_path) / "AAPL_ohlcv_yahoo.pkl"
        assert expected_file.exists()

    def test_store_updates_index(self, temp_storage, sample_ohlcv):
        """Should update index with new entry."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        assert "AAPL_ohlcv_yahoo" in temp_storage.index

        # Check index file exists
        index_file = Path(temp_storage.base_path) / "index.json"
        assert index_file.exists()

    def test_store_merges_existing_data(self, temp_storage, sample_ohlcv, additional_ohlcv):
        """Should merge with existing data."""
        # Store initial data
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Store overlapping data
        temp_storage.store("AAPL", additional_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Load and verify
        filepath = temp_storage.index["AAPL_ohlcv_yahoo"]
        merged = pd.read_pickle(filepath)

        # Should have combined date range
        assert len(merged) > len(sample_ohlcv)
        assert merged.index.min() == sample_ohlcv.index.min()
        assert merged.index.max() == additional_ohlcv.index.max()

    def test_store_deduplicates(self, temp_storage, sample_ohlcv):
        """Should remove duplicate timestamps."""
        # Store same data twice
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        filepath = temp_storage.index["AAPL_ohlcv_yahoo"]
        stored = pd.read_pickle(filepath)

        # Should not have duplicates
        assert len(stored) == len(sample_ohlcv)
        assert not stored.index.duplicated().any()

    def test_store_keeps_latest_duplicate(self, temp_storage, sample_ohlcv):
        """Should keep latest value for duplicate timestamps."""
        # Modify data with same timestamps
        modified = sample_ohlcv.copy()
        modified["close"] = modified["close"] + 10

        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("AAPL", modified, DataType.OHLCV, DataSource.YAHOO)

        filepath = temp_storage.index["AAPL_ohlcv_yahoo"]
        stored = pd.read_pickle(filepath)

        # Should have modified values (kept 'last')
        assert stored["close"].equals(modified["close"])

    def test_store_sorts_index(self, temp_storage, sample_ohlcv):
        """Should sort data by index."""
        # Create unsorted data by reindexing in reverse
        unsorted = sample_ohlcv.iloc[::-1].copy()

        temp_storage.store("AAPL", unsorted, DataType.OHLCV, DataSource.YAHOO)

        filepath = temp_storage.index["AAPL_ohlcv_yahoo"]
        stored = pd.read_pickle(filepath)

        # Should be sorted
        assert stored.index.is_monotonic_increasing

    def test_store_with_metadata(self, temp_storage, sample_ohlcv):
        """Should store metadata separately."""
        metadata = {"source_api": "yahoo_finance", "version": "1.0"}

        temp_storage.store(
            "AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO, metadata=metadata
        )

        # Check metadata file
        meta_file = Path(temp_storage.metadata_path) / "AAPL_ohlcv_yahoo_meta.json"
        assert meta_file.exists()

        with open(meta_file) as f:
            stored_meta = json.load(f)

        assert stored_meta == metadata


class TestFetchData:
    """Test data fetching."""

    def test_fetch_single_symbol(self, temp_storage, sample_ohlcv):
        """Should fetch data for single symbol."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            data_type=DataType.OHLCV,
        )

        result = temp_storage.fetch(query)

        assert result is not None
        assert len(result) == len(sample_ohlcv)

    def test_fetch_filters_by_date(self, temp_storage, sample_ohlcv):
        """Should filter results by date range."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Query subset of dates
        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 3),
            end_date=datetime(2023, 1, 7),
            data_type=DataType.OHLCV,
        )

        result = temp_storage.fetch(query)

        assert result is not None
        assert len(result) == 5  # 5 days in range
        assert result.index.min() >= query.start_date
        assert result.index.max() <= query.end_date

    def test_fetch_multiple_symbols(self, temp_storage, sample_ohlcv):
        """Should fetch and combine multiple symbols."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("GOOGL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        query = DataQuery(
            symbols=["AAPL", "GOOGL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            data_type=DataType.OHLCV,
        )

        result = temp_storage.fetch(query)

        assert result is not None
        # Should combine both symbols (may have duplicates in test data)
        assert len(result) >= len(sample_ohlcv)

    def test_fetch_missing_symbol(self, temp_storage):
        """Should return None for missing symbol."""
        query = DataQuery(
            symbols=["NONEXISTENT"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            data_type=DataType.OHLCV,
        )

        result = temp_storage.fetch(query)
        assert result is None

    def test_fetch_empty_date_range(self, temp_storage, sample_ohlcv):
        """Should return None for date range with no data."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            data_type=DataType.OHLCV,
        )

        result = temp_storage.fetch(query)
        assert result is None

    def test_fetch_deduplicates_results(self, temp_storage, sample_ohlcv):
        """Should deduplicate combined results."""
        # Store same data under different sources
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.POLYGON)

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            data_type=DataType.OHLCV,
        )

        result = temp_storage.fetch(query)

        # Should deduplicate
        assert not result.index.duplicated().any()

    def test_fetch_handles_corrupt_file(self, temp_storage, sample_ohlcv):
        """Should handle corrupted pickle files gracefully."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Corrupt the file
        filepath = Path(temp_storage.index["AAPL_ohlcv_yahoo"])
        filepath.write_text("corrupted data")

        query = DataQuery(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            data_type=DataType.OHLCV,
        )

        # Should not crash
        result = temp_storage.fetch(query)
        assert result is None


class TestDeleteBefore:
    """Test data pruning."""

    def test_delete_before_cutoff(self, temp_storage, sample_ohlcv):
        """Should delete data before cutoff date."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        cutoff = datetime(2023, 1, 6)
        deleted = temp_storage.delete_before(cutoff)

        # Should delete first 5 records (before 2023-01-06)
        assert deleted == 5

        # Verify remaining data
        filepath = temp_storage.index["AAPL_ohlcv_yahoo"]
        remaining = pd.read_pickle(filepath)
        assert len(remaining) == 5
        assert remaining.index.min() >= cutoff

    def test_delete_before_removes_empty_files(self, temp_storage, sample_ohlcv):
        """Should remove files with no data after cutoff."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Delete all data
        cutoff = datetime(2025, 1, 1)
        temp_storage.delete_before(cutoff)

        # File should be removed
        filepath = Path(temp_storage.ohlcv_path) / "AAPL_ohlcv_yahoo.pkl"
        assert not filepath.exists()

        # Index should be updated
        assert "AAPL_ohlcv_yahoo" not in temp_storage.index

    def test_delete_before_updates_index(self, temp_storage, sample_ohlcv, additional_ohlcv):
        """Should update index file after deletion."""
        # sample_ohlcv: 2023-01-01 to 2023-01-10
        # additional_ohlcv: 2023-01-08 to 2023-01-17
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("GOOGL", additional_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Delete everything before 2023-01-11 (AAPL completely gone, GOOGL keeps 2023-01-11 onwards)
        cutoff = datetime(2023, 1, 11)
        temp_storage.delete_before(cutoff)

        # Load index file
        index_file = Path(temp_storage.base_path) / "index.json"
        with open(index_file) as f:
            saved_index = json.load(f)

        # AAPL should be removed (all data before cutoff), GOOGL should remain (has data after cutoff)
        assert "AAPL_ohlcv_yahoo" not in saved_index
        assert "GOOGL_ohlcv_yahoo" in saved_index

    def test_delete_before_multiple_files(self, temp_storage, sample_ohlcv):
        """Should prune multiple files."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("GOOGL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("MSFT", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        cutoff = datetime(2023, 1, 6)
        deleted = temp_storage.delete_before(cutoff)

        # Should delete from all files
        assert deleted == 15  # 5 per symbol × 3 symbols

    def test_delete_before_handles_missing_files(self, temp_storage, sample_ohlcv):
        """Should handle files that no longer exist."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Delete file manually
        filepath = Path(temp_storage.index["AAPL_ohlcv_yahoo"])
        filepath.unlink()

        # Should not crash
        cutoff = datetime(2023, 1, 6)
        deleted = temp_storage.delete_before(cutoff)
        assert deleted == 0


class TestStorageStats:
    """Test storage statistics."""

    def test_stats_empty_storage(self, temp_storage):
        """Should return zero stats for empty storage."""
        stats = temp_storage.get_stats()

        assert stats["total_records"] == 0
        assert stats["total_size_mb"] == 0
        assert stats["oldest_record"] is None
        assert stats["newest_record"] is None
        assert stats["symbols_count"] == 0

    def test_stats_with_data(self, temp_storage, sample_ohlcv):
        """Should calculate correct statistics."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("GOOGL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        stats = temp_storage.get_stats()

        assert stats["total_records"] == 20  # 10 per symbol
        assert stats["total_size_mb"] > 0
        assert stats["oldest_record"] == datetime(2023, 1, 1)
        assert stats["newest_record"] == datetime(2023, 1, 10)
        assert stats["symbols_count"] == 2

    def test_stats_date_range(self, temp_storage, sample_ohlcv, additional_ohlcv):
        """Should track correct date range."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("GOOGL", additional_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        stats = temp_storage.get_stats()

        assert stats["oldest_record"] == sample_ohlcv.index.min()
        assert stats["newest_record"] == additional_ohlcv.index.max()

    def test_stats_file_size(self, temp_storage, sample_ohlcv):
        """Should calculate file sizes."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        stats = temp_storage.get_stats()

        filepath = Path(temp_storage.index["AAPL_ohlcv_yahoo"])
        actual_size_mb = filepath.stat().st_size / (1024 * 1024)

        assert stats["total_size_mb"] == pytest.approx(actual_size_mb, rel=0.01)

    def test_stats_handles_errors(self, temp_storage, sample_ohlcv):
        """Should handle file errors gracefully."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Corrupt file
        filepath = Path(temp_storage.index["AAPL_ohlcv_yahoo"])
        filepath.write_text("corrupted")

        # Should not crash
        stats = temp_storage.get_stats()
        # Stats will be partial/incomplete but shouldn't raise


class TestIndexManagement:
    """Test index persistence and management."""

    def test_index_persistence(self, temp_storage, sample_ohlcv):
        """Index should persist across instances."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        # Create new instance
        new_storage = DataStorage(base_path=temp_storage.base_path)

        # Should load saved index
        assert "AAPL_ohlcv_yahoo" in new_storage.index

    def test_index_update_on_store(self, temp_storage, sample_ohlcv):
        """Should update index on each store."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        index_file = Path(temp_storage.base_path) / "index.json"
        first_mtime = index_file.stat().st_mtime

        # Store another symbol
        temp_storage.store("GOOGL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        second_mtime = index_file.stat().st_mtime
        assert second_mtime >= first_mtime

    def test_index_format(self, temp_storage, sample_ohlcv):
        """Index should have correct key format."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        with open(Path(temp_storage.base_path) / "index.json") as f:
            index = json.load(f)

        assert "AAPL_ohlcv_yahoo" in index
        assert index["AAPL_ohlcv_yahoo"].endswith("AAPL_ohlcv_yahoo.pkl")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_store_empty_dataframe(self, temp_storage):
        """Should handle empty DataFrame."""
        empty_df = pd.DataFrame()
        result = temp_storage.store("AAPL", empty_df, DataType.OHLCV, DataSource.YAHOO)

        assert result is True

    def test_multiple_data_types(self, temp_storage, sample_ohlcv):
        """Should support different data types for same symbol."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("AAPL", sample_ohlcv, DataType.QUOTE, DataSource.YAHOO)

        assert "AAPL_ohlcv_yahoo" in temp_storage.index
        assert "AAPL_quote_yahoo" in temp_storage.index

    def test_multiple_sources(self, temp_storage, sample_ohlcv):
        """Should support different sources for same symbol."""
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)
        temp_storage.store("AAPL", sample_ohlcv, DataType.OHLCV, DataSource.POLYGON)

        assert "AAPL_ohlcv_yahoo" in temp_storage.index
        assert "AAPL_ohlcv_polygon" in temp_storage.index

    def test_special_characters_in_symbol(self, temp_storage, sample_ohlcv):
        """Should handle symbols with special characters."""
        # Some symbols have special chars like BRK.B
        temp_storage.store("BRK.B", sample_ohlcv, DataType.OHLCV, DataSource.YAHOO)

        assert "BRK.B_ohlcv_yahoo" in temp_storage.index
