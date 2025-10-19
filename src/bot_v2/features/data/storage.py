"""
Local data storage implementation.

Complete isolation - no external dependencies.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from bot_v2.features.data.types import DataQuery, DataType, DataSource
from bot_v2.utilities.logging_patterns import get_logger


logger = get_logger(__name__, component="data_storage")

DEFAULT_STORAGE_ROOT = Path("var/data/coinbase_trader/shared/storage")
LEGACY_RUNTIME_STORAGE_ROOT = Path("var/data/perps_bot/shared/storage")
LEGACY_STORAGE_ROOT = Path("data_storage")


class DataStorage:
    """Persistent data storage."""

    def __init__(self, base_path: str | os.PathLike[str] | None = None):
        """
        Initialize data storage.

        Args:
            base_path: Base directory for storage
        """
        resolved_base = self._resolve_base_path(base_path)
        self.base_path = str(resolved_base)
        os.makedirs(self.base_path, exist_ok=True)

        # Create subdirectories
        self.ohlcv_path = os.path.join(self.base_path, "ohlcv")
        self.metadata_path = os.path.join(self.base_path, "metadata")
        os.makedirs(self.ohlcv_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)

        # Index for fast lookups
        self.index: dict[str, str] = self._load_index()

    def store(
        self,
        symbol: str,
        data: pd.DataFrame,
        data_type: DataType,
        source: DataSource,
        metadata: dict | None = None,
    ) -> bool:
        """
        Store data to disk.

        Args:
            symbol: Stock symbol
            data: Data to store
            data_type: Type of data
            source: Data source
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        try:
            # Generate filename
            filename = f"{symbol}_{data_type.value}_{source.value}.pkl"
            filepath = os.path.join(self.ohlcv_path, filename)

            # Load existing data if any
            if os.path.exists(filepath):
                existing_data = cast(pd.DataFrame, pd.read_pickle(filepath))  # nosec B301
                # Merge with new data (avoid duplicates)
                data = pd.concat([existing_data, data])
                data = data[~data.index.duplicated(keep="last")]
                data.sort_index(inplace=True)

            # Save data
            data.to_pickle(filepath)  # nosec B301

            # Update index
            self._update_index(symbol, data_type, source, filepath)

            # Store metadata if provided
            if metadata:
                meta_filename = f"{symbol}_{data_type.value}_{source.value}_meta.json"
                meta_filepath = os.path.join(self.metadata_path, meta_filename)
                with open(meta_filepath, "w") as f:
                    json.dump(metadata, f)

            return True

        except Exception as e:
            print(f"Storage error: {e}")
            return False

    def fetch(self, query: DataQuery) -> pd.DataFrame | None:
        """
        Fetch data based on query.

        Args:
            query: Data query

        Returns:
            DataFrame or None
        """
        results: list[pd.DataFrame] = []

        for symbol in query.symbols:
            # Find file in index
            key = f"{symbol}_{query.data_type.value}"

            for index_key, filepath in self.index.items():
                if index_key.startswith(key):
                    try:
                        # Load data
                        data = cast(pd.DataFrame, pd.read_pickle(filepath))  # nosec B301

                        # Filter by date range
                        mask = (data.index >= query.start_date) & (data.index <= query.end_date)
                        filtered_data = data[mask]

                        if not filtered_data.empty:
                            results.append(filtered_data)
                    except (FileNotFoundError, pickle.UnpicklingError, ValueError, OSError) as exc:
                        logger.warning(
                            "Failed to load %s for %s (%s): %s",
                            filepath,
                            symbol,
                            query.data_type.value,
                            exc,
                            exc_info=True,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Unexpected error loading %s for %s",
                            filepath,
                            symbol,
                        )

        if results:
            # Combine results
            combined = cast(pd.DataFrame, pd.concat(results))
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            return combined

        return None

    def delete_before(self, cutoff: datetime) -> int:
        """
        Delete data before cutoff date.

        Args:
            cutoff: Cutoff date

        Returns:
            Number of records deleted
        """
        deleted_count = 0

        for filepath in self.index.values():
            try:
                data = cast(pd.DataFrame, pd.read_pickle(filepath))  # nosec B301
                original_len = len(data)

                # Keep only data after cutoff
                data = data[data.index >= cutoff]

                if len(data) < original_len:
                    deleted_count += original_len - len(data)

                    if len(data) > 0:
                        # Save filtered data
                        data.to_pickle(filepath)  # nosec B301
                    else:
                        # Delete empty file
                        os.remove(filepath)
                        # Remove from index
                        keys_to_remove = [k for k, v in self.index.items() if v == filepath]
                        for key in keys_to_remove:
                            del self.index[key]
            except (FileNotFoundError, pickle.UnpicklingError, ValueError, OSError) as exc:
                logger.warning("Failed to prune %s: %s", filepath, exc, exc_info=True)
            except Exception as exc:
                logger.exception("Unexpected error pruning %s", filepath)

        # Save updated index
        self._save_index()

        return deleted_count

    def get_stats(self) -> dict:
        """Get storage statistics."""
        total_records = 0
        total_size_mb = 0.0
        oldest_record: datetime | None = None
        newest_record: datetime | None = None
        symbols: set[str] = set()

        for filepath in self.index.values():
            try:
                # Get file size
                size_bytes = os.path.getsize(filepath)
                total_size_mb += size_bytes / (1024 * 1024)

                # Load data for stats
                data = cast(pd.DataFrame, pd.read_pickle(filepath))  # nosec B301
                total_records += len(data)

                # Track date range
                if not data.empty:
                    min_date = data.index.min()
                    max_date = data.index.max()

                    if oldest_record is None or min_date < oldest_record:
                        oldest_record = min_date
                    if newest_record is None or max_date > newest_record:
                        newest_record = max_date

                # Extract symbol from filename
                filename = os.path.basename(filepath)
                symbol = filename.split("_")[0]
                symbols.add(symbol)

            except (FileNotFoundError, pickle.UnpicklingError, ValueError, OSError) as exc:
                logger.warning("Failed to load %s for stats: %s", filepath, exc, exc_info=True)
            except Exception as exc:
                logger.exception("Unexpected error collecting stats from %s", filepath)

        return {
            "total_records": total_records,
            "total_size_mb": total_size_mb,
            "oldest_record": oldest_record,
            "newest_record": newest_record,
            "symbols_count": len(symbols),
        }

    @staticmethod
    def _resolve_base_path(base_path: str | os.PathLike[str] | None) -> Path:
        """
        Resolve the storage root, migrating legacy directories when possible.

        Preference order:
        1. Explicit path provided by the caller.
        2. Migrated legacy runtime dir (`var/data/perps_bot/shared/storage`) when the new root is absent.
        3. Migrated legacy `data_storage` directory when present.
        4. Default `var/data/coinbase_trader/shared/storage`.
        """
        if base_path is not None:
            return Path(base_path)

        default_root = DEFAULT_STORAGE_ROOT
        legacy_runtime_root = LEGACY_RUNTIME_STORAGE_ROOT
        legacy_root = LEGACY_STORAGE_ROOT

        if legacy_runtime_root.exists():
            try:
                default_root.parent.mkdir(parents=True, exist_ok=True)
                if not default_root.exists():
                    legacy_runtime_root.rename(default_root)
                    logger.info("Migrated legacy runtime storage to %s", default_root)
                return default_root
            except OSError as exc:
                logger.warning(
                    "Failed to migrate legacy runtime storage to %s (%s); using %s",
                    default_root,
                    exc,
                    legacy_runtime_root,
                    exc_info=True,
                )
                return legacy_runtime_root.resolve()

        if legacy_root.exists():
            try:
                default_root.parent.mkdir(parents=True, exist_ok=True)
                if not default_root.exists():
                    legacy_root.rename(default_root)
                    logger.info("Migrated legacy data storage to %s", default_root)
                return default_root
            except OSError as exc:
                logger.warning(
                    "Failed to migrate legacy data storage to %s (%s); using %s",
                    default_root,
                    exc,
                    legacy_root,
                    exc_info=True,
                )
                return legacy_root.resolve()

        return default_root

    def _load_index(self) -> dict[str, str]:
        """Load storage index."""
        index_file = os.path.join(self.base_path, "index.json")

        if os.path.exists(index_file):
            with open(index_file) as f:
                return cast(dict[str, str], json.load(f))

        # Build index from files
        index: dict[str, str] = {}
        for filename in os.listdir(self.ohlcv_path):
            if filename.endswith(".pkl"):
                parts = filename[:-4].split("_")
                if len(parts) >= 3:
                    symbol = parts[0]
                    data_type = parts[1]
                    source = parts[2]
                    key = f"{symbol}_{data_type}_{source}"
                    filepath = os.path.join(self.ohlcv_path, filename)
                    index[key] = filepath

        return index

    def _save_index(self) -> None:
        """Save storage index."""
        index_file = os.path.join(self.base_path, "index.json")
        with open(index_file, "w") as f:
            json.dump(self.index, f)

    def _update_index(
        self, symbol: str, data_type: DataType, source: DataSource, filepath: str
    ) -> None:
        """Update storage index."""
        key = f"{symbol}_{data_type.value}_{source.value}"
        self.index[key] = filepath
        self._save_index()
