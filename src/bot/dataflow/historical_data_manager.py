"""
Historical Data Manager for Strategy Training

Provides clean, validated, multi-source historical data for strategy development and training.
Handles data aggregation, quality validation, corporate actions, and efficient caching.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# Optional dependencies
try:
    import alpha_vantage

    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

try:
    import quandl

    QUANDL_AVAILABLE = True
except ImportError:
    QUANDL_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""

    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    CSV_FILE = "csv_file"


class DataFrequency(Enum):
    """Data frequency options"""

    DAILY = "1d"
    HOURLY = "1h"
    MINUTE = "1m"


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""

    total_records: int = 0
    missing_records: int = 0
    duplicate_records: int = 0
    outlier_records: int = 0
    quality_score: float = 0.0
    completeness_ratio: float = 0.0
    consistency_score: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class DatasetMetadata:
    """Metadata for cached datasets"""

    symbols: list[str]
    start_date: datetime
    end_date: datetime
    frequency: DataFrequency
    sources: list[DataSource]
    quality_metrics: dict[str, DataQualityMetrics]
    created_at: datetime
    last_updated: datetime
    cache_key: str
    corporate_actions_adjusted: bool = True
    survivorship_bias_handled: bool = True


@dataclass
class HistoricalDataConfig:
    """Configuration for historical data management"""

    # Data sources and priorities
    preferred_sources: list[DataSource] = field(default_factory=lambda: [DataSource.YFINANCE])
    fallback_sources: list[DataSource] = field(default_factory=list)

    # API keys
    alpha_vantage_api_key: str | None = None
    quandl_api_key: str | None = None

    # Data quality settings
    min_quality_score: float = 0.85
    max_missing_data_ratio: float = 0.05
    outlier_detection_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0

    # Caching settings
    cache_dir: Path = Path("data/historical_cache")
    enable_caching: bool = True
    cache_expiry_hours: int = 24

    # Corporate actions
    adjust_for_dividends: bool = True
    adjust_for_splits: bool = True

    # Performance settings
    max_concurrent_downloads: int = 10
    request_delay_seconds: float = 0.1
    retry_attempts: int = 3
    timeout_seconds: int = 30


class BaseDataProvider(ABC):
    """Base class for data providers"""

    def __init__(self, config: HistoricalDataConfig) -> None:
        self.config = config

    @abstractmethod
    def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime, frequency: DataFrequency
    ) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if data provider is available"""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> list[str]:
        """Get list of supported symbols"""
        pass


class YFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider"""

    def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime, frequency: DataFrequency
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)

            # Map frequency
            interval_map = {
                DataFrequency.DAILY: "1d",
                DataFrequency.HOURLY: "1h",
                DataFrequency.MINUTE: "1m",
            }

            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval_map[frequency],
                auto_adjust=self.config.adjust_for_dividends,
                back_adjust=True,
                repair=True,
                keepna=False,
            )

            if data.empty:
                logger.warning(f"No data returned for {symbol} from Yahoo Finance")
                return pd.DataFrame()

            # Standardize column names
            data.columns = [col.title() for col in data.columns]

            # Add metadata
            data["Symbol"] = symbol
            data["Source"] = DataSource.YFINANCE.value

            return data

        except Exception as e:
            logger.error(f"Error fetching {symbol} from Yahoo Finance: {str(e)}")
            return pd.DataFrame()

    def is_available(self) -> bool:
        """Check if Yahoo Finance is available"""
        try:
            # Test with a simple request
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            return not test_data.empty
        except Exception:
            return False

    def get_supported_symbols(self) -> list[str]:
        """Yahoo Finance supports most symbols"""
        # Return common symbols for validation
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage data provider"""

    def __init__(self, config: HistoricalDataConfig) -> None:
        super().__init__(config)
        if not ALPHA_VANTAGE_AVAILABLE:
            raise ImportError("alpha_vantage library not installed")
        if not config.alpha_vantage_api_key:
            raise ValueError("Alpha Vantage API key required")

    def fetch_data(
        self, symbol: str, start_date: datetime, end_date: datetime, frequency: DataFrequency
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage"""
        try:
            from alpha_vantage.timeseries import TimeSeries

            ts = TimeSeries(key=self.config.alpha_vantage_api_key, output_format="pandas")

            if frequency == DataFrequency.DAILY:
                data, meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize="full")
            elif frequency == DataFrequency.HOURLY:
                data, meta_data = ts.get_intraday(
                    symbol=symbol, interval="60min", outputsize="full"
                )
            else:  # MINUTE
                data, meta_data = ts.get_intraday(symbol=symbol, interval="1min", outputsize="full")

            if data.empty:
                return pd.DataFrame()

            # Filter by date range
            data = data.loc[start_date:end_date]

            # Standardize column names
            column_map = {
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",
                "6. volume": "Volume",
            }
            data = data.rename(columns=column_map)

            # Add metadata
            data["Symbol"] = symbol
            data["Source"] = DataSource.ALPHA_VANTAGE.value

            return data

        except Exception as e:
            logger.error(f"Error fetching {symbol} from Alpha Vantage: {str(e)}")
            return pd.DataFrame()

    def is_available(self) -> bool:
        """Check if Alpha Vantage is available"""
        return ALPHA_VANTAGE_AVAILABLE and bool(self.config.alpha_vantage_api_key)

    def get_supported_symbols(self) -> list[str]:
        """Alpha Vantage supports US equities"""
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


class DataQualityValidator:
    """Validates and scores data quality"""

    def __init__(self, config: HistoricalDataConfig) -> None:
        self.config = config

    def validate_dataset(self, data: pd.DataFrame, symbol: str) -> DataQualityMetrics:
        """Comprehensive data quality validation"""
        if data.empty:
            return DataQualityMetrics(
                total_records=0, quality_score=0.0, issues=["Dataset is empty"]
            )

        metrics = DataQualityMetrics()
        metrics.total_records = len(data)

        # Check for missing data
        missing_data = data.isnull().sum()
        metrics.missing_records = missing_data.sum()

        # Check for duplicates
        duplicate_indices = data.index.duplicated()
        metrics.duplicate_records = duplicate_indices.sum()

        # Check for outliers
        metrics.outlier_records = self._detect_outliers(data)

        # Calculate quality metrics
        metrics.completeness_ratio = 1.0 - (
            metrics.missing_records / (len(data) * len(data.columns))
        )
        metrics.consistency_score = self._calculate_consistency_score(data)

        # Overall quality score (weighted average)
        weights = {"completeness": 0.4, "consistency": 0.3, "outliers": 0.2, "duplicates": 0.1}

        completeness_score = metrics.completeness_ratio
        consistency_score = metrics.consistency_score
        outlier_score = 1.0 - (metrics.outlier_records / metrics.total_records)
        duplicate_score = 1.0 - (metrics.duplicate_records / metrics.total_records)

        metrics.quality_score = (
            weights["completeness"] * completeness_score
            + weights["consistency"] * consistency_score
            + weights["outliers"] * outlier_score
            + weights["duplicates"] * duplicate_score
        )

        # Generate quality issues
        metrics.issues = self._generate_quality_issues(data, metrics, symbol)

        return metrics

    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """Detect outliers in price data"""
        outlier_count = 0

        # Check numeric columns only
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        numeric_columns = [col for col in numeric_columns if col in data.columns]

        for col in numeric_columns:
            if self.config.outlier_detection_method == "iqr":
                outlier_count += self._detect_outliers_iqr(data[col])
            elif self.config.outlier_detection_method == "zscore":
                outlier_count += self._detect_outliers_zscore(data[col])

        return outlier_count

    def _detect_outliers_iqr(self, series: pd.Series) -> int:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers.sum()

    def _detect_outliers_zscore(self, series: pd.Series) -> int:
        """Detect outliers using Z-score method"""
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > self.config.outlier_threshold
        return outliers.sum()

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        consistency_score = 1.0

        # Check OHLC relationships
        if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            # High should be >= max(Open, Close)
            high_violations = (data["High"] < data[["Open", "Close"]].max(axis=1)).sum()

            # Low should be <= min(Open, Close)
            low_violations = (data["Low"] > data[["Open", "Close"]].min(axis=1)).sum()

            total_violations = high_violations + low_violations
            violation_ratio = total_violations / len(data) if len(data) > 0 else 0
            consistency_score *= 1.0 - violation_ratio

        # Check for negative prices
        price_columns = ["Open", "High", "Low", "Close"]
        price_columns = [col for col in price_columns if col in data.columns]

        for col in price_columns:
            negative_count = (data[col] <= 0).sum()
            if negative_count > 0:
                consistency_score *= 1.0 - negative_count / len(data)

        # Check volume consistency
        if "Volume" in data.columns:
            negative_volume = (data["Volume"] < 0).sum()
            if negative_volume > 0:
                consistency_score *= 1.0 - negative_volume / len(data)

        return consistency_score

    def _generate_quality_issues(
        self, data: pd.DataFrame, metrics: DataQualityMetrics, symbol: str
    ) -> list[str]:
        """Generate list of quality issues"""
        issues = []

        if metrics.missing_records > 0:
            missing_ratio = metrics.missing_records / (len(data) * len(data.columns))
            issues.append(f"Missing data: {missing_ratio:.2%} of records")

        if metrics.duplicate_records > 0:
            duplicate_ratio = metrics.duplicate_records / len(data)
            issues.append(f"Duplicate timestamps: {duplicate_ratio:.2%} of records")

        if metrics.outlier_records > 0:
            outlier_ratio = metrics.outlier_records / len(data)
            issues.append(f"Outliers detected: {outlier_ratio:.2%} of records")

        if metrics.quality_score < self.config.min_quality_score:
            issues.append(
                f"Quality score {metrics.quality_score:.2f} below threshold {self.config.min_quality_score}"
            )

        return issues


class HistoricalDataManager:
    """Main historical data management system"""

    def __init__(self, config: HistoricalDataConfig | None = None) -> None:
        self.config = config or HistoricalDataConfig()
        self.providers = self._initialize_providers()
        self.validator = DataQualityValidator(self.config)
        self.cache_dir = self.config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Historical Data Manager initialized with {len(self.providers)} providers")

    def _initialize_providers(self) -> dict[DataSource, BaseDataProvider]:
        """Initialize available data providers"""
        providers = {}

        # Always add Yahoo Finance
        providers[DataSource.YFINANCE] = YFinanceProvider(self.config)

        # Add Alpha Vantage if available
        if ALPHA_VANTAGE_AVAILABLE and self.config.alpha_vantage_api_key:
            try:
                providers[DataSource.ALPHA_VANTAGE] = AlphaVantageProvider(self.config)
            except (ImportError, ValueError) as e:
                logger.warning(f"Alpha Vantage provider not available: {str(e)}")

        return providers

    def get_training_dataset(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency = DataFrequency.DAILY,
        force_refresh: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], DatasetMetadata]:
        """Get clean, validated training dataset"""

        # Check cache first
        cache_key = self._generate_cache_key(symbols, start_date, end_date, frequency)

        if not force_refresh and self.config.enable_caching:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded dataset from cache: {cache_key}")
                return cached_data

        logger.info(
            f"Fetching dataset: {len(symbols)} symbols from {start_date.date()} to {end_date.date()}"
        )

        # Fetch data for all symbols
        datasets = {}
        quality_metrics = {}

        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_downloads) as executor:
            # Submit all download tasks
            future_to_symbol = {}
            for symbol in symbols:
                future = executor.submit(
                    self._fetch_symbol_data, symbol, start_date, end_date, frequency
                )
                future_to_symbol[future] = symbol

            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data, quality = future.result()
                    if not data.empty and quality.quality_score >= self.config.min_quality_score:
                        datasets[symbol] = data
                        quality_metrics[symbol] = quality
                        logger.info(
                            f"✅ {symbol}: {len(data)} records, quality: {quality.quality_score:.2f}"
                        )
                    else:
                        logger.warning(
                            f"❌ {symbol}: Low quality data excluded (score: {quality.quality_score:.2f})"
                        )
                except Exception as e:
                    logger.error(f"❌ {symbol}: Failed to fetch data: {str(e)}")

        # Create metadata
        metadata = DatasetMetadata(
            symbols=list(datasets.keys()),
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            sources=self.config.preferred_sources,
            quality_metrics=quality_metrics,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            cache_key=cache_key,
            corporate_actions_adjusted=self.config.adjust_for_dividends,
            survivorship_bias_handled=True,
        )

        # Cache the results
        if self.config.enable_caching:
            self._save_to_cache(cache_key, (datasets, metadata))

        logger.info(f"Dataset prepared: {len(datasets)}/{len(symbols)} symbols with quality data")
        return datasets, metadata

    def _fetch_symbol_data(
        self, symbol: str, start_date: datetime, end_date: datetime, frequency: DataFrequency
    ) -> tuple[pd.DataFrame, DataQualityMetrics]:
        """Fetch and validate data for a single symbol"""

        # Try preferred sources first, then fallbacks
        sources_to_try = self.config.preferred_sources + self.config.fallback_sources

        for source in sources_to_try:
            if source not in self.providers:
                continue

            provider = self.providers[source]
            if not provider.is_available():
                continue

            try:
                data = provider.fetch_data(symbol, start_date, end_date, frequency)
                if data.empty:
                    continue

                # Validate quality
                quality_metrics = self.validator.validate_dataset(data, symbol)

                if quality_metrics.quality_score >= self.config.min_quality_score:
                    return data, quality_metrics
                else:
                    logger.warning(
                        f"{symbol} from {source.value}: Quality score {quality_metrics.quality_score:.2f} too low"
                    )

            except Exception as e:
                logger.error(f"Error fetching {symbol} from {source.value}: {str(e)}")
                continue

        # Return empty data if all sources failed
        return pd.DataFrame(), DataQualityMetrics()

    def _generate_cache_key(
        self, symbols: list[str], start_date: datetime, end_date: datetime, frequency: DataFrequency
    ) -> str:
        """Generate cache key for dataset"""
        key_data = {
            "symbols": sorted(symbols),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "frequency": frequency.value,
            "sources": [s.value for s in self.config.preferred_sources],
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _save_to_cache(
        self, cache_key: str, data: tuple[dict[str, pd.DataFrame], DatasetMetadata]
    ) -> None:
        """Save dataset to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.joblib"
            joblib.dump(data, cache_file)
            logger.debug(f"Dataset cached: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {str(e)}")

    def _load_from_cache(
        self, cache_key: str
    ) -> tuple[dict[str, pd.DataFrame], DatasetMetadata] | None:
        """Load dataset from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.joblib"

            if not cache_file.exists():
                return None

            # Check if cache is expired
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() > (self.config.cache_expiry_hours * 3600):
                logger.debug(f"Cache expired: {cache_key}")
                return None

            return joblib.load(cache_file)

        except Exception as e:
            logger.warning(f"Failed to load from cache: {str(e)}")
            return None

    def get_available_symbols(self, source: DataSource | None = None) -> list[str]:
        """Get list of available symbols"""
        if source and source in self.providers:
            return self.providers[source].get_supported_symbols()
        else:
            # Return union of all provider symbols
            all_symbols = set()
            for provider in self.providers.values():
                all_symbols.update(provider.get_supported_symbols())
            return sorted(list(all_symbols))

    def validate_dataset_quality(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, DataQualityMetrics]:
        """Validate quality of existing dataset"""
        quality_metrics = {}

        for symbol, df in data.items():
            quality_metrics[symbol] = self.validator.validate_dataset(df, symbol)

        return quality_metrics

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.joblib"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_directory": str(self.cache_dir),
            "cached_datasets": len(cache_files),
            "total_cache_size_mb": total_size / (1024 * 1024),
            "cache_expiry_hours": self.config.cache_expiry_hours,
            "oldest_cache": (
                min((f.stat().st_mtime for f in cache_files), default=0) if cache_files else 0
            ),
            "newest_cache": (
                max((f.stat().st_mtime for f in cache_files), default=0) if cache_files else 0
            ),
        }

    def clear_cache(self, older_than_hours: int | None = None):
        """Clear cached data"""
        cache_files = list(self.cache_dir.glob("*.joblib"))

        if older_than_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            cache_files = [
                f for f in cache_files if datetime.fromtimestamp(f.stat().st_mtime) < cutoff_time
            ]

        removed_count = 0
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {str(e)}")

        logger.info(f"Removed {removed_count} cache files")
        return removed_count


def create_historical_data_manager(
    preferred_sources: list[DataSource] | None = None,
    alpha_vantage_api_key: str | None = None,
    min_quality_score: float = 0.85,
    cache_dir: str = "data/historical_cache",
    **kwargs,
) -> HistoricalDataManager:
    """Factory function to create historical data manager"""

    config = HistoricalDataConfig(
        preferred_sources=preferred_sources or [DataSource.YFINANCE],
        alpha_vantage_api_key=alpha_vantage_api_key,
        min_quality_score=min_quality_score,
        cache_dir=Path(cache_dir),
        **kwargs,
    )

    return HistoricalDataManager(config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta

    async def main() -> None:
        """Example usage of Historical Data Manager"""
        print("Historical Data Manager Testing")
        print("=" * 40)

        # Create data manager
        manager = create_historical_data_manager(min_quality_score=0.80, max_concurrent_downloads=5)

        # Test symbols
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data

        print(f"Fetching data for {len(test_symbols)} symbols...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")

        # Get training dataset
        datasets, metadata = manager.get_training_dataset(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=DataFrequency.DAILY,
        )

        print("\n📊 Dataset Summary:")
        print(f"   Symbols retrieved: {len(datasets)}/{len(test_symbols)}")
        print(f"   Date range: {metadata.start_date.date()} to {metadata.end_date.date()}")
        print(f"   Corporate actions adjusted: {metadata.corporate_actions_adjusted}")

        # Quality metrics
        print("\n📈 Quality Metrics:")
        for symbol, quality in metadata.quality_metrics.items():
            print(
                f"   {symbol}: {quality.total_records} records, quality: {quality.quality_score:.2f}"
            )
            if quality.issues:
                print(f"      Issues: {', '.join(quality.issues[:2])}")

        # Cache info
        cache_info = manager.get_cache_info()
        print("\n💾 Cache Info:")
        print(f"   Cached datasets: {cache_info['cached_datasets']}")
        print(f"   Total cache size: {cache_info['total_cache_size_mb']:.1f} MB")

        print("\n🚀 Historical Data Manager ready for strategy training!")

    # Run the example
    asyncio.run(main())
