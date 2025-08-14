from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.dataflow.validate import adjust_to_adjclose, validate_daily_bars
from bot.logging import get_logger
from bot.utils.validation import SymbolValidator, DateValidator, DataFrameValidator

logger = get_logger("pipeline")


@dataclass
class DataQualityMetrics:
    """Data quality metrics for pipeline monitoring."""

    total_symbols_requested: int = 0
    symbols_loaded_successfully: int = 0
    symbols_failed: int = 0
    total_data_points: int = 0
    missing_data_points: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    validation_errors: int = 0
    adjustment_applied: int = 0
    avg_load_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_symbols_requested == 0:
            return 0.0
        return (self.symbols_loaded_successfully / self.total_symbols_requested) * 100

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.cache_hits / total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "total_requested": self.total_symbols_requested,
            "loaded_successfully": self.symbols_loaded_successfully,
            "failed": self.symbols_failed,
            "success_rate_pct": round(self.success_rate, 2),
            "cache_hit_rate_pct": round(self.cache_hit_rate, 2),
            "validation_errors": self.validation_errors,
            "adjustments_applied": self.adjustment_applied,
            "avg_load_time_ms": round(self.avg_load_time_ms, 2),
            "error_count": len(self.errors),
        }


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""

    # Cache settings
    use_cache: bool = True
    cache_ttl_hours: int = 24

    # Validation settings
    strict_validation: bool = True
    min_data_points: int = 10

    # Performance settings
    timeout_seconds: float = 30.0
    retry_attempts: int = 3

    # Data adjustment settings
    apply_adjustments: bool = True

    # Error handling
    fail_on_missing_symbols: bool = False
    max_missing_data_pct: float = 10.0  # Max % of missing data points allowed


class DataPipeline:
    """Unified data pipeline for backtesting and live trading.

    Features:
    - Multi-source data fetching with fallback
    - Intelligent caching with TTL
    - Comprehensive data validation
    - Quality metrics and reporting
    - Error handling with graceful degradation
    - Support for batch processing
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.source = YFinanceSource()
        self._cache: Dict[str, tuple[pd.DataFrame, datetime]] = {}
        self.metrics = DataQualityMetrics()

        logger.info(f"DataPipeline initialized with config: {self.config}")

    def fetch_and_validate(
        self, symbols: list[str], start: datetime, end: datetime, use_cache: Optional[bool] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch and validate data for multiple symbols.

        Args:
            symbols: List of stock symbols to fetch
            start: Start date for data
            end: End date for data
            use_cache: Override global cache setting

        Returns:
            Dict of symbol -> validated OHLCV DataFrame

        Raises:
            ValueError: If no valid symbols provided or critical validation fails
        """
        start_time = time.time()

        # Validate inputs
        validated_symbols = self._validate_symbols(symbols)
        start_date, end_date = self._validate_date_range(start, end)

        use_cache_flag = use_cache if use_cache is not None else self.config.use_cache

        # Reset metrics for this batch
        self.metrics = DataQualityMetrics()
        self.metrics.total_symbols_requested = len(validated_symbols)

        result = {}
        load_times = []

        logger.info(
            f"Fetching data for {len(validated_symbols)} symbols from {start_date.date()} to {end_date.date()}"
        )

        for symbol in validated_symbols:
            symbol_start_time = time.time()

            try:
                df = self._fetch_symbol_data(symbol, start_date, end_date, use_cache_flag)

                if df is not None and not df.empty:
                    result[symbol] = df
                    self.metrics.symbols_loaded_successfully += 1
                    self.metrics.total_data_points += len(df)

                    load_time = (time.time() - symbol_start_time) * 1000
                    load_times.append(load_time)

                    logger.debug(f"Successfully loaded {len(df)} data points for {symbol}")
                else:
                    self._handle_symbol_failure(symbol, "No data returned")

            except Exception as e:
                error_msg = f"Failed to load {symbol}: {str(e)}"
                self._handle_symbol_failure(symbol, error_msg)
                logger.warning(error_msg)

        # Calculate final metrics
        if load_times:
            self.metrics.avg_load_time_ms = sum(load_times) / len(load_times)

        total_time = (time.time() - start_time) * 1000

        # Log final results
        metrics_dict = self.metrics.to_dict()
        logger.info(f"Pipeline completed in {total_time:.2f}ms: {metrics_dict}")

        # Check if we meet minimum requirements
        if self.config.fail_on_missing_symbols and self.metrics.symbols_failed > 0:
            raise ValueError(f"Failed to load {self.metrics.symbols_failed} symbols")

        if not result:
            raise ValueError("No data could be loaded for any symbol")

        return result

    def _fetch_symbol_data(
        self, symbol: str, start_date: datetime, end_date: datetime, use_cache: bool
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol with caching and validation."""

        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"

        # Check cache first
        if use_cache and cache_key in self._cache:
            cached_df, cache_time = self._cache[cache_key]

            # Check if cache is still valid
            cache_age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            if cache_age_hours < self.config.cache_ttl_hours:
                self.metrics.cache_hits += 1
                logger.debug(f"Cache hit for {symbol} (age: {cache_age_hours:.1f}h)")
                return cached_df.copy()
            else:
                # Cache expired, remove it
                del self._cache[cache_key]
                logger.debug(f"Cache expired for {symbol} (age: {cache_age_hours:.1f}h)")

        self.metrics.cache_misses += 1

        # Fetch fresh data
        try:
            df = self.source.get_daily_bars(
                symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

            if df.empty:
                logger.warning(f"Empty dataset returned for {symbol}")
                return None

            # Apply adjustments if configured
            df_processed = df.copy()
            if self.config.apply_adjustments:
                df_processed, was_adjusted = adjust_to_adjclose(df_processed)
                if was_adjusted:
                    self.metrics.adjustment_applied += 1
                    logger.debug(f"Applied price adjustments for {symbol}")

            # Validate data
            try:
                validate_daily_bars(df_processed, symbol)

                # Additional quality checks
                self._perform_quality_checks(df_processed, symbol)

            except ValueError as e:
                self.metrics.validation_errors += 1

                if self.config.strict_validation:
                    raise
                else:
                    logger.warning(f"Validation warning for {symbol}: {e}")
                    # Continue with potentially problematic data

            # Cache successful result
            if use_cache:
                self._cache[cache_key] = (df_processed.copy(), datetime.now())
                logger.debug(f"Cached data for {symbol}")

            return df_processed

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def _validate_symbols(self, symbols: list[str]) -> list[str]:
        """Validate and normalize symbol list."""
        if not symbols:
            raise ValueError("No symbols provided")

        try:
            return SymbolValidator.validate_symbols(symbols)
        except ValueError as e:
            logger.error(f"Symbol validation failed: {e}")
            raise

    def _validate_date_range(self, start: datetime, end: datetime) -> tuple[datetime, datetime]:
        """Validate date range."""
        if start >= end:
            raise ValueError(f"Start date ({start.date()}) must be before end date ({end.date()})")

        # Check for reasonable date range (not too far in the future)
        if start > datetime.now():
            raise ValueError(f"Start date ({start.date()}) cannot be in the future")

        return start, end

    def _perform_quality_checks(self, df: pd.DataFrame, symbol: str) -> None:
        """Perform additional data quality checks."""

        # Check minimum data points
        if len(df) < self.config.min_data_points:
            raise ValueError(
                f"{symbol}: insufficient data points ({len(df)} < {self.config.min_data_points})"
            )

        # Check for excessive missing values
        required_cols = ["Open", "High", "Low", "Close"]
        available_cols = [col for col in required_cols if col in df.columns]

        if available_cols:
            missing_pct = (
                df[available_cols].isna().sum().sum() / (len(df) * len(available_cols))
            ) * 100

            if missing_pct > self.config.max_missing_data_pct:
                self.metrics.missing_data_points += int(df[available_cols].isna().sum().sum())
                raise ValueError(
                    f"{symbol}: excessive missing data ({missing_pct:.1f}% > {self.config.max_missing_data_pct}%)"
                )

        # Check for unrealistic values (basic sanity)
        price_cols = [col for col in ["Open", "High", "Low", "Close"] if col in df.columns]
        if price_cols:
            for col in price_cols:
                if (df[col] <= 0).any():
                    raise ValueError(f"{symbol}: non-positive prices detected in {col}")

                if (df[col] > 100000).any():  # Arbitrary high threshold
                    raise ValueError(f"{symbol}: unrealistically high prices in {col}")

    def _handle_symbol_failure(self, symbol: str, error_msg: str) -> None:
        """Handle symbol loading failure."""
        self.metrics.symbols_failed += 1
        self.metrics.errors.append(f"{symbol}: {error_msg}")

        if self.config.fail_on_missing_symbols:
            logger.error(error_msg)
        else:
            logger.warning(error_msg)

    def warm_cache(
        self, symbols: list[str], start: datetime, end: datetime, quiet: bool = False
    ) -> Dict[str, bool]:
        """Pre-load data into cache for given symbols.

        Args:
            symbols: List of symbols to warm
            start: Start date
            end: End date
            quiet: Suppress logging

        Returns:
            Dict of symbol -> success status
        """
        if not quiet:
            logger.info(f"Warming cache for {len(symbols)} symbols")

        results = {}

        for symbol in symbols:
            try:
                df = self._fetch_symbol_data(symbol, start, end, use_cache=True)
                results[symbol] = df is not None and not df.empty

                if not quiet:
                    if results[symbol]:
                        logger.info(f"Cache warmed for {symbol}: {len(df)} data points")
                    else:
                        logger.warning(f"Failed to warm cache for {symbol}")

            except Exception as e:
                results[symbol] = False
                if not quiet:
                    logger.warning(f"Error warming cache for {symbol}: {e}")

        return results

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cache for specific symbol or all symbols.

        Args:
            symbol: Symbol to clear, or None for all
        """
        if symbol is None:
            cache_size = len(self._cache)
            self._cache.clear()
            logger.info(f"Cleared entire cache ({cache_size} entries)")
        else:
            # Remove all cache entries for this symbol
            keys_to_remove = [key for key in self._cache.keys() if key.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for {symbol} ({len(keys_to_remove)} entries)")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        total_entries = len(self._cache)
        total_memory_mb = 0

        # Estimate memory usage (rough calculation)
        for df, _ in self._cache.values():
            total_memory_mb += df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Cache age distribution
        now = datetime.now()
        age_buckets = {"<1h": 0, "1-6h": 0, "6-24h": 0, ">24h": 0}

        for _, cache_time in self._cache.values():
            age_hours = (now - cache_time).total_seconds() / 3600
            if age_hours < 1:
                age_buckets["<1h"] += 1
            elif age_hours < 6:
                age_buckets["1-6h"] += 1
            elif age_hours < 24:
                age_buckets["6-24h"] += 1
            else:
                age_buckets[">24h"] += 1

        return {
            "total_entries": total_entries,
            "estimated_memory_mb": round(total_memory_mb, 2),
            "age_distribution": age_buckets,
            "ttl_hours": self.config.cache_ttl_hours,
        }

    def get_metrics(self) -> DataQualityMetrics:
        """Get current quality metrics."""
        return self.metrics

    def health_check(self, test_symbol: str = "AAPL") -> Dict[str, Any]:
        """Perform health check on the pipeline.

        Args:
            test_symbol: Symbol to use for testing

        Returns:
            Health check results
        """
        health = {"status": "healthy", "errors": [], "warnings": [], "tests": {}}

        # Test data source connectivity
        try:
            test_start = datetime(2023, 1, 1)
            test_end = datetime(2023, 1, 31)

            start_time = time.time()
            result = self.fetch_and_validate([test_symbol], test_start, test_end, use_cache=False)
            response_time = (time.time() - start_time) * 1000

            health["tests"]["data_fetch"] = {
                "success": test_symbol in result,
                "response_time_ms": round(response_time, 2),
                "data_points": len(result.get(test_symbol, [])) if test_symbol in result else 0,
            }

            if test_symbol not in result:
                health["errors"].append(f"Failed to fetch test data for {test_symbol}")
                health["status"] = "degraded"

        except Exception as e:
            health["errors"].append(f"Data fetch test failed: {str(e)}")
            health["status"] = "unhealthy"
            health["tests"]["data_fetch"] = {"success": False, "error": str(e)}

        # Test cache functionality
        try:
            cache_info = self.get_cache_info()
            health["tests"]["cache"] = {
                "entries": cache_info["total_entries"],
                "memory_mb": cache_info["estimated_memory_mb"],
            }
        except Exception as e:
            health["warnings"].append(f"Cache check failed: {str(e)}")

        # Test validation
        try:
            test_df = pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [105.0],
                    "Low": [99.0],
                    "Close": [102.0],
                    "Volume": [1000],
                },
                index=pd.DatetimeIndex(["2023-01-01"]),
            )

            validate_daily_bars(test_df, "TEST")
            health["tests"]["validation"] = {"success": True}

        except Exception as e:
            health["errors"].append(f"Validation test failed: {str(e)}")
            health["status"] = "unhealthy"
            health["tests"]["validation"] = {"success": False, "error": str(e)}

        return health

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about configured data sources."""
        sources_info = []

        for source_config in self.multi_source_config.sources:
            source_info = {
                "type": source_config.source_type.value,
                "priority": source_config.priority,
                "enabled": source_config.enabled,
                "timeout": source_config.timeout_seconds,
                "max_retries": source_config.max_retries,
                "available": source_config.source_type in self.data_sources,
            }
            sources_info.append(source_info)

        return {
            "total_sources": len(self.multi_source_config.sources),
            "available_sources": len(self.data_sources),
            "failover_enabled": self.multi_source_config.failover_enabled,
            "parallel_fetch": self.multi_source_config.parallel_fetch,
            "primary_source": type(self.primary_source).__name__,
            "sources": sources_info,
        }
