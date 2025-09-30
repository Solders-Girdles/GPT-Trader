"""
Data fetching and validation for backtesting.
"""

import logging
from datetime import datetime
from typing import Any, cast

import pandas as pd
from bot_v2.config import get_config
from bot_v2.data_providers import get_data_provider
from bot_v2.errors import DataError, NetworkError, TimeoutError
from bot_v2.errors.handler import RecoveryStrategy, get_error_handler
from bot_v2.validation import DateValidator, SymbolValidator, validate_inputs

logger = logging.getLogger(__name__)


@validate_inputs(symbol=SymbolValidator(), start=DateValidator(), end=DateValidator())
def fetch_historical_data(
    symbol: str, start: datetime, end: datetime, interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical market data for backtesting.

    Args:
        symbol: Stock symbol
        start: Start date
        end: End date
        interval: Data interval

    Returns:
        DataFrame with OHLCV data

    Raises:
        DataError: If data fetching fails or data is invalid
        NetworkError: If network issues occur
        TimeoutError: If request times out
    """
    logger.info(
        f"Fetching historical data for {symbol} from {start.date()} to {end.date()}",
        extra={
            "symbol": symbol,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "interval": interval,
        },
    )
    # Get configuration for retry settings
    get_config("backtest")
    error_handler = get_error_handler()

    # Fetch data with retry logic
    try:
        data = cast(
            pd.DataFrame,
            error_handler.with_retry(
                _fetch_data_with_validation,
                symbol,
                start,
                end,
                interval,
                recovery_strategy=RecoveryStrategy.RETRY,
            ),
        )
    except Exception as e:
        # Determine error type and wrap appropriately
        if "timeout" in str(e).lower():
            raise TimeoutError(
                f"Timeout fetching data for {symbol}",
                operation="fetch_historical_data",
                timeout_seconds=30,
                context={"symbol": symbol, "date_range": f"{start.date()} to {end.date()}"},
            ) from e
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            raise NetworkError(
                f"Network error fetching data for {symbol}",
                url=f"data_provider://{symbol}",
                context={"symbol": symbol, "date_range": f"{start.date()} to {end.date()}"},
            ) from e
        else:
            raise DataError(
                f"Failed to fetch historical data for {symbol}",
                symbol=symbol,
                context={
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "interval": interval,
                    "error": str(e),
                },
            ) from e

    logger.info(
        f"Successfully fetched {len(data)} data points for {symbol}",
        extra={
            "symbol": symbol,
            "rows_fetched": len(data),
            "date_range": f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else "empty",
        },
    )

    return data


def _fetch_data_with_validation(
    symbol: str, start: datetime, end: datetime, interval: str = "1d"
) -> pd.DataFrame:
    """
    Internal function to fetch and validate data.

    Args:
        symbol: Stock symbol
        start: Start date
        end: End date
        interval: Data interval

    Returns:
        Validated DataFrame with OHLCV data

    Raises:
        DataError: If data issues are found
    """
    # Get data provider
    try:
        provider = get_data_provider()
    except Exception as e:
        raise DataError(
            "Failed to get data provider", symbol=symbol, context={"provider_error": str(e)}
        ) from e

    provider_typed: Any = provider

    # Fetch raw data (support providers with either start/end or period API)
    try:
        fetched: Any
        try:
            # Prefer explicit date range if supported
            fetched = provider_typed.get_historical_data(
                symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
            )
        except TypeError:
            # Fallback: compute days to period string
            days = max(1, (end - start).days)
            period = f"{days}d"
            fetched = provider_typed.get_historical_data(symbol, period=period, interval=interval)
    except Exception as e:
        raise DataError(
            f"Data provider failed for {symbol}",
            symbol=symbol,
            context={"provider_type": type(provider).__name__, "provider_error": str(e)},
        ) from e

    if not isinstance(fetched, pd.DataFrame):
        raise DataError(
            "Data provider returned unexpected payload",
            symbol=symbol,
            context={
                "provider_type": type(provider).__name__,
                "payload_type": type(fetched).__name__,
            },
        )

    assert isinstance(fetched, pd.DataFrame)
    data: pd.DataFrame = fetched

    # Check if data is empty
    if data.empty:
        raise DataError(
            f"No data available for {symbol} in the specified date range",
            symbol=symbol,
            context={
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "interval": interval,
            },
        )

    # Validate column presence
    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        raise DataError(
            f"Missing required columns: {missing_columns}",
            symbol=symbol,
            context={"available_columns": list(data.columns), "missing_columns": missing_columns},
        )

    # Standardize columns and convert to lowercase
    try:
        data = data[expected_columns].copy()
        data.columns = data.columns.str.lower()
    except Exception as e:
        raise DataError(
            f"Failed to standardize columns for {symbol}",
            symbol=symbol,
            context={"columns": list(data.columns), "standardization_error": str(e)},
        ) from e

    # Validate data quality using local validation
    try:
        from bot_v2.features.backtest.validation import validate_data as validate_market_data

        if not validate_market_data(data):
            raise DataError(
                f"Data quality validation failed for {symbol}",
                symbol=symbol,
                context={
                    "data_shape": data.shape,
                    "date_range": (
                        f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else "empty"
                    ),
                },
            )
    except Exception as e:
        # If it's already a DataError, re-raise
        if isinstance(e, DataError):
            raise
        # Otherwise wrap in DataError
        raise DataError(
            f"Data validation error for {symbol}",
            symbol=symbol,
            context={"validation_error": str(e)},
        ) from e

    # Additional quality checks
    _perform_additional_quality_checks(data, symbol)

    return data


def _perform_additional_quality_checks(data: pd.DataFrame, symbol: str) -> None:
    """
    Perform additional data quality checks.

    Args:
        data: Market data to check
        symbol: Stock symbol for context

    Raises:
        DataError: If quality issues are found
    """
    config = get_config("backtest")

    # Check minimum data points
    min_points = config.get("min_data_points", 30)
    if len(data) < min_points:
        raise DataError(
            f"Insufficient data points for {symbol}: need {min_points}, got {len(data)}",
            symbol=symbol,
            context={
                "required_points": min_points,
                "available_points": len(data),
                "shortfall": min_points - len(data),
            },
        )

    # Check data requirements from config
    data_req = config.get("data_requirements", {})

    # Check minimum volume if specified
    min_volume = data_req.get("min_volume")
    if min_volume and (data["volume"] < min_volume).any():
        low_volume_count = (data["volume"] < min_volume).sum()
        logger.warning(
            f"Low volume periods detected for {symbol}: {low_volume_count} periods below {min_volume}",
            extra={
                "symbol": symbol,
                "low_volume_count": low_volume_count,
                "min_volume_threshold": min_volume,
            },
        )

    # Check price range if specified
    min_price = data_req.get("min_price")
    max_price = data_req.get("max_price")

    if min_price and (data["close"] < min_price).any():
        low_price_count = (data["close"] < min_price).sum()
        logger.warning(
            f"Low price periods detected for {symbol}: {low_price_count} periods below ${min_price}",
            extra={
                "symbol": symbol,
                "low_price_count": low_price_count,
                "min_price_threshold": min_price,
            },
        )

    if max_price and (data["close"] > max_price).any():
        high_price_count = (data["close"] > max_price).sum()
        logger.warning(
            f"High price periods detected for {symbol}: {high_price_count} periods above ${max_price}",
            extra={
                "symbol": symbol,
                "high_price_count": high_price_count,
                "max_price_threshold": max_price,
            },
        )

    # Check for data gaps (missing trading days)
    if len(data) > 1:
        date_diffs = data.index.to_series().diff().dt.days
        large_gaps = date_diffs > 7  # More than a week gap
        if large_gaps.any():
            gap_count = large_gaps.sum()
            logger.warning(
                f"Large data gaps detected for {symbol}: {gap_count} gaps > 7 days",
                extra={"symbol": symbol, "gap_count": gap_count, "max_gap_days": date_diffs.max()},
            )


def get_data_quality_report(data: pd.DataFrame, symbol: str) -> dict:
    """
    Generate a data quality report.

    Args:
        data: Market data
        symbol: Stock symbol

    Returns:
        Dictionary with quality metrics
    """
    try:
        issues: list[str] = []
        report: dict[str, Any] = {
            "symbol": symbol,
            "total_periods": len(data),
            "date_range": {
                "start": data.index[0].isoformat() if len(data) > 0 else None,
                "end": data.index[-1].isoformat() if len(data) > 0 else None,
            },
            "null_counts": data.isnull().sum().to_dict(),
            "price_stats": {
                "min_close": float(data["close"].min()) if len(data) > 0 else None,
                "max_close": float(data["close"].max()) if len(data) > 0 else None,
                "avg_volume": float(data["volume"].mean()) if len(data) > 0 else None,
            },
            "data_quality_issues": issues,
        }

        # Check for common issues
        if data.isnull().any().any():
            issues.append("null_values_present")

        if len(data) > 1:
            # Check for duplicate dates
            if data.index.duplicated().any():
                issues.append("duplicate_dates")

            # Check for extreme returns
            returns = data["close"].pct_change().abs()
            if (returns > 0.5).any():
                issues.append("extreme_returns")

        return report

    except Exception as e:
        logger.warning(f"Failed to generate quality report for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e),
            "data_quality_issues": ["report_generation_failed"],
        }
