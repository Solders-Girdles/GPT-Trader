"""
Local data quality checking.

Complete isolation - no external dependencies.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from bot_v2.features.data.types import DataQuality


class DataQualityChecker:
    """Checks data quality."""

    def __init__(self):
        """Initialize quality checker."""
        self.quality_history: list[DataQuality] = []
        self.max_history = 100

    def check_quality(self, data: pd.DataFrame) -> DataQuality:
        """
        Check quality of data.

        Args:
            data: DataFrame to check

        Returns:
            DataQuality metrics
        """
        if data.empty:
            return DataQuality(completeness=0.0, accuracy=0.0, consistency=0.0, timeliness=0.0)

        # Check completeness (non-null values)
        total_values = data.size
        non_null_values = data.count().sum()
        completeness = non_null_values / total_values if total_values > 0 else 0

        # Check accuracy (valid price relationships for OHLC data)
        accuracy = self._check_accuracy(data)

        # Check consistency (price movements within reasonable bounds)
        consistency = self._check_consistency(data)

        # Check timeliness (how recent the data is)
        timeliness = self._check_timeliness(data)

        quality = DataQuality(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
        )

        # Store in history
        self.quality_history.append(quality)
        if len(self.quality_history) > self.max_history:
            self.quality_history.pop(0)

        return quality

    def _check_accuracy(self, data: pd.DataFrame) -> float:
        """
        Check data accuracy (OHLC relationships).

        Args:
            data: DataFrame to check

        Returns:
            Accuracy score (0-1)
        """
        if "high" not in data.columns or "low" not in data.columns:
            return 1.0  # Not OHLC data, assume accurate

        valid_checks = []

        # Check high >= low
        if "high" in data.columns and "low" in data.columns:
            valid_checks.append((data["high"] >= data["low"]).mean())

        # Check high >= open and close
        if "high" in data.columns and "open" in data.columns:
            valid_checks.append((data["high"] >= data["open"]).mean())
        if "high" in data.columns and "close" in data.columns:
            valid_checks.append((data["high"] >= data["close"]).mean())

        # Check low <= open and close
        if "low" in data.columns and "open" in data.columns:
            valid_checks.append((data["low"] <= data["open"]).mean())
        if "low" in data.columns and "close" in data.columns:
            valid_checks.append((data["low"] <= data["close"]).mean())

        # Check for positive prices
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                valid_checks.append((data[col] > 0).mean())

        return np.mean(valid_checks) if valid_checks else 1.0

    def _check_consistency(self, data: pd.DataFrame) -> float:
        """
        Check data consistency (reasonable price movements).

        Args:
            data: DataFrame to check

        Returns:
            Consistency score (0-1)
        """
        if "close" not in data.columns:
            return 1.0

        # Calculate daily returns
        returns = data["close"].pct_change().dropna()

        if len(returns) == 0:
            return 1.0

        # Check for outliers (returns > 50% in a day are suspicious)
        reasonable_returns = (returns.abs() < 0.5).mean()

        # Check for zero variance (suspicious if no movement at all)
        has_variance = returns.std() > 0

        # Check for gaps in data
        if isinstance(data.index, pd.DatetimeIndex):
            expected_days = pd.bdate_range(start=data.index.min(), end=data.index.max())
            coverage = len(data) / len(expected_days) if len(expected_days) > 0 else 1.0
        else:
            coverage = 1.0

        consistency = reasonable_returns * 0.5 + float(has_variance) * 0.25 + coverage * 0.25

        return consistency

    def _check_timeliness(self, data: pd.DataFrame) -> float:
        """
        Check data timeliness.

        Args:
            data: DataFrame to check

        Returns:
            Timeliness score (0-1)
        """
        if data.empty:
            return 0.0

        # Get most recent data point
        if isinstance(data.index, pd.DatetimeIndex):
            latest_date = data.index.max()
            age_days = (datetime.now() - latest_date).days

            # Score based on age
            if age_days == 0:
                return 1.0  # Today's data
            elif age_days <= 1:
                return 0.95  # Yesterday's data
            elif age_days <= 7:
                return 0.8  # Within a week
            elif age_days <= 30:
                return 0.6  # Within a month
            elif age_days <= 90:
                return 0.4  # Within 3 months
            else:
                return max(0.1, 1.0 - (age_days / 365))  # Older data

        return 0.5  # Default if not datetime index

    def validate_ohlcv(self, data: pd.DataFrame) -> list[str]:
        """
        Validate OHLCV data and return list of issues.

        Args:
            data: DataFrame to validate

        Returns:
            List of validation issues
        """
        issues = []

        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")

        # Check for nulls
        null_counts = data[data.columns.intersection(required_columns)].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

        # Check OHLC relationships
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            invalid_high_low = (data["high"] < data["low"]).sum()
            if invalid_high_low > 0:
                issues.append(f"High < Low in {invalid_high_low} rows")

            invalid_high = ((data["high"] < data["open"]) | (data["high"] < data["close"])).sum()
            if invalid_high > 0:
                issues.append(f"High below Open/Close in {invalid_high} rows")

            invalid_low = ((data["low"] > data["open"]) | (data["low"] > data["close"])).sum()
            if invalid_low > 0:
                issues.append(f"Low above Open/Close in {invalid_low} rows")

        # Check for negative prices
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Negative {col} prices in {negative_count} rows")

        # Check for negative volume
        if "volume" in data.columns:
            negative_volume = (data["volume"] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume in {negative_volume} rows")

        return issues

    def get_quality_trend(self) -> dict[str, float]:
        """
        Get quality trend from history.

        Returns:
            Dict of metric -> average score
        """
        if not self.quality_history:
            return {"completeness": 0.0, "accuracy": 0.0, "consistency": 0.0, "timeliness": 0.0}

        return {
            "completeness": np.mean([q.completeness for q in self.quality_history]),
            "accuracy": np.mean([q.accuracy for q in self.quality_history]),
            "consistency": np.mean([q.consistency for q in self.quality_history]),
            "timeliness": np.mean([q.timeliness for q in self.quality_history]),
        }
