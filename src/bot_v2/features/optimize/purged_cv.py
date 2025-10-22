"""
Purged and embargoed cross-validation for time-series strategy validation.

Implements walk-forward validation with embargo periods to prevent data leakage
in backtesting. Based on Marcos López de Prado's "Advances in Financial Machine Learning".

Key concepts:
- Purging: Remove training data that overlaps with test data in time
- Embargo: Add buffer period after test set to prevent look-ahead bias
- Walk-forward: Sequential train/test splits that respect temporal ordering
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="purged_cv")


@dataclass
class CVSplit:
    """Single train/test split with temporal boundaries."""

    split_id: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    embargo_end_idx: int  # Index where embargo ends (next train can start)

    @property
    def train_size(self) -> int:
        """Number of bars in training set."""
        return self.train_end_idx - self.train_start_idx

    @property
    def test_size(self) -> int:
        """Number of bars in test set."""
        return self.test_end_idx - self.test_start_idx

    def get_train_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract training data."""
        return data.iloc[self.train_start_idx : self.train_end_idx].copy()

    def get_test_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract test data."""
        return data.iloc[self.test_start_idx : self.test_end_idx].copy()


class PurgedWalkForwardCV:
    """
    Walk-forward cross-validation with purging and embargo.

    Creates sequential train/test splits where:
    1. Training data always precedes test data (no look-ahead)
    2. Embargo period prevents using data immediately after test (no leakage)
    3. Next training set starts after embargo ends

    Example timeline:
    ```
    [Train 1] [Test 1] [Embargo 1] [Train 2] [Test 2] [Embargo 2] ...
    ```

    Use cases:
    - Validate strategy parameters without overfitting
    - Estimate out-of-sample performance
    - Compare baseline vs enhanced strategies
    """

    def __init__(
        self,
        *,
        n_splits: int = 5,
        train_size: int | None = None,
        test_size: int | None = None,
        embargo_pct: float = 0.01,  # 1% embargo by default
        min_train_size: int = 100,
        min_test_size: int = 20,
    ):
        """
        Initialize purged walk-forward CV.

        Args:
            n_splits: Number of train/test splits
            train_size: Fixed training size in bars (None = expanding window)
            test_size: Test size in bars (None = auto-calculated)
            embargo_pct: Embargo as percentage of test size (0.01 = 1%)
            min_train_size: Minimum bars required for training
            min_test_size: Minimum bars required for testing
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size

    def split(self, data: pd.DataFrame) -> list[CVSplit]:
        """
        Generate train/test splits with embargo.

        Args:
            data: Time-series data (must be sorted by time)

        Returns:
            List of CVSplit objects
        """
        n_samples = len(data)

        if n_samples < self.min_train_size + self.min_test_size:
            raise ValueError(
                f"Insufficient data: {n_samples} bars (need >= {self.min_train_size + self.min_test_size})"
            )

        # Calculate test size if not provided
        test_size = self.test_size
        if test_size is None:
            # Auto-calculate: divide remaining data into n_splits test sets
            # Reserve min_train_size for first training set
            available = n_samples - self.min_train_size
            test_size = max(self.min_test_size, available // self.n_splits)

        # Calculate embargo size
        embargo_size = max(1, int(test_size * self.embargo_pct))

        logger.info(
            "PurgedWalkForwardCV setup | n_splits=%d | test_size=%d | embargo_size=%d | total_bars=%d",
            self.n_splits,
            test_size,
            embargo_size,
            n_samples,
        )

        splits: list[CVSplit] = []
        train_start = 0

        for split_id in range(self.n_splits):
            # Test set starts after training
            if self.train_size is None:
                # Expanding window: train from start to current point
                train_end = train_start + self.min_train_size + split_id * test_size
            else:
                # Rolling window: fixed-size training window
                train_end = train_start + self.train_size

            test_start = train_end
            test_end = test_start + test_size
            embargo_end = test_end + embargo_size

            # Check if we have enough data for this split
            if test_end > n_samples:
                logger.warning(
                    "Insufficient data for split %d | test_end=%d > n_samples=%d",
                    split_id,
                    test_end,
                    n_samples,
                )
                break

            if train_end - train_start < self.min_train_size:
                logger.warning(
                    "Training set too small for split %d | size=%d < min=%d",
                    split_id,
                    train_end - train_start,
                    self.min_train_size,
                )
                break

            split = CVSplit(
                split_id=split_id,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
                embargo_end_idx=embargo_end,
            )

            splits.append(split)

            logger.debug(
                "Split %d | train=[%d:%d] (%d bars) | test=[%d:%d] (%d bars) | embargo_end=%d",
                split_id,
                train_start,
                train_end,
                split.train_size,
                test_start,
                test_end,
                split.test_size,
                embargo_end,
            )

            # Next training set starts after embargo
            if self.train_size is None:
                # Expanding window: keep same start
                train_start = 0
            else:
                # Rolling window: slide forward past embargo
                train_start = embargo_end

        if not splits:
            raise ValueError(f"Could not create any valid splits (data too small: {n_samples} bars)")

        logger.info("Created %d walk-forward splits", len(splits))
        return splits

    def get_split_dates(self, data: pd.DataFrame, timestamp_col: str = "timestamp") -> list[dict]:
        """
        Get human-readable date ranges for each split.

        Args:
            data: Time-series data with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            List of dicts with date ranges for each split
        """
        splits = self.split(data)
        split_dates = []

        for split in splits:
            train_data = split.get_train_data(data)
            test_data = split.get_test_data(data)

            split_dates.append(
                {
                    "split_id": split.split_id,
                    "train_start": train_data[timestamp_col].iloc[0],
                    "train_end": train_data[timestamp_col].iloc[-1],
                    "test_start": test_data[timestamp_col].iloc[0],
                    "test_end": test_data[timestamp_col].iloc[-1],
                    "train_bars": split.train_size,
                    "test_bars": split.test_size,
                }
            )

        return split_dates


class AnchoredWalkForwardCV(PurgedWalkForwardCV):
    """
    Anchored walk-forward: training always starts from beginning (expanding window).

    This is equivalent to PurgedWalkForwardCV with train_size=None (the default).
    Provided as explicit class for clarity.

    Timeline:
    ```
    [-----Train 1-----] [Test 1] [Embargo]
    [---------Train 2---------] [Test 2] [Embargo]
    [---------------Train 3---------------] [Test 3] [Embargo]
    ```
    """

    def __init__(
        self,
        *,
        n_splits: int = 5,
        test_size: int | None = None,
        embargo_pct: float = 0.01,
        min_train_size: int = 100,
        min_test_size: int = 20,
    ):
        super().__init__(
            n_splits=n_splits,
            train_size=None,  # Expanding window
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
        )


class RollingWalkForwardCV(PurgedWalkForwardCV):
    """
    Rolling walk-forward: training window slides forward with fixed size.

    Timeline:
    ```
    [Train 1] [Test 1] [Embargo]
              [Train 2] [Test 2] [Embargo]
                        [Train 3] [Test 3] [Embargo]
    ```
    """

    def __init__(
        self,
        *,
        n_splits: int = 5,
        train_size: int = 200,  # Fixed window
        test_size: int | None = None,
        embargo_pct: float = 0.01,
        min_train_size: int = 100,
        min_test_size: int = 20,
    ):
        super().__init__(
            n_splits=n_splits,
            train_size=train_size,  # Fixed window
            test_size=test_size,
            embargo_pct=embargo_pct,
            min_train_size=min_train_size,
            min_test_size=min_test_size,
        )


def validate_temporal_ordering(data: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    """
    Verify data is sorted by timestamp.

    Args:
        data: DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Raises:
        ValueError: If data is not sorted
    """
    if timestamp_col not in data.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")

    timestamps = data[timestamp_col]
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"Data is not sorted by {timestamp_col} (required for time-series CV)")


__all__ = [
    "CVSplit",
    "PurgedWalkForwardCV",
    "AnchoredWalkForwardCV",
    "RollingWalkForwardCV",
    "validate_temporal_ordering",
]
