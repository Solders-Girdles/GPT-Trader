"""Unit tests for walk-forward configuration and window generation."""

from __future__ import annotations

from datetime import datetime

import pytest

from gpt_trader.features.optimize.walk_forward import (
    WalkForwardConfig,
    _months_between,
    generate_windows,
)


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = WalkForwardConfig()
        assert config.train_months == 6
        assert config.test_months == 1
        assert config.anchor_start is False
        assert config.min_trades_per_window == 10
        assert config.overlap_months == 0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = WalkForwardConfig(
            train_months=3,
            test_months=2,
            anchor_start=True,
            min_trades_per_window=20,
        )
        assert config.train_months == 3
        assert config.test_months == 2
        assert config.anchor_start is True
        assert config.min_trades_per_window == 20

    def test_invalid_train_months_raises(self) -> None:
        """Test that invalid train_months raises ValueError."""
        with pytest.raises(ValueError, match="train_months must be at least 1"):
            WalkForwardConfig(train_months=0)

    def test_invalid_test_months_raises(self) -> None:
        """Test that invalid test_months raises ValueError."""
        with pytest.raises(ValueError, match="test_months must be at least 1"):
            WalkForwardConfig(test_months=0)

    def test_overlap_exceeds_train_raises(self) -> None:
        """Test that overlap >= train_months raises ValueError."""
        with pytest.raises(ValueError, match="overlap_months must be less than"):
            WalkForwardConfig(train_months=3, overlap_months=3)

        with pytest.raises(ValueError, match="overlap_months must be less than"):
            WalkForwardConfig(train_months=3, overlap_months=5)


class TestGenerateWindows:
    """Tests for generate_windows function."""

    def test_basic_rolling_window(self) -> None:
        """Test basic rolling window generation."""
        config = WalkForwardConfig(train_months=3, test_months=1)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        assert len(windows) >= 8  # 12 - 3 - 1 = 8 possible windows

        assert windows[0].train_start == datetime(2024, 1, 1)
        assert windows[0].train_end == datetime(2024, 4, 1)
        assert windows[0].test_start == datetime(2024, 4, 1)
        assert windows[0].test_end == datetime(2024, 5, 1)

    def test_anchored_window(self) -> None:
        """Test anchored (expanding) window generation."""
        config = WalkForwardConfig(train_months=3, test_months=1, anchor_start=True)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 8, 1),
            config=config,
        )

        for window in windows:
            assert window.train_start == datetime(2024, 1, 1)

    def test_window_continuity(self) -> None:
        """Test that windows are continuous (no gaps between test end and next train start)."""
        config = WalkForwardConfig(train_months=3, test_months=1)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        for i in range(len(windows) - 1):
            current_test_end = windows[i].test_end
            next_test_start = windows[i + 1].test_start
            assert _months_between(current_test_end, next_test_start) < 0.1

    def test_insufficient_data_raises(self) -> None:
        """Test that insufficient data raises ValueError."""
        config = WalkForwardConfig(train_months=6, test_months=1)

        with pytest.raises(ValueError, match="Date range too short"):
            generate_windows(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 5, 1),
                config=config,
            )

    def test_exactly_one_window(self) -> None:
        """Test generating exactly one window."""
        config = WalkForwardConfig(train_months=3, test_months=1)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 5, 1),  # Exactly 4 months: 3 train + 1 test
            config=config,
        )

        assert len(windows) == 1
        assert windows[0].train_start == datetime(2024, 1, 1)
        assert windows[0].test_end == datetime(2024, 5, 1)

    def test_window_ids_sequential(self) -> None:
        """Test that window IDs are sequential starting from 0."""
        config = WalkForwardConfig(train_months=2, test_months=1)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        for i, window in enumerate(windows):
            assert window.window_id == i

    def test_test_follows_train(self) -> None:
        """Test that test period immediately follows train period."""
        config = WalkForwardConfig(train_months=4, test_months=2)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        for window in windows:
            assert window.test_start == window.train_end


class TestEdgeCases:
    """Test edge cases for walk-forward analysis."""

    def test_large_train_window(self) -> None:
        """Test with large train window (12 months)."""
        config = WalkForwardConfig(train_months=12, test_months=1)
        windows = generate_windows(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        assert len(windows) >= 1
        assert windows[0].train_days >= 365

    def test_large_test_window(self) -> None:
        """Test with large test window (3 months)."""
        config = WalkForwardConfig(train_months=6, test_months=3)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        assert len(windows) >= 1
        for window in windows:
            assert window.test_days >= 28 * 3  # At least ~3 months

    def test_windows_dont_exceed_end_date(self) -> None:
        """Test that no window test period exceeds the end date."""
        config = WalkForwardConfig(train_months=3, test_months=1)
        end_date = datetime(2024, 12, 15)  # Mid-month end

        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=end_date,
            config=config,
        )

        for window in windows:
            assert window.test_end <= end_date
