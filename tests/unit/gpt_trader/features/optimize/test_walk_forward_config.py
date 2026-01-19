"""Unit tests for WalkForwardConfig."""

from __future__ import annotations

import pytest

from gpt_trader.features.optimize.walk_forward import WalkForwardConfig


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
