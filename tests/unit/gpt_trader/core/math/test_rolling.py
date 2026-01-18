"""Unit tests for RollingStats."""

import math
import statistics
from decimal import Decimal

import pytest

from gpt_trader.core.math.rolling import RollingStats


class TestRollingStats:
    def test_initial_state(self):
        stats = RollingStats(window_size=5)
        assert stats.count == 0
        assert not stats.is_full
        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.std_dev == 0.0

    def test_window_filling(self):
        stats = RollingStats(window_size=3)

        stats.update(1.0)
        assert stats.count == 1
        assert not stats.is_full
        assert stats.mean == 1.0

        stats.update(2.0)
        assert stats.count == 2
        assert not stats.is_full
        assert stats.mean == 1.5

        stats.update(3.0)
        assert stats.count == 3
        assert stats.is_full
        assert stats.mean == 2.0

    def test_sliding_window(self):
        stats = RollingStats(window_size=3)

        # Fill window: [1, 2, 3]
        stats.update(1.0)
        stats.update(2.0)
        stats.update(3.0)

        assert stats.mean == 2.0

        # Slide: [2, 3, 4]
        stats.update(4.0)
        assert stats.count == 3
        assert stats.mean == 3.0
        assert math.isclose(stats.variance, 1.0)

        # Slide: [3, 4, 5]
        stats.update(5.0)
        assert stats.mean == 4.0
        assert math.isclose(stats.variance, 1.0)

    def test_accuracy_vs_statistics(self):
        data = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0]
        window_size = 4
        stats = RollingStats(window_size=window_size)

        for i, x in enumerate(data):
            stats.update(x)

            current_window = data[max(0, i - window_size + 1) : i + 1]

            assert math.isclose(stats.mean, statistics.mean(current_window))
            if len(current_window) > 1:
                assert math.isclose(stats.variance, statistics.variance(current_window))
                assert math.isclose(stats.std_dev, statistics.stdev(current_window))

    def test_decimal_support(self):
        stats = RollingStats(window_size=3)
        stats.update(Decimal("1.0"))
        stats.update(Decimal("2.0"))
        stats.update(Decimal("3.0"))

        assert stats.mean == 2.0
        assert math.isclose(stats.variance, 1.0)

    def test_clear(self):
        stats = RollingStats(window_size=3)
        stats.update(1.0)
        stats.update(2.0)
        stats.clear()

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.variance == 0.0

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            RollingStats(window_size=0)
