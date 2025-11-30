"""Unit tests for IncrementalStats."""

import math
import statistics
from decimal import Decimal

from gpt_trader.core.math.incremental import IncrementalStats


class TestIncrementalStats:
    def test_initial_state(self):
        stats = IncrementalStats()
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.std_dev == 0.0
        assert stats.min_val == float("inf")
        assert stats.max_val == float("-inf")

    def test_single_value(self):
        stats = IncrementalStats()
        stats.update(10.0)

        assert stats.count == 1
        assert stats.mean == 10.0
        assert stats.variance == 0.0
        assert stats.std_dev == 0.0
        assert stats.min_val == 10.0
        assert stats.max_val == 10.0

    def test_multiple_values_accuracy(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = IncrementalStats()

        for x in data:
            stats.update(x)

        assert stats.count == 5
        assert stats.mean == statistics.mean(data)
        assert math.isclose(stats.variance, statistics.variance(data))
        assert math.isclose(stats.std_dev, statistics.stdev(data))
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0

    def test_decimal_support(self):
        data = [Decimal("1.5"), Decimal("2.5"), Decimal("3.5")]
        stats = IncrementalStats()

        for x in data:
            stats.update(x)

        assert stats.count == 3
        assert stats.mean == 2.5
        assert math.isclose(stats.variance, 1.0)

    def test_reset(self):
        stats = IncrementalStats()
        stats.update(100)
        stats.reset()

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.variance == 0.0

    def test_numerical_stability(self):
        # Welford's algorithm handles large offsets better than naive sum of squares
        large_base = 1e9
        data = [large_base + 1.0, large_base + 2.0, large_base + 3.0]
        stats = IncrementalStats()

        for x in data:
            stats.update(x)

        assert stats.mean == large_base + 2.0
        assert math.isclose(stats.variance, 1.0)
