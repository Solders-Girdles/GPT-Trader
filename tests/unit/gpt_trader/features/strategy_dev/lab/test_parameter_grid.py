"""Tests for parameter grid module."""

import pytest

from gpt_trader.features.strategy_dev.lab.parameter_grid import (
    ParameterGrid,
    ParameterRange,
    create_common_grids,
)


class TestParameterRange:
    """Tests for ParameterRange."""

    def test_discrete_values(self):
        """Test parameter with discrete values."""
        param = ParameterRange(
            name="period",
            values=[10, 20, 30, 40],
        )

        values = param.get_values()
        assert values == [10, 20, 30, 40]

    def test_range_values(self):
        """Test parameter with range."""
        param = ParameterRange(
            name="threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.2,
        )

        values = param.get_values()
        assert len(values) == 6  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

    def test_sample(self):
        """Test sampling from parameter."""
        param = ParameterRange(
            name="value",
            values=[1, 2, 3, 4, 5],
        )

        # Sample should return a value from the list
        for _ in range(10):
            sample = param.sample()
            assert sample in [1, 2, 3, 4, 5]

    def test_requires_values_or_range(self):
        """Test validation requires values or range."""
        with pytest.raises(ValueError):
            ParameterRange(name="invalid")


class TestParameterGrid:
    """Tests for ParameterGrid."""

    def test_add_parameter(self):
        """Test adding parameters."""
        grid = ParameterGrid()
        grid.add_parameter("period", values=[10, 20, 30])
        grid.add_parameter("threshold", min_value=0.5, max_value=0.9, step=0.1)

        assert len(grid.parameters) == 2
        assert grid.get_param_names() == ["period", "threshold"]

    def test_iteration(self):
        """Test iterating over combinations."""
        grid = ParameterGrid()
        grid.add_parameter("a", values=[1, 2])
        grid.add_parameter("b", values=[10, 20])

        combinations = list(grid)

        assert len(combinations) == 4
        assert {"a": 1, "b": 10} in combinations
        assert {"a": 2, "b": 20} in combinations

    def test_len(self):
        """Test getting total combinations."""
        grid = ParameterGrid()
        grid.add_parameter("a", values=[1, 2, 3])
        grid.add_parameter("b", values=[10, 20])

        assert len(grid) == 6

    def test_constraints(self):
        """Test parameter constraints."""
        grid = ParameterGrid()
        grid.add_parameter("fast", values=[5, 10, 15, 20])
        grid.add_parameter("slow", values=[10, 20, 30, 40])
        grid.add_constraint(lambda p: p["fast"] < p["slow"])

        combinations = list(grid)

        # All combinations should have fast < slow
        for combo in combinations:
            assert combo["fast"] < combo["slow"]

    def test_sample(self):
        """Test random sampling."""
        grid = ParameterGrid()
        grid.add_parameter("a", values=[1, 2, 3, 4, 5])
        grid.add_parameter("b", values=[10, 20, 30, 40, 50])

        samples = grid.sample(count=5, seed=42)

        assert len(samples) == 5
        for sample in samples:
            assert "a" in sample
            assert "b" in sample

    def test_latin_hypercube_sample(self):
        """Test Latin Hypercube sampling."""
        grid = ParameterGrid()
        grid.add_parameter("a", values=list(range(100)))
        grid.add_parameter("b", values=list(range(100)))

        samples = grid.latin_hypercube_sample(count=10, seed=42)

        assert len(samples) == 10

    def test_summary(self):
        """Test grid summary."""
        grid = ParameterGrid()
        grid.add_parameter("period", values=[10, 20, 30])
        grid.add_parameter("threshold", min_value=0.5, max_value=0.9, step=0.1)

        summary = grid.summary()

        assert summary["total_combinations"] > 0
        assert "period" in summary["parameters"]
        assert "threshold" in summary["parameters"]

    def test_from_parameter_dict(self):
        """Test creating from simple dict."""
        param_dict = {
            "period": [10, 20, 30],
            "threshold": [0.5, 0.6, 0.7],
        }

        grid = ParameterGrid.from_parameter_dict(param_dict)

        assert len(grid.parameters) == 2
        assert len(grid) == 9

    def test_to_from_dict(self):
        """Test serialization."""
        grid = ParameterGrid()
        grid.add_parameter("a", values=[1, 2, 3])
        grid.add_parameter("b", min_value=0.0, max_value=1.0, step=0.5)

        data = grid.to_dict()
        restored = ParameterGrid.from_dict(data)

        assert len(restored) == len(grid)


class TestCommonGrids:
    """Tests for common grid factory."""

    def test_create_common_grids(self):
        """Test creating common grids."""
        grids = create_common_grids()

        assert "moving_average" in grids
        assert "rsi" in grids
        assert "bollinger_bands" in grids
        assert "position_sizing" in grids

    def test_moving_average_constraint(self):
        """Test MA grid has fast < slow constraint."""
        grids = create_common_grids()
        ma_grid = grids["moving_average"]

        for combo in ma_grid:
            assert combo["fast_period"] < combo["slow_period"]

    def test_rsi_constraint(self):
        """Test RSI grid has oversold < overbought constraint."""
        grids = create_common_grids()
        rsi_grid = grids["rsi"]

        for combo in rsi_grid:
            assert combo["oversold"] < combo["overbought"]
