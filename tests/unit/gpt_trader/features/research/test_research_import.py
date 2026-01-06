"""
Stub tests for gpt_trader.features.research module.

These tests verify the research module can be imported and basic types work.
Add more comprehensive backtesting tests as the module evolves.
"""

from gpt_trader.features.research import (
    BacktestSimulator,
    HistoricalDataPoint,
    PerformanceMetrics,
)


class TestResearchModuleImport:
    """Test that research module exports are available."""

    def test_all_exports_importable(self) -> None:
        """Verify all __all__ exports are importable."""
        import gpt_trader.features.research

        for name in gpt_trader.features.research.__all__:
            assert hasattr(gpt_trader.features.research, name), f"Missing export: {name}"

    def test_module_has_docstring(self) -> None:
        """Verify module has documentation."""
        import gpt_trader.features.research

        assert gpt_trader.features.research.__doc__ is not None
        assert "Research" in gpt_trader.features.research.__doc__


class TestHistoricalDataPoint:
    """Test HistoricalDataPoint data structure."""

    def test_data_point_is_class(self) -> None:
        """Verify HistoricalDataPoint is defined."""
        assert HistoricalDataPoint is not None


class TestBacktestSimulator:
    """Test BacktestSimulator class."""

    def test_simulator_is_class(self) -> None:
        """Verify BacktestSimulator is defined."""
        assert BacktestSimulator is not None


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""

    def test_metrics_is_class(self) -> None:
        """Verify PerformanceMetrics is defined."""
        assert PerformanceMetrics is not None
