"""Tests for performance timing utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.utilities.performance.timing as timing_module
from gpt_trader.utilities.performance.timing import (
    PerformanceTimer,
    measure_performance,
    measure_performance_decorator,
)


def _patch_time(monkeypatch, values: list[float]) -> None:
    iterator = iter(values)
    monkeypatch.setattr(timing_module.time, "time", lambda: next(iterator))


class TestMeasurePerformance:
    """Tests for measure_performance context manager."""

    def test_records_metric(self, monkeypatch) -> None:
        mock_collector = MagicMock()
        _patch_time(monkeypatch, [1000.0, 1000.02])

        with measure_performance("test_op", collector=mock_collector):
            pass

        mock_collector.record.assert_called_once()
        metric = mock_collector.record.call_args[0][0]
        assert metric.name == "test_op"
        assert metric.unit == "s"
        assert metric.value >= 0.01

    def test_uses_provided_tags(self) -> None:
        mock_collector = MagicMock()

        with measure_performance("test_op", tags={"key": "value"}, collector=mock_collector):
            pass

        metric = mock_collector.record.call_args[0][0]
        assert metric.tags == {"key": "value"}

    def test_default_tags_empty(self) -> None:
        mock_collector = MagicMock()

        with measure_performance("test_op", collector=mock_collector):
            pass

        metric = mock_collector.record.call_args[0][0]
        assert metric.tags == {}

    def test_uses_global_collector_when_none_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_collector = MagicMock()
        mock_get_collector = MagicMock(return_value=mock_collector)
        monkeypatch.setattr(timing_module, "get_collector", mock_get_collector)

        with measure_performance("test_op"):
            pass

        mock_collector.record.assert_called_once()


class TestMeasurePerformanceDecorator:
    """Tests for measure_performance_decorator."""

    def test_calls_function(self) -> None:
        mock_collector = MagicMock()

        @measure_performance_decorator("test_func", collector=mock_collector)
        def my_func() -> int:
            return 42

        result = my_func()
        assert result == 42

    def test_records_metric(self) -> None:
        mock_collector = MagicMock()

        @measure_performance_decorator("test_func", collector=mock_collector)
        def my_func() -> int:
            return 42

        my_func()
        mock_collector.record.assert_called_once()

    def test_uses_function_name_when_no_operation_name(self) -> None:
        mock_collector = MagicMock()

        @measure_performance_decorator(collector=mock_collector)
        def my_func() -> int:
            return 42

        my_func()
        metric = mock_collector.record.call_args[0][0]
        assert "my_func" in metric.name

    def test_passes_tags(self) -> None:
        mock_collector = MagicMock()

        @measure_performance_decorator("test_func", tags={"env": "test"}, collector=mock_collector)
        def my_func() -> int:
            return 42

        my_func()
        metric = mock_collector.record.call_args[0][0]
        assert metric.tags == {"env": "test"}


class TestPerformanceTimer:
    """Tests for PerformanceTimer class."""

    def test_init(self) -> None:
        timer = PerformanceTimer("test_op")
        assert timer.operation_name == "test_op"
        assert timer.tags == {}
        assert timer.start_time is None
        assert timer.end_time is None

    def test_init_with_tags(self) -> None:
        timer = PerformanceTimer("test_op", tags={"key": "value"})
        assert timer.tags == {"key": "value"}

    def test_start(self) -> None:
        timer = PerformanceTimer("test_op")
        timer.start()
        assert timer.start_time is not None

    def test_stop_records_metric(self, monkeypatch) -> None:
        mock_collector = MagicMock()
        _patch_time(monkeypatch, [1000.0, 1000.02])
        timer = PerformanceTimer("test_op", collector=mock_collector)

        timer.start()
        duration = timer.stop()

        assert duration >= 0.01
        mock_collector.record.assert_called_once()

    def test_stop_without_start_raises(self) -> None:
        mock_collector = MagicMock()
        timer = PerformanceTimer("test_op", collector=mock_collector)

        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.stop()

    def test_context_manager_enter(self) -> None:
        mock_collector = MagicMock()
        timer = PerformanceTimer("test_op", collector=mock_collector)

        with timer:
            assert timer.start_time is not None

    def test_context_manager_exit(self, monkeypatch) -> None:
        mock_collector = MagicMock()
        _patch_time(monkeypatch, [1000.0, 1000.02])
        timer = PerformanceTimer("test_op", collector=mock_collector)

        with timer:
            pass

        assert timer.end_time is not None
        mock_collector.record.assert_called_once()
