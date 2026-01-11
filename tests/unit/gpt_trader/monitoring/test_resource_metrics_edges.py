"""Edge-case tests for process resource metrics collection."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _StubProcess:
    def __init__(self, *, rss_bytes: float, cpu_percent: object) -> None:
        self._rss_bytes = rss_bytes
        self._cpu_percent = cpu_percent

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self._rss_bytes)

    def cpu_percent(self) -> object:
        return self._cpu_percent


class _ErrorProcess:
    def memory_info(self) -> SimpleNamespace:
        raise RuntimeError("memory_info failed")

    def cpu_percent(self) -> float:
        raise RuntimeError("cpu_percent failed")


def _build_service():
    from gpt_trader.features.live_trade.engines.system_maintenance import (
        SystemMaintenanceService,
    )

    return SystemMaintenanceService(status_reporter=MagicMock())


def _psutil_stub(process: object) -> SimpleNamespace:
    return SimpleNamespace(Process=lambda: process)


def test_collect_process_metrics_missing_psutil_returns_na() -> None:
    service = _build_service()

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("missing psutil")
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", mock_import):
        memory_usage, cpu_usage = service._collect_process_metrics()

    assert memory_usage == "N/A"
    assert cpu_usage == "N/A"


def test_collect_process_metrics_handles_read_errors() -> None:
    service = _build_service()
    psutil_stub = _psutil_stub(_ErrorProcess())

    with patch.dict("sys.modules", {"psutil": psutil_stub}):
        memory_usage, cpu_usage = service._collect_process_metrics()

    assert memory_usage == "Unknown"
    assert cpu_usage == "Unknown"


@pytest.mark.parametrize(
    ("rss_bytes", "expected_memory"),
    [
        (0.0, "0.0MB"),
        (1024.0 * 1024.0 * 1024.0, "1024.0MB"),
    ],
)
def test_collect_process_metrics_formats_memory_bounds(
    rss_bytes: float, expected_memory: str
) -> None:
    service = _build_service()
    process = _StubProcess(rss_bytes=rss_bytes, cpu_percent=12.5)
    psutil_stub = _psutil_stub(process)

    with patch.dict("sys.modules", {"psutil": psutil_stub}):
        with patch(
            "gpt_trader.features.live_trade.engines.system_maintenance.record_gauge"
        ) as mock_gauge:
            memory_usage, cpu_usage = service._collect_process_metrics()

    assert memory_usage == expected_memory
    assert cpu_usage == "12.5%"
    mock_gauge.assert_called_once_with("gpt_trader_process_memory_mb", rss_bytes / 1024 / 1024)


def test_collect_process_metrics_invalid_cpu_percent_falls_back() -> None:
    service = _build_service()
    process = _StubProcess(rss_bytes=1024.0 * 1024.0, cpu_percent=None)
    psutil_stub = _psutil_stub(process)

    with patch.dict("sys.modules", {"psutil": psutil_stub}):
        with patch("gpt_trader.features.live_trade.engines.system_maintenance.record_gauge"):
            memory_usage, cpu_usage = service._collect_process_metrics()

    assert memory_usage == "1.0MB"
    assert cpu_usage == "Unknown"


def test_collect_process_metrics_invalid_memory_value_falls_back() -> None:
    service = _build_service()
    process = _StubProcess(rss_bytes=-1.0, cpu_percent=5.0)
    psutil_stub = _psutil_stub(process)

    with patch.dict("sys.modules", {"psutil": psutil_stub}):
        with patch(
            "gpt_trader.features.live_trade.engines.system_maintenance.record_gauge"
        ) as mock_gauge:
            memory_usage, cpu_usage = service._collect_process_metrics()

    assert memory_usage == "Unknown"
    assert cpu_usage == "5.0%"
    assert not mock_gauge.called
