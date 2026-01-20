"""Tests for system monitor metrics publisher."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import gpt_trader.monitoring.system as system_module
import gpt_trader.monitoring.system.metrics as metrics_module
from gpt_trader.monitoring.system.metrics import MetricsPublisher


@pytest.fixture
def emit_metric_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_emit = MagicMock()
    monkeypatch.setattr(metrics_module, "emit_metric", mock_emit)
    return mock_emit


@pytest.fixture
def plog_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(system_module, "get_logger", lambda: mock_logger)
    return mock_logger


@pytest.fixture
def metrics_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(metrics_module, "logger", mock_logger)
    return mock_logger


@pytest.fixture
def publisher(tmp_path: Path) -> MetricsPublisher:
    return MetricsPublisher(
        event_store=MagicMock(),
        bot_id="bot-123",
        profile="TEST",
        base_dir=tmp_path,
    )


class TestMetricsPublisherInit:
    """Tests for MetricsPublisher initialization."""

    def test_stores_config(self) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="PROD",
        )
        assert publisher._bot_id == "bot-123"
        assert publisher._profile == "PROD"
        assert publisher._event_store is event_store


class TestMetricsPublisherPublish:
    """Tests for MetricsPublisher.publish method."""

    def test_publishes_to_event_store(
        self,
        emit_metric_mock: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        publisher: MetricsPublisher,
    ) -> None:
        monkeypatch.setattr(publisher, "_log_update", MagicMock())
        publisher.publish({"cycle": 1, "pnl": 100.0})

        emit_metric_mock.assert_called_once()
        call_args = emit_metric_mock.call_args
        assert call_args[0][1] == "bot-123"
        assert call_args[0][2]["event_type"] == "cycle_metrics"
        assert call_args[0][2]["cycle"] == 1

    def test_writes_snapshot(
        self,
        emit_metric_mock: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        publisher: MetricsPublisher,
    ) -> None:
        monkeypatch.setattr(publisher, "_log_update", MagicMock())
        publisher.publish({"cycle": 1, "pnl": 100.0})

        # Check file was created
        metrics_path = publisher._base_dir / publisher._profile / "metrics.json"
        assert metrics_path.exists()

        with metrics_path.open() as f:
            data = json.load(f)
            assert data["cycle"] == 1
            assert data["pnl"] == 100.0


class TestMetricsPublisherWriteSnapshot:
    """Tests for MetricsPublisher._write_snapshot method."""

    def test_creates_directories(self, tmp_path: Path, publisher: MetricsPublisher) -> None:
        publisher._write_snapshot({"test": "data"})

        expected_dir = tmp_path / "TEST"
        assert expected_dir.exists()

    def test_handles_write_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        metrics_logger: MagicMock,
        publisher: MetricsPublisher,
    ) -> None:
        # Mock _target_dirs to return an unwritable path
        monkeypatch.setattr(publisher, "_target_dirs", lambda: [Path("/nonexistent/path")])
        # Should not raise, just log
        publisher._write_snapshot({"test": "data"})
        metrics_logger.debug.assert_called_once()


class TestMetricsPublisherLogUpdate:
    """Tests for MetricsPublisher._log_update method."""

    def test_logs_metrics(self, plog_mock: MagicMock, publisher: MetricsPublisher) -> None:
        publisher._log_update({"cycle": 1, "pnl": 100.0})

        plog_mock.log_event.assert_called_once()

    def test_filters_large_fields(self, plog_mock: MagicMock, publisher: MetricsPublisher) -> None:
        publisher._log_update(
            {
                "cycle": 1,
                "pnl": 100.0,
                "positions": [{"sym": "BTC"}],  # Should be filtered
                "decisions": [{"action": "buy"}],  # Should be filtered
            }
        )

        call_kwargs = plog_mock.log_event.call_args[1]
        assert "cycle" in call_kwargs
        assert "pnl" in call_kwargs
        assert "positions" not in call_kwargs
        assert "decisions" not in call_kwargs

    def test_handles_log_errors(
        self,
        plog_mock: MagicMock,
        metrics_logger: MagicMock,
        publisher: MetricsPublisher,
    ) -> None:
        plog_mock.log_event.side_effect = Exception("Log error")
        publisher._log_update({"cycle": 1})
        plog_mock.log_event.assert_called_once()
        metrics_logger.debug.assert_called_once()


class TestMetricsPublisherWriteHealthStatus:
    """Tests for MetricsPublisher.write_health_status method."""

    def test_writes_health_status(self, tmp_path: Path, publisher: MetricsPublisher) -> None:
        publisher.write_health_status(ok=True, message="System healthy")

        health_path = tmp_path / "TEST" / "health.json"
        assert health_path.exists()

        with health_path.open() as f:
            data = json.load(f)
            assert data["ok"] is True
            assert data["message"] == "System healthy"
            assert data["error"] == ""
            assert "timestamp" in data

    def test_writes_unhealthy_status(self, tmp_path: Path, publisher: MetricsPublisher) -> None:
        publisher.write_health_status(ok=False, error="Connection lost")

        health_path = tmp_path / "TEST" / "health.json"

        with health_path.open() as f:
            data = json.load(f)
            assert data["ok"] is False
            assert data["error"] == "Connection lost"


class TestTargetDirs:
    """Tests for MetricsPublisher._target_dirs method."""

    def test_returns_profile_specific_path(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="CANARY",
            base_dir=tmp_path,
        )

        dirs = publisher._target_dirs()
        assert len(dirs) == 1
        assert dirs[0] == tmp_path / "CANARY"
