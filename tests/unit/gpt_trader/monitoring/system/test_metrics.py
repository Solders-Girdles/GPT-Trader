"""Tests for system monitor metrics publisher."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from gpt_trader.monitoring.system.metrics import MetricsPublisher


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

    def test_publishes_to_event_store(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

        with patch("gpt_trader.monitoring.system.metrics.emit_metric") as mock_emit:
            with patch.object(publisher, "_log_update"):
                publisher.publish({"cycle": 1, "pnl": 100.0})

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][1] == "bot-123"
            assert call_args[0][2]["event_type"] == "cycle_metrics"
            assert call_args[0][2]["cycle"] == 1

    def test_writes_snapshot(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

        with patch("gpt_trader.monitoring.system.metrics.emit_metric"):
            with patch.object(publisher, "_log_update"):
                publisher.publish({"cycle": 1, "pnl": 100.0})

        # Check file was created
        metrics_path = tmp_path / "TEST" / "metrics.json"
        assert metrics_path.exists()

        with metrics_path.open() as f:
            data = json.load(f)
            assert data["cycle"] == 1
            assert data["pnl"] == 100.0


class TestMetricsPublisherWriteSnapshot:
    """Tests for MetricsPublisher._write_snapshot method."""

    def test_creates_directories(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

        publisher._write_snapshot({"test": "data"})

        expected_dir = tmp_path / "TEST"
        assert expected_dir.exists()

    def test_handles_write_errors(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

        # Mock _target_dirs to return an unwritable path
        with patch.object(publisher, "_target_dirs", return_value=[Path("/nonexistent/path")]):
            # Should not raise, just log
            publisher._write_snapshot({"test": "data"})


class TestMetricsPublisherLogUpdate:
    """Tests for MetricsPublisher._log_update method."""

    def test_logs_metrics(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

        with patch("gpt_trader.monitoring.system.get_logger") as mock_get_plog:
            mock_logger = MagicMock()
            mock_get_plog.return_value = mock_logger

            publisher._log_update({"cycle": 1, "pnl": 100.0})

            mock_logger.log_event.assert_called_once()

    def test_filters_large_fields(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

        with patch("gpt_trader.monitoring.system.get_logger") as mock_get_plog:
            mock_logger = MagicMock()
            mock_get_plog.return_value = mock_logger

            publisher._log_update(
                {
                    "cycle": 1,
                    "pnl": 100.0,
                    "positions": [{"sym": "BTC"}],  # Should be filtered
                    "decisions": [{"action": "buy"}],  # Should be filtered
                }
            )

            call_kwargs = mock_logger.log_event.call_args[1]
            assert "cycle" in call_kwargs
            assert "pnl" in call_kwargs
            assert "positions" not in call_kwargs
            assert "decisions" not in call_kwargs

    def test_handles_log_errors(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

        with patch("gpt_trader.monitoring.system.get_logger") as mock_get_plog:
            mock_logger = MagicMock()
            mock_logger.log_event.side_effect = Exception("Log error")
            mock_get_plog.return_value = mock_logger

            # Should not raise
            publisher._log_update({"cycle": 1})


class TestMetricsPublisherWriteHealthStatus:
    """Tests for MetricsPublisher.write_health_status method."""

    def test_writes_health_status(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

        publisher.write_health_status(ok=True, message="System healthy")

        health_path = tmp_path / "TEST" / "health.json"
        assert health_path.exists()

        with health_path.open() as f:
            data = json.load(f)
            assert data["ok"] is True
            assert data["message"] == "System healthy"
            assert data["error"] == ""
            assert "timestamp" in data

    def test_writes_unhealthy_status(self, tmp_path: Path) -> None:
        event_store = MagicMock()
        publisher = MetricsPublisher(
            event_store=event_store,
            bot_id="bot-123",
            profile="TEST",
            base_dir=tmp_path,
        )

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
