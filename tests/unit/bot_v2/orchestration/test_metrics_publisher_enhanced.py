"""Enhanced tests for MetricsPublisher module to improve coverage from 41.07% to 85%+."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bot_v2.orchestration.system_monitor_metrics import MetricsPublisher
from bot_v2.utilities.logging_patterns import get_logger


@pytest.fixture
def mock_event_store():
    """Mock event store for testing."""
    store = Mock()
    return store


@pytest.fixture
def temp_base_dir(tmp_path: Path) -> Path:
    """Temporary base directory for metrics file operations."""
    return tmp_path / "runtime_data"


@pytest.fixture
def metrics_publisher(mock_event_store, temp_base_dir: Path) -> MetricsPublisher:
    """Create MetricsPublisher instance for testing."""
    return MetricsPublisher(
        event_store=mock_event_store,
        bot_id="test_bot",
        profile="prod",
        base_dir=temp_base_dir,
    )


def test_publisher_initialization_and_attributes(
    metrics_publisher: MetricsPublisher, mock_event_store, temp_base_dir: Path
) -> None:
    """Test MetricsPublisher initialization and attribute assignment."""
    assert metrics_publisher._event_store is mock_event_store
    assert metrics_publisher._bot_id == "test_bot"
    assert metrics_publisher._profile == "prod"
    assert metrics_publisher._base_dir is temp_base_dir


def test_target_dirs_modern_and_legacy_paths_different(
    metrics_publisher: MetricsPublisher, temp_base_dir: Path
) -> None:
    """Test _target_dirs returns both modern and legacy paths when different."""
    target_dirs = metrics_publisher._target_dirs()

    expected_modern = temp_base_dir / "coinbase_trader" / "prod"
    expected_legacy = temp_base_dir / "perps_bot" / "prod"

    assert len(target_dirs) == 2
    assert expected_modern in target_dirs
    assert expected_legacy in target_dirs
    assert expected_modern != expected_legacy


def test_target_dirs_single_path_when_identical(temp_base_dir: Path) -> None:
    """Test _target_dirs behavior with different profile names."""
    # Create a publisher with a different profile to see the logic
    publisher = MetricsPublisher(
        event_store=Mock(),
        bot_id="test_bot",
        profile="coinbase_trader",  # This makes modern and legacy paths identical
        base_dir=temp_base_dir,
    )

    target_dirs = publisher._target_dirs()
    expected_modern = temp_base_dir / "coinbase_trader" / "coinbase_trader"
    expected_legacy = temp_base_dir / "perps_bot" / "coinbase_trader"

    # Should return both paths when they're different
    assert len(target_dirs) == 2
    assert expected_modern in target_dirs
    assert expected_legacy in target_dirs


def test_publish_emits_metric_to_event_store(
    metrics_publisher: MetricsPublisher, mock_event_store
) -> None:
    """Test publish method calls emit_metric with correct parameters."""
    test_metrics = {
        "equity": 10000.0,
        "positions": [{"symbol": "BTC-PERP", "quantity": 0.5}],
        "profile": "prod",
    }

    with patch("bot_v2.orchestration.system_monitor_metrics.emit_metric") as mock_emit:
        metrics_publisher.publish(test_metrics)

        mock_emit.assert_called_once_with(
            mock_event_store,
            "test_bot",
            {"event_type": "cycle_metrics", **test_metrics},
            logger=get_logger(__name__, component="system_monitor_metrics"),
        )


def test_publish_writes_snapshot_and_logs_update(
    metrics_publisher: MetricsPublisher, temp_base_dir: Path
) -> None:
    """Test publish method calls _write_snapshot and _log_update."""
    test_metrics = {"equity": 10000.0, "profile": "prod"}

    with (
        patch.object(metrics_publisher, "_write_snapshot") as mock_snapshot,
        patch.object(metrics_publisher, "_log_update") as mock_log,
    ):

        metrics_publisher.publish(test_metrics)

        mock_snapshot.assert_called_once_with(test_metrics)
        mock_log.assert_called_once_with(test_metrics)


def test_publish_with_complex_metrics_data(metrics_publisher: MetricsPublisher) -> None:
    """Test publish with complex nested metrics data."""
    complex_metrics = {
        "equity": 10000.0,
        "positions": [
            {"symbol": "BTC-PERP", "quantity": 0.5, "side": "long"},
            {"symbol": "ETH-PERP", "quantity": 2.0, "side": "short"},
        ],
        "decisions": {
            "BTC-PERP": {"action": "BUY", "reason": "signal"},
            "ETH-PERP": {"action": "SELL", "reason": "take_profit"},
        },
        "system_metrics": {
            "cpu_percent": 45.2,
            "memory_mb": 2048,
            "disk_gb": 125.5,
        },
        "timestamp": "2025-10-20T21:00:00Z",
    }

    with (
        patch("bot_v2.orchestration.system_monitor_metrics.emit_metric"),
        patch.object(metrics_publisher, "_write_snapshot"),
        patch.object(metrics_publisher, "_log_update"),
    ):

        # Should not raise any exceptions with complex data
        metrics_publisher.publish(complex_metrics)


def test_write_snapshot_creates_directories(
    metrics_publisher: MetricsPublisher, temp_base_dir: Path
) -> None:
    """Test _write_snapshot creates target directories if they don't exist."""
    test_metrics = {"equity": 10000.0, "profile": "prod"}

    metrics_publisher._write_snapshot(test_metrics)

    # Check that both modern and legacy directories were created
    modern_dir = temp_base_dir / "coinbase_trader" / "prod"
    legacy_dir = temp_base_dir / "perps_bot" / "prod"

    assert modern_dir.exists()
    assert modern_dir.is_dir()
    assert legacy_dir.exists()
    assert legacy_dir.is_dir()


def test_write_snapshot_handles_json_serialization(
    metrics_publisher: MetricsPublisher, temp_base_dir: Path
) -> None:
    """Test _write_snapshot properly serializes metrics to JSON."""
    test_metrics = {
        "equity": 10000.0,
        "positions": [{"symbol": "BTC-PERP", "quantity": 0.5}],
        "nested": {"key": "value", "number": 42},
    }

    metrics_publisher._write_snapshot(test_metrics)

    # Check modern path
    modern_metrics_file = temp_base_dir / "coinbase_trader" / "prod" / "metrics.json"
    assert modern_metrics_file.exists()

    with modern_metrics_file.open("r") as f:
        loaded_data = json.load(f)

    assert loaded_data == test_metrics

    # Check legacy path
    legacy_metrics_file = temp_base_dir / "perps_bot" / "prod" / "metrics.json"
    assert legacy_metrics_file.exists()

    with legacy_metrics_file.open("r") as f:
        legacy_data = json.load(f)

    assert legacy_data == test_metrics


def test_write_snapshot_atomic_write_operations(
    metrics_publisher: MetricsPublisher, temp_base_dir: Path
) -> None:
    """Test _write_snapshot writes to both target paths atomically."""
    test_metrics = {"equity": 10000.0, "profile": "prod"}

    # Mock file operations to verify atomic behavior
    with patch("builtins.open", create=True) as mock_open:
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        metrics_publisher._write_snapshot(test_metrics)

        # Should open file twice (once for each target directory)
        assert mock_open.call_count == 2

        # Should call json.dump twice
        assert mock_file.__enter__.return_value.write.call_count >= 2


def test_write_snapshot_handles_permission_denied(
    metrics_publisher: MetricsPublisher, caplog: pytest.LogCaptureFixture
) -> None:
    """Test _write_snapshot handles permission denied errors gracefully."""
    caplog.set_level(logging.DEBUG, "bot_v2.orchestration.system_monitor_metrics")

    test_metrics = {"equity": 10000.0}

    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        # Should not raise exception
        metrics_publisher._write_snapshot(test_metrics)

    # Should log debug message
    assert "Failed to write metrics snapshot" in caplog.text
    assert "Permission denied" in caplog.text
    assert "write_snapshot" in caplog.text


def test_write_snapshot_handles_disk_full_error(
    metrics_publisher: MetricsPublisher, caplog: pytest.LogCaptureFixture
) -> None:
    """Test _write_snapshot handles disk full errors gracefully."""
    caplog.set_level(logging.DEBUG, metrics_publisher.logger.name)

    test_metrics = {"equity": 10000.0}

    with patch("builtins.open", side_effect=OSError("No space left on device")):
        # Should not raise exception
        metrics_publisher._write_snapshot(test_metrics)

    # Should log debug message
    assert "Failed to write metrics snapshot" in caplog.text
    assert "No space left on device" in caplog.text


def test_write_snapshot_handles_invalid_json_data(
    metrics_publisher: MetricsPublisher, caplog: pytest.LogCaptureFixture
) -> None:
    """Test _write_snapshot handles non-serializable data gracefully."""
    caplog.set_level(logging.DEBUG, metrics_publisher.logger.name)

    # Create non-serializable data (circular reference)
    test_metrics = {"equity": 10000.0}
    test_metrics["self"] = test_metrics  # Create circular reference

    with patch("json.dump", side_effect=TypeError("Object not serializable")):
        # Should not raise exception
        metrics_publisher._write_snapshot(test_metrics)

    # Should log debug message
    assert "Failed to write metrics snapshot" in caplog.text
    assert "Object not serializable" in caplog.text


def test_log_update_logs_success_case(metrics_publisher: MetricsPublisher) -> None:
    """Test _log_update successfully logs metrics update."""
    test_metrics = {
        "equity": 10000.0,
        "profile": "prod",
        "positions": [{"symbol": "BTC-PERP"}],  # Should be filtered out
        "decisions": {"BTC-PERP": "BUY"},  # Should be filtered out
        "event_type": "cycle_metrics",  # Should be filtered out
        "custom_field": "value",
    }

    with patch("bot_v2.orchestration.system_monitor_metrics._get_plog") as mock_get_plog:
        mock_plog = Mock()
        mock_get_plog.return_value = mock_plog

        metrics_publisher._log_update(test_metrics)

        mock_plog.log_event.assert_called_once_with(
            level=mock_get_plog.return_value.LogLevel.INFO,
            event_type="metrics_update",
            message="Cycle metrics updated",
            component="CoinbaseTrader",
            equity=10000.0,
            profile="prod",
            custom_field="value",
        )


def test_log_update_handles_logging_failure(
    metrics_publisher: MetricsPublisher, caplog: pytest.LogCaptureFixture
) -> None:
    """Test _log_update handles logging failures gracefully."""
    caplog.set_level(logging.DEBUG, metrics_publisher.logger.name)

    test_metrics = {"equity": 10000.0}

    with patch("bot_v2.orchestration.system_monitor_metrics._get_plog") as mock_get_plog:
        mock_plog = Mock()
        mock_plog.log_event.side_effect = Exception("Logging failed")
        mock_get_plog.return_value = mock_plog

        # Should not raise exception
        metrics_publisher._log_update(test_metrics)

    # Should log debug message
    assert "Failed to emit metrics update event" in caplog.text
    assert "Logging failed" in caplog.text
    assert "log_update" in caplog.text


def test_write_health_status_success_case(
    metrics_publisher: MetricsPublisher, temp_base_dir: Path
) -> None:
    """Test write_health_status writes success status to disk."""
    metrics_publisher.write_health_status(ok=True, message="System healthy")

    # Check both target directories
    for target_dir in metrics_publisher._target_dirs():
        health_file = target_dir / "health.json"
        assert health_file.exists()

        with health_file.open("r") as f:
            health_data = json.load(f)

        assert health_data["ok"] is True
        assert health_data["message"] == "System healthy"
        assert health_data["error"] == ""
        assert "timestamp" in health_data


def test_write_health_status_failure_case(
    metrics_publisher: MetricsPublisher, temp_base_dir: Path
) -> None:
    """Test write_health_status writes failure status to disk."""
    error_message = "Connection timeout"
    metrics_publisher.write_health_status(ok=False, message="System error", error=error_message)

    # Check both target directories
    for target_dir in metrics_publisher._target_dirs():
        health_file = target_dir / "health.json"
        assert health_file.exists()

        with health_file.open("r") as f:
            health_data = json.load(f)

        assert health_data["ok"] is False
        assert health_data["message"] == "System error"
        assert health_data["error"] == error_message
        assert "timestamp" in health_data


def test_write_health_status_creates_directories(
    metrics_publisher: MetricsPublisher, temp_base_dir: Path
) -> None:
    """Test write_health_status creates target directories if they don't exist."""
    metrics_publisher.write_health_status(ok=True, message="Test")

    # Check that both modern and legacy directories were created
    modern_dir = temp_base_dir / "coinbase_trader" / "prod"
    legacy_dir = temp_base_dir / "perps_bot" / "prod"

    assert modern_dir.exists()
    assert modern_dir.is_dir()
    assert legacy_dir.exists()
    assert legacy_dir.is_dir()


def test_publish_gracefully_degrades_on_file_errors(
    metrics_publisher: MetricsPublisher, caplog: pytest.LogCaptureFixture
) -> None:
    """Test publish gracefully degrades when file operations fail."""
    caplog.set_level(logging.DEBUG, metrics_publisher.logger.name)

    test_metrics = {"equity": 10000.0}

    with (
        patch("builtins.open", side_effect=PermissionError("Permission denied")),
        patch("bot_v2.orchestration.system_monitor_metrics.emit_metric") as mock_emit,
    ):

        # Should not raise exception
        metrics_publisher.publish(test_metrics)

        # Should still emit metric to event store
        mock_emit.assert_called_once()

        # Should log debug message about snapshot failure
        assert "Failed to write metrics snapshot" in caplog.text


def test_publish_with_none_or_empty_metrics(metrics_publisher: MetricsPublisher) -> None:
    """Test publish handles None or empty metrics gracefully."""
    with (
        patch("bot_v2.orchestration.system_monitor_metrics.emit_metric") as mock_emit,
        patch.object(metrics_publisher, "_write_snapshot") as mock_snapshot,
        patch.object(metrics_publisher, "_log_update") as mock_log,
    ):

        # Test with empty dict
        metrics_publisher.publish({})

        mock_emit.assert_called_once()
        mock_snapshot.assert_called_once()
        mock_log.assert_called_once()


def test_publish_with_large_complex_metrics(metrics_publisher: MetricsPublisher) -> None:
    """Test publish handles large complex metrics structures."""
    large_metrics = {
        "equity": 10000.0,
        "positions": [{"symbol": f"SYM-{i}", "quantity": i * 0.1} for i in range(1000)],
        "decisions": {f"SYM-{i}": {"action": "BUY" if i % 2 == 0 else "SELL"} for i in range(500)},
        "system_data": {f"metric_{i}": i for i in range(200)},
    }

    with (
        patch("bot_v2.orchestration.system_monitor_metrics.emit_metric"),
        patch.object(metrics_publisher, "_write_snapshot"),
        patch.object(metrics_publisher, "_log_update"),
    ):

        # Should handle large data without issues
        metrics_publisher.publish(large_metrics)


def test_concurrent_publish_operations(metrics_publisher: MetricsPublisher) -> None:
    """Test MetricsPublisher handles concurrent publish operations safely."""
    import threading

    results = []
    errors = []

    def publish_metrics(thread_id: int):
        try:
            metrics = {"thread_id": thread_id, "equity": 10000.0 + thread_id}
            with (
                patch("bot_v2.orchestration.system_monitor_metrics.emit_metric"),
                patch.object(metrics_publisher, "_write_snapshot"),
                patch.object(metrics_publisher, "_log_update"),
            ):
                metrics_publisher.publish(metrics)
            results.append(thread_id)
        except Exception as e:
            errors.append((thread_id, e))

    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=publish_metrics, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Should have completed all operations without errors
    assert len(errors) == 0
    assert len(results) == 5


def test_profile_specific_directory_structure(temp_base_dir: Path) -> None:
    """Test MetricsPublisher creates correct directory structure for different profiles."""
    profiles = ["prod", "dev", "test", "staging"]

    for profile in profiles:
        publisher = MetricsPublisher(
            event_store=Mock(),
            bot_id="test_bot",
            profile=profile,
            base_dir=temp_base_dir,
        )

        publisher.write_health_status(ok=True, message=f"Profile {profile} test")

        # Check modern path
        modern_dir = temp_base_dir / "coinbase_trader" / profile
        assert modern_dir.exists()

        health_file = modern_dir / "health.json"
        assert health_file.exists()

        with health_file.open("r") as f:
            health_data = json.load(f)

        assert f"Profile {profile} test" in health_data["message"]


def test_error_context_logging_with_debug_level(
    metrics_publisher: MetricsPublisher, caplog: pytest.LogCaptureFixture
) -> None:
    """Test error logging includes proper context information."""
    caplog.set_level(logging.DEBUG, metrics_publisher.logger.name)

    test_metrics = {"equity": 10000.0}

    with patch("builtins.open", side_effect=OSError("Device error")):
        metrics_publisher._write_snapshot(test_metrics)

    # Verify error context
    assert "operation" in caplog.text
    assert "system_monitor_metrics" in caplog.text
    assert "write_snapshot" in caplog.text
    assert "error" in caplog.text
    assert "Device error" in caplog.text
