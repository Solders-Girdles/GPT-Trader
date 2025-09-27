"""Tests for comprehensive logging system."""

import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bot.config import TradingConfig, set_config
from bot.logging import (
    Alert,
    AlertManager,
    CorrelationTracker,
    DatabaseLogHandler,
    HealthMonitor,
    LogQueryInterface,
    MLLogger,
    RiskLogger,
    StreamingLogHandler,
    StrategyLogger,
    SystemLogger,
    TradingLogger,
    get_log_monitor,
)


class TestDatabaseLogHandler(unittest.TestCase):
    """Test database logging handler."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)
        self.handler = DatabaseLogHandler(self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.db_path.unlink(missing_ok=True)

    def test_database_initialization(self):
        """Test database schema initialization."""
        # Database should be created and have the correct table
        query_interface = LogQueryInterface(self.db_path)
        
        # Should not raise an error
        logs = query_interface.query_logs(limit=1)
        self.assertIsInstance(logs, pd.DataFrame)

    def test_log_persistence(self):
        """Test log entry persistence."""
        import logging
        
        # Create a test log record
        logger = logging.getLogger("test")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_fields = {"event_type": "test_event", "symbol": "TEST"}
        
        # Emit the record
        self.handler.emit(record)
        
        # Query the database
        query_interface = LogQueryInterface(self.db_path)
        logs = query_interface.query_logs(limit=10)
        
        # Verify the log was persisted
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs.iloc[0]["message"], "Test message")
        self.assertEqual(logs.iloc[0]["level"], "INFO")


class TestStreamingLogHandler(unittest.TestCase):
    """Test streaming log handler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = StreamingLogHandler()
        self.received_logs = []

    def log_callback(self, log_data):
        """Callback for receiving logs."""
        self.received_logs.append(log_data)

    def test_subscription(self):
        """Test log subscription."""
        self.handler.subscribe(self.log_callback)
        
        # Create and emit a test log
        import logging
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test streaming",
            args=(),
            exc_info=None
        )
        
        self.handler.emit(record)
        
        # Verify callback was called
        self.assertEqual(len(self.received_logs), 1)
        self.assertEqual(self.received_logs[0]["message"], "Test streaming")

    def test_buffer_management(self):
        """Test log buffer management."""
        import logging
        
        # Emit multiple logs
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Message {i}",
                args=(),
                exc_info=None
            )
            self.handler.emit(record)
        
        # Check buffer
        recent_logs = self.handler.get_recent_logs(3)
        self.assertEqual(len(recent_logs), 3)
        # Should be the last 3 messages
        self.assertEqual(recent_logs[-1]["message"], "Message 4")


class TestSpecializedLoggers(unittest.TestCase):
    """Test specialized domain loggers."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config
        self.temp_config = TradingConfig()
        self.temp_config.logging.structured_logging = True
        set_config(self.temp_config)

    def test_trading_logger(self):
        """Test trading-specific logging."""
        trading_logger = TradingLogger()
        
        # Test signal logging
        trading_logger.log_signal_generated(
            strategy_name="test_strategy",
            symbol="TEST",
            signal_type="buy",
            confidence=0.75
        )
        
        # Test order logging
        trading_logger.log_order_submitted(
            order_id="TEST-001",
            symbol="TEST",
            side="buy",
            quantity=100,
            order_type="market"
        )
        
        # Should not raise any exceptions
        self.assertTrue(True)

    def test_strategy_logger(self):
        """Test strategy-specific logging."""
        strategy_logger = StrategyLogger("test_strategy")
        
        # Test strategy lifecycle logging
        strategy_logger.log_strategy_start({"param1": "value1"})
        strategy_logger.log_data_processing("TEST", 100, 50.0)
        strategy_logger.log_decision_process("TEST", "buy", 0.8)
        strategy_logger.log_performance_update(10.5, 1.2, 5.0, 0.6, 50)
        
        # Should not raise any exceptions
        self.assertTrue(True)

    def test_ml_logger(self):
        """Test ML-specific logging."""
        ml_logger = MLLogger()
        
        # Test ML lifecycle logging
        ml_logger.log_model_training_start(
            "test_model", 1000, ["feature1", "feature2"], {"lr": 0.01}
        )
        ml_logger.log_model_training_complete(
            "test_model", 3600, 0.01, 0.85, 5.0
        )
        ml_logger.log_prediction(
            "test_model", {"feature1": 1.0}, 0.75, 0.9, 2.5
        )
        
        # Should not raise any exceptions
        self.assertTrue(True)

    def test_risk_logger(self):
        """Test risk management logging."""
        risk_logger = RiskLogger()
        
        # Test risk event logging
        risk_logger.log_risk_check("position_size", True, "value", 1000, 2000)
        risk_logger.log_circuit_breaker_trigger(
            "daily_loss", "max_loss_exceeded", 5000, 4000, "halt_trading"
        )
        risk_logger.log_var_calculation(100000, 2000, 4500, 0.95, "historical")
        
        # Should not raise any exceptions
        self.assertTrue(True)

    def test_system_logger(self):
        """Test system health logging."""
        system_logger = SystemLogger()
        
        # Test system event logging
        system_logger.log_system_startup(["component1", "component2"], 10.5)
        system_logger.log_resource_usage(50.0, 60.0, 70.0)
        system_logger.log_health_check("database", "healthy", 25.0)
        system_logger.log_error_rate("api", 5, 100, 60)
        
        # Should not raise any exceptions
        self.assertTrue(True)


class TestAlertManager(unittest.TestCase):
    """Test alert management system."""

    def setUp(self):
        """Set up test fixtures."""
        self.alert_manager = AlertManager()
        self.triggered_alerts = []

    def alert_callback(self, alert):
        """Callback for alert notifications."""
        self.triggered_alerts.append(alert)

    def test_alert_creation(self):
        """Test alert creation."""
        alert = self.alert_manager.create_alert(
            level="warning",
            title="Test Alert",
            message="Test message",
            component="test"
        )
        
        self.assertEqual(alert.level, "warning")
        self.assertEqual(alert.title, "Test Alert")
        self.assertFalse(alert.acknowledged)
        self.assertFalse(alert.resolved)

    def test_alert_subscription(self):
        """Test alert subscriptions."""
        self.alert_manager.subscribe(self.alert_callback)
        
        # Create an alert
        self.alert_manager.create_alert(
            level="error",
            title="Test Error",
            message="Error occurred",
            component="test"
        )
        
        # Verify callback was triggered
        self.assertEqual(len(self.triggered_alerts), 1)
        self.assertEqual(self.triggered_alerts[0].title, "Test Error")

    def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        alert = self.alert_manager.create_alert(
            level="info",
            title="Test Info",
            message="Info message",
            component="test"
        )
        
        # Acknowledge the alert
        result = self.alert_manager.acknowledge_alert(alert.id)
        self.assertTrue(result)
        self.assertTrue(alert.acknowledged)

    def test_alert_resolution(self):
        """Test alert resolution."""
        alert = self.alert_manager.create_alert(
            level="warning",
            title="Test Warning",
            message="Warning message",
            component="test"
        )
        
        # Resolve the alert
        result = self.alert_manager.resolve_alert(alert.id)
        self.assertTrue(result)
        self.assertTrue(alert.resolved)

    def test_active_alerts(self):
        """Test active alert filtering."""
        # Create multiple alerts
        alert1 = self.alert_manager.create_alert("info", "Alert 1", "Message 1", "test")
        alert2 = self.alert_manager.create_alert("warning", "Alert 2", "Message 2", "test")
        alert3 = self.alert_manager.create_alert("error", "Alert 3", "Message 3", "test")
        
        # Resolve one alert
        self.alert_manager.resolve_alert(alert2.id)
        
        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 2)
        
        # Verify the resolved alert is not in active list
        active_ids = [alert.id for alert in active_alerts]
        self.assertNotIn(alert2.id, active_ids)


class TestCorrelationTracker(unittest.TestCase):
    """Test correlation tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = CorrelationTracker()

    def test_request_tracking(self):
        """Test request lifecycle tracking."""
        correlation_id = "test-001"
        
        # Start request
        self.tracker.start_request(correlation_id, "test_operation", context="test")
        
        # Add events
        self.tracker.add_event(correlation_id, "step1", data="value1")
        self.tracker.add_event(correlation_id, "step2", data="value2")
        
        # Complete request
        self.tracker.complete_request(correlation_id, success=True, result="success")
        
        # Get trace
        trace = self.tracker.get_request_trace(correlation_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace["operation"], "test_operation")
        self.assertEqual(len(trace["events"]), 2)
        self.assertTrue(trace["success"])

    def test_active_request_retrieval(self):
        """Test retrieving active request data."""
        correlation_id = "active-001"
        
        # Start request but don't complete
        self.tracker.start_request(correlation_id, "active_operation")
        self.tracker.add_event(correlation_id, "event1")
        
        # Should be able to retrieve active request
        trace = self.tracker.get_request_trace(correlation_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace["operation"], "active_operation")
        self.assertEqual(len(trace["events"]), 1)
        self.assertNotIn("success", trace)  # Not completed yet


class TestHealthMonitor(unittest.TestCase):
    """Test health monitoring."""

    def setUp(self):
        """Set up test fixtures."""
        self.health_monitor = HealthMonitor()

    def test_component_registration(self):
        """Test component registration."""
        def mock_health_check():
            return "healthy", 25.0, {"detail": "test"}
        
        self.health_monitor.register_component("test_component", mock_health_check)
        
        # Perform manual health check
        self.health_monitor._perform_health_checks()
        
        # Get status
        status = self.health_monitor.get_health_status("test_component")
        self.assertIsNotNone(status)
        self.assertEqual(status.status, "healthy")
        self.assertEqual(status.response_time_ms, 25.0)

    def test_overall_health_calculation(self):
        """Test overall health status calculation."""
        def healthy_check():
            return "healthy", 10.0, {}
        
        def unhealthy_check():
            return "unhealthy", 100.0, {}
        
        # Register components
        self.health_monitor.register_component("healthy_comp", healthy_check)
        self.health_monitor.register_component("unhealthy_comp", unhealthy_check)
        
        # Perform checks
        self.health_monitor._perform_health_checks()
        
        # Overall health should be unhealthy due to one unhealthy component
        overall_health = self.health_monitor.get_overall_health()
        self.assertEqual(overall_health, "unhealthy")


class TestLogQueryInterface(unittest.TestCase):
    """Test log query interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)
        self.query_interface = LogQueryInterface(self.db_path)
        
        # Populate with test data
        self._populate_test_data()

    def tearDown(self):
        """Clean up test fixtures."""
        self.db_path.unlink(missing_ok=True)

    def _populate_test_data(self):
        """Populate database with test log data."""
        import sqlite3
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert test log entries
            test_data = [
                (time.time() - 3600, "INFO", "test.logger", "Test message 1", "test", "func1", 10, 
                 json.dumps({"event_type": "test_event", "symbol": "TEST1"}), None, None),
                (time.time() - 1800, "ERROR", "test.logger", "Test error", "test", "func2", 20,
                 json.dumps({"event_type": "error_event"}), "ValueError", "Test error message"),
                (time.time() - 900, "INFO", "trading.logger", "Trade executed", "trading", "execute", 30,
                 json.dumps({"event_type": "order_executed", "symbol": "TEST2", "pnl": 150.0}), None, None),
            ]
            
            for data in test_data:
                conn.execute("""
                    INSERT INTO log_entries 
                    (timestamp, level, logger, message, module, function, line, extra_fields, exception_type, exception_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, data)

    def test_basic_query(self):
        """Test basic log querying."""
        logs = self.query_interface.query_logs(limit=10)
        self.assertEqual(len(logs), 3)

    def test_filtered_query(self):
        """Test filtered log queries."""
        # Filter by level
        error_logs = self.query_interface.query_logs(level="ERROR")
        self.assertEqual(len(error_logs), 1)
        self.assertEqual(error_logs.iloc[0]["message"], "Test error")
        
        # Filter by event type
        trade_logs = self.query_interface.query_logs(event_type="order_executed")
        self.assertEqual(len(trade_logs), 1)
        self.assertEqual(trade_logs.iloc[0]["symbol"], "TEST2")

    def test_error_summary(self):
        """Test error summary generation."""
        summary = self.query_interface.get_error_summary(hours_back=24)
        self.assertEqual(summary["total_errors"], 1)
        self.assertTrue(len(summary["component_errors"]) > 0)

    def test_trading_summary(self):
        """Test trading summary generation."""
        summary = self.query_interface.get_trading_summary(hours_back=24)
        self.assertEqual(summary["total_realized_pnl"], 150.0)
        self.assertEqual(summary["completed_trades"], 1)

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        anomalies = self.query_interface.detect_anomalies(hours_back=24)
        
        # Should detect patterns in our test data
        self.assertIsInstance(anomalies["anomalies_detected"], int)
        self.assertIsInstance(anomalies["anomalies"], list)


@patch('bot.config.get_config')
def test_log_monitor_integration(mock_get_config):
    """Test log monitor integration."""
    # Mock config
    mock_config = MagicMock()
    mock_config.database.database_path.parent = Path("/tmp")
    mock_get_config.return_value = mock_config
    
    # Get monitor instance
    monitor = get_log_monitor()
    
    # Test basic functionality
    summary = monitor.get_monitoring_summary()
    assert "timestamp" in summary
    assert "overall_health" in summary
    assert "system_status" in summary
    
    # Test stream handler
    stream_handler = monitor.get_stream_handler()
    assert isinstance(stream_handler, StreamingLogHandler)


if __name__ == "__main__":
    unittest.main()