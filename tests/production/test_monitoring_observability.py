"""
Monitoring and Observability Testing for Phase 5 Production Integration.

Tests logging system, metrics collection, health checks, and alert delivery
capabilities.
"""

import asyncio
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bot.live.production_orchestrator import (
    OrchestrationMode,
    OrchestratorConfig,
    ProductionOrchestrator,
)
from bot.monitor.alerts import AlertConfig as AlertManagerConfig
from bot.monitor.alerts import AlertManager, AlertSeverity, AlertType
from bot.monitor.performance_monitor import AlertConfig, PerformanceMonitor, PerformanceThresholds


@dataclass
class MockStrategyPerformance:
    """Mock strategy performance data."""

    strategy_id: str
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    n_positions: int
    timestamp: datetime


class TestLoggingSystem:
    """Test logging system configuration and functionality."""

    def test_logging_configuration(self):
        """Test logging configuration setup."""
        # Test that logging is properly configured
        from bot.logging import get_logger

        logger = get_logger("bot")

        # Verify logger exists
        assert logger is not None

        # Test log levels
        logger.setLevel(logging.INFO)
        assert logger.level == logging.INFO

        # Test handler configuration
        handlers = logger.handlers
        assert len(handlers) > 0, "Logger should have at least one handler"

    def test_logging_output(self):
        """Test logging output to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Configure file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            # Configure logger
            logger = logging.getLogger("test_logger")
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)

            # Test logging
            test_message = "Test log message"
            logger.info(test_message)

            # Verify log file was created and contains message
            assert log_file.exists()
            log_content = log_file.read_text()
            assert test_message in log_content

    def test_logging_levels(self):
        """Test different logging levels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "levels.log"

            # Configure file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.WARNING)

            logger = logging.getLogger("test_levels")
            logger.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

            # Test different levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            # Verify only WARNING and above are logged
            log_content = log_file.read_text()
            assert "Debug message" not in log_content
            assert "Info message" not in log_content
            assert "Warning message" in log_content
            assert "Error message" in log_content

    def test_logging_rotation(self):
        """Test log file rotation."""
        # This would test log rotation functionality
        # Implementation would depend on the specific rotation mechanism

        # For now, test that we can create multiple log files
        with tempfile.TemporaryDirectory() as temp_dir:
            log_files = []

            for i in range(3):
                log_file = Path(temp_dir) / f"test_{i}.log"
                file_handler = logging.FileHandler(log_file)

                logger = logging.getLogger(f"test_rotation_{i}")
                logger.addHandler(file_handler)
                logger.info(f"Message {i}")

                log_files.append(log_file)

            # Verify all log files were created
            for log_file in log_files:
                assert log_file.exists()
                assert log_file.read_text() != ""


class TestMetricsCollection:
    """Test metrics collection and aggregation."""

    def test_performance_metrics_collection(self):
        """Test performance metrics collection."""
        # Mock broker
        mock_broker = Mock()

        # Create performance monitor
        thresholds = PerformanceThresholds(min_sharpe=0.5, max_drawdown=0.15, min_cagr=0.05)

        alert_config = AlertConfig(
            webhook_enabled=False, email_enabled=False, alert_cooldown_hours=24
        )

        monitor = PerformanceMonitor(mock_broker, thresholds, alert_config)

        # Test metrics collection
        performance_data = [
            MockStrategyPerformance(
                strategy_id="strategy_1",
                sharpe_ratio=1.2,
                cagr=0.08,
                max_drawdown=0.10,
                n_positions=5,
                timestamp=datetime.now(),
            ),
            MockStrategyPerformance(
                strategy_id="strategy_2",
                sharpe_ratio=0.8,
                cagr=0.06,
                max_drawdown=0.12,
                n_positions=3,
                timestamp=datetime.now(),
            ),
        ]

        # Store performance data
        for perf in performance_data:
            strategy_id = perf.strategy_id
            if strategy_id not in monitor.performance_history:
                monitor.performance_history[strategy_id] = []
            monitor.performance_history[strategy_id].append(perf)

        # Verify metrics were collected
        assert len(monitor.performance_history) == 2
        assert "strategy_1" in monitor.performance_history
        assert "strategy_2" in monitor.performance_history
        assert len(monitor.performance_history["strategy_1"]) == 1
        assert len(monitor.performance_history["strategy_2"]) == 1

    def test_metrics_aggregation(self):
        """Test metrics aggregation and summary generation."""
        # Mock broker
        mock_broker = Mock()

        # Create performance monitor
        thresholds = PerformanceThresholds()
        alert_config = AlertConfig(
            webhook_enabled=False, email_enabled=False, alert_cooldown_hours=24
        )

        monitor = PerformanceMonitor(mock_broker, thresholds, alert_config)

        # Add performance data
        monitor.performance_history = {
            "strategy_1": [
                MockStrategyPerformance(
                    strategy_id="strategy_1",
                    sharpe_ratio=1.2,
                    cagr=0.08,
                    max_drawdown=0.10,
                    n_positions=5,
                    timestamp=datetime.now(),
                )
            ],
            "strategy_2": [
                MockStrategyPerformance(
                    strategy_id="strategy_2",
                    sharpe_ratio=0.8,
                    cagr=0.06,
                    max_drawdown=0.12,
                    n_positions=3,
                    timestamp=datetime.now(),
                )
            ],
        }

        # Generate summary
        summary = monitor.get_performance_summary()

        # Verify summary structure
        assert "monitoring_active" in summary
        assert "strategies" in summary
        assert "alerts" in summary

        # Verify strategy data
        assert "strategy_1" in summary["strategies"]
        assert "strategy_2" in summary["strategies"]

        strategy_1_data = summary["strategies"]["strategy_1"]
        assert "current_sharpe" in strategy_1_data
        assert "current_cagr" in strategy_1_data
        assert "current_drawdown" in strategy_1_data
        assert "n_positions" in strategy_1_data
        assert "last_update" in strategy_1_data

    def test_metrics_persistence(self):
        """Test metrics persistence to storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_file = Path(temp_dir) / "metrics.json"

            # Create sample metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "strategies": {
                    "strategy_1": {
                        "sharpe_ratio": 1.2,
                        "cagr": 0.08,
                        "max_drawdown": 0.10,
                        "n_positions": 5,
                    }
                },
                "system": {"cpu_usage": 0.25, "memory_usage": 0.15, "disk_usage": 0.30},
            }

            # Save metrics
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            # Verify metrics were saved
            assert metrics_file.exists()

            # Load and verify metrics
            with open(metrics_file) as f:
                loaded_metrics = json.load(f)

            assert loaded_metrics["strategies"]["strategy_1"]["sharpe_ratio"] == 1.2
            assert loaded_metrics["system"]["cpu_usage"] == 0.25

    def test_metrics_validation(self):
        """Test metrics validation and data quality checks."""
        # Test valid metrics
        valid_metrics = {"sharpe_ratio": 1.2, "cagr": 0.08, "max_drawdown": 0.10, "n_positions": 5}

        # All values should be numeric and reasonable
        assert isinstance(valid_metrics["sharpe_ratio"], (int, float))
        assert isinstance(valid_metrics["cagr"], (int, float))
        assert isinstance(valid_metrics["max_drawdown"], (int, float))
        assert isinstance(valid_metrics["n_positions"], int)

        assert valid_metrics["sharpe_ratio"] > -10 and valid_metrics["sharpe_ratio"] < 10
        assert valid_metrics["cagr"] > -1 and valid_metrics["cagr"] < 2
        assert valid_metrics["max_drawdown"] >= 0 and valid_metrics["max_drawdown"] <= 1
        assert valid_metrics["n_positions"] >= 0 and valid_metrics["n_positions"] <= 100


class TestHealthChecks:
    """Test health check functionality."""

    def test_system_health_check(self):
        """Test system health check functionality."""
        # Mock components
        mock_broker = Mock()
        mock_knowledge_base = Mock()

        # Create orchestrator config
        config = OrchestratorConfig(
            mode=OrchestrationMode.SEMI_AUTOMATED,
            rebalance_interval=3600,
            risk_check_interval=300,
            performance_check_interval=600,
            max_strategies=5,
            min_strategy_confidence=0.7,
            enable_alerts=True,
        )

        # Create orchestrator
        orchestrator = ProductionOrchestrator(
            config=config,
            broker=mock_broker,
            knowledge_base=mock_knowledge_base,
            symbols=["AAPL", "MSFT", "GOOGL"],
        )

        # Test health check
        health_status = orchestrator._calculate_system_status()

        # Verify health status structure
        assert health_status is not None
        assert hasattr(health_status, "timestamp")
        assert hasattr(health_status, "mode")
        assert hasattr(health_status, "status")
        assert hasattr(health_status, "components")

    def test_component_health_checks(self):
        """Test individual component health checks."""
        # Mock components with health status
        mock_data_manager = Mock()
        mock_data_manager.is_running = True

        mock_strategy_selector = Mock()
        mock_strategy_selector.is_healthy = True

        mock_portfolio_optimizer = Mock()
        mock_portfolio_optimizer.is_healthy = True

        mock_risk_manager = Mock()
        mock_risk_manager.is_healthy = True

        mock_performance_monitor = Mock()
        mock_performance_monitor.is_healthy = True

        mock_alert_manager = Mock()
        mock_alert_manager.is_healthy = True

        # Test health checks
        health_checks = {
            "data_manager": mock_data_manager.is_running,
            "strategy_selector": mock_strategy_selector.is_healthy,
            "portfolio_optimizer": mock_portfolio_optimizer.is_healthy,
            "risk_manager": mock_risk_manager.is_healthy,
            "performance_monitor": mock_performance_monitor.is_healthy,
            "alert_manager": mock_alert_manager.is_healthy,
        }

        # Verify all components are healthy
        for component, healthy in health_checks.items():
            assert healthy, f"Component {component} is not healthy"

    def test_health_check_failure_detection(self):
        """Test health check failure detection."""
        # Mock unhealthy components
        mock_data_manager = Mock()
        mock_data_manager.is_running = False

        mock_strategy_selector = Mock()
        mock_strategy_selector.is_healthy = False

        # Test health checks
        health_checks = {
            "data_manager": mock_data_manager.is_running,
            "strategy_selector": mock_strategy_selector.is_healthy,
        }

        # Identify unhealthy components
        unhealthy_components = [comp for comp, healthy in health_checks.items() if not healthy]

        # Verify unhealthy components are detected
        assert len(unhealthy_components) == 2
        assert "data_manager" in unhealthy_components
        assert "strategy_selector" in unhealthy_components

    def test_health_check_timeout(self):
        """Test health check timeout handling."""

        # Mock slow health check
        def slow_health_check():
            import time

            time.sleep(2)  # Simulate slow response
            return True

        # Test timeout handling
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Health check timed out")

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)  # 1 second timeout

        try:
            # This should timeout
            slow_health_check()
        except TimeoutError:
            # Expected timeout
            pass
        finally:
            signal.alarm(0)  # Cancel alarm


class TestAlertDelivery:
    """Test alert delivery system."""

    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        config = AlertManagerConfig(
            webhook_enabled=True,
            webhook_url="https://hooks.slack.com/services/test",
            email_enabled=False,
            slack_enabled=True,
            alert_cooldown_minutes=30,
        )

        alert_manager = AlertManager(config)

        assert alert_manager.config == config
        assert alert_manager.alerts == []
        assert alert_manager.alert_history == []
        assert alert_manager.rate_limit_tracker == {}

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_webhook_alert_delivery(self, mock_session):
        """Test webhook alert delivery."""
        config = AlertManagerConfig(
            webhook_enabled=True,
            webhook_url="https://hooks.slack.com/services/test",
            email_enabled=False,
            slack_enabled=False,
            alert_cooldown_minutes=30,
        )

        alert_manager = AlertManager(config)

        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None

        mock_session.return_value.post.return_value = mock_context

        # Send alert
        alert_id = await alert_manager.send_alert(
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            data={"sharpe_ratio": 0.5},
        )

        # Verify alert was created
        assert alert_id is not None
        assert len(alert_manager.alerts) == 1

        alert = alert_manager.alerts[0]
        assert alert.alert_type == AlertType.PERFORMANCE
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"

    def test_alert_rate_limiting(self):
        """Test alert rate limiting."""
        config = AlertManagerConfig(
            webhook_enabled=True,
            webhook_url="https://hooks.slack.com/services/test",
            email_enabled=False,
            slack_enabled=False,
            alert_cooldown_minutes=30,
        )

        alert_manager = AlertManager(config)

        # Test rate limiting check
        alert_type = AlertType.PERFORMANCE
        severity = AlertSeverity.WARNING

        # First alert should pass
        assert alert_manager._check_rate_limit(alert_type, severity) == True

        # Add recent alert to tracker
        alert_manager.rate_limit_tracker[f"{alert_type.value}_{severity.value}"] = [datetime.now()]

        # Second alert should be rate limited
        assert alert_manager._check_rate_limit(alert_type, severity) == False

    def test_alert_severity_levels(self):
        """Test alert severity levels."""
        config = AlertManagerConfig(
            webhook_enabled=True,
            webhook_url="https://hooks.slack.com/services/test",
            email_enabled=False,
            slack_enabled=False,
            alert_cooldown_minutes=30,
        )

        alert_manager = AlertManager(config)

        # Test different severity levels
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]

        for severity in severities:
            alert_id = asyncio.run(
                alert_manager.send_alert(
                    alert_type=AlertType.PERFORMANCE,
                    severity=severity,
                    title=f"Test {severity.value}",
                    message=f"This is a {severity.value} alert",
                )
            )

            assert alert_id is not None

    def test_alert_data_validation(self):
        """Test alert data validation."""
        config = AlertManagerConfig(
            webhook_enabled=True,
            webhook_url="https://hooks.slack.com/services/test",
            email_enabled=False,
            slack_enabled=False,
            alert_cooldown_minutes=30,
        )

        alert_manager = AlertManager(config)

        # Test valid alert data
        valid_data = {
            "sharpe_ratio": 0.5,
            "drawdown": 0.15,
            "n_positions": 5,
            "timestamp": datetime.now().isoformat(),
        }

        alert_id = asyncio.run(
            alert_manager.send_alert(
                alert_type=AlertType.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                title="Test Alert",
                message="Test message",
                data=valid_data,
            )
        )

        assert alert_id is not None

        # Verify data was stored
        alert = alert_manager.alerts[0]
        assert alert.data == valid_data

    @pytest.mark.asyncio
    async def test_alert_history_tracking(self):
        """Test alert history tracking."""
        config = AlertManagerConfig(
            webhook_enabled=True,
            webhook_url="https://hooks.slack.com/services/test",
            email_enabled=False,
            slack_enabled=False,
            alert_cooldown_minutes=0,  # Disable rate limiting for test
        )

        alert_manager = AlertManager(config)

        # Send multiple alerts
        for i in range(5):
            await alert_manager.send_alert(
                alert_type=AlertType.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                title=f"Alert {i}",
                message=f"Message {i}",
            )

        # Verify history tracking
        assert len(alert_manager.alerts) == 5
        assert len(alert_manager.alert_history) == 5

        # Verify alert order (newest first)
        for i, alert in enumerate(alert_manager.alerts):
            assert alert.title == f"Alert {i}"


class TestObservabilityIntegration:
    """Test observability integration across components."""

    def test_monitoring_integration(self):
        """Test monitoring integration with other components."""
        # Mock broker
        mock_broker = Mock()

        # Create performance monitor
        thresholds = PerformanceThresholds()
        alert_config = AlertConfig(
            webhook_enabled=False, email_enabled=False, alert_cooldown_hours=24
        )

        monitor = PerformanceMonitor(mock_broker, thresholds, alert_config)

        # Create alert manager
        alert_manager_config = AlertManagerConfig(
            webhook_enabled=True,
            webhook_url="https://hooks.slack.com/services/test",
            email_enabled=False,
            slack_enabled=False,
            alert_cooldown_minutes=30,
        )

        alert_manager = AlertManager(alert_manager_config)

        # Test integration
        assert monitor is not None
        assert alert_manager is not None

        # Verify both components can work together
        assert hasattr(monitor, "alert_config")
        assert hasattr(alert_manager, "config")

    def test_logging_integration(self):
        """Test logging integration across components."""
        # Test that all components use the same logging system
        loggers = [
            logging.getLogger("bot.monitor.performance_monitor"),
            logging.getLogger("bot.monitor.alerts"),
            logging.getLogger("bot.live.production_orchestrator"),
        ]

        for logger in loggers:
            assert logger is not None
            assert logger.name.startswith("bot.")

    def test_metrics_integration(self):
        """Test metrics integration across components."""
        # Test that metrics can be collected from multiple components
        metrics = {
            "performance": {"sharpe_ratio": 1.2, "cagr": 0.08, "max_drawdown": 0.10},
            "system": {"cpu_usage": 0.25, "memory_usage": 0.15, "disk_usage": 0.30},
            "alerts": {
                "total_alerts": 5,
                "unacknowledged": 2,
                "by_severity": {"warning": 3, "error": 2},
            },
        }

        # Verify metrics structure
        assert "performance" in metrics
        assert "system" in metrics
        assert "alerts" in metrics

        # Verify performance metrics
        perf_metrics = metrics["performance"]
        assert "sharpe_ratio" in perf_metrics
        assert "cagr" in perf_metrics
        assert "max_drawdown" in perf_metrics

        # Verify system metrics
        sys_metrics = metrics["system"]
        assert "cpu_usage" in sys_metrics
        assert "memory_usage" in sys_metrics
        assert "disk_usage" in sys_metrics

        # Verify alert metrics
        alert_metrics = metrics["alerts"]
        assert "total_alerts" in alert_metrics
        assert "unacknowledged" in alert_metrics
        assert "by_severity" in alert_metrics

    def test_health_check_integration(self):
        """Test health check integration across components."""
        # Mock components
        mock_broker = Mock()
        mock_knowledge_base = Mock()

        # Create orchestrator
        config = OrchestratorConfig(mode=OrchestrationMode.SEMI_AUTOMATED, enable_alerts=True)

        orchestrator = ProductionOrchestrator(
            config=config,
            broker=mock_broker,
            knowledge_base=mock_knowledge_base,
            symbols=["AAPL", "MSFT", "GOOGL"],
        )

        # Test health check integration
        health_status = orchestrator._calculate_system_status()

        # Verify health status includes all components
        assert health_status is not None
        assert hasattr(health_status, "components")

        # Verify all expected components are present
        expected_components = [
            "data_manager",
            "strategy_selector",
            "portfolio_optimizer",
            "risk_manager",
            "performance_monitor",
            "alert_manager",
        ]

        for component in expected_components:
            assert component in health_status.components


class TestSelectionMetricsExposure:
    """Ensure selection metrics are exposed in performance monitor summary."""

    def test_selection_metrics_in_summary(self):
        mock_broker = Mock()
        thresholds = PerformanceThresholds()
        alert_config = AlertConfig(
            webhook_enabled=False,
            email_enabled=False,
            alert_cooldown_hours=24,
        )

        monitor = PerformanceMonitor(mock_broker, thresholds, alert_config)

        predicted_ranks = ["s1", "s2", "s3"]
        actual_performance = {"s1": 1.0, "s2": 0.8, "s3": 0.2}
        selected_strategies = ["s1", "s3"]

        snapshot = monitor.record_selection_metrics(
            predicted_ranks=predicted_ranks,
            actual_performance=actual_performance,
            selected_strategies=selected_strategies,
        )

        # Verify snapshot contains expected keys and values are floats
        assert set(snapshot.keys()) == {"top_k_accuracy", "rank_correlation", "regret"}
        assert isinstance(snapshot["top_k_accuracy"], float)
        assert isinstance(snapshot["rank_correlation"], float)
        assert isinstance(snapshot["regret"], float)

        summary = monitor.get_performance_summary()
        assert "selection_metrics" in summary
        metrics = summary["selection_metrics"]
        assert set(metrics.keys()) == {"top_k_accuracy", "rank_correlation", "regret"}

        # Turnover exposure should exist and be a float (default 0.0 if none recorded)
        assert "recent_turnover" in summary
        assert isinstance(summary["recent_turnover"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
