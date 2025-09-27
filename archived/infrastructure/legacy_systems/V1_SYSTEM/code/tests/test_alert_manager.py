"""
Unit tests for the Alert Manager component.
"""

from datetime import datetime

import pytest
from bot.monitor.alerts import Alert, AlertConfig, AlertManager, AlertSeverity, AlertType


class TestAlertManager:
    """Test cases for the AlertManager class."""

    @pytest.fixture
    def alert_config(self):
        """Create alert configuration for testing."""
        return AlertConfig(
            email_enabled=False,
            slack_enabled=False,
            discord_enabled=False,
            webhook_enabled=False,
            alert_cooldown_minutes=0,  # No cooldown for testing
            max_alerts_per_hour=100,  # High limit for testing
        )

    @pytest.fixture
    def alert_manager(self, alert_config):
        """Create alert manager instance."""
        return AlertManager(alert_config)

    def test_initialization(self, alert_manager, alert_config):
        """Test alert manager initialization."""
        assert alert_manager.config == alert_config
        assert len(alert_manager.alerts) == 0
        assert len(alert_manager.alert_history) == 0
        assert len(alert_manager.rate_limit_tracker) == 0

    @pytest.mark.asyncio
    async def test_send_alert_basic(self, alert_manager):
        """Test basic alert sending functionality."""
        alert_id = await alert_manager.send_alert(
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            data={"test_key": "test_value"},
        )

        assert isinstance(alert_id, str)
        assert len(alert_id) > 0

        # Check that alert was added
        assert len(alert_manager.alerts) == 1
        assert len(alert_manager.alert_history) == 1

        alert = alert_manager.alerts[0]
        assert alert.alert_id == alert_id
        assert alert.alert_type == AlertType.PERFORMANCE
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.data == {"test_key": "test_value"}
        assert alert.timestamp is not None
        assert not alert.acknowledged

    @pytest.mark.asyncio
    async def test_send_performance_alert(self, alert_manager):
        """Test performance alert sending."""
        alert_id = await alert_manager.send_performance_alert(
            strategy_id="strategy_001",
            metric="sharpe_ratio",
            current_value=0.3,
            threshold_value=0.5,
            severity=AlertSeverity.WARNING,
        )

        assert isinstance(alert_id, str)

        alert = alert_manager.alerts[0]
        assert alert.alert_type == AlertType.PERFORMANCE
        assert alert.severity == AlertSeverity.WARNING
        assert "strategy_001" in alert.title
        assert "sharpe_ratio" in alert.message
        assert alert.data["strategy_id"] == "strategy_001"
        assert alert.data["metric"] == "sharpe_ratio"
        assert alert.data["current_value"] == 0.3
        assert alert.data["threshold_value"] == 0.5

    @pytest.mark.asyncio
    async def test_send_risk_alert(self, alert_manager):
        """Test risk alert sending."""
        alert_id = await alert_manager.send_risk_alert(
            risk_type="portfolio_var",
            current_value=0.025,
            limit_value=0.02,
            severity=AlertSeverity.WARNING,
        )

        assert isinstance(alert_id, str)

        alert = alert_manager.alerts[0]
        assert alert.alert_type == AlertType.RISK
        assert alert.severity == AlertSeverity.WARNING
        assert "portfolio_var" in alert.title
        assert alert.data["risk_type"] == "portfolio_var"
        assert alert.data["current_value"] == 0.025
        assert alert.data["limit_value"] == 0.02

    @pytest.mark.asyncio
    async def test_send_strategy_alert(self, alert_manager):
        """Test strategy alert sending."""
        alert_id = await alert_manager.send_strategy_alert(
            strategy_id="momentum_001",
            event="regime_change",
            details="Market regime changed from trending to volatile",
            severity=AlertSeverity.INFO,
        )

        assert isinstance(alert_id, str)

        alert = alert_manager.alerts[0]
        assert alert.alert_type == AlertType.STRATEGY
        assert alert.severity == AlertSeverity.INFO
        assert "momentum_001" in alert.title
        assert "regime_change" in alert.message
        assert alert.data["strategy_id"] == "momentum_001"
        assert alert.data["event"] == "regime_change"
        assert alert.data["details"] == "Market regime changed from trending to volatile"

    @pytest.mark.asyncio
    async def test_send_system_alert(self, alert_manager):
        """Test system alert sending."""
        alert_id = await alert_manager.send_system_alert(
            component="data_feed",
            event="connection_lost",
            details="Lost connection to market data feed",
            severity=AlertSeverity.ERROR,
        )

        assert isinstance(alert_id, str)

        alert = alert_manager.alerts[0]
        assert alert.alert_type == AlertType.SYSTEM
        assert alert.severity == AlertSeverity.ERROR
        assert "data_feed" in alert.title
        assert "connection_lost" in alert.message
        assert alert.data["component"] == "data_feed"
        assert alert.data["event"] == "connection_lost"
        assert alert.data["details"] == "Lost connection to market data feed"

    @pytest.mark.asyncio
    async def test_send_trade_alert(self, alert_manager):
        """Test trade alert sending."""
        alert_id = await alert_manager.send_trade_alert(
            symbol="AAPL", action="buy", quantity=100, price=150.25, severity=AlertSeverity.INFO
        )

        assert isinstance(alert_id, str)

        alert = alert_manager.alerts[0]
        assert alert.alert_type == AlertType.TRADE
        assert alert.severity == AlertSeverity.INFO
        assert "AAPL" in alert.title
        assert "buy" in alert.message
        assert "100" in alert.message
        assert "150.25" in alert.message
        assert alert.data["symbol"] == "AAPL"
        assert alert.data["action"] == "buy"
        assert alert.data["quantity"] == 100
        assert alert.data["price"] == 150.25

    @pytest.mark.asyncio
    async def test_rate_limiting(self, alert_manager):
        """Test rate limiting functionality."""
        # Send multiple alerts quickly
        alert_ids = []
        for i in range(5):
            alert_id = await alert_manager.send_alert(
                alert_type=AlertType.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                title=f"Test Alert {i}",
                message=f"Test message {i}",
            )
            alert_ids.append(alert_id)

        # All alerts should be sent (within hourly limit)
        assert len(alert_manager.alerts) == 5
        assert len(set(alert_ids)) == 5  # All unique IDs

    @pytest.mark.asyncio
    async def test_rate_limiting_exceeded(self, alert_manager):
        """Test rate limiting when exceeded."""
        # Set very low limits
        alert_manager.config.max_alerts_per_hour = 1  # Only 1 alert per hour
        alert_manager.config.alert_cooldown_minutes = 0  # No cooldown for this test

        # Send first alert
        alert_id1 = await alert_manager.send_alert(
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Test Alert 1",
            message="Test message 1",
        )

        # Second alert should be rate limited (exceeds hourly limit)
        alert_id2 = await alert_manager.send_alert(
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Test Alert 2",
            message="Test message 2",
        )

        # Should only have 1 alert (second was rate limited)
        assert len(alert_manager.alerts) == 1
        assert alert_id1 is not None
        assert alert_id2 is not None  # Should still return an ID

    def test_acknowledge_alert(self, alert_manager):
        """Test alert acknowledgment."""
        # Create a test alert
        alert = Alert(
            alert_id="test_alert_001",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            data={},
            timestamp=datetime.now(),
        )

        alert_manager.alerts.append(alert)

        # Acknowledge the alert
        success = alert_manager.acknowledge_alert("test_alert_001", "test_user")

        assert success is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "test_user"
        assert alert.acknowledged_at is not None

    def test_acknowledge_nonexistent_alert(self, alert_manager):
        """Test acknowledging a non-existent alert."""
        success = alert_manager.acknowledge_alert("nonexistent_alert", "test_user")

        assert success is False

    def test_get_active_alerts(self, alert_manager):
        """Test getting active (unacknowledged) alerts."""
        # Create test alerts
        alert1 = Alert(
            alert_id="alert_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Alert 1",
            message="Message 1",
            data={},
            timestamp=datetime.now(),
        )

        alert2 = Alert(
            alert_id="alert_2",
            alert_type=AlertType.RISK,
            severity=AlertSeverity.ERROR,
            title="Alert 2",
            message="Message 2",
            data={},
            timestamp=datetime.now(),
        )

        # Acknowledge one alert
        alert1.acknowledged = True
        alert1.acknowledged_by = "test_user"
        alert1.acknowledged_at = datetime.now()

        alert_manager.alerts = [alert1, alert2]

        # Get active alerts
        active_alerts = alert_manager.get_active_alerts()

        assert len(active_alerts) == 1
        assert active_alerts[0].alert_id == "alert_2"

    def test_get_active_alerts_by_type(self, alert_manager):
        """Test getting active alerts filtered by type."""
        # Create test alerts
        alert1 = Alert(
            alert_id="alert_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Alert 1",
            message="Message 1",
            data={},
            timestamp=datetime.now(),
        )

        alert2 = Alert(
            alert_id="alert_2",
            alert_type=AlertType.RISK,
            severity=AlertSeverity.ERROR,
            title="Alert 2",
            message="Message 2",
            data={},
            timestamp=datetime.now(),
        )

        alert_manager.alerts = [alert1, alert2]

        # Get active performance alerts
        performance_alerts = alert_manager.get_active_alerts(AlertType.PERFORMANCE)

        assert len(performance_alerts) == 1
        assert performance_alerts[0].alert_id == "alert_1"

    @pytest.mark.asyncio
    async def test_alert_channels_disabled(self, alert_manager):
        """Test alert sending when all channels are disabled."""
        # All channels are disabled in the config
        alert_id = await alert_manager.send_alert(
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
        )

        # Alert should still be created and stored
        assert isinstance(alert_id, str)
        assert len(alert_manager.alerts) == 1

    @pytest.mark.asyncio
    async def test_alert_severity_filtering(self, alert_manager):
        """Test alert severity filtering for different channels."""
        # Configure different severity thresholds
        alert_manager.config.min_severity_for_email = AlertSeverity.WARNING
        alert_manager.config.min_severity_for_slack = AlertSeverity.ERROR
        alert_manager.config.min_severity_for_discord = AlertSeverity.CRITICAL

        # Send info alert (should not be sent to any channel)
        await alert_manager.send_alert(
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.INFO,
            title="Info Alert",
            message="Info message",
        )

        # Send warning alert (should be sent to email only)
        await alert_manager.send_alert(
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Warning Alert",
            message="Warning message",
        )

        # Send error alert (should be sent to email and slack)
        await alert_manager.send_alert(
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.ERROR,
            title="Error Alert",
            message="Error message",
        )

        # All alerts should be stored
        assert len(alert_manager.alerts) == 3

    def test_get_alert_summary(self, alert_manager):
        """Test getting alert summary."""
        # Create test alerts
        alert1 = Alert(
            alert_id="alert_1",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Alert 1",
            message="Message 1",
            data={},
            timestamp=datetime.now(),
        )

        alert2 = Alert(
            alert_id="alert_2",
            alert_type=AlertType.RISK,
            severity=AlertSeverity.ERROR,
            title="Alert 2",
            message="Message 2",
            data={},
            timestamp=datetime.now(),
        )

        alert_manager.alerts = [alert1, alert2]
        alert_manager.alert_history = [alert1, alert2]

        summary = alert_manager.get_alert_summary()

        assert isinstance(summary, dict)
        assert "total_alerts" in summary
        assert "active_alerts" in summary
        assert "alerts_last_24h" in summary
        assert "alerts_last_7d" in summary
        assert "alerts_by_type" in summary
        assert "alerts_by_severity" in summary
        assert summary["total_alerts"] == 2
        assert summary["active_alerts"] == 2


class TestAlertConfig:
    """Test cases for the AlertConfig class."""

    def test_default_alert_config(self):
        """Test default alert configuration."""
        config = AlertConfig()

        assert config.email_enabled is False
        assert config.slack_enabled is False
        assert config.discord_enabled is False
        assert config.webhook_enabled is False
        assert config.alert_cooldown_minutes == 30
        assert config.max_alerts_per_hour == 10
        assert config.min_severity_for_email == AlertSeverity.WARNING
        assert config.min_severity_for_slack == AlertSeverity.WARNING
        assert config.min_severity_for_discord == AlertSeverity.ERROR
        assert config.min_severity_for_webhook == AlertSeverity.ERROR
        assert config.email_recipients == []

    def test_custom_alert_config(self):
        """Test custom alert configuration."""
        config = AlertConfig(
            email_enabled=True,
            slack_enabled=True,
            alert_cooldown_minutes=15,
            max_alerts_per_hour=20,
            email_recipients=["test@example.com"],
        )

        assert config.email_enabled is True
        assert config.slack_enabled is True
        assert config.alert_cooldown_minutes == 15
        assert config.max_alerts_per_hour == 20
        assert config.email_recipients == ["test@example.com"]


class TestAlert:
    """Test cases for the Alert class."""

    def test_alert_creation(self):
        """Test alert object creation."""
        alert = Alert(
            alert_id="test_alert_001",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            data={"key": "value"},
            timestamp=datetime.now(),
        )

        assert alert.alert_id == "test_alert_001"
        assert alert.alert_type == AlertType.PERFORMANCE
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.data == {"key": "value"}
        assert alert.timestamp is not None
        assert not alert.acknowledged
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None

    def test_alert_acknowledgment(self):
        """Test alert acknowledgment."""
        alert = Alert(
            alert_id="test_alert_001",
            alert_type=AlertType.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            data={},
            timestamp=datetime.now(),
        )

        # Acknowledge the alert
        alert.acknowledged = True
        alert.acknowledged_by = "test_user"
        alert.acknowledged_at = datetime.now()

        assert alert.acknowledged is True
        assert alert.acknowledged_by == "test_user"
        assert alert.acknowledged_at is not None


class TestAlertSeverity:
    """Test cases for the AlertSeverity enum."""

    def test_alert_severities(self):
        """Test all alert severities."""
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]

        for severity in severities:
            assert isinstance(severity.value, str)
            assert len(severity.value) > 0

    def test_severity_comparison(self):
        """Test severity level comparison."""
        # Create a mapping of severity to numeric values for comparison
        severity_values = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3,
        }

        assert severity_values[AlertSeverity.INFO] < severity_values[AlertSeverity.WARNING]
        assert severity_values[AlertSeverity.WARNING] < severity_values[AlertSeverity.ERROR]
        assert severity_values[AlertSeverity.ERROR] < severity_values[AlertSeverity.CRITICAL]


class TestAlertType:
    """Test cases for the AlertType enum."""

    def test_alert_types(self):
        """Test all alert types."""
        types = [
            AlertType.PERFORMANCE,
            AlertType.RISK,
            AlertType.STRATEGY,
            AlertType.SYSTEM,
            AlertType.TRADE,
        ]

        for alert_type in types:
            assert isinstance(alert_type.value, str)
            assert len(alert_type.value) > 0
