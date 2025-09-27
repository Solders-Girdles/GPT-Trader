"""
Integration tests for canary profile configuration and runtime guards.

Tests:
- YAML profile loading
- Runtime guard triggers
- Alert dispatching
- Trading window enforcement
- Risk limit enforcement
"""

import pytest
pytestmark = pytest.mark.integration
import asyncio
import yaml
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bot_v2.monitoring.runtime_guards import (
    RuntimeGuardManager, DailyLossGuard, StaleMarkGuard,
    ErrorRateGuard, GuardConfig, AlertSeverity, GuardStatus
)
from bot_v2.monitoring.alerts import (
    AlertDispatcher, Alert, LogChannel, create_risk_alert
)


class TestCanaryYAMLProfile:
    """Test canary.yaml profile parsing and validation."""
    
    def test_yaml_is_valid(self):
        """Test that canary.yaml is valid YAML."""
        yaml_path = Path(__file__).parent.parent.parent / "config" / "profiles" / "canary.yaml"
        
        assert yaml_path.exists(), "canary.yaml should exist"
        
        # Load and parse YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config is not None, "YAML should parse successfully"
        assert config['profile_name'] == 'canary'
        assert config['environment'] == 'production'
    
    def test_critical_safety_settings(self):
        """Test that critical safety settings are present."""
        yaml_path = Path(__file__).parent.parent.parent / "config" / "profiles" / "canary.yaml"
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Trading settings
        trading = config['trading']
        assert trading['mode'] == 'reduce_only'
        assert trading['symbols'] == ['BTC-PERP']
        assert trading['position_sizing']['max_position_size'] == 0.01
        assert trading['position_sizing']['max_notional_value'] == 500
        
        # Risk settings
        risk = config['risk_management']
        assert risk['daily_loss_limit'] == 10.00
        assert risk['max_leverage'] == 1.0
        assert risk['circuit_breakers']['consecutive_losses'] == 2
        assert risk['circuit_breakers']['stale_mark_seconds'] == 60
        
        # Session settings
        session = config['session']
        assert session['start_time'] == '14:00'
        assert session['end_time'] == '15:00'
        assert session['max_session_duration_minutes'] == 60
    
    def test_alert_configuration(self):
        """Test alert channel configuration."""
        yaml_path = Path(__file__).parent.parent.parent / "config" / "profiles" / "canary.yaml"
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        alerts = config['monitoring']['alerts']
        assert alerts['enabled'] is True
        
        # Check channels
        channels = alerts['channels']
        channel_types = [c['type'] for c in channels]
        assert 'log' in channel_types
        assert 'slack' in channel_types
        assert 'pagerduty' in channel_types
        
        # Check conditions
        conditions = alerts['conditions']
        condition_names = [c['name'] for c in conditions]
        assert 'daily_loss_breach' in condition_names
        assert 'stale_marks' in condition_names


class TestRuntimeGuards:
    """Test runtime guard functionality."""
    
    def test_daily_loss_guard(self):
        """Test daily loss limit guard."""
        config = GuardConfig(
            name="daily_loss",
            threshold=10.0,
            severity=AlertSeverity.ERROR,
            auto_shutdown=True
        )
        guard = DailyLossGuard(config)
        
        # First loss within limit
        context = {'pnl': -5.0}
        alert = guard.check(context)
        assert alert is None
        assert guard.status == GuardStatus.WARNING  # 50% of limit
        
        # Second loss breaches limit
        context = {'pnl': -6.0}
        alert = guard.check(context)
        assert alert is not None
        assert alert.severity == AlertSeverity.ERROR
        assert "Daily loss limit breached" in alert.message
        assert guard.status == GuardStatus.BREACHED
    
    def test_stale_mark_guard(self):
        """Test stale mark detection."""
        config = GuardConfig(
            name="stale_marks",
            threshold=60,  # 60 seconds
            severity=AlertSeverity.WARNING
        )
        guard = StaleMarkGuard(config)
        
        # Fresh mark
        context = {
            'symbol': 'BTC-PERP',
            'mark_timestamp': datetime.now()
        }
        alert = guard.check(context)
        assert alert is None
        
        # Stale mark
        context = {
            'symbol': 'BTC-PERP',
            'mark_timestamp': datetime.now() - timedelta(seconds=90)
        }
        alert = guard.check(context)
        assert alert is not None
        assert "Stale marks detected" in alert.message
    
    def test_error_rate_guard(self):
        """Test error rate monitoring."""
        config = GuardConfig(
            name="error_rate",
            threshold=3,  # 3 errors
            window_seconds=60,
            severity=AlertSeverity.ERROR
        )
        guard = ErrorRateGuard(config)
        
        # Simulate errors
        for i in range(3):
            context = {'error': True}
            alert = guard.check(context)
            if i < 2:
                assert alert is None  # Below threshold
        
        # Fourth error triggers alert
        context = {'error': True}
        alert = guard.check(context)
        assert alert is not None
        assert "High error rate" in alert.message
    
    def test_guard_cooldown(self):
        """Test guard cooldown prevents alert spam."""
        config = GuardConfig(
            name="test_guard",
            threshold=10.0,
            cooldown_seconds=5
        )
        guard = DailyLossGuard(config)
        
        # First breach
        context = {'pnl': -11.0}
        alert1 = guard.check(context)
        assert alert1 is not None
        
        # Immediate second check during cooldown
        alert2 = guard.check(context)
        assert alert2 is None  # Cooldown prevents alert


class TestAlertDispatcher:
    """Test alert dispatching system."""
    
    @pytest.mark.asyncio
    async def test_log_channel(self):
        """Test log channel always works."""
        dispatcher = AlertDispatcher()
        
        alert = create_risk_alert(
            "Test Alert",
            "This is a test",
            AlertSeverity.WARNING
        )
        
        results = await dispatcher.dispatch(alert)
        assert 'log' in results
        assert results['log'] is True
    
    @pytest.mark.asyncio
    async def test_severity_filtering(self):
        """Test channels filter by severity."""
        dispatcher = AlertDispatcher()
        
        # Add channel with high threshold
        mock_channel = AsyncMock()
        mock_channel.min_severity = AlertSeverity.ERROR
        mock_channel.send = AsyncMock(return_value=False)
        dispatcher.add_channel('test', mock_channel)
        
        # Send low severity alert
        alert = create_risk_alert(
            "Low Priority",
            "Not urgent",
            AlertSeverity.INFO
        )
        
        results = await dispatcher.dispatch(alert)
        # Should not be sent to test channel due to severity
        mock_channel.send.assert_not_called()
    
    def test_alert_history(self):
        """Test alert history tracking."""
        dispatcher = AlertDispatcher()
        
        # Create multiple alerts
        for i in range(5):
            alert = Alert(
                timestamp=datetime.now(),
                source="test",
                severity=AlertSeverity.INFO if i % 2 == 0 else AlertSeverity.WARNING,
                title=f"Alert {i}",
                message=f"Message {i}"
            )
            asyncio.run(dispatcher.dispatch(alert))
        
        # Check history
        all_alerts = dispatcher.get_recent_alerts(count=10)
        assert len(all_alerts) == 5
        
        # Filter by severity
        warnings = dispatcher.get_recent_alerts(severity=AlertSeverity.WARNING)
        assert len(warnings) == 2


class TestTradingWindowGuard:
    """Test trading window enforcement."""
    
    def test_within_window(self):
        """Test detection of being within trading window."""
        from scripts.run_perps_bot import BotConfig
        
        config = BotConfig(
            profile="test",
            trading_window_start=time(9, 0),
            trading_window_end=time(17, 0),
            trading_days=['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        )
        
        # Mock current time to be Wednesday 10:00
        from scripts.run_perps_bot import TradingBot
        bot = TradingBot(config)
        with patch.object(bot, "_now", return_value=datetime(2024, 1, 3, 10, 0), create=True):
            assert bot.is_within_trading_window() is True
    
    def test_outside_window_time(self):
        """Test detection of being outside trading hours."""
        from scripts.run_perps_bot import BotConfig
        
        config = BotConfig(
            profile="test",
            trading_window_start=time(9, 0),
            trading_window_end=time(17, 0),
            trading_days=['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        )
        
        # Mock current time to be Wednesday 20:00 (outside hours)
        from scripts.run_perps_bot import TradingBot
        bot = TradingBot(config)
        with patch.object(bot, "_now", return_value=datetime(2024, 1, 3, 20, 0), create=True):
            assert bot.is_within_trading_window() is False
    
    def test_weekend_blocked(self):
        """Test that weekends are blocked."""
        from scripts.run_perps_bot import BotConfig
        
        config = BotConfig(
            profile="test",
            trading_window_start=time(9, 0),
            trading_window_end=time(17, 0),
            trading_days=['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        )
        
        # Mock current time to be Saturday 10:00
        from scripts.run_perps_bot import TradingBot
        bot = TradingBot(config)
        with patch.object(bot, "_now", return_value=datetime(2024, 1, 6, 10, 0), create=True):
            assert bot.is_within_trading_window() is False


class TestCanaryProfileIntegration:
    """Integration tests for full canary profile."""
    
    @pytest.mark.asyncio
    async def test_canary_profile_loading(self):
        """Test loading canary profile in bot."""
        from scripts.run_perps_bot import BotConfig
        
        config = BotConfig.from_profile('canary')
        
        assert config.profile.value == 'canary'
        assert config.reduce_only_mode is True
        assert config.max_position_size == Decimal('500')
        assert config.daily_loss_limit == Decimal('10')
        assert config.symbols == ['BTC-PERP']
        assert config.time_in_force == 'IOC'
    
    @pytest.mark.asyncio  
    async def test_guard_manager_creation(self):
        """Test guard manager is created with correct guards."""
        from bot_v2.monitoring.runtime_guards import create_default_guards
        
        config = {
            'risk_management': {
                'daily_loss_limit': 10,
                'max_drawdown_pct': 2,
                'circuit_breakers': {
                    'stale_mark_seconds': 60,
                    'error_threshold': 3
                }
            }
        }
        
        manager = create_default_guards(config)
        
        assert 'daily_loss' in manager.guards
        assert 'stale_marks' in manager.guards
        assert 'error_rate' in manager.guards
        assert 'max_drawdown' in manager.guards
        
        # Check thresholds
        assert manager.guards['daily_loss'].config.threshold == 10
        assert manager.guards['stale_marks'].config.threshold == 60
        assert manager.guards['error_rate'].config.threshold == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
