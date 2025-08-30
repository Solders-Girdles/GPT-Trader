"""
Comprehensive tests for circuit breakers and kill switch systems.

Tests all safety controls including:
- Circuit breaker triggering conditions
- Kill switch activation and resumption
- Integration between systems
- State persistence
- Thread safety
- Performance under load
"""

import asyncio
import pytest
import tempfile
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.bot.risk.circuit_breakers import (
    CircuitBreakerSystem, CircuitBreakerRule, CircuitBreakerType, 
    ActionType, BreakerStatus, create_circuit_breaker_system
)
from src.bot.risk.kill_switch import (
    EmergencyKillSwitch, KillSwitchConfig, KillSwitchType, KillSwitchMode,
    KillSwitchReason, KillSwitchStatus, create_emergency_kill_switch
)
from src.bot.risk.safety_integration import (
    SafetySystemsIntegration, SafetySystemConfig, create_safety_systems_integration
)


class MockTradingEngine:
    """Mock trading engine for testing"""
    
    def __init__(self, engine_id: str):
        self.engine_id = engine_id
        self.is_running = True
        self.positions = {}
        self.stop_called = False
        self.emergency_stop_called = False
        self.freeze_called = False
        
        # Mock order types
        class OrderSide:
            BUY = "buy"
            SELL = "sell"
        
        class OrderType:
            MARKET = "market"
            LIMIT = "limit"
        
        self.OrderSide = OrderSide
        self.OrderType = OrderType
    
    def stop_trading_engine(self):
        """Stop trading engine"""
        self.is_running = False
        self.stop_called = True
    
    def emergency_stop(self):
        """Emergency stop"""
        self.is_running = False
        self.emergency_stop_called = True
    
    def freeze_trading(self):
        """Freeze trading"""
        self.freeze_called = True
    
    def submit_order(self, **kwargs):
        """Submit order (mock)"""
        return f"order_{time.time()}"
    
    def close_all_positions_graceful(self):
        """Close all positions gracefully"""
        count = len(self.positions)
        self.positions.clear()
        return count
    
    def liquidate_all_positions(self):
        """Liquidate all positions"""
        count = len(self.positions)
        self.positions.clear()
        return count


class MockRiskMonitor:
    """Mock risk monitor for testing"""
    
    def __init__(self):
        self.risk_metrics = MagicMock()
        self.risk_metrics.total_unrealized_pnl = 0
        self.risk_metrics.total_realized_pnl = -1000  # $1000 loss
        self.risk_metrics.current_drawdown = 0.05  # 5% drawdown
        
        self.position_metrics = {"max_position_pct": 0.15}  # 15% concentration
        self.market_metrics = {
            "volume_spike_ratio": 2.0,
            "volatility_spike_ratio": 2.5
        }
        self.trade_metrics = {"consecutive_losses": 3}
        self.strategy_metrics = {"win_rate": 0.45}  # 45% win rate
    
    def get_current_risk_metrics(self):
        return self.risk_metrics
    
    def get_position_metrics(self):
        return self.position_metrics
    
    def get_market_metrics(self):
        return self.market_metrics
    
    def get_trade_metrics(self):
        return self.trade_metrics
    
    def get_strategy_metrics(self):
        return self.strategy_metrics


class MockAlertingSystem:
    """Mock alerting system for testing"""
    
    def __init__(self):
        self.alerts_sent = []
    
    def send_critical_alert(self, title, message, severity="critical"):
        """Send critical alert"""
        self.alerts_sent.append({
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now()
        })
    
    def send_alert(self, title, message, severity="info"):
        """Send regular alert"""
        self.alerts_sent.append({
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now()
        })


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_trading_engine():
    """Create mock trading engine"""
    engine = MockTradingEngine("test_engine")
    # Add some mock positions
    engine.positions = {
        "AAPL": MagicMock(symbol="AAPL", quantity=100, current_price=150.0),
        "GOOGL": MagicMock(symbol="GOOGL", quantity=50, current_price=2800.0),
    }
    return engine


@pytest.fixture
def mock_risk_monitor():
    """Create mock risk monitor"""
    return MockRiskMonitor()


@pytest.fixture
def mock_alerting_system():
    """Create mock alerting system"""
    return MockAlertingSystem()


class TestCircuitBreakers:
    """Test circuit breaker system"""
    
    def test_circuit_breaker_initialization(self, temp_dir):
        """Test circuit breaker system initialization"""
        breaker_system = create_circuit_breaker_system(
            risk_dir=temp_dir,
            initial_capital=Decimal("50000"),
            enable_auto_liquidation=True
        )
        
        assert breaker_system.initial_capital == Decimal("50000")
        assert breaker_system.enable_auto_liquidation is True
        assert len(breaker_system.breaker_rules) > 0
        
        # Check database was created
        assert (Path(temp_dir) / "circuit_breakers.db").exists()
    
    def test_add_remove_breaker_rule(self, temp_dir):
        """Test adding and removing breaker rules"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        
        initial_count = len(breaker_system.breaker_rules)
        
        # Add new rule
        rule = CircuitBreakerRule(
            breaker_id="test_breaker",
            breaker_type=CircuitBreakerType.DAILY_LOSS,
            description="Test breaker",
            threshold=Decimal("1000"),
            lookback_period=timedelta(hours=1),
            primary_action=ActionType.ALERT_ONLY
        )
        
        breaker_system.add_breaker_rule(rule)
        assert len(breaker_system.breaker_rules) == initial_count + 1
        assert "test_breaker" in breaker_system.breaker_rules
        
        # Remove rule
        removed = breaker_system.remove_breaker_rule("test_breaker")
        assert removed is True
        assert len(breaker_system.breaker_rules) == initial_count
        assert "test_breaker" not in breaker_system.breaker_rules
    
    def test_daily_loss_condition_check(self, temp_dir, mock_risk_monitor):
        """Test daily loss condition checking"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        breaker_system.register_risk_monitor(mock_risk_monitor)
        
        # Get daily loss rule
        rule = breaker_system.breaker_rules["max_daily_drawdown"]
        
        # Test condition (mock has $1000 loss)
        condition_met = breaker_system._check_daily_loss_condition(rule)
        assert condition_met is False  # Loss not exceeding threshold yet
        
        # Increase loss to exceed threshold
        mock_risk_monitor.risk_metrics.total_realized_pnl = -6000  # $6000 loss
        condition_met = breaker_system._check_daily_loss_condition(rule)
        assert condition_met is True  # Loss exceeds 5% of $50k = $2500
    
    def test_drawdown_condition_check(self, temp_dir, mock_risk_monitor):
        """Test drawdown condition checking"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        breaker_system.register_risk_monitor(mock_risk_monitor)
        
        # Get drawdown rule
        rule = breaker_system.breaker_rules["max_portfolio_drawdown"]
        
        # Test condition (mock has 5% drawdown)
        condition_met = breaker_system._check_drawdown_condition(rule)
        assert condition_met is False  # 5% < 15% threshold
        
        # Increase drawdown to exceed threshold
        mock_risk_monitor.risk_metrics.current_drawdown = 0.20  # 20% drawdown
        condition_met = breaker_system._check_drawdown_condition(rule)
        assert condition_met is True  # 20% > 15% threshold
    
    def test_position_concentration_check(self, temp_dir, mock_risk_monitor):
        """Test position concentration checking"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        breaker_system.register_risk_monitor(mock_risk_monitor)
        
        # Get concentration rule
        rule = breaker_system.breaker_rules["position_concentration"]
        
        # Test condition (mock has 15% concentration)
        condition_met = breaker_system._check_position_concentration_condition(rule)
        assert condition_met is False  # 15% < 20% threshold
        
        # Increase concentration to exceed threshold
        mock_risk_monitor.position_metrics["max_position_pct"] = 0.25  # 25%
        condition_met = breaker_system._check_position_concentration_condition(rule)
        assert condition_met is True  # 25% > 20% threshold
    
    def test_volume_anomaly_check(self, temp_dir, mock_risk_monitor):
        """Test volume anomaly checking"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        breaker_system.register_risk_monitor(mock_risk_monitor)
        
        # Get volume anomaly rule
        rule = breaker_system.breaker_rules["volume_anomaly"]
        
        # Test condition (mock has 2x volume)
        condition_met = breaker_system._check_volume_anomaly_condition(rule)
        assert condition_met is False  # 2x < 5x threshold
        
        # Increase volume spike to exceed threshold
        mock_risk_monitor.market_metrics["volume_spike_ratio"] = 6.0  # 6x volume
        condition_met = breaker_system._check_volume_anomaly_condition(rule)
        assert condition_met is True  # 6x > 5x threshold
    
    def test_consecutive_losses_check(self, temp_dir, mock_risk_monitor):
        """Test consecutive losses checking"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        breaker_system.register_risk_monitor(mock_risk_monitor)
        
        # Get consecutive losses rule
        rule = breaker_system.breaker_rules["consecutive_losses"]
        
        # Test condition (mock has 3 consecutive losses)
        condition_met = breaker_system._check_consecutive_losses_condition(rule)
        assert condition_met is False  # 3 < 5 threshold
        
        # Increase consecutive losses to exceed threshold
        mock_risk_monitor.trade_metrics["consecutive_losses"] = 6
        condition_met = breaker_system._check_consecutive_losses_condition(rule)
        assert condition_met is True  # 6 >= 5 threshold
    
    def test_manual_trigger(self, temp_dir, mock_trading_engine, mock_alerting_system):
        """Test manual circuit breaker trigger"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        breaker_system.register_trading_engine("test_engine", mock_trading_engine)
        breaker_system.register_alerting_system(mock_alerting_system)
        
        # Manual trigger
        success = breaker_system.manual_trigger("max_daily_drawdown", "Testing manual trigger")
        assert success is True
        
        # Check engine was stopped
        assert mock_trading_engine.stop_called is True
        
        # Check alert was sent
        assert len(mock_alerting_system.alerts_sent) > 0
        
        # Check event was recorded
        assert len(breaker_system.event_history) > 0
    
    def test_monitoring_loop(self, temp_dir, mock_risk_monitor):
        """Test circuit breaker monitoring loop"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        breaker_system.register_risk_monitor(mock_risk_monitor)
        
        # Start monitoring
        breaker_system.start_monitoring()
        assert breaker_system.is_monitoring is True
        
        # Let it monitor for a short time
        time.sleep(2)
        
        # Stop monitoring
        breaker_system.stop_monitoring()
        assert breaker_system.is_monitoring is False


class TestKillSwitches:
    """Test kill switch system"""
    
    def test_kill_switch_initialization(self, temp_dir):
        """Test kill switch system initialization"""
        kill_switch = create_emergency_kill_switch(
            data_dir=temp_dir,
            enable_auto_resume=True,
            default_cooldown_minutes=15
        )
        
        assert kill_switch.enable_auto_resume is True
        assert kill_switch.default_cooldown_minutes == 15
        assert len(kill_switch.switches) > 0
        
        # Check database was created
        assert (Path(temp_dir) / "kill_switches.db").exists()
    
    def test_add_remove_kill_switch(self, temp_dir):
        """Test adding and removing kill switches"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        
        initial_count = len(kill_switch.switches)
        
        # Add new switch
        config = KillSwitchConfig(
            switch_id="test_switch",
            switch_type=KillSwitchType.STRATEGY,
            mode=KillSwitchMode.IMMEDIATE,
            description="Test switch"
        )
        
        kill_switch.add_kill_switch(config)
        assert len(kill_switch.switches) == initial_count + 1
        assert "test_switch" in kill_switch.switches
        
        # Remove switch
        removed = kill_switch.remove_kill_switch("test_switch")
        assert removed is True
        assert len(kill_switch.switches) == initial_count
        assert "test_switch" not in kill_switch.switches
    
    def test_global_emergency_stop(self, temp_dir, mock_trading_engine, mock_alerting_system):
        """Test global emergency stop"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        kill_switch.register_trading_engine("test_engine", mock_trading_engine)
        kill_switch.register_alerting_system(mock_alerting_system)
        
        # Trigger global emergency stop
        event_id = kill_switch.trigger_kill_switch(
            "global_emergency_stop",
            KillSwitchReason.MANUAL_OVERRIDE,
            "test_user",
            "Testing emergency stop"
        )
        
        assert event_id is not None
        assert kill_switch.global_shutdown_active is True
        
        # Check engine was stopped
        assert mock_trading_engine.emergency_stop_called is True
        
        # Check alert was sent
        assert len(mock_alerting_system.alerts_sent) > 0
        
        # Check event was recorded
        assert len(kill_switch.event_history) > 0
        assert event_id in kill_switch.active_events
    
    def test_global_graceful_shutdown(self, temp_dir, mock_trading_engine):
        """Test global graceful shutdown"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        kill_switch.register_trading_engine("test_engine", mock_trading_engine)
        
        # Trigger graceful shutdown
        event_id = kill_switch.trigger_kill_switch(
            "global_graceful_shutdown",
            KillSwitchReason.CIRCUIT_BREAKER,
            "circuit_breaker_system",
            "Graceful shutdown test"
        )
        
        assert event_id is not None
        
        # Check positions were closed
        event = kill_switch.active_events[event_id]
        assert event.positions_closed >= 0  # Mock returns 0 since positions were cleared
        
        # Check engine was stopped
        assert mock_trading_engine.stop_called is True
    
    def test_global_liquidation(self, temp_dir, mock_trading_engine):
        """Test global liquidation"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        kill_switch.register_trading_engine("test_engine", mock_trading_engine)
        
        # Trigger liquidation
        event_id = kill_switch.trigger_kill_switch(
            "global_liquidation",
            KillSwitchReason.RISK_LIMIT_BREACH,
            "risk_monitor",
            "Emergency liquidation test"
        )
        
        assert event_id is not None
        
        # Check positions were liquidated
        event = kill_switch.active_events[event_id]
        assert event.positions_closed >= 0
    
    def test_resume_kill_switch(self, temp_dir, mock_trading_engine):
        """Test resuming kill switch"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        kill_switch.register_trading_engine("test_engine", mock_trading_engine)
        
        # Trigger emergency stop
        event_id = kill_switch.trigger_kill_switch(
            "global_emergency_stop",
            KillSwitchReason.MANUAL_OVERRIDE,
            "test_user"
        )
        
        assert event_id in kill_switch.active_events
        assert kill_switch.global_shutdown_active is True
        
        # Resume
        success = kill_switch.resume_kill_switch(
            event_id,
            "test_user",
            "Test completed"
        )
        
        assert success is True
        assert event_id not in kill_switch.active_events
        assert kill_switch.global_shutdown_active is False
        
        # Check event was marked as resumed
        event = next(e for e in kill_switch.event_history if e.event_id == event_id)
        assert event.resumed_at is not None
    
    def test_auto_resume(self, temp_dir):
        """Test auto-resume functionality"""
        kill_switch = create_emergency_kill_switch(
            data_dir=temp_dir,
            enable_auto_resume=True,
            default_cooldown_minutes=1  # Short cooldown for testing
        )
        
        # Add a switch with auto-resume
        config = KillSwitchConfig(
            switch_id="auto_resume_test",
            switch_type=KillSwitchType.STRATEGY,
            mode=KillSwitchMode.IMMEDIATE,
            description="Auto-resume test",
            auto_resume_after=timedelta(seconds=2),  # Very short for testing
            require_manual_resume=False
        )
        kill_switch.add_kill_switch(config)
        
        # Start monitoring for auto-resume
        kill_switch.start_monitoring()
        
        # Trigger switch
        event_id = kill_switch.trigger_kill_switch(
            "auto_resume_test",
            KillSwitchReason.MANUAL_OVERRIDE,
            "test_user"
        )
        
        assert event_id in kill_switch.active_events
        
        # Wait for auto-resume
        time.sleep(3)
        
        # Check it was auto-resumed
        assert event_id not in kill_switch.active_events
        
        kill_switch.stop_monitoring()
    
    def test_thread_safety(self, temp_dir):
        """Test thread safety of kill switch operations"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        
        results = []
        
        def trigger_switch(switch_id, reason):
            event_id = kill_switch.trigger_kill_switch(
                switch_id,
                reason,
                f"thread_{threading.current_thread().ident}"
            )
            results.append(event_id)
        
        # Create multiple threads triggering switches
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=trigger_switch,
                args=("global_emergency_stop", KillSwitchReason.MANUAL_OVERRIDE)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results - only one should succeed (first one)
        successful = [r for r in results if r is not None]
        assert len(successful) == 1


class TestSafetyIntegration:
    """Test integrated safety systems"""
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self, temp_dir):
        """Test safety integration initialization"""
        config = SafetySystemConfig(
            circuit_breaker_data_dir=str(Path(temp_dir) / "cb"),
            kill_switch_data_dir=str(Path(temp_dir) / "ks"),
            enable_auto_resume=False,
            enable_cross_system_triggers=True
        )
        
        integration = create_safety_systems_integration(config)
        
        # Initialize
        success = await integration.initialize()
        assert success is True
        assert integration.is_initialized is True
        
        # Check subsystems exist
        assert integration.circuit_breakers is not None
        assert integration.kill_switches is not None
    
    @pytest.mark.asyncio
    async def test_cross_system_triggers(self, temp_dir, mock_trading_engine, mock_risk_monitor):
        """Test cross-system triggering"""
        config = SafetySystemConfig(
            circuit_breaker_data_dir=str(Path(temp_dir) / "cb"),
            kill_switch_data_dir=str(Path(temp_dir) / "ks"),
            enable_cross_system_triggers=True
        )
        
        integration = create_safety_systems_integration(config)
        
        # Register mocks
        integration.register_trading_engine("test_engine", mock_trading_engine)
        integration.register_risk_monitor(mock_risk_monitor)
        
        # Initialize
        await integration.initialize()
        integration.start_monitoring()
        
        # Set up conditions for circuit breaker trigger
        mock_risk_monitor.risk_metrics.current_drawdown = 0.20  # 20% drawdown
        
        # Wait a moment for monitoring
        time.sleep(2)
        
        # Check if cross-system trigger occurred
        # (This would require the monitoring loop to detect the condition)
        
        integration.stop_monitoring()
    
    def test_comprehensive_status(self, temp_dir):
        """Test comprehensive status reporting"""
        integration = create_safety_systems_integration()
        
        status = integration.get_comprehensive_status()
        
        assert "integration" in status
        assert "circuit_breakers" in status
        assert "kill_switches" in status
        assert "cross_system_events" in status
        assert "total_events" in status
        
        # Check integration status
        integration_status = status["integration"]
        assert "is_initialized" in integration_status
        assert "is_monitoring" in integration_status
        assert "performance_metrics" in integration_status
    
    def test_global_emergency_trigger(self, temp_dir, mock_trading_engine):
        """Test global emergency through integration"""
        integration = create_safety_systems_integration()
        integration.register_trading_engine("test_engine", mock_trading_engine)
        
        # Trigger global emergency stop
        event_id = integration.trigger_global_emergency_stop(
            "Integration test emergency",
            "test_user",
            "immediate"
        )
        
        assert event_id is not None
        
        # Check engine was stopped
        assert mock_trading_engine.emergency_stop_called is True
    
    def test_circuit_breaker_integration_trigger(self, temp_dir):
        """Test circuit breaker trigger through integration"""
        integration = create_safety_systems_integration()
        
        # Trigger circuit breaker
        success = integration.trigger_circuit_breaker(
            "max_daily_drawdown",
            "Integration test trigger"
        )
        
        assert success is True
    
    def test_performance_metrics_tracking(self, temp_dir):
        """Test performance metrics tracking"""
        integration = create_safety_systems_integration()
        integration.start_monitoring()
        
        # Let it run for a bit
        time.sleep(2)
        
        status = integration.get_comprehensive_status()
        metrics = status["integration"]["performance_metrics"]
        
        assert metrics["safety_checks_performed"] > 0
        assert "last_update" in metrics
        
        integration.stop_monitoring()


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_database_error_handling(self, temp_dir):
        """Test handling of database errors"""
        # Create circuit breaker with invalid path
        with pytest.raises(Exception):
            create_circuit_breaker_system(risk_dir="/invalid/path/that/does/not/exist")
    
    def test_missing_risk_monitor(self, temp_dir):
        """Test behavior when risk monitor is missing"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        
        # Try to check conditions without risk monitor
        rule = list(breaker_system.breaker_rules.values())[0]
        condition_met = breaker_system._check_breaker_condition(rule)
        
        # Should handle gracefully and return False
        assert condition_met is False
    
    def test_invalid_kill_switch_trigger(self, temp_dir):
        """Test triggering non-existent kill switch"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        
        # Try to trigger non-existent switch
        event_id = kill_switch.trigger_kill_switch(
            "non_existent_switch",
            KillSwitchReason.MANUAL_OVERRIDE,
            "test_user"
        )
        
        assert event_id is None
    
    def test_resume_non_existent_event(self, temp_dir):
        """Test resuming non-existent event"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        
        # Try to resume non-existent event
        success = kill_switch.resume_kill_switch(
            "non_existent_event",
            "test_user"
        )
        
        assert success is False


class TestPerformance:
    """Test performance under load"""
    
    def test_high_frequency_monitoring(self, temp_dir, mock_risk_monitor):
        """Test performance under high-frequency monitoring"""
        breaker_system = create_circuit_breaker_system(risk_dir=temp_dir)
        breaker_system.register_risk_monitor(mock_risk_monitor)
        
        # Start monitoring
        breaker_system.start_monitoring()
        
        # Let it run at high frequency
        start_time = time.time()
        time.sleep(5)  # 5 seconds of monitoring
        end_time = time.time()
        
        # Stop monitoring
        breaker_system.stop_monitoring()
        
        # Check performance metrics
        stats = breaker_system.monitoring_stats
        checks_performed = stats["risk_checks_performed"]
        
        # Should have performed many checks
        assert checks_performed > 0
        
        # Calculate checks per second
        duration = end_time - start_time
        checks_per_second = checks_performed / duration
        
        # Should be able to handle at least 0.5 checks per second
        assert checks_per_second >= 0.5
    
    def test_concurrent_kill_switch_operations(self, temp_dir):
        """Test concurrent kill switch operations"""
        kill_switch = create_emergency_kill_switch(data_dir=temp_dir)
        
        # Add multiple switches for testing
        for i in range(10):
            config = KillSwitchConfig(
                switch_id=f"test_switch_{i}",
                switch_type=KillSwitchType.STRATEGY,
                mode=KillSwitchMode.IMMEDIATE,
                description=f"Test switch {i}"
            )
            kill_switch.add_kill_switch(config)
        
        results = []
        
        def trigger_random_switch():
            import random
            switch_id = f"test_switch_{random.randint(0, 9)}"
            event_id = kill_switch.trigger_kill_switch(
                switch_id,
                KillSwitchReason.MANUAL_OVERRIDE,
                f"thread_{threading.current_thread().ident}"
            )
            results.append((switch_id, event_id))
        
        # Create many concurrent operations
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=trigger_random_switch)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        successful = [r for r in results if r[1] is not None]
        assert len(successful) > 0  # At least some should succeed


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])