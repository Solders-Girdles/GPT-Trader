"""
Tests for Risk Limit Monitor
Phase 3, Week 3: RISK-008
Test suite for risk limit monitoring and enforcement
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from src.bot.risk.risk_limit_monitor import (
    RiskLimitMonitor,
    LimitType,
    LimitBreach,
    RiskLimit
)


class TestRiskLimitMonitor:
    """Test suite for RiskLimitMonitor"""
    
    @pytest.fixture
    def sample_limits(self):
        """Create sample risk limits"""
        return [
            RiskLimit(
                name="max_var",
                limit_type=LimitType.HARD,
                threshold=-0.05,  # 5% VaR limit
                action=BreachAction.STOP_TRADING,
                description="Maximum VaR limit"
            ),
            RiskLimit(
                name="position_concentration",
                limit_type=LimitType.SOFT,
                threshold=0.25,  # 25% max position
                action=BreachAction.ALERT,
                description="Position concentration limit"
            ),
            RiskLimit(
                name="drawdown_warning",
                limit_type=LimitType.WARNING,
                threshold=-0.10,  # 10% drawdown warning
                action=BreachAction.LOG,
                description="Drawdown warning level"
            )
        ]
    
    @pytest.fixture
    def monitor(self, sample_limits):
        """Create RiskLimitMonitor instance"""
        return RiskLimitMonitor(limits=sample_limits)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample risk metrics"""
        return {
            'var_95': -0.03,
            'cvar_95': -0.045,
            'max_position_size': 0.20,
            'gross_exposure': 0.85,
            'net_exposure': 0.75,
            'current_drawdown': -0.08,
            'sharpe_ratio': 1.2,
            'portfolio_value': 1000000,
            'max_sector_concentration': 0.30
        }
    
    def test_monitor_initialization(self, monitor, sample_limits):
        """Test monitor initialization"""
        assert monitor is not None
        assert len(monitor.limits) == len(sample_limits)
        assert monitor.breach_history == []
        assert monitor.last_check is None
    
    def test_add_limit(self, monitor):
        """Test adding new limit"""
        new_limit = RiskLimit(
            name="leverage_limit",
            limit_type=LimitType.HARD,
            threshold=2.0,
            action=BreachAction.REDUCE_POSITIONS,
            description="Maximum leverage limit"
        )
        
        initial_count = len(monitor.limits)
        monitor.add_limit(new_limit)
        
        assert len(monitor.limits) == initial_count + 1
        assert new_limit in monitor.limits
    
    def test_remove_limit(self, monitor):
        """Test removing existing limit"""
        limit_name = "max_var"
        initial_count = len(monitor.limits)
        
        result = monitor.remove_limit(limit_name)
        
        assert result is True
        assert len(monitor.limits) == initial_count - 1
        assert not any(limit.name == limit_name for limit in monitor.limits)
    
    def test_update_limit_threshold(self, monitor):
        """Test updating limit threshold"""
        limit_name = "max_var"
        new_threshold = -0.08
        
        result = monitor.update_limit_threshold(limit_name, new_threshold)
        
        assert result is True
        limit = next(l for l in monitor.limits if l.name == limit_name)
        assert limit.threshold == new_threshold
    
    def test_check_limits_no_breach(self, monitor, sample_metrics):
        """Test limit checking with no breaches"""
        # Modify metrics to be within all limits
        safe_metrics = sample_metrics.copy()
        safe_metrics['var_95'] = -0.02  # Within 5% limit
        safe_metrics['max_position_size'] = 0.15  # Within 25% limit
        safe_metrics['current_drawdown'] = -0.05  # Within 10% warning
        
        breaches = monitor.check_limits(safe_metrics)
        
        assert len(breaches) == 0
        assert monitor.last_check is not None
    
    def test_check_limits_with_breach(self, monitor, sample_metrics):
        """Test limit checking with breaches"""
        # Modify metrics to breach limits
        breach_metrics = sample_metrics.copy()
        breach_metrics['var_95'] = -0.08  # Breaches 5% limit
        breach_metrics['max_position_size'] = 0.30  # Breaches 25% limit
        
        breaches = monitor.check_limits(breach_metrics)
        
        assert len(breaches) > 0
        # Check for VaR breach
        var_breach = next((b for b in breaches if b.limit_name == "max_var"), None)
        assert var_breach is not None
        assert var_breach.limit_type == LimitType.HARD
        assert var_breach.action == BreachAction.STOP_TRADING
        
        # Check breach is recorded in history
        assert len(monitor.breach_history) > 0
    
    def test_hard_limit_breach_action(self, monitor):
        """Test hard limit breach triggers immediate action"""
        breach_metrics = {
            'var_95': -0.10,  # Severe breach of 5% limit
            'max_position_size': 0.20,
            'current_drawdown': -0.05
        }
        
        with patch.object(monitor, '_execute_action') as mock_action:
            breaches = monitor.check_limits(breach_metrics)
            
            hard_breaches = [b for b in breaches if b.limit_type == LimitType.HARD]
            assert len(hard_breaches) > 0
            
            # Should execute action for hard breaches
            mock_action.assert_called()
    
    def test_soft_limit_breach_alert(self, monitor):
        """Test soft limit breach generates alert"""
        breach_metrics = {
            'var_95': -0.02,
            'max_position_size': 0.30,  # Breaches 25% soft limit
            'current_drawdown': -0.05
        }
        
        with patch('src.bot.risk.risk_limit_monitor.send_alert') as mock_alert:
            breaches = monitor.check_limits(breach_metrics)
            
            soft_breaches = [b for b in breaches if b.limit_type == LimitType.SOFT]
            assert len(soft_breaches) > 0
            
            # Should send alert for soft breaches
            mock_alert.assert_called()
    
    def test_warning_limit_logging(self, monitor):
        """Test warning limit breach logs appropriately"""
        breach_metrics = {
            'var_95': -0.02,
            'max_position_size': 0.20,
            'current_drawdown': -0.12  # Breaches 10% warning limit
        }
        
        with patch('logging.warning') as mock_log:
            breaches = monitor.check_limits(breach_metrics)
            
            warning_breaches = [b for b in breaches if b.limit_type == LimitType.WARNING]
            assert len(warning_breaches) > 0
            
            # Should log warning
            mock_log.assert_called()
    
    def test_dynamic_limit_adjustment(self, monitor):
        """Test dynamic limit adjustment based on volatility"""
        # Add dynamic limit
        dynamic_limit = DynamicLimit(
            name="dynamic_var",
            base_threshold=-0.05,
            volatility_adjustment=True,
            market_condition_adjustment=True,
            min_threshold=-0.15,
            max_threshold=-0.02
        )
        
        monitor.add_dynamic_limit(dynamic_limit)
        
        # Test with high volatility period
        high_vol_metrics = {
            'var_95': -0.08,
            'realized_volatility': 0.35,  # High volatility
            'market_regime': 'stressed'
        }
        
        adjusted_threshold = monitor._adjust_dynamic_threshold(
            dynamic_limit, high_vol_metrics
        )
        
        # Threshold should be more lenient in high volatility
        assert adjusted_threshold < dynamic_limit.base_threshold
        assert adjusted_threshold >= dynamic_limit.min_threshold
    
    def test_limit_breach_history(self, monitor, sample_metrics):
        """Test breach history tracking"""
        # Create breach
        breach_metrics = sample_metrics.copy()
        breach_metrics['var_95'] = -0.08
        
        monitor.check_limits(breach_metrics)
        
        # Check history
        assert len(monitor.breach_history) > 0
        breach = monitor.breach_history[0]
        assert breach.timestamp is not None
        assert breach.limit_name == "max_var"
        assert breach.actual_value == -0.08
        assert breach.threshold == -0.05
    
    def test_get_breach_summary(self, monitor):
        """Test breach summary generation"""
        # Create some breaches
        for i in range(5):
            breach = LimitBreach(
                limit_name="test_limit",
                limit_type=LimitType.SOFT,
                threshold=-0.05,
                actual_value=-0.06 - i*0.01,
                severity=1.0 + i*0.2,
                action=BreachAction.ALERT,
                timestamp=datetime.now() - timedelta(hours=i)
            )
            monitor.breach_history.append(breach)
        
        summary = monitor.get_breach_summary(hours=24)
        
        assert 'total_breaches' in summary
        assert 'breach_by_type' in summary
        assert 'breach_by_severity' in summary
        assert summary['total_breaches'] == 5
    
    def test_limit_breach_cooldown(self, monitor):
        """Test breach cooldown to prevent spam"""
        breach_metrics = {
            'var_95': -0.08,
            'max_position_size': 0.20,
            'current_drawdown': -0.05
        }
        
        # Set cooldown period
        monitor.breach_cooldown = timedelta(minutes=5)
        
        # First breach should trigger action
        with patch.object(monitor, '_execute_action') as mock_action:
            breaches1 = monitor.check_limits(breach_metrics)
            assert len(breaches1) > 0
            action_calls1 = mock_action.call_count
        
        # Second breach within cooldown should not trigger action again
        with patch.object(monitor, '_execute_action') as mock_action:
            breaches2 = monitor.check_limits(breach_metrics)
            assert len(breaches2) > 0  # Still detected
            action_calls2 = mock_action.call_count
            # Should not execute action again during cooldown
            assert action_calls2 == 0
    
    def test_batch_limit_update(self, monitor):
        """Test batch update of multiple limits"""
        updates = {
            'max_var': -0.07,
            'position_concentration': 0.30,
            'drawdown_warning': -0.15
        }
        
        results = monitor.batch_update_limits(updates)
        
        assert all(results.values())
        
        # Verify updates
        for limit in monitor.limits:
            if limit.name in updates:
                assert limit.threshold == updates[limit.name]
    
    def test_limit_validation(self, monitor):
        """Test limit validation logic"""
        # Invalid limit (positive VaR threshold)
        invalid_limit = RiskLimit(
            name="invalid_var",
            limit_type=LimitType.HARD,
            threshold=0.05,  # VaR should be negative
            action=BreachAction.STOP_TRADING,
            description="Invalid VaR limit"
        )
        
        with pytest.raises(ValueError):
            monitor.validate_and_add_limit(invalid_limit, metric_type='var')
    
    def test_circuit_breaker_integration(self, monitor):
        """Test integration with circuit breaker system"""
        # Add circuit breaker limit
        circuit_limit = RiskLimit(
            name="circuit_breaker",
            limit_type=LimitType.HARD,
            threshold=-0.15,  # 15% loss circuit breaker
            action=BreachAction.HALT_SYSTEM,
            description="Emergency circuit breaker"
        )
        
        monitor.add_limit(circuit_limit)
        
        # Test severe breach
        severe_metrics = {
            'current_drawdown': -0.20,  # 20% drawdown
            'var_95': -0.03,
            'max_position_size': 0.20
        }
        
        with patch.object(monitor, '_halt_system') as mock_halt:
            breaches = monitor.check_limits(severe_metrics)
            
            circuit_breach = next(
                (b for b in breaches if b.action == BreachAction.HALT_SYSTEM), 
                None
            )
            assert circuit_breach is not None
            mock_halt.assert_called()
    
    def test_limit_performance_monitoring(self, monitor):
        """Test performance monitoring of limit checks"""
        import time
        
        large_metrics = {f'metric_{i}': np.random.random() for i in range(100)}
        large_metrics.update({
            'var_95': -0.03,
            'max_position_size': 0.20,
            'current_drawdown': -0.05
        })
        
        start_time = time.time()
        breaches = monitor.check_limits(large_metrics)
        check_time = time.time() - start_time
        
        # Should complete quickly even with many metrics
        assert check_time < 0.1  # Less than 100ms
        assert monitor.last_check_duration is not None
    
    def test_limit_serialization(self, monitor):
        """Test limit configuration serialization"""
        config = monitor.export_configuration()
        
        assert 'limits' in config
        assert 'settings' in config
        assert len(config['limits']) == len(monitor.limits)
        
        # Test deserialization
        new_monitor = RiskLimitMonitor.from_configuration(config)
        assert len(new_monitor.limits) == len(monitor.limits)
    
    def test_custom_breach_handler(self, monitor):
        """Test custom breach handler functionality"""
        custom_actions = []
        
        def custom_handler(breach: LimitBreach):
            custom_actions.append(f"Custom action for {breach.limit_name}")
        
        monitor.add_custom_handler("position_concentration", custom_handler)
        
        breach_metrics = {
            'var_95': -0.02,
            'max_position_size': 0.30,  # Breaches position limit
            'current_drawdown': -0.05
        }
        
        breaches = monitor.check_limits(breach_metrics)
        
        # Custom handler should have been called
        assert len(custom_actions) > 0
        assert "position_concentration" in custom_actions[0]
    
    def test_limit_hierarchy(self, monitor):
        """Test limit hierarchy and precedence"""
        # Add multiple limits for same metric
        critical_limit = RiskLimit(
            name="critical_var",
            limit_type=LimitType.HARD,
            threshold=-0.08,
            action=BreachAction.HALT_SYSTEM,
            description="Critical VaR limit",
            priority=1  # Highest priority
        )
        
        monitor.add_limit(critical_limit)
        
        breach_metrics = {
            'var_95': -0.10,  # Breaches both limits
            'max_position_size': 0.20,
            'current_drawdown': -0.05
        }
        
        breaches = monitor.check_limits(breach_metrics)
        
        # Should trigger highest priority action
        var_breaches = [b for b in breaches if 'var' in b.limit_name]
        assert len(var_breaches) >= 2
        
        # Critical breach should have higher priority
        critical_breach = next(
            (b for b in var_breaches if b.limit_name == "critical_var"), 
            None
        )
        assert critical_breach is not None
        assert critical_breach.action == BreachAction.HALT_SYSTEM
    
    def test_market_hours_consideration(self, monitor):
        """Test limit behavior during market hours vs after hours"""
        # Set market hours sensitivity
        monitor.market_hours_only = True
        
        with patch('src.bot.risk.risk_limit_monitor.is_market_open') as mock_market:
            # Test during market hours
            mock_market.return_value = True
            
            breach_metrics = {
                'var_95': -0.08,
                'max_position_size': 0.20,
                'current_drawdown': -0.05
            }
            
            breaches_market = monitor.check_limits(breach_metrics)
            assert len(breaches_market) > 0
            
            # Test after market hours
            mock_market.return_value = False
            
            breaches_after = monitor.check_limits(breach_metrics)
            
            # Behavior might differ based on configuration
            # Some limits might be less strict after hours
    
    def test_risk_budget_allocation(self, monitor):
        """Test risk budget allocation and monitoring"""
        # Set risk budgets
        risk_budgets = {
            'equity': 0.60,    # 60% of risk budget
            'fixed_income': 0.25,  # 25% of risk budget
            'alternatives': 0.15   # 15% of risk budget
        }
        
        monitor.set_risk_budgets(risk_budgets)
        
        portfolio_allocation = {
            'equity': 0.70,        # Over-allocated
            'fixed_income': 0.20,  # Under-allocated
            'alternatives': 0.10   # Under-allocated
        }
        
        budget_breaches = monitor.check_risk_budget_utilization(portfolio_allocation)
        
        # Equity should be flagged as over-allocated
        equity_breach = next(
            (b for b in budget_breaches if 'equity' in b['category']), 
            None
        )
        assert equity_breach is not None
        assert equity_breach['over_allocated'] is True


class TestRiskLimitMonitorIntegration:
    """Integration tests for RiskLimitMonitor"""
    
    @pytest.fixture
    def mock_risk_engine(self):
        """Mock risk metrics engine"""
        mock = Mock()
        mock.calculate_var.return_value = -0.04
        mock.calculate_exposure_metrics.return_value = Mock(
            gross_exposure=0.85,
            net_exposure=0.75,
            concentration_ratio=0.25
        )
        return mock
    
    @pytest.fixture
    def mock_alerting_system(self):
        """Mock alerting system"""
        mock = Mock()
        mock.send_alert.return_value = True
        return mock
    
    def test_integration_with_risk_engine(self, mock_risk_engine, sample_limits):
        """Test integration with risk metrics engine"""
        monitor = RiskLimitMonitor(
            limits=sample_limits,
            risk_engine=mock_risk_engine
        )
        
        # Monitor should fetch metrics from engine
        breaches = monitor.check_all_limits()
        
        assert mock_risk_engine.calculate_var.called
        assert mock_risk_engine.calculate_exposure_metrics.called
    
    def test_integration_with_alerting(self, mock_alerting_system, sample_limits):
        """Test integration with alerting system"""
        monitor = RiskLimitMonitor(
            limits=sample_limits,
            alerting_system=mock_alerting_system
        )
        
        breach_metrics = {
            'var_95': -0.08,  # Breach
            'max_position_size': 0.30,  # Breach
            'current_drawdown': -0.05
        }
        
        breaches = monitor.check_limits(breach_metrics)
        
        # Should send alerts for breaches
        assert mock_alerting_system.send_alert.called
        
        # Check alert content
        alert_calls = mock_alerting_system.send_alert.call_args_list
        assert len(alert_calls) > 0
    
    @patch('src.bot.risk.risk_limit_monitor.execute_trade_halt')
    def test_integration_with_trading_system(self, mock_halt, sample_limits):
        """Test integration with trading system for hard limits"""
        monitor = RiskLimitMonitor(limits=sample_limits)
        
        severe_breach_metrics = {
            'var_95': -0.12,  # Severe breach requiring trading halt
            'max_position_size': 0.20,
            'current_drawdown': -0.05
        }
        
        breaches = monitor.check_limits(severe_breach_metrics)
        
        # Should halt trading for hard limit breaches
        hard_breaches = [b for b in breaches if b.limit_type == LimitType.HARD]
        if hard_breaches:
            mock_halt.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])