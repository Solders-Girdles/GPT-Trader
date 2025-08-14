"""
Risk Limit Monitoring System
Phase 3, Week 3: RISK-007
Real-time monitoring and enforcement of risk limits
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of risk limits"""
    HARD = "hard"        # Cannot be exceeded
    SOFT = "soft"        # Can be exceeded with approval
    WARNING = "warning"  # Notification only
    DYNAMIC = "dynamic"  # Adjusts based on conditions


class LimitStatus(Enum):
    """Status of limit checks"""
    OK = "ok"
    WARNING = "warning"
    BREACH = "breach"
    CRITICAL = "critical"


class RiskMetricType(Enum):
    """Types of risk metrics to monitor"""
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    CVAR_95 = "cvar_95"
    CVAR_99 = "cvar_99"
    MAX_DRAWDOWN = "max_drawdown"
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    POSITION_SIZE = "position_size"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"


@dataclass
class RiskLimit:
    """Definition of a risk limit"""
    name: str
    metric_type: RiskMetricType
    limit_type: LimitType
    
    # Limit values
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    warning_threshold: float = 0.8  # 80% of limit triggers warning
    
    # Dynamic limit parameters
    is_dynamic: bool = False
    base_limit: Optional[float] = None
    volatility_adjustment: bool = False
    
    # Actions
    auto_hedge: bool = False
    auto_reduce: bool = False
    notification_required: bool = True
    
    # Metadata
    description: str = ""
    owner: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    
    def check_limit(self, value: float) -> LimitStatus:
        """Check if value breaches limit"""
        if self.max_value is not None:
            if value > self.max_value:
                return LimitStatus.BREACH
            elif value > self.max_value * self.warning_threshold:
                return LimitStatus.WARNING
        
        if self.min_value is not None:
            if value < self.min_value:
                return LimitStatus.BREACH
            elif value < self.min_value / self.warning_threshold:
                return LimitStatus.WARNING
        
        return LimitStatus.OK


@dataclass
class LimitBreach:
    """Record of a limit breach"""
    timestamp: datetime
    limit_name: str
    metric_type: RiskMetricType
    current_value: float
    limit_value: float
    breach_percentage: float
    status: LimitStatus
    action_taken: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'limit_name': self.limit_name,
            'metric_type': self.metric_type.value,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'breach_percentage': self.breach_percentage,
            'status': self.status.value,
            'action_taken': self.action_taken,
            'resolved': self.resolved
        }


class RiskLimitMonitor:
    """
    Real-time risk limit monitoring system.
    
    Features:
    - Multiple limit types (hard, soft, warning, dynamic)
    - Real-time breach detection
    - Automatic actions on breach
    - Historical breach tracking
    - Dynamic limit adjustment
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize risk limit monitor.
        
        Args:
            config_path: Path to limit configuration file
        """
        self.limits: Dict[str, RiskLimit] = {}
        self.current_metrics: Dict[RiskMetricType, float] = {}
        self.breach_history: List[LimitBreach] = []
        self.active_breaches: Dict[str, LimitBreach] = {}
        
        # Callbacks
        self.breach_callbacks: List[Callable] = []
        self.resolution_callbacks: List[Callable] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Load configuration if provided
        if config_path:
            self.load_limits_from_config(config_path)
        else:
            self._setup_default_limits()
    
    def _setup_default_limits(self):
        """Setup default risk limits"""
        # VaR limits
        self.add_limit(RiskLimit(
            name="VaR_95_Limit",
            metric_type=RiskMetricType.VAR_95,
            limit_type=LimitType.HARD,
            max_value=0.05,  # 5% VaR limit
            warning_threshold=0.8,
            description="95% VaR should not exceed 5% of capital"
        ))
        
        self.add_limit(RiskLimit(
            name="VaR_99_Limit",
            metric_type=RiskMetricType.VAR_99,
            limit_type=LimitType.HARD,
            max_value=0.10,  # 10% VaR limit
            warning_threshold=0.8,
            description="99% VaR should not exceed 10% of capital"
        ))
        
        # Exposure limits
        self.add_limit(RiskLimit(
            name="Gross_Exposure_Limit",
            metric_type=RiskMetricType.GROSS_EXPOSURE,
            limit_type=LimitType.SOFT,
            max_value=2.0,  # 200% gross exposure
            warning_threshold=0.9,
            description="Gross exposure should not exceed 200% of capital"
        ))
        
        self.add_limit(RiskLimit(
            name="Net_Exposure_Limit",
            metric_type=RiskMetricType.NET_EXPOSURE,
            limit_type=LimitType.SOFT,
            max_value=1.0,  # 100% net long
            min_value=-0.5,  # 50% net short
            warning_threshold=0.8,
            description="Net exposure limits"
        ))
        
        # Drawdown limit
        self.add_limit(RiskLimit(
            name="Max_Drawdown_Limit",
            metric_type=RiskMetricType.MAX_DRAWDOWN,
            limit_type=LimitType.HARD,
            max_value=0.20,  # 20% max drawdown
            warning_threshold=0.75,
            auto_reduce=True,
            description="Maximum drawdown limit with auto-reduction"
        ))
        
        # Concentration limit
        self.add_limit(RiskLimit(
            name="Position_Concentration",
            metric_type=RiskMetricType.CONCENTRATION,
            limit_type=LimitType.WARNING,
            max_value=0.20,  # 20% in single position
            warning_threshold=0.8,
            description="Single position concentration limit"
        ))
        
        # Greeks limits (for options)
        self.add_limit(RiskLimit(
            name="Delta_Limit",
            metric_type=RiskMetricType.DELTA,
            limit_type=LimitType.SOFT,
            max_value=1000,
            min_value=-1000,
            description="Portfolio delta limits"
        ))
        
        self.add_limit(RiskLimit(
            name="Gamma_Limit",
            metric_type=RiskMetricType.GAMMA,
            limit_type=LimitType.WARNING,
            max_value=100,
            description="Portfolio gamma limit"
        ))
        
        self.add_limit(RiskLimit(
            name="Vega_Limit",
            metric_type=RiskMetricType.VEGA,
            limit_type=LimitType.WARNING,
            max_value=500,
            description="Portfolio vega limit"
        ))
    
    def add_limit(self, limit: RiskLimit) -> None:
        """Add a risk limit to monitor"""
        self.limits[limit.name] = limit
        logger.info(f"Added risk limit: {limit.name}")
    
    def remove_limit(self, limit_name: str) -> None:
        """Remove a risk limit"""
        if limit_name in self.limits:
            del self.limits[limit_name]
            logger.info(f"Removed risk limit: {limit_name}")
    
    def update_metric(self, metric_type: RiskMetricType, value: float) -> List[LimitBreach]:
        """
        Update a risk metric and check for breaches.
        
        Args:
            metric_type: Type of metric
            value: Current value
            
        Returns:
            List of any limit breaches detected
        """
        self.current_metrics[metric_type] = value
        breaches = []
        
        # Check all limits for this metric
        for limit_name, limit in self.limits.items():
            if limit.metric_type == metric_type:
                status = limit.check_limit(value)
                
                if status in [LimitStatus.WARNING, LimitStatus.BREACH, LimitStatus.CRITICAL]:
                    breach = self._create_breach(limit, value, status)
                    breaches.append(breach)
                    self._handle_breach(breach, limit)
                elif limit_name in self.active_breaches:
                    # Limit is back to normal
                    self._resolve_breach(limit_name)
        
        return breaches
    
    def update_all_metrics(self, metrics: Dict[RiskMetricType, float]) -> List[LimitBreach]:
        """
        Update multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            List of all breaches detected
        """
        all_breaches = []
        
        for metric_type, value in metrics.items():
            breaches = self.update_metric(metric_type, value)
            all_breaches.extend(breaches)
        
        return all_breaches
    
    def _create_breach(self, limit: RiskLimit, value: float, status: LimitStatus) -> LimitBreach:
        """Create a breach record"""
        limit_value = limit.max_value if value > 0 else limit.min_value
        
        if limit_value:
            breach_percentage = abs((value - limit_value) / limit_value)
        else:
            breach_percentage = 0
        
        return LimitBreach(
            timestamp=datetime.now(),
            limit_name=limit.name,
            metric_type=limit.metric_type,
            current_value=value,
            limit_value=limit_value,
            breach_percentage=breach_percentage,
            status=status
        )
    
    def _handle_breach(self, breach: LimitBreach, limit: RiskLimit) -> None:
        """Handle a limit breach"""
        # Add to active breaches
        self.active_breaches[breach.limit_name] = breach
        self.breach_history.append(breach)
        
        # Log the breach
        if breach.status == LimitStatus.WARNING:
            logger.warning(f"Risk limit warning: {breach.limit_name} at {breach.current_value:.2f}")
        elif breach.status == LimitStatus.BREACH:
            logger.error(f"Risk limit breach: {breach.limit_name} at {breach.current_value:.2f}")
        elif breach.status == LimitStatus.CRITICAL:
            logger.critical(f"Critical risk limit breach: {breach.limit_name} at {breach.current_value:.2f}")
        
        # Take automatic actions
        actions_taken = []
        
        if limit.auto_hedge:
            actions_taken.append("Auto-hedge initiated")
            self._initiate_auto_hedge(breach)
        
        if limit.auto_reduce:
            actions_taken.append("Position reduction initiated")
            self._initiate_position_reduction(breach)
        
        if limit.notification_required:
            actions_taken.append("Notification sent")
            self._send_notification(breach)
        
        breach.action_taken = ", ".join(actions_taken) if actions_taken else None
        
        # Trigger callbacks
        for callback in self.breach_callbacks:
            try:
                callback(breach)
            except Exception as e:
                logger.error(f"Breach callback error: {e}")
    
    def _resolve_breach(self, limit_name: str) -> None:
        """Mark a breach as resolved"""
        if limit_name in self.active_breaches:
            breach = self.active_breaches[limit_name]
            breach.resolved = True
            breach.resolution_time = datetime.now()
            
            del self.active_breaches[limit_name]
            
            logger.info(f"Risk limit breach resolved: {limit_name}")
            
            # Trigger resolution callbacks
            for callback in self.resolution_callbacks:
                try:
                    callback(breach)
                except Exception as e:
                    logger.error(f"Resolution callback error: {e}")
    
    def _initiate_auto_hedge(self, breach: LimitBreach) -> None:
        """Initiate automatic hedging (placeholder)"""
        logger.info(f"Auto-hedge initiated for {breach.limit_name}")
        # Implementation would depend on specific hedging strategy
    
    def _initiate_position_reduction(self, breach: LimitBreach) -> None:
        """Initiate position reduction (placeholder)"""
        logger.info(f"Position reduction initiated for {breach.limit_name}")
        # Implementation would depend on position management system
    
    def _send_notification(self, breach: LimitBreach) -> None:
        """Send breach notification (placeholder)"""
        logger.info(f"Notification sent for {breach.limit_name}")
        # Implementation would depend on notification system
    
    def adjust_dynamic_limits(self, volatility: Optional[float] = None) -> None:
        """
        Adjust dynamic limits based on market conditions.
        
        Args:
            volatility: Current market volatility
        """
        for limit in self.limits.values():
            if limit.is_dynamic and limit.base_limit:
                if volatility and limit.volatility_adjustment:
                    # Adjust limit based on volatility
                    # Lower limits in high volatility
                    adjustment_factor = 1.0 / (1.0 + volatility)
                    
                    if limit.max_value:
                        limit.max_value = limit.base_limit * adjustment_factor
                    
                    logger.info(f"Adjusted {limit.name} to {limit.max_value:.2f} based on volatility")
    
    def get_limit_utilization(self) -> Dict[str, float]:
        """
        Get current utilization of all limits.
        
        Returns:
            Dictionary of limit utilization percentages
        """
        utilization = {}
        
        for limit_name, limit in self.limits.items():
            if limit.metric_type in self.current_metrics:
                current_value = self.current_metrics[limit.metric_type]
                
                if limit.max_value:
                    utilization[limit_name] = abs(current_value / limit.max_value)
                elif limit.min_value:
                    utilization[limit_name] = abs(current_value / limit.min_value)
                else:
                    utilization[limit_name] = 0
        
        return utilization
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive status summary.
        
        Returns:
            Status summary dictionary
        """
        utilization = self.get_limit_utilization()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_breaches': len(self.active_breaches),
            'breach_details': [breach.to_dict() for breach in self.active_breaches.values()],
            'limit_utilization': utilization,
            'highest_utilization': max(utilization.values()) if utilization else 0,
            'total_limits': len(self.limits),
            'metrics_monitored': len(self.current_metrics),
            'historical_breaches_24h': len([
                b for b in self.breach_history
                if b.timestamp > datetime.now() - timedelta(hours=24)
            ])
        }
    
    def add_breach_callback(self, callback: Callable) -> None:
        """Add callback for breach events"""
        self.breach_callbacks.append(callback)
    
    def add_resolution_callback(self, callback: Callable) -> None:
        """Add callback for breach resolution"""
        self.resolution_callbacks.append(callback)
    
    def save_breach_history(self, filepath: str) -> None:
        """Save breach history to file"""
        history_data = [breach.to_dict() for breach in self.breach_history]
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Breach history saved to {filepath}")
    
    def load_limits_from_config(self, config_path: str) -> None:
        """Load limits from configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            for limit_config in config.get('limits', []):
                limit = RiskLimit(
                    name=limit_config['name'],
                    metric_type=RiskMetricType[limit_config['metric_type']],
                    limit_type=LimitType[limit_config['limit_type']],
                    max_value=limit_config.get('max_value'),
                    min_value=limit_config.get('min_value'),
                    warning_threshold=limit_config.get('warning_threshold', 0.8),
                    description=limit_config.get('description', '')
                )
                self.add_limit(limit)
            
            logger.info(f"Loaded {len(self.limits)} limits from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load limits from config: {e}")


def demonstrate_risk_limits():
    """Demonstrate risk limit monitoring"""
    print("Risk Limit Monitoring System Demo")
    print("=" * 60)
    
    # Initialize monitor
    monitor = RiskLimitMonitor()
    
    # Add breach callback
    def on_breach(breach: LimitBreach):
        print(f"\n⚠️  BREACH ALERT: {breach.limit_name}")
        print(f"   Value: {breach.current_value:.3f} (Limit: {breach.limit_value:.3f})")
        print(f"   Status: {breach.status.value}")
    
    monitor.add_breach_callback(on_breach)
    
    print("\nRisk Limits Configured:")
    for name, limit in monitor.limits.items():
        print(f"  • {name}: {limit.metric_type.value}")
        if limit.max_value:
            print(f"    Max: {limit.max_value:.2f}")
        if limit.min_value:
            print(f"    Min: {limit.min_value:.2f}")
    
    # Simulate metric updates
    print("\n" + "=" * 40)
    print("Simulating Risk Metrics...")
    print("=" * 40)
    
    # Normal conditions
    print("\n1. Normal Conditions:")
    metrics_normal = {
        RiskMetricType.VAR_95: 0.03,  # 3% VaR
        RiskMetricType.VAR_99: 0.05,  # 5% VaR
        RiskMetricType.GROSS_EXPOSURE: 1.2,  # 120%
        RiskMetricType.NET_EXPOSURE: 0.4,  # 40% net long
        RiskMetricType.MAX_DRAWDOWN: 0.08,  # 8% drawdown
        RiskMetricType.CONCENTRATION: 0.15  # 15% concentration
    }
    
    breaches = monitor.update_all_metrics(metrics_normal)
    print(f"   Breaches: {len(breaches)}")
    
    utilization = monitor.get_limit_utilization()
    print("   Utilization:")
    for limit_name, util in utilization.items():
        print(f"     {limit_name}: {util:.1%}")
    
    # Warning conditions
    print("\n2. Warning Conditions:")
    metrics_warning = {
        RiskMetricType.VAR_95: 0.042,  # 4.2% VaR (84% of limit)
        RiskMetricType.MAX_DRAWDOWN: 0.16,  # 16% drawdown (80% of limit)
        RiskMetricType.GROSS_EXPOSURE: 1.85  # 185% exposure
    }
    
    breaches = monitor.update_all_metrics(metrics_warning)
    print(f"   Breaches: {len(breaches)}")
    
    # Breach conditions
    print("\n3. Breach Conditions:")
    metrics_breach = {
        RiskMetricType.VAR_95: 0.06,  # 6% VaR (exceeds 5% limit)
        RiskMetricType.MAX_DRAWDOWN: 0.25,  # 25% drawdown (exceeds 20% limit)
        RiskMetricType.NET_EXPOSURE: 1.2  # 120% net long (exceeds 100% limit)
    }
    
    breaches = monitor.update_all_metrics(metrics_breach)
    print(f"   Breaches: {len(breaches)}")
    
    # Status summary
    print("\n" + "=" * 40)
    print("Status Summary:")
    print("=" * 40)
    
    summary = monitor.get_status_summary()
    print(f"  Active Breaches: {summary['active_breaches']}")
    print(f"  Highest Utilization: {summary['highest_utilization']:.1%}")
    print(f"  24h Historical Breaches: {summary['historical_breaches_24h']}")
    
    if summary['breach_details']:
        print("\n  Active Breach Details:")
        for breach in summary['breach_details']:
            print(f"    • {breach['limit_name']}: {breach['current_value']:.3f} ({breach['status']})")
    
    # Test dynamic limit adjustment
    print("\n4. Dynamic Limit Adjustment:")
    print("   Adjusting limits for high volatility (0.3)...")
    monitor.adjust_dynamic_limits(volatility=0.3)
    
    print("\n✅ Risk Limit Monitoring operational!")


if __name__ == "__main__":
    demonstrate_risk_limits()