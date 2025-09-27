"""
Runtime Guard Alerting System

Monitors critical runtime conditions and triggers alerts when thresholds are breached.
Implements circuit breakers for daily loss limits, stale marks, and error rates.
"""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

# Import AlertSeverity from alerts module to avoid duplication
try:
    from .alerts import AlertSeverity
except ImportError:
    # Fallback if alerts module not available
    class AlertSeverity(Enum):
        """Alert severity levels."""
        DEBUG = "debug"
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

logger = logging.getLogger(__name__)


class GuardStatus(Enum):
    """Guard status states."""
    HEALTHY = "healthy"
    WARNING = "warning"
    BREACHED = "breached"
    DISABLED = "disabled"


@dataclass
class Alert:
    """Alert data structure."""
    timestamp: datetime
    guard_name: str
    severity: AlertSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'guard_name': self.guard_name,
            'severity': self.severity.value,
            'message': self.message,
            'context': self.context
        }


@dataclass
class GuardConfig:
    """Configuration for a runtime guard."""
    name: str
    enabled: bool = True
    threshold: float = 0.0
    window_seconds: int = 60
    severity: AlertSeverity = AlertSeverity.WARNING
    auto_shutdown: bool = False
    cooldown_seconds: int = 300  # Prevent alert spam


class RuntimeGuard:
    """Base class for runtime guards."""
    
    def __init__(self, config: GuardConfig):
        self.config = config
        self.status = GuardStatus.HEALTHY if config.enabled else GuardStatus.DISABLED
        self.last_check = datetime.now()
        self.last_alert = None
        self.breach_count = 0
        self.alerts: List[Alert] = []
        
    def check(self, context: Dict[str, Any]) -> Optional[Alert]:
        """
        Check guard condition and return alert if breached.
        
        Args:
            context: Current runtime context
            
        Returns:
            Alert if condition breached, None otherwise
        """
        if not self.config.enabled:
            return None
            
        # Check cooldown
        if self.last_alert:
            elapsed = (datetime.now() - self.last_alert).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return None
        
        # Perform guard-specific check
        is_breached, message = self._evaluate(context)
        
        if is_breached:
            self.status = GuardStatus.BREACHED
            self.breach_count += 1
            alert = Alert(
                timestamp=datetime.now(),
                guard_name=self.config.name,
                severity=self.config.severity,
                message=message,
                context=context
            )
            self.alerts.append(alert)
            self.last_alert = datetime.now()
            return alert
            
        # Check if we should downgrade from breached to warning
        if self.status == GuardStatus.BREACHED:
            self.status = GuardStatus.WARNING
            
        self.last_check = datetime.now()
        return None
        
    def _evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """
        Evaluate guard condition. Override in subclasses.
        
        Returns:
            (is_breached, message)
        """
        raise NotImplementedError
        
    def reset(self):
        """Reset guard state."""
        self.status = GuardStatus.HEALTHY if self.config.enabled else GuardStatus.DISABLED
        self.breach_count = 0
        self.last_alert = None


class DailyLossGuard(RuntimeGuard):
    """Monitor daily loss limits."""
    
    def __init__(self, config: GuardConfig):
        super().__init__(config)
        self.daily_pnl = Decimal('0')
        self.last_reset = datetime.now().date()
        
    def _evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check if daily loss limit is breached."""
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_pnl = Decimal('0')
            self.last_reset = current_date
            
        # Update PnL
        pnl = Decimal(str(context.get('pnl', 0)))
        self.daily_pnl += pnl
        
        # Check threshold
        if self.daily_pnl < -abs(self.config.threshold):
            loss_amount = abs(self.daily_pnl)
            message = (f"Daily loss limit breached: ${loss_amount:.2f} "
                      f"(limit: ${self.config.threshold:.2f})")
            return True, message
            
        # Check warning level (50% of limit) to surface early risk
        warning_threshold = abs(self.config.threshold) * 0.5
        if self.daily_pnl <= -warning_threshold and self.status == GuardStatus.HEALTHY:
            self.status = GuardStatus.WARNING
            
        return False, ""


class StaleMarkGuard(RuntimeGuard):
    """Monitor for stale market data."""
    
    def __init__(self, config: GuardConfig):
        super().__init__(config)
        self.last_marks: Dict[str, datetime] = {}
        
    def _evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check if marks are stale."""
        symbol = context.get('symbol')
        mark_time = context.get('mark_timestamp')
        
        if not symbol or not mark_time:
            return False, ""
            
        # Convert mark_time to datetime if needed
        if isinstance(mark_time, str):
            mark_time = datetime.fromisoformat(mark_time)
        elif isinstance(mark_time, (int, float)):
            mark_time = datetime.fromtimestamp(mark_time)
            
        self.last_marks[symbol] = mark_time
        
        # Check staleness
        age_seconds = (datetime.now() - mark_time).total_seconds()
        if age_seconds > self.config.threshold:
            message = (f"Stale marks detected for {symbol}: "
                      f"{age_seconds:.1f}s old (limit: {self.config.threshold}s)")
            return True, message
            
        return False, ""


class ErrorRateGuard(RuntimeGuard):
    """Monitor error rates."""
    
    def __init__(self, config: GuardConfig):
        super().__init__(config)
        self.error_times: List[datetime] = []
        
    def _evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check if error rate exceeds threshold."""
        if context.get('error'):
            self.error_times.append(datetime.now())
            
        # Clean old errors outside window
        cutoff = datetime.now() - timedelta(seconds=self.config.window_seconds)
        self.error_times = [t for t in self.error_times if t > cutoff]
        
        # Check threshold
        error_count = len(self.error_times)
        if error_count > self.config.threshold:
            message = (f"High error rate: {error_count} errors in "
                      f"{self.config.window_seconds}s (limit: {int(self.config.threshold)})")
            return True, message
            
        return False, ""


class PositionStuckGuard(RuntimeGuard):
    """Monitor for positions that aren't being managed."""
    
    def __init__(self, config: GuardConfig):
        super().__init__(config)
        self.position_times: Dict[str, datetime] = {}
        
    def _evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check if any positions are stuck."""
        positions = context.get('positions', {})
        
        for symbol, position in positions.items():
            # Handle both 'size' and 'qty' field names
            size = position.get('size', position.get('qty', 0))
            if size != 0:
                if symbol not in self.position_times:
                    self.position_times[symbol] = datetime.now()
            else:
                self.position_times.pop(symbol, None)
                
        # Check for stuck positions
        stuck_positions = []
        for symbol, open_time in self.position_times.items():
            age_seconds = (datetime.now() - open_time).total_seconds()
            if age_seconds > self.config.threshold:
                stuck_positions.append((symbol, age_seconds))
                
        if stuck_positions:
            details = ', '.join([f"{sym}: {age:.0f}s" for sym, age in stuck_positions])
            message = f"Stuck positions detected: {details}"
            return True, message
            
        return False, ""


class DrawdownGuard(RuntimeGuard):
    """Monitor maximum drawdown."""
    
    def __init__(self, config: GuardConfig):
        super().__init__(config)
        self.peak_equity = Decimal('0')
        self.current_drawdown = Decimal('0')
        
    def _evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """Check if drawdown exceeds limit."""
        equity = Decimal(str(context.get('equity', 0)))
        
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity * 100
            
            if self.current_drawdown > self.config.threshold:
                message = (f"Maximum drawdown breached: {self.current_drawdown:.2f}% "
                          f"(limit: {self.config.threshold:.2f}%)")
                return True, message
                
        return False, ""


class RuntimeGuardManager:
    """Manages all runtime guards and alert routing."""
    
    def __init__(self):
        self.guards: Dict[str, RuntimeGuard] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.shutdown_callback: Optional[Callable[[], None]] = None
        self.is_running = False
        
    def add_guard(self, guard: RuntimeGuard):
        """Add a runtime guard."""
        self.guards[guard.config.name] = guard
        logger.info(f"Added runtime guard: {guard.config.name}")
        
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
        
    def set_shutdown_callback(self, callback: Callable[[], None]):
        """Set shutdown callback for auto-shutdown guards."""
        self.shutdown_callback = callback
        
    def check_all(self, context: Dict[str, Any]) -> List[Alert]:
        """
        Check all guards with current context.
        
        Args:
            context: Runtime context containing current state
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        for guard in self.guards.values():
            alert = guard.check(context)
            if alert:
                alerts.append(alert)
                self._handle_alert(alert, guard)
                
        return alerts
        
    def _handle_alert(self, alert: Alert, guard: RuntimeGuard):
        """Handle an alert."""
        # Log alert
        log_method = getattr(logger, alert.severity.value, logger.info)
        log_method(f"[{alert.guard_name}] {alert.message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
                
        # Check for auto-shutdown
        if guard.config.auto_shutdown and self.shutdown_callback:
            logger.critical(f"Auto-shutdown triggered by {alert.guard_name}")
            self.shutdown_callback()
            
    def get_status(self) -> Dict[str, Any]:
        """Get status of all guards."""
        return {
            guard_name: {
                'status': guard.status.value,
                'breach_count': guard.breach_count,
                'last_check': guard.last_check.isoformat() if guard.last_check else None,
                'last_alert': guard.last_alert.isoformat() if guard.last_alert else None,
                'enabled': guard.config.enabled
            }
            for guard_name, guard in self.guards.items()
        }
        
    def reset_guard(self, guard_name: str):
        """Reset a specific guard."""
        if guard_name in self.guards:
            self.guards[guard_name].reset()
            logger.info(f"Reset guard: {guard_name}")
            
    def reset_all(self):
        """Reset all guards."""
        for guard in self.guards.values():
            guard.reset()
        logger.info("Reset all runtime guards")


def create_default_guards(config: Dict[str, Any]) -> RuntimeGuardManager:
    """
    Create default runtime guards from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RuntimeGuardManager
    """
    manager = RuntimeGuardManager()
    
    # Daily loss guard
    loss_limit = config.get('risk_management', {}).get('daily_loss_limit', 100)
    manager.add_guard(DailyLossGuard(GuardConfig(
        name="daily_loss",
        threshold=loss_limit,
        severity=AlertSeverity.ERROR,
        auto_shutdown=True
    )))
    
    # Stale mark guard
    stale_seconds = config.get('risk_management', {}).get('circuit_breakers', {}).get('stale_mark_seconds', 60)
    manager.add_guard(StaleMarkGuard(GuardConfig(
        name="stale_marks",
        threshold=stale_seconds,
        severity=AlertSeverity.WARNING
    )))
    
    # Error rate guard
    error_threshold = config.get('risk_management', {}).get('circuit_breakers', {}).get('error_threshold', 10)
    manager.add_guard(ErrorRateGuard(GuardConfig(
        name="error_rate",
        threshold=error_threshold,
        window_seconds=300,
        severity=AlertSeverity.ERROR,
        auto_shutdown=True
    )))
    
    # Position stuck guard
    manager.add_guard(PositionStuckGuard(GuardConfig(
        name="position_stuck",
        threshold=1800,  # 30 minutes
        severity=AlertSeverity.WARNING
    )))
    
    # Drawdown guard
    max_drawdown = config.get('risk_management', {}).get('max_drawdown_pct', 5.0)
    manager.add_guard(DrawdownGuard(GuardConfig(
        name="max_drawdown",
        threshold=max_drawdown,
        severity=AlertSeverity.ERROR,
        auto_shutdown=True
    )))
    
    return manager


# Example alert handlers

def log_alert_handler(alert: Alert):
    """Simple log-based alert handler."""
    logger.info(f"ALERT: {json.dumps(alert.to_dict(), indent=2)}")


def slack_alert_handler(alert: Alert, webhook_url: str):
    """Send alerts to Slack."""
    import requests
    
    # Color based on severity
    colors = {
        AlertSeverity.DEBUG: "#808080",
        AlertSeverity.INFO: "#0000FF",
        AlertSeverity.WARNING: "#FFA500",
        AlertSeverity.ERROR: "#FF0000",
        AlertSeverity.CRITICAL: "#8B0000"
    }
    
    payload = {
        "attachments": [{
            "color": colors.get(alert.severity, "#808080"),
            "title": f"🚨 {alert.guard_name}",
            "text": alert.message,
            "fields": [
                {"title": "Severity", "value": alert.severity.value, "short": True},
                {"title": "Time", "value": alert.timestamp.strftime("%H:%M:%S"), "short": True}
            ],
            "footer": "Trading Bot Alert System",
            "ts": int(alert.timestamp.timestamp())
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


def email_alert_handler(alert: Alert, smtp_config: Dict[str, str]):
    """Send alerts via email."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    if alert.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        return  # Only send email for serious alerts
    
    msg = MIMEMultipart()
    msg['From'] = smtp_config['from']
    msg['To'] = smtp_config['to']
    msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.guard_name}"
    
    body = f"""
Trading Bot Alert

Guard: {alert.guard_name}
Severity: {alert.severity.value}
Time: {alert.timestamp}

Message:
{alert.message}

Context:
{json.dumps(alert.context, indent=2)}
"""
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
            if smtp_config.get('use_tls'):
                server.starttls()
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


if __name__ == "__main__":
    # Example usage
    config = {
        'risk_management': {
            'daily_loss_limit': 10.0,
            'max_drawdown_pct': 2.0,
            'circuit_breakers': {
                'stale_mark_seconds': 60,
                'error_threshold': 3
            }
        }
    }
    
    # Create manager with default guards
    manager = create_default_guards(config)
    
    # Add alert handlers
    manager.add_alert_handler(log_alert_handler)
    
    # Simulate runtime context
    context = {
        'pnl': -5.0,
        'equity': 1000,
        'symbol': 'BTC-PERP',
        'mark_timestamp': datetime.now() - timedelta(seconds=30),
        'positions': {},
        'error': False
    }
    
    # Check guards
    alerts = manager.check_all(context)
    print(f"Triggered {len(alerts)} alerts")
    
    # Get status
    status = manager.get_status()
    print(f"Guard status: {json.dumps(status, indent=2)}")
