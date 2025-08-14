"""
Unified Monitoring System for GPT-Trader
Consolidates monitoring, performance tracking, and health checks
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from ..core.database import get_database, DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for unified monitor"""
    
    # Monitoring intervals
    metrics_interval: int = 60  # seconds
    health_check_interval: int = 30  # seconds
    alert_check_interval: int = 10  # seconds
    
    # Thresholds
    max_drawdown_threshold: float = 0.10  # 10%
    min_sharpe_threshold: float = 0.5
    max_position_threshold: float = 0.30  # 30% of portfolio
    
    # Alert settings
    alert_cooldown: int = 300  # 5 minutes between same alerts
    max_alerts_per_hour: int = 20
    
    # Storage
    metrics_retention_days: int = 90
    snapshot_interval: int = 3600  # 1 hour


class UnifiedMonitor:
    """
    Centralized monitoring system that consolidates:
    - Performance monitoring
    - Health checking
    - Alert management
    - Metrics collection
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.db_manager = get_database()
        
        # Components (will be initialized by submodules)
        self.metrics_collector = None
        self.health_checker = None
        self.alert_manager = None
        
        # State
        self.is_running = False
        self._threads = []
        self._stop_event = threading.Event()
        
        # Metrics cache
        self._metrics_cache = {}
        self._last_snapshot = None
        
        logger.info("Unified monitor initialized")
    
    def start(self) -> None:
        """Start all monitoring threads"""
        if self.is_running:
            logger.warning("Monitor already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # Start monitoring threads
        self._threads = [
            threading.Thread(target=self._metrics_loop, name="metrics"),
            threading.Thread(target=self._health_loop, name="health"),
            threading.Thread(target=self._alert_loop, name="alerts"),
            threading.Thread(target=self._snapshot_loop, name="snapshots"),
        ]
        
        for thread in self._threads:
            thread.daemon = True
            thread.start()
        
        logger.info("Monitor started with %d threads", len(self._threads))
    
    def stop(self) -> None:
        """Stop all monitoring threads"""
        if not self.is_running:
            return
        
        logger.info("Stopping monitor...")
        self.is_running = False
        self._stop_event.set()
        
        # Wait for threads to complete
        for thread in self._threads:
            thread.join(timeout=5)
        
        self._threads.clear()
        logger.info("Monitor stopped")
    
    def _metrics_loop(self) -> None:
        """Collect performance metrics periodically"""
        while not self._stop_event.wait(self.config.metrics_interval):
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
    
    def _health_loop(self) -> None:
        """Check system health periodically"""
        while not self._stop_event.wait(self.config.health_check_interval):
            try:
                self._check_health()
            except Exception as e:
                logger.error(f"Error checking health: {e}")
    
    def _alert_loop(self) -> None:
        """Check for alert conditions periodically"""
        while not self._stop_event.wait(self.config.alert_check_interval):
            try:
                self._check_alerts()
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
    
    def _snapshot_loop(self) -> None:
        """Take performance snapshots periodically"""
        while not self._stop_event.wait(self.config.snapshot_interval):
            try:
                self._take_snapshot()
            except Exception as e:
                logger.error(f"Error taking snapshot: {e}")
    
    def _collect_metrics(self) -> None:
        """Collect current metrics"""
        metrics = {}
        
        with self.db_manager.get_connection() as conn:
            # Portfolio metrics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_positions,
                    SUM(CAST(unrealized_pnl AS REAL)) as unrealized_pnl,
                    SUM(CAST(market_value AS REAL)) as portfolio_value
                FROM positions
                WHERE closed_at IS NULL
            """)
            row = cursor.fetchone()
            if row:
                metrics['total_positions'] = row[0]
                metrics['unrealized_pnl'] = row[1] or 0
                metrics['portfolio_value'] = row[2] or 0
            
            # Trading metrics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as trades_today,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CAST(realized_pnl AS REAL)) as daily_pnl
                FROM positions
                WHERE DATE(closed_at) = DATE('now')
            """)
            row = cursor.fetchone()
            if row:
                metrics['trades_today'] = row[0]
                metrics['winning_trades'] = row[1] or 0
                metrics['daily_pnl'] = row[2] or 0
            
            # Calculate derived metrics
            if metrics.get('trades_today', 0) > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['trades_today']
            else:
                metrics['win_rate'] = 0
            
            # Store metrics
            self._metrics_cache = metrics
            
            # Record in database
            conn.execute("""
                INSERT INTO component_metrics (
                    component_id, metric_name, metric_value, metric_data
                ) VALUES (?, ?, ?, ?)
            """, ('monitor', 'portfolio_metrics', metrics.get('portfolio_value', 0), 
                  str(metrics)))
            
            conn.commit()
    
    def _check_health(self) -> None:
        """Check system health"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        with self.db_manager.get_connection() as conn:
            # Check component health
            cursor = conn.execute("""
                SELECT component_id, status, health_status, updated_at
                FROM components
                WHERE updated_at > datetime('now', '-5 minutes')
            """)
            
            for row in cursor:
                component_id = row[0]
                health_status['components'][component_id] = {
                    'status': row[1],
                    'health': row[2],
                    'last_update': row[3]
                }
            
            # Check for stale components
            cursor = conn.execute("""
                SELECT component_id, updated_at
                FROM components
                WHERE updated_at < datetime('now', '-5 minutes')
            """)
            
            stale_components = []
            for row in cursor:
                stale_components.append(row[0])
                health_status['components'][row[0]] = {
                    'status': 'stale',
                    'health': 'unknown',
                    'last_update': row[1]
                }
            
            # Generate health alerts if needed
            if stale_components:
                self._create_alert(
                    'SYSTEM_HEALTH',
                    'WARNING',
                    f"Stale components detected: {', '.join(stale_components)}"
                )
    
    def _check_alerts(self) -> None:
        """Check for alert conditions"""
        if not self._metrics_cache:
            return
        
        # Check drawdown
        if 'current_drawdown' in self._metrics_cache:
            if abs(self._metrics_cache['current_drawdown']) > self.config.max_drawdown_threshold:
                self._create_alert(
                    'RISK_ALERT',
                    'CRITICAL',
                    f"Drawdown exceeded threshold: {self._metrics_cache['current_drawdown']:.2%}"
                )
        
        # Check position concentration
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT symbol, 
                       CAST(market_value AS REAL) / 
                       (SELECT SUM(CAST(market_value AS REAL)) FROM positions WHERE closed_at IS NULL) as concentration
                FROM positions
                WHERE closed_at IS NULL
                HAVING concentration > ?
            """, (self.config.max_position_threshold,))
            
            for row in cursor:
                self._create_alert(
                    'POSITION_ALERT',
                    'WARNING',
                    f"Position concentration too high: {row[0]} = {row[1]:.2%}"
                )
    
    def _take_snapshot(self) -> None:
        """Take performance snapshot"""
        if not self._metrics_cache:
            return
        
        with self.db_manager.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO performance_snapshots (
                    snapshot_date, total_pnl, unrealized_pnl, realized_pnl,
                    daily_pnl, portfolio_value, total_positions, total_trades,
                    success_rate, snapshot_data
                ) VALUES (DATE('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self._metrics_cache.get('total_pnl', 0),
                self._metrics_cache.get('unrealized_pnl', 0),
                self._metrics_cache.get('realized_pnl', 0),
                self._metrics_cache.get('daily_pnl', 0),
                self._metrics_cache.get('portfolio_value', 0),
                self._metrics_cache.get('total_positions', 0),
                self._metrics_cache.get('trades_today', 0),
                self._metrics_cache.get('win_rate', 0),
                str(self._metrics_cache)
            ))
            conn.commit()
        
        self._last_snapshot = datetime.now()
        logger.debug("Performance snapshot saved")
    
    def _create_alert(self, event_type: str, severity: str, message: str) -> None:
        """Create an alert"""
        with self.db_manager.get_connection() as conn:
            # Check for recent similar alerts (cooldown)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM alert_events
                WHERE event_type = ? AND message = ?
                AND created_at > datetime('now', '-{} seconds')
            """.format(self.config.alert_cooldown), (event_type, message))
            
            if cursor.fetchone()[0] > 0:
                return  # Skip due to cooldown
            
            # Create alert
            import uuid
            alert_id = str(uuid.uuid4())
            
            conn.execute("""
                INSERT INTO alert_events (
                    event_id, rule_id, severity, title, message,
                    event_type, event_data, created_at
                ) VALUES (?, 'monitor', ?, ?, ?, ?, ?, datetime('now'))
            """, (alert_id, severity, event_type, message, event_type, '{}'))
            
            conn.commit()
            logger.info(f"Alert created: {severity} - {message}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self._metrics_cache.copy()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT component_id, status, health_status, updated_at
                FROM components
                ORDER BY component_id
            """)
            
            components = {}
            for row in cursor:
                components[row[0]] = {
                    'status': row[1],
                    'health': row[2],
                    'last_update': row[3]
                }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': self._calculate_overall_health(components),
                'components': components
            }
    
    def _calculate_overall_health(self, components: Dict[str, Any]) -> str:
        """Calculate overall system health"""
        if not components:
            return 'unknown'
        
        statuses = [c.get('health', 'unknown') for c in components.values()]
        
        if 'critical' in statuses or 'error' in statuses:
            return 'unhealthy'
        elif 'warning' in statuses or 'unknown' in statuses:
            return 'degraded'
        else:
            return 'healthy'
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT event_id, severity, title, message, created_at
                FROM alert_events
                WHERE created_at > datetime('now', '-{} hours')
                ORDER BY created_at DESC
                LIMIT 100
            """.format(hours))
            
            alerts = []
            for row in cursor:
                alerts.append({
                    'id': row[0],
                    'severity': row[1],
                    'title': row[2],
                    'message': row[3],
                    'timestamp': row[4]
                })
            
            return alerts