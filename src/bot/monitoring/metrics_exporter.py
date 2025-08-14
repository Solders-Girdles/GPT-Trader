"""
Prometheus Metrics Exporter
Phase 2.5 - Day 4

Exports application metrics to Prometheus for monitoring.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)

from ..database.database_manager import get_db_manager
from ..database.models import Position, SystemMetric

logger = logging.getLogger(__name__)


class MetricsExporter:
    """
    Exports application metrics to Prometheus.

    Metrics:
    - Database performance (QPS, latency, connections)
    - Data pipeline (throughput, latency, errors)
    - ML models (accuracy, predictions, latency)
    - Trading (positions, orders, PnL)
    - System (CPU, memory, disk)
    """

    def __init__(self, port: int = 8000):
        self.port = port
        self.registry = CollectorRegistry()
        self.db = get_db_manager()

        # Database metrics
        self.db_queries_total = Counter(
            "postgres_queries_total",
            "Total number of database queries",
            ["query_type"],
            registry=self.registry,
        )

        self.db_query_duration = Histogram(
            "postgres_query_duration_seconds",
            "Database query duration",
            ["query_type"],
            registry=self.registry,
        )

        self.db_connections_active = Gauge(
            "postgres_connections_active", "Active database connections", registry=self.registry
        )

        self.db_connections_idle = Gauge(
            "postgres_connections_idle", "Idle database connections", registry=self.registry
        )

        # Data pipeline metrics
        self.datafeed_messages_received = Counter(
            "datafeed_messages_received_total",
            "Total messages received",
            ["source"],
            registry=self.registry,
        )

        self.datafeed_messages_validated = Counter(
            "datafeed_messages_validated_total", "Total messages validated", registry=self.registry
        )

        self.datafeed_messages_rejected = Counter(
            "datafeed_messages_rejected_total",
            "Total messages rejected",
            ["reason"],
            registry=self.registry,
        )

        self.datasource_status = Gauge(
            "datasource_status",
            "Data source status (1=active, 0=inactive)",
            ["source"],
            registry=self.registry,
        )

        self.datasource_latency = Gauge(
            "datasource_latency_ms",
            "Data source latency in milliseconds",
            ["source"],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "redis_cache_hits_total", "Total cache hits", registry=self.registry
        )

        self.cache_misses = Counter(
            "redis_cache_misses_total", "Total cache misses", registry=self.registry
        )

        self.cache_hit_rate = Gauge(
            "redis_cache_hit_rate", "Cache hit rate", registry=self.registry
        )

        # ML model metrics
        self.ml_predictions = Counter(
            "ml_predictions_total",
            "Total ML predictions made",
            ["model_name"],
            registry=self.registry,
        )

        self.ml_prediction_errors = Counter(
            "ml_prediction_errors_total",
            "Total ML prediction errors",
            ["model_name"],
            registry=self.registry,
        )

        self.ml_model_accuracy = Gauge(
            "ml_model_accuracy", "ML model accuracy", ["model_name"], registry=self.registry
        )

        self.ml_model_last_updated = Gauge(
            "ml_model_last_updated_timestamp",
            "Timestamp of last model update",
            ["model_name"],
            registry=self.registry,
        )

        # Trading metrics
        self.positions_open = Gauge(
            "positions_open_total", "Total open positions", registry=self.registry
        )

        self.positions_value = Gauge(
            "positions_value_usd", "Total value of open positions", registry=self.registry
        )

        self.position_risk_score = Gauge(
            "position_risk_score", "Risk score for position", ["symbol"], registry=self.registry
        )

        self.orders_submitted = Counter(
            "orders_submitted_total",
            "Total orders submitted",
            ["order_type"],
            registry=self.registry,
        )

        self.order_execution_failures = Counter(
            "order_execution_failures_total",
            "Total order execution failures",
            ["reason"],
            registry=self.registry,
        )

        self.portfolio_value = Gauge(
            "portfolio_value_usd", "Total portfolio value", registry=self.registry
        )

        self.portfolio_drawdown = Gauge(
            "portfolio_drawdown_percent", "Portfolio drawdown percentage", registry=self.registry
        )

        # System metrics
        self.system_cpu_percent = Gauge(
            "system_cpu_percent", "System CPU usage percentage", registry=self.registry
        )

        self.system_memory_mb = Gauge(
            "process_resident_memory_bytes", "Process memory usage in bytes", registry=self.registry
        )

        self.system_disk_usage = Gauge(
            "system_disk_usage_percent", "Disk usage percentage", registry=self.registry
        )

        # Error metrics
        self.errors_total = Counter(
            "errors_total", "Total errors", ["error_type"], registry=self.registry
        )

        logger.info(f"MetricsExporter initialized on port {port}")

    async def update_database_metrics(self):
        """Update database-related metrics"""
        try:
            # Get connection pool status
            pool_status = self.db.get_pool_status()
            self.db_connections_active.set(pool_status.get("checked_out", 0))
            self.db_connections_idle.set(
                pool_status.get("size", 0) - pool_status.get("checked_out", 0)
            )

            # Get cache metrics
            cache_stats = self.db.get_cache_stats()
            if cache_stats:
                hits = cache_stats.get("hits", 0)
                misses = cache_stats.get("misses", 0)
                total = hits + misses

                if total > 0:
                    hit_rate = hits / total
                    self.cache_hit_rate.set(hit_rate)

        except Exception as e:
            logger.error(f"Error updating database metrics: {e}")
            self.errors_total.labels(error_type="metrics_update").inc()

    async def update_data_pipeline_metrics(self):
        """Update data pipeline metrics"""
        try:
            # Query recent metrics from database
            recent_metrics = (
                self.db.session.query(SystemMetric)
                .filter(
                    SystemMetric.component == "realtime_feed",
                    SystemMetric.timestamp >= datetime.utcnow() - timedelta(minutes=5),
                )
                .all()
            )

            for metric in recent_metrics:
                if metric.metric_name == "datafeed_messages_received":
                    # This would be aggregated from actual feed
                    pass
                elif metric.metric_name == "datafeed_messages_validated":
                    # This would be aggregated from actual feed
                    pass

            # Get data source status (mock for now)
            sources = ["alpaca", "polygon", "yahoo", "iex"]
            for source in sources:
                # In production, get actual status
                self.datasource_status.labels(source=source).set(1)
                self.datasource_latency.labels(source=source).set(50)

        except Exception as e:
            logger.error(f"Error updating pipeline metrics: {e}")
            self.errors_total.labels(error_type="metrics_update").inc()

    async def update_trading_metrics(self):
        """Update trading-related metrics"""
        try:
            # Count open positions
            open_positions = self.db.session.query(Position).filter(Position.status == "open").all()

            self.positions_open.set(len(open_positions))

            # Calculate total value
            total_value = Decimal("0")
            for position in open_positions:
                if position.current_price and position.quantity:
                    value = position.current_price * position.quantity
                    total_value += value

                    # Mock risk score
                    risk_score = 0.3  # Would be calculated based on actual risk model
                    self.position_risk_score.labels(symbol=position.symbol).set(risk_score)

            self.positions_value.set(float(total_value))

            # Mock portfolio metrics
            self.portfolio_value.set(100000)  # Would come from actual portfolio
            self.portfolio_drawdown.set(5.2)  # Would be calculated

        except Exception as e:
            logger.error(f"Error updating trading metrics: {e}")
            self.errors_total.labels(error_type="metrics_update").inc()

    async def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_percent.set(cpu_percent)

            # Memory usage
            process = psutil.Process()
            memory_bytes = process.memory_info().rss
            self.system_memory_mb.set(memory_bytes)

            # Disk usage
            disk_usage = psutil.disk_usage("/")
            self.system_disk_usage.set(disk_usage.percent)

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            self.errors_total.labels(error_type="metrics_update").inc()

    async def update_ml_metrics(self):
        """Update ML model metrics"""
        try:
            # Mock ML metrics - would come from actual ML components
            models = ["hmm_regime", "xgboost_selector", "portfolio_optimizer"]

            for model in models:
                self.ml_model_accuracy.labels(model_name=model).set(0.75)
                self.ml_model_last_updated.labels(model_name=model).set(
                    datetime.utcnow().timestamp()
                )

        except Exception as e:
            logger.error(f"Error updating ML metrics: {e}")
            self.errors_total.labels(error_type="metrics_update").inc()

    async def update_all_metrics(self):
        """Update all metrics"""
        await self.update_database_metrics()
        await self.update_data_pipeline_metrics()
        await self.update_trading_metrics()
        await self.update_system_metrics()
        await self.update_ml_metrics()

    async def run_metrics_loop(self):
        """Run continuous metrics update loop"""
        logger.info("Starting metrics update loop")

        while True:
            try:
                await self.update_all_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                self.errors_total.labels(error_type="metrics_loop").inc()
                await asyncio.sleep(30)  # Wait longer on error

    def start(self):
        """Start the metrics HTTP server"""
        # Start HTTP server for Prometheus to scrape
        start_http_server(self.port, registry=self.registry)
        logger.info(f"Metrics server started on port {self.port}")

        # Start metrics update loop
        loop = asyncio.get_event_loop()
        loop.create_task(self.run_metrics_loop())

    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format"""
        return generate_latest(self.registry)


class CustomCollector:
    """Custom collector for complex metrics"""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter

    def collect(self):
        """Collect custom metrics"""
        # Add any custom metric collection logic here
        pass


def start_metrics_server(port: int = 8000):
    """Start the metrics server"""
    exporter = MetricsExporter(port=port)
    exporter.start()

    logger.info(f"Metrics available at http://localhost:{port}/metrics")

    return exporter


if __name__ == "__main__":
    import asyncio

    # Start metrics server
    exporter = start_metrics_server()

    # Keep running
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logger.info("Metrics server stopped")
