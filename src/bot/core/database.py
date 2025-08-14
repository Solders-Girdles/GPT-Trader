"""
GPT-Trader Unified Database Architecture

Centralized database management system that replaces 8 separate SQLite databases
with a single, coherent schema providing:

- Unified data model for all trading operations
- Transactional consistency across components
- Connection pooling and performance optimization
- Database migration and schema management
- Query optimization and indexing strategies

This module consolidates data from:
- Live Trading Engine (orders, positions, executions)
- Risk Monitor (risk metrics, position tracking)
- Circuit Breakers (rules, events, statistics)
- Alerting System (rules, events, delivery attempts)
- Strategy Health (metrics, reports, actions)
- Streaming Data (quotes, trades, bars)
- Dashboard (snapshots, sessions)
"""

import logging
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationException, DatabaseException, raise_config_error

logger = logging.getLogger(__name__)


class TransactionIsolationLevel(Enum):
    """SQLite transaction isolation levels"""

    DEFERRED = "DEFERRED"
    IMMEDIATE = "IMMEDIATE"
    EXCLUSIVE = "EXCLUSIVE"


@dataclass
class DatabaseConfig:
    """Database configuration"""

    database_path: Path

    # Connection settings
    timeout: float = 30.0
    check_same_thread: bool = False

    # Performance settings
    journal_mode: str = "WAL"  # Write-Ahead Logging for better concurrency
    synchronous: str = "NORMAL"  # Balance between safety and performance
    cache_size: int = -64000  # 64MB cache (negative means KB)
    temp_store: str = "MEMORY"  # Store temp tables in memory

    # Connection pooling
    max_connections: int = 20
    connection_timeout: float = 30.0

    # Backup settings
    backup_enabled: bool = True
    backup_interval: timedelta = field(default_factory=lambda: timedelta(hours=6))
    backup_retention_days: int = 7

    # Schema migration
    migration_enabled: bool = True
    migration_dir: Path = field(default_factory=lambda: Path("migrations"))

    def __post_init__(self) -> None:
        """Validate configuration"""
        if not self.database_path:
            raise_config_error("Database path cannot be empty")

        # Ensure Path objects
        if isinstance(self.database_path, str):
            self.database_path = Path(self.database_path)
        if isinstance(self.migration_dir, str):
            self.migration_dir = Path(self.migration_dir)

        # Validate settings
        if self.timeout <= 0:
            raise_config_error("Database timeout must be positive")
        if self.max_connections <= 0:
            raise_config_error("Max connections must be positive")


class ConnectionPool:
    """Thread-safe SQLite connection pool"""

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self._connections: list[sqlite3.Connection] = []
        self._available_connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._created_connections = 0

        logger.info(f"Initializing connection pool with max {config.max_connections} connections")

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimized settings"""
        try:
            conn = sqlite3.connect(
                str(self.config.database_path),
                timeout=self.config.timeout,
                check_same_thread=self.config.check_same_thread,
            )

            # Apply performance optimizations
            conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")
            conn.execute(f"PRAGMA cache_size = {self.config.cache_size}")
            conn.execute(f"PRAGMA temp_store = {self.config.temp_store}")

            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")

            # Set row factory for named access
            conn.row_factory = sqlite3.Row

            self._created_connections += 1
            logger.debug(f"Created database connection #{self._created_connections}")

            return conn

        except Exception as e:
            raise DatabaseException(
                f"Failed to create database connection: {str(e)}",
                operation="create_connection",
                context={"database_path": str(self.config.database_path)},
            )

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection from the pool"""
        conn = None
        try:
            with self._lock:
                if self._available_connections:
                    conn = self._available_connections.pop()
                elif len(self._connections) < self.config.max_connections:
                    conn = self._create_connection()
                    self._connections.append(conn)
                else:
                    # Wait for connection to become available
                    # In a production system, this would use a proper queue
                    raise DatabaseException(
                        "Connection pool exhausted",
                        operation="get_connection",
                        context={"max_connections": self.config.max_connections},
                    )

            yield conn

        except Exception:
            if conn:
                try:
                    conn.rollback()
                except sqlite3.Error as e:
                    # Rollback failed - connection may be in bad state
                    logger.debug(f"Failed to rollback transaction: {e}")
                    pass
            raise

        finally:
            if conn:
                with self._lock:
                    self._available_connections.append(conn)

    def close_all(self) -> None:
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except sqlite3.Error as e:
                    # Connection close failed - log for debugging
                    logger.debug(f"Failed to close database connection: {e}")
                    pass
            self._connections.clear()
            self._available_connections.clear()
            logger.info("Closed all database connections")


class SchemaManager:
    """Database schema management and migration"""

    def __init__(self, config: DatabaseConfig, connection_pool: ConnectionPool) -> None:
        self.config = config
        self.connection_pool = connection_pool
        self.schema_version = 1

    def initialize_schema(self) -> None:
        """Initialize the unified database schema"""
        with self.connection_pool.get_connection() as conn:
            try:
                # Create schema version tracking
                self._create_version_table(conn)

                # Create all tables
                self._create_system_tables(conn)
                self._create_trading_tables(conn)
                self._create_risk_tables(conn)
                self._create_monitoring_tables(conn)
                self._create_market_data_tables(conn)
                self._create_ml_tables(conn)  # Phase 2 ML tables

                # Create indexes
                self._create_indexes(conn)

                # Create views
                self._create_views(conn)

                # Update schema version
                self._update_schema_version(conn)

                conn.commit()
                logger.info(
                    f"Database schema initialized successfully (version {self.schema_version})"
                )

            except Exception as e:
                conn.rollback()
                raise DatabaseException(
                    f"Failed to initialize database schema: {str(e)}", operation="initialize_schema"
                )

    def _create_version_table(self, conn: sqlite3.Connection) -> None:
        """Create schema version tracking table"""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    def _create_system_tables(self, conn: sqlite3.Connection) -> None:
        """Create system and component management tables"""

        # Component registry
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS components (
                component_id TEXT PRIMARY KEY,
                component_type TEXT NOT NULL,
                status TEXT NOT NULL,
                health_status TEXT,
                config_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Component metrics
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS component_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_data TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            )
        """
        )

        # System events
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_events (
                event_id TEXT PRIMARY KEY,
                component_id TEXT,
                event_type TEXT NOT NULL,
                event_category TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                event_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            )
        """
        )

        # Configuration store
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS configuration (
                config_key TEXT PRIMARY KEY,
                config_value TEXT NOT NULL,
                config_type TEXT NOT NULL,
                component_id TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            )
        """
        )

    def _create_trading_tables(self, conn: sqlite3.Connection) -> None:
        """Create trading-related tables"""

        # Orders
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                component_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity TEXT NOT NULL,
                limit_price TEXT,
                stop_price TEXT,
                status TEXT NOT NULL,
                filled_quantity TEXT DEFAULT '0',
                remaining_quantity TEXT DEFAULT '0',
                average_fill_price TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                submitted_at TIMESTAMP,
                filled_at TIMESTAMP,
                cancelled_at TIMESTAMP,
                broker_order_id TEXT,
                execution_venue TEXT,
                commission TEXT DEFAULT '0',
                order_data TEXT,
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            )
        """
        )

        # Positions
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                component_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity TEXT NOT NULL,
                average_price TEXT NOT NULL,
                current_price TEXT,
                market_value TEXT,
                unrealized_pnl TEXT DEFAULT '0',
                realized_pnl TEXT DEFAULT '0',
                opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                position_data TEXT,
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            )
        """
        )

        # Execution reports
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                executed_quantity TEXT NOT NULL,
                executed_price TEXT NOT NULL,
                execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                venue TEXT NOT NULL,
                commission TEXT DEFAULT '0',
                execution_quality TEXT,
                slippage_bps REAL,
                execution_data TEXT,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """
        )

        # Strategy performance
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT NOT NULL,
                measurement_date DATE NOT NULL,
                total_pnl TEXT NOT NULL,
                realized_pnl TEXT NOT NULL,
                unrealized_pnl TEXT NOT NULL,
                total_return_pct REAL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                current_drawdown REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                calmar_ratio REAL DEFAULT 0,
                performance_data TEXT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_id, measurement_date)
            )
        """
        )

    def _create_risk_tables(self, conn: sqlite3.Connection) -> None:
        """Create risk management tables"""

        # Risk metrics
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_date DATE NOT NULL,
                total_capital TEXT NOT NULL,
                invested_capital TEXT NOT NULL,
                portfolio_value TEXT NOT NULL,
                total_pnl TEXT NOT NULL,
                unrealized_pnl TEXT NOT NULL,
                realized_pnl TEXT NOT NULL,
                current_drawdown REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                portfolio_var_95 TEXT DEFAULT '0',
                portfolio_cvar_95 TEXT DEFAULT '0',
                portfolio_volatility_30d REAL DEFAULT 0,
                sharpe_ratio_30d REAL DEFAULT 0,
                portfolio_beta REAL DEFAULT 1,
                avg_correlation REAL DEFAULT 0,
                risk_data TEXT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(metric_date)
            )
        """
        )

        # Circuit breaker rules
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS circuit_breaker_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                breaker_type TEXT NOT NULL,
                threshold_value TEXT NOT NULL,
                lookback_period_seconds INTEGER NOT NULL,
                primary_action TEXT NOT NULL,
                secondary_actions TEXT,
                cooldown_period_seconds INTEGER,
                max_triggers_per_day INTEGER,
                status TEXT NOT NULL,
                configuration TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Circuit breaker events
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS circuit_breaker_events (
                event_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                breaker_type TEXT NOT NULL,
                trigger_value TEXT NOT NULL,
                threshold_value TEXT NOT NULL,
                strategy_id TEXT,
                symbol TEXT,
                actions_taken TEXT,
                positions_closed INTEGER DEFAULT 0,
                strategies_halted INTEGER DEFAULT 0,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                event_data TEXT,
                FOREIGN KEY (rule_id) REFERENCES circuit_breaker_rules(rule_id)
            )
        """
        )

    def _create_monitoring_tables(self, conn: sqlite3.Connection) -> None:
        """Create monitoring and alerting tables"""

        # Alert rules
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                event_types TEXT NOT NULL,
                severity_levels TEXT NOT NULL,
                channels TEXT NOT NULL,
                configuration TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Alert events
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_events (
                event_id TEXT PRIMARY KEY,
                rule_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                component_id TEXT,
                event_type TEXT NOT NULL,
                event_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                first_attempt_at TIMESTAMP,
                delivered_at TIMESTAMP,
                delivery_status TEXT,
                FOREIGN KEY (rule_id) REFERENCES alert_rules(rule_id),
                FOREIGN KEY (component_id) REFERENCES components(component_id)
            )
        """
        )

        # Alert delivery attempts
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_delivery_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_data TEXT,
                FOREIGN KEY (event_id) REFERENCES alert_events(event_id)
            )
        """
        )

        # Performance snapshots
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date DATE NOT NULL,
                total_pnl TEXT NOT NULL,
                unrealized_pnl TEXT NOT NULL,
                realized_pnl TEXT NOT NULL,
                daily_pnl TEXT NOT NULL,
                portfolio_value TEXT NOT NULL,
                total_positions INTEGER DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0,
                current_drawdown REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                snapshot_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(snapshot_date)
            )
        """
        )

    def _create_market_data_tables(self, conn: sqlite3.Connection) -> None:
        """Create market data tables"""

        # Quotes
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bid_price TEXT NOT NULL,
                bid_size INTEGER NOT NULL,
                ask_price TEXT NOT NULL,
                ask_size INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data_source TEXT DEFAULT 'UNKNOWN'
            )
        """
        )

        # Trades
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price TEXT NOT NULL,
                size INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                exchange TEXT DEFAULT 'UNKNOWN',
                trade_id TEXT,
                data_source TEXT DEFAULT 'UNKNOWN'
            )
        """
        )

        # OHLCV bars
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bar_type TEXT NOT NULL,
                open_price TEXT NOT NULL,
                high_price TEXT NOT NULL,
                low_price TEXT NOT NULL,
                close_price TEXT NOT NULL,
                volume INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data_source TEXT DEFAULT 'UNKNOWN',
                UNIQUE(symbol, bar_type, timestamp)
            )
        """
        )

    def _create_ml_tables(self, conn: sqlite3.Connection) -> None:
        """Create ML-specific tables for Phase 2"""

        # ML Models registry
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_models (
                model_id TEXT PRIMARY KEY,
                model_name TEXT,
                model_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                version TEXT,
                parameters TEXT,
                training_date TIMESTAMP NOT NULL,
                training_data_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                status TEXT DEFAULT 'active',
                performance_metrics TEXT,
                file_path TEXT
            )
        """
        )

        # ML Training runs
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_training_runs (
                run_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                training_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                training_end TIMESTAMP,
                data_start_date DATE,
                data_end_date DATE,
                features_used TEXT NOT NULL,
                cross_validation_score REAL,
                test_score REAL,
                training_metrics TEXT,
                status TEXT DEFAULT 'running',
                error_message TEXT,
                FOREIGN KEY (model_id) REFERENCES ml_models(model_id)
            )
        """
        )

        # ML Predictions
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ml_predictions (
                prediction_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                symbol TEXT,
                prediction_date DATE NOT NULL,
                prediction_type TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                confidence REAL,
                features_json TEXT,
                actual_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES ml_models(model_id)
            )
        """
        )

        # Market Regime Detection
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                regime_id INTEGER NOT NULL,
                regime_name TEXT,
                confidence REAL,
                volatility REAL,
                trend_strength REAL,
                regime_data TEXT,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, model_version)
            )
        """
        )

        # Portfolio Optimization Results
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_optimizations (
                optimization_id TEXT PRIMARY KEY,
                optimization_date DATE NOT NULL,
                optimization_type TEXT NOT NULL,
                target_return REAL,
                risk_tolerance REAL,
                constraints TEXT,
                optimal_weights TEXT NOT NULL,
                expected_return REAL,
                expected_risk REAL,
                sharpe_ratio REAL,
                optimization_metrics TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Feature Store
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                feature_date DATE NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL NOT NULL,
                feature_group TEXT,
                data_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, feature_date, feature_name)
            )
        """
        )

        # Rebalancing Log
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rebalancing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rebalance_time TIMESTAMP NOT NULL,
                trigger_type TEXT,
                old_weights TEXT,
                new_weights TEXT,
                transaction_cost REAL,
                executed BOOLEAN DEFAULT 0,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for performance"""

        indexes = [
            # System indexes
            "CREATE INDEX IF NOT EXISTS idx_components_type ON components(component_type)",
            "CREATE INDEX IF NOT EXISTS idx_components_status ON components(status)",
            "CREATE INDEX IF NOT EXISTS idx_component_metrics_component ON component_metrics(component_id)",
            "CREATE INDEX IF NOT EXISTS idx_component_metrics_recorded ON component_metrics(recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_component ON system_events(component_id)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_created ON system_events(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type)",
            # Trading indexes
            "CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id)",
            "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_positions_updated ON positions(last_updated)",
            "CREATE INDEX IF NOT EXISTS idx_executions_order ON executions(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_executions_symbol ON executions(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_executions_time ON executions(execution_time)",
            # Risk indexes
            "CREATE INDEX IF NOT EXISTS idx_risk_metrics_date ON risk_metrics(metric_date)",
            "CREATE INDEX IF NOT EXISTS idx_circuit_breaker_events_rule ON circuit_breaker_events(rule_id)",
            "CREATE INDEX IF NOT EXISTS idx_circuit_breaker_events_triggered ON circuit_breaker_events(triggered_at)",
            # Monitoring indexes
            "CREATE INDEX IF NOT EXISTS idx_alert_events_rule ON alert_events(rule_id)",
            "CREATE INDEX IF NOT EXISTS idx_alert_events_created ON alert_events(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_alert_events_severity ON alert_events(severity)",
            "CREATE INDEX IF NOT EXISTS idx_alert_delivery_event ON alert_delivery_attempts(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_performance_snapshots_date ON performance_snapshots(snapshot_date)",
            # Market data indexes
            "CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON quotes(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_bars_symbol_type_time ON bars(symbol, bar_type, timestamp)",
            # ML indexes
            "CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(model_type)",
            "CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status)",
            "CREATE INDEX IF NOT EXISTS idx_ml_training_runs_model ON ml_training_runs(model_id)",
            "CREATE INDEX IF NOT EXISTS idx_ml_training_runs_status ON ml_training_runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_id)",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_date ON ml_predictions(prediction_date)",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_market_regimes_date ON market_regimes(date)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_optimizations_date ON portfolio_optimizations(optimization_date)",
            "CREATE INDEX IF NOT EXISTS idx_feature_store_symbol_date ON feature_store(symbol, feature_date)",
            "CREATE INDEX IF NOT EXISTS idx_feature_store_name ON feature_store(feature_name)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

    def _create_views(self, conn: sqlite3.Connection) -> None:
        """Create useful database views"""

        # Active positions view
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_active_positions AS
            SELECT
                p.*,
                s.total_pnl as strategy_pnl,
                s.win_rate as strategy_win_rate
            FROM positions p
            LEFT JOIN strategy_performance s ON p.strategy_id = s.strategy_id
            WHERE p.closed_at IS NULL AND p.quantity != '0'
        """
        )

        # Daily performance view
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_daily_performance AS
            SELECT
                DATE(created_at) as trade_date,
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CAST(realized_pnl AS REAL)) as daily_pnl
            FROM positions
            WHERE closed_at IS NOT NULL
            GROUP BY DATE(created_at)
            ORDER BY trade_date DESC
        """
        )

        # Component health view
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_component_health AS
            SELECT
                c.component_id,
                c.component_type,
                c.status,
                c.health_status,
                c.updated_at,
                COUNT(se.event_id) as error_count,
                MAX(se.created_at) as last_error
            FROM components c
            LEFT JOIN system_events se ON c.component_id = se.component_id
                AND se.severity IN ('ERROR', 'CRITICAL')
                AND se.created_at > datetime('now', '-24 hours')
            GROUP BY c.component_id, c.component_type, c.status, c.health_status, c.updated_at
        """
        )

    def _update_schema_version(self, conn: sqlite3.Connection) -> None:
        """Update schema version tracking"""
        conn.execute(
            """
            INSERT OR REPLACE INTO schema_versions (version, description)
            VALUES (?, ?)
        """,
            (self.schema_version, "Initial unified schema"),
        )


class DatabaseManager:
    """
    Centralized database management for GPT-Trader

    Provides unified access to all trading data with:
    - Connection pooling and optimization
    - Transaction management
    - Query helpers and utilities
    - Schema management and migrations
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: DatabaseConfig | None = None) -> "DatabaseManager":
        """Singleton pattern for database manager"""
        with cls._lock:
            if cls._instance is None:
                if config is None:
                    raise ConfigurationException("DatabaseConfig required for first initialization")
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: DatabaseConfig = None) -> None:
        """Initialize database manager"""
        if self._initialized:
            return

        if config is None:
            raise ConfigurationException("DatabaseConfig required for initialization")

        self.config = config

        # Create database directory
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection pool
        self.connection_pool = ConnectionPool(config)

        # Initialize schema manager
        self.schema_manager = SchemaManager(config, self.connection_pool)

        # Initialize database schema
        self.schema_manager.initialize_schema()

        self._initialized = True
        logger.info("Database manager initialized successfully")

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection"""
        with self.connection_pool.get_connection() as conn:
            yield conn

    @contextmanager
    def transaction(
        self, isolation_level: TransactionIsolationLevel = TransactionIsolationLevel.DEFERRED
    ) -> Generator[sqlite3.Connection, None, None]:
        """Execute operations within a transaction"""
        with self.get_connection() as conn:
            try:
                conn.execute(f"BEGIN {isolation_level.value}")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def execute_query(self, query: str, parameters: tuple | None = None) -> sqlite3.Cursor:
        """Execute a query and return cursor"""
        with self.get_connection() as conn:
            try:
                if parameters:
                    return conn.execute(query, parameters)
                else:
                    return conn.execute(query)
            except Exception as e:
                raise DatabaseException(
                    f"Query execution failed: {str(e)}",
                    operation="execute_query",
                    context={"query": query[:100] + "..." if len(query) > 100 else query},
                )

    def fetch_one(self, query: str, parameters: tuple | None = None) -> sqlite3.Row | None:
        """Fetch single row from query"""
        cursor = self.execute_query(query, parameters)
        return cursor.fetchone()

    def fetch_all(self, query: str, parameters: tuple | None = None) -> list[sqlite3.Row]:
        """Fetch all rows from query"""
        cursor = self.execute_query(query, parameters)
        return cursor.fetchall()

    def insert_record(self, table: str, data: dict[str, Any]) -> str:
        """Insert a record into table"""
        # Validate table name to prevent SQL injection
        self._validate_table_name(table)

        columns = list(data.keys())
        placeholders = ["?" for _ in columns]
        values = [data[col] for col in columns]

        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        with self.get_connection() as conn:
            try:
                cursor = conn.execute(query, values)
                conn.commit()
                return str(cursor.lastrowid)
            except Exception as e:
                raise DatabaseException(
                    f"Insert failed: {str(e)}",
                    operation="insert_record",
                    table=table,
                    context={"columns": columns},
                )

    def update_record(
        self, table: str, data: dict[str, Any], where_clause: str, where_params: tuple = None
    ) -> int:
        """Update records in table"""
        # Validate table name to prevent SQL injection
        self._validate_table_name(table)

        set_clauses = [f"{col} = ?" for col in data.keys()]
        values = list(data.values())

        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {where_clause}"

        if where_params:
            values.extend(where_params)

        with self.get_connection() as conn:
            try:
                cursor = conn.execute(query, values)
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                raise DatabaseException(
                    f"Update failed: {str(e)}",
                    operation="update_record",
                    table=table,
                    context={"where_clause": where_clause},
                )

    def delete_records(self, table: str, where_clause: str, where_params: tuple = None) -> int:
        """Delete records from table"""
        # Validate table name to prevent SQL injection
        self._validate_table_name(table)

        query = f"DELETE FROM {table} WHERE {where_clause}"

        with self.get_connection() as conn:
            try:
                cursor = conn.execute(query, where_params or ())
                conn.commit()
                return cursor.rowcount
            except Exception as e:
                raise DatabaseException(
                    f"Delete failed: {str(e)}",
                    operation="delete_records",
                    table=table,
                    context={"where_clause": where_clause},
                )

    def get_table_info(self, table: str) -> list[dict[str, Any]]:
        """Get table schema information"""
        # Validate table name to prevent SQL injection
        self._validate_table_name(table)

        cursor = self.execute_query(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        return [dict(row) for row in columns]

    def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics"""
        stats = {}

        # Database file size
        if self.config.database_path.exists():
            stats["file_size_mb"] = self.config.database_path.stat().st_size / (1024 * 1024)

        # Table counts
        table_counts = {}
        tables = [
            "components",
            "orders",
            "positions",
            "executions",
            "risk_metrics",
            "circuit_breaker_rules",
            "circuit_breaker_events",
            "alert_rules",
            "alert_events",
            "quotes",
            "trades",
            "bars",
        ]

        for table in tables:
            try:
                # Validate table name to prevent SQL injection
                self._validate_table_name(table)
                cursor = self.execute_query(f"SELECT COUNT(*) FROM {table}")
                table_counts[table] = cursor.fetchone()[0]
            except (sqlite3.Error, ValueError) as e:
                # Database error or invalid table name - assume empty
                logger.debug(f"Failed to get count for table {table}: {e}")
                table_counts[table] = 0

        stats["table_counts"] = table_counts
        stats["connection_pool_size"] = len(self.connection_pool._connections)

        return stats

    def vacuum_database(self) -> None:
        """Vacuum database to reclaim space and optimize"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuum completed")

    def backup_database(self, backup_path: Path | None = None) -> Path:
        """Create database backup"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config.database_path.parent / f"backup_{timestamp}.db"

        with self.get_connection() as source_conn:
            backup_conn = sqlite3.connect(str(backup_path))
            try:
                source_conn.backup(backup_conn)
                logger.info(f"Database backup created: {backup_path}")
                return backup_path
            finally:
                backup_conn.close()

    def _validate_table_name(self, table: str) -> None:
        """Validate table name to prevent SQL injection"""
        # Define allowed table names (whitelist approach)
        allowed_tables = {
            "components",
            "component_metrics",
            "system_events",
            "configuration",
            "orders",
            "positions",
            "executions",
            "strategy_performance",
            "risk_metrics",
            "circuit_breaker_rules",
            "circuit_breaker_events",
            "alert_rules",
            "alert_events",
            "alert_delivery_attempts",
            "performance_snapshots",
            "quotes",
            "trades",
            "bars",
            "schema_versions",
            # ML tables
            "ml_models",
            "ml_training_runs",
            "ml_predictions",
            "market_regimes",
            "portfolio_optimizations",
            "feature_store",
        }

        if not table or table not in allowed_tables:
            raise DatabaseException(
                f"Invalid table name: {table}",
                operation="validate_table_name",
                context={"allowed_tables": list(allowed_tables)},
            )

    def close(self) -> None:
        """Close database manager and all connections"""
        self.connection_pool.close_all()
        logger.info("Database manager closed")


# Global database manager instance
_db_manager: DatabaseManager | None = None


def initialize_database(config: DatabaseConfig) -> DatabaseManager:
    """Initialize global database manager"""
    global _db_manager
    _db_manager = DatabaseManager(config)
    return _db_manager


def get_database() -> DatabaseManager:
    """Get global database manager instance"""
    if _db_manager is None:
        raise ConfigurationException("Database not initialized. Call initialize_database() first.")
    return _db_manager
