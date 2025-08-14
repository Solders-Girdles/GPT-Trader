"""
PostgreSQL Database Manager with Connection Pooling
Phase 2.5 - Production Database Abstraction Layer

Provides high-performance database access with connection pooling,
query optimization, and automatic retry logic.
"""

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, TypeVar

import redis
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import get_config
from .models import Base

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseConfig:
    """Database configuration"""

    def __init__(self):
        config = get_config()

        # PostgreSQL settings
        self.host = config.get("database.host", "localhost")
        self.port = config.get("database.port", 5432)
        self.database = config.get("database.name", "gpt_trader")
        self.username = config.get("database.username", "trader")
        # Get password from environment variable, no fallback for security
        self.password = config.get("database.password") or os.getenv("DATABASE_PASSWORD")
        if not self.password:
            raise ValueError(
                "Database password must be set via config or DATABASE_PASSWORD environment variable"
            )

        # Connection pool settings
        self.pool_size = config.get("database.pool_size", 20)
        self.max_overflow = config.get("database.max_overflow", 40)
        self.pool_timeout = config.get("database.pool_timeout", 30)
        self.pool_recycle = config.get("database.pool_recycle", 3600)
        self.pool_pre_ping = config.get("database.pool_pre_ping", True)

        # Query settings
        self.statement_timeout = config.get("database.statement_timeout", 30000)  # 30 seconds
        self.lock_timeout = config.get("database.lock_timeout", 10000)  # 10 seconds

        # Redis cache settings
        self.redis_host = config.get("redis.host", "localhost")
        self.redis_port = config.get("redis.port", 6379)
        self.redis_db = config.get("redis.db", 0)
        self.cache_ttl = config.get("redis.cache_ttl", 3600)  # 1 hour

    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return (
            f"postgresql+psycopg2://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    @property
    def async_connection_string(self) -> str:
        """Get async PostgreSQL connection string"""
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class DatabaseManager:
    """
    High-performance database manager with connection pooling and caching.

    Features:
    - Connection pooling with QueuePool
    - Automatic retry on connection failures
    - Query result caching with Redis
    - Transaction management
    - Performance monitoring
    """

    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig()
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._scoped_session: scoped_session | None = None
        self._redis_client: redis.Redis | None = None

        # Performance metrics
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self._initialize()

    def _initialize(self):
        """Initialize database connections and session factory"""
        try:
            # Create engine with connection pooling
            self._engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=False,  # Set to True for SQL debugging
                connect_args={
                    "options": f"-c statement_timeout={self.config.statement_timeout} "
                    f"-c lock_timeout={self.config.lock_timeout}"
                },
            )

            # Add event listeners
            self._setup_event_listeners()

            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine, expire_on_commit=False, autoflush=False, autocommit=False
            )

            # Create scoped session for thread-safety
            self._scoped_session = scoped_session(self._session_factory)

            # Initialize Redis client for caching
            try:
                self._redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True,
                    connection_pool=redis.ConnectionPool(
                        max_connections=50,
                        host=self.config.redis_host,
                        port=self.config.redis_port,
                        db=self.config.redis_db,
                    ),
                )
                self._redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis connection failed, caching disabled: {e}")
                self._redis_client = None

            # Create tables if they don't exist
            Base.metadata.create_all(self._engine)

            logger.info("Database manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise

    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring"""

        @event.listens_for(self._engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Configure connection on connect"""
            connection_record.info["pid"] = dbapi_conn.get_backend_pid()
            logger.debug(f"New connection established: PID {connection_record.info['pid']}")

        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log connection checkout from pool"""
            pid = connection_record.info.get("pid", "unknown")
            logger.debug(f"Connection checked out from pool: PID {pid}")

        @event.listens_for(self._engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Log connection return to pool"""
            pid = connection_record.info.get("pid", "unknown")
            logger.debug(f"Connection returned to pool: PID {pid}")

    @contextmanager
    def session_scope(self) -> Session:
        """
        Provide a transactional scope for database operations.

        Usage:
            with db_manager.session_scope() as session:
                session.add(model)
                session.commit()
        """
        session = self._scoped_session()
        try:
            yield session
            session.commit()
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Integrity error: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        """
        Execute a raw SQL query with retry logic.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result dictionaries
        """
        self.query_count += 1

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            return [dict(row) for row in result]

    def get_one(self, model: type[T], **filters) -> T | None:
        """
        Get a single record by filters.

        Args:
            model: SQLAlchemy model class
            **filters: Filter conditions

        Returns:
            Model instance or None
        """
        with self.session_scope() as session:
            return session.query(model).filter_by(**filters).first()

    def get_many(
        self,
        model: type[T],
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
        **filters,
    ) -> list[T]:
        """
        Get multiple records with pagination.

        Args:
            model: SQLAlchemy model class
            limit: Maximum number of records
            offset: Number of records to skip
            order_by: Column to order by
            **filters: Filter conditions

        Returns:
            List of model instances
        """
        with self.session_scope() as session:
            query = session.query(model).filter_by(**filters)

            if order_by:
                query = query.order_by(text(order_by))

            if offset:
                query = query.offset(offset)

            if limit:
                query = query.limit(limit)

            return query.all()

    def create(self, model: type[T], **data) -> T:
        """
        Create a new record.

        Args:
            model: SQLAlchemy model class
            **data: Model data

        Returns:
            Created model instance
        """
        with self.session_scope() as session:
            instance = model(**data)
            session.add(instance)
            session.flush()  # Get ID without committing
            session.refresh(instance)
            return instance

    def update(self, model: type[T], filters: dict, **updates) -> int:
        """
        Update records matching filters.

        Args:
            model: SQLAlchemy model class
            filters: Filter conditions
            **updates: Fields to update

        Returns:
            Number of updated records
        """
        with self.session_scope() as session:
            count = session.query(model).filter_by(**filters).update(updates)
            return count

    def delete(self, model: type[T], **filters) -> int:
        """
        Delete records matching filters.

        Args:
            model: SQLAlchemy model class
            **filters: Filter conditions

        Returns:
            Number of deleted records
        """
        with self.session_scope() as session:
            count = session.query(model).filter_by(**filters).delete()
            return count

    def bulk_insert(self, model: type[T], records: list[dict]) -> int:
        """
        Bulk insert multiple records.

        Args:
            model: SQLAlchemy model class
            records: List of record dictionaries

        Returns:
            Number of inserted records
        """
        if not records:
            return 0

        with self.session_scope() as session:
            session.bulk_insert_mappings(model, records)
            return len(records)

    def cache_get(self, key: str) -> Any | None:
        """Get value from cache"""
        if not self._redis_client:
            return None

        try:
            value = self._redis_client.get(key)
            if value:
                self.cache_hits += 1
                return json.loads(value)
            else:
                self.cache_misses += 1
                return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def cache_set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache"""
        if not self._redis_client:
            return

        try:
            ttl = ttl or self.config.cache_ttl
            self._redis_client.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def cache_delete(self, pattern: str) -> None:
        """Delete cache keys matching pattern"""
        if not self._redis_client:
            return

        try:
            keys = self._redis_client.keys(pattern)
            if keys:
                self._redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")

    def get_pool_status(self) -> dict[str, Any]:
        """Get connection pool status"""
        pool = self._engine.pool
        return {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
            "query_count": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
        }

    def health_check(self) -> bool:
        """Check database health"""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def close(self) -> None:
        """Close all connections"""
        if self._scoped_session:
            self._scoped_session.remove()

        if self._engine:
            self._engine.dispose()

        if self._redis_client:
            self._redis_client.close()

        logger.info("Database manager closed")


# Singleton instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get or create database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def close_db_manager() -> None:
    """Close database manager"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None
