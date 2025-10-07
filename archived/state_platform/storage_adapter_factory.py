"""
Storage Adapter Factory for State Management

Centralizes adapter initialization and connection bootstrap logic
for Redis, PostgreSQL, and S3 storage backends.
"""

import logging

from bot_v2.state.utils.adapters import (
    DefaultPostgresAdapter,
    DefaultRedisAdapter,
    DefaultS3Adapter,
    PostgresAdapter,
    RedisAdapter,
    S3Adapter,
)

logger = logging.getLogger(__name__)


class StorageAdapterFactory:
    """
    Factory for initializing and validating storage adapters.

    Centralizes connection logic and graceful degradation when
    backends are unavailable.
    """

    @staticmethod
    def create_redis_adapter(host: str, port: int, db: int) -> RedisAdapter | None:
        """
        Initialize Redis connection.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number

        Returns:
            Initialized RedisAdapter or None if connection fails
        """
        try:
            adapter = DefaultRedisAdapter(host=host, port=port, db=db)
            if adapter.ping():
                logger.info("Redis connection established")
                return adapter
            else:
                logger.warning("Redis ping failed")
                return None
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            return None

    @staticmethod
    def create_postgres_adapter(
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> PostgresAdapter | None:
        """
        Initialize PostgreSQL connection and create tables.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password

        Returns:
            Initialized PostgresAdapter or None if connection/setup fails
        """
        try:
            adapter = DefaultPostgresAdapter(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )
            # Create tables if not exist
            StorageAdapterFactory._create_postgres_tables(adapter)
            logger.info("PostgreSQL connection established")
            return adapter
        except Exception as e:
            logger.warning(f"PostgreSQL initialization failed: {e}")
            return None

    @staticmethod
    def create_s3_adapter(region: str, bucket: str) -> S3Adapter | None:
        """
        Initialize S3 client and verify bucket exists.

        Args:
            region: AWS region
            bucket: S3 bucket name

        Returns:
            Initialized S3Adapter or None if connection/verification fails
        """
        try:
            adapter = DefaultS3Adapter(region=region)
            # Verify bucket exists
            adapter.head_bucket(bucket=bucket)
            logger.info("S3 connection established")
            return adapter
        except Exception as e:
            logger.warning(f"S3 initialization failed: {e}")
            return None

    @staticmethod
    def validate_postgres_adapter(
        adapter: PostgresAdapter, bucket: str | None = None
    ) -> PostgresAdapter | None:
        """
        Validate provided PostgreSQL adapter by creating tables.

        Args:
            adapter: PostgreSQL adapter to validate
            bucket: Unused, kept for signature compatibility

        Returns:
            The adapter if valid, None if validation fails
        """
        try:
            StorageAdapterFactory._create_postgres_tables(adapter)
            return adapter
        except Exception as e:
            logger.warning(f"PostgreSQL table creation failed: {e}")
            return None

    @staticmethod
    def validate_s3_adapter(adapter: S3Adapter, bucket: str) -> S3Adapter | None:
        """
        Validate provided S3 adapter by checking bucket exists.

        Args:
            adapter: S3 adapter to validate
            bucket: S3 bucket name to verify

        Returns:
            The adapter if valid, None if validation fails
        """
        try:
            adapter.head_bucket(bucket=bucket)
            return adapter
        except Exception as e:
            logger.warning(f"S3 bucket verification failed: {e}")
            return None

    @staticmethod
    def _create_postgres_tables(adapter: PostgresAdapter) -> None:
        """
        Create necessary PostgreSQL tables.

        Args:
            adapter: PostgreSQL adapter to use

        Raises:
            Exception: If table creation fails
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS state_warm (
            key VARCHAR(255) PRIMARY KEY,
            data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            size_bytes INTEGER,
            checksum VARCHAR(64),
            version INTEGER DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_state_warm_last_accessed
        ON state_warm(last_accessed);

        CREATE TABLE IF NOT EXISTS state_metadata (
            key VARCHAR(255) PRIMARY KEY,
            category VARCHAR(10),
            location VARCHAR(255),
            created_at TIMESTAMP,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            size_bytes INTEGER,
            checksum VARCHAR(64)
        );
        """

        try:
            adapter.execute(create_table_sql)
            adapter.commit()
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL tables: {e}")
            adapter.rollback()
            raise

    @classmethod
    def create_adapters(
        cls,
        redis_config: dict | None = None,
        postgres_config: dict | None = None,
        s3_config: dict | None = None,
    ) -> tuple[RedisAdapter | None, PostgresAdapter | None, S3Adapter | None]:
        """
        Create all storage adapters based on configuration.

        Args:
            redis_config: Redis configuration dict with keys: host, port, db
            postgres_config: Postgres configuration dict with keys: host, port, database, user, password
            s3_config: S3 configuration dict with keys: region, bucket

        Returns:
            Tuple of (redis_adapter, postgres_adapter, s3_adapter), each may be None
        """
        redis_adapter = None
        postgres_adapter = None
        s3_adapter = None

        if redis_config:
            redis_adapter = cls.create_redis_adapter(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
            )

        if postgres_config:
            postgres_adapter = cls.create_postgres_adapter(
                host=postgres_config["host"],
                port=postgres_config["port"],
                database=postgres_config["database"],
                user=postgres_config["user"],
                password=postgres_config["password"],
            )

        if s3_config:
            s3_adapter = cls.create_s3_adapter(
                region=s3_config["region"],
                bucket=s3_config["bucket"],
            )

        return redis_adapter, postgres_adapter, s3_adapter
