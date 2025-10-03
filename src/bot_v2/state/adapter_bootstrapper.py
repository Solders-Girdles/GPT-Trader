"""
Adapter Bootstrapper for StateManager

Handles creation and validation of storage adapters with fallback logic.
Extracted from StateManager.__init__ to improve separation of concerns.
"""

from dataclasses import dataclass

from bot_v2.state.storage_adapter_factory import StorageAdapterFactory
from bot_v2.state.utils.adapters import (
    PostgresAdapter,
    RedisAdapter,
    S3Adapter,
)


@dataclass
class BootstrappedAdapters:
    """
    Bundle of initialized storage adapters.

    Attributes:
        redis: Redis adapter for hot tier storage
        postgres: PostgreSQL adapter for warm tier storage
        s3: S3 adapter for cold tier storage
    """

    redis: RedisAdapter | None
    postgres: PostgresAdapter | None
    s3: S3Adapter | None


class AdapterBootstrapper:
    """
    Handles adapter creation and validation for StateManager.

    Provides fallback logic: if an adapter is provided, validate it;
    otherwise create a new one from config.
    """

    def __init__(self, factory: StorageAdapterFactory):
        """
        Initialize bootstrapper with a storage adapter factory.

        Args:
            factory: Factory for creating storage adapters
        """
        self.factory = factory

    def bootstrap(
        self,
        config,  # StateConfig type hint omitted to avoid circular import
        redis_adapter: RedisAdapter | None = None,
        postgres_adapter: PostgresAdapter | None = None,
        s3_adapter: S3Adapter | None = None,
    ) -> BootstrappedAdapters:
        """
        Bootstrap storage adapters with fallback to creation.

        Args:
            config: State configuration
            redis_adapter: Optional pre-configured Redis adapter
            postgres_adapter: Optional pre-configured Postgres adapter
            s3_adapter: Optional pre-configured S3 adapter

        Returns:
            BootstrappedAdapters with all adapters ready
        """
        return BootstrappedAdapters(
            redis=self._bootstrap_redis(config, redis_adapter),
            postgres=self._bootstrap_postgres(config, postgres_adapter),
            s3=self._bootstrap_s3(config, s3_adapter),
        )

    def _bootstrap_redis(self, config, adapter: RedisAdapter | None) -> RedisAdapter | None:
        """
        Bootstrap Redis adapter.

        Args:
            config: State configuration
            adapter: Optional pre-configured adapter

        Returns:
            Configured Redis adapter or None
        """
        if adapter is None:
            return self.factory.create_redis_adapter(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
            )
        return adapter

    def _bootstrap_postgres(
        self, config, adapter: PostgresAdapter | None
    ) -> PostgresAdapter | None:
        """
        Bootstrap and validate Postgres adapter.

        Args:
            config: State configuration
            adapter: Optional pre-configured adapter

        Returns:
            Configured and validated Postgres adapter or None
        """
        if adapter is None:
            return self.factory.create_postgres_adapter(
                host=config.postgres_host,
                port=config.postgres_port,
                database=config.postgres_database,
                user=config.postgres_user,
                password=config.postgres_password,
            )
        # Validate provided adapter by creating tables
        return self.factory.validate_postgres_adapter(adapter)

    def _bootstrap_s3(self, config, adapter: S3Adapter | None) -> S3Adapter | None:
        """
        Bootstrap and validate S3 adapter.

        Args:
            config: State configuration
            adapter: Optional pre-configured adapter

        Returns:
            Configured and validated S3 adapter or None
        """
        if adapter is None:
            return self.factory.create_s3_adapter(
                region=config.s3_region,
                bucket=config.s3_bucket,
            )
        # Validate provided adapter by checking bucket
        return self.factory.validate_s3_adapter(adapter, config.s3_bucket)
