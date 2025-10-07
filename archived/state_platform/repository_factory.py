"""
Repository Factory for StateManager

Handles creation of tier-specific repositories from storage adapters.
Extracted from StateManager.__init__ to improve separation of concerns.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.monitoring.metrics_collector import MetricsCollector

from bot_v2.state.adapter_bootstrapper import BootstrappedAdapters
from bot_v2.state.repositories import (
    PostgresStateRepository,
    RedisStateRepository,
    S3StateRepository,
)


class RepositoryFactory:
    """
    Creates tier-specific repositories from storage adapters.

    Handles null adapter cases gracefully, returning None for repositories
    when the underlying adapter is unavailable.
    """

    @staticmethod
    def create_repositories(
        adapters: BootstrappedAdapters,
        config,  # StateConfig type hint omitted to avoid circular import
        metrics_collector: "MetricsCollector | None" = None,
    ):  # Returns StateRepositories (from state_manager.py)
        """
        Create tier-specific repositories from adapters.

        Args:
            adapters: Bootstrapped storage adapters
            config: State configuration
            metrics_collector: Optional metrics collector for telemetry

        Returns:
            StateRepositories bundle with redis, postgres, and s3 repositories
        """
        # Import here to avoid circular dependency
        from bot_v2.state.state_manager import StateRepositories

        redis_repo = (
            RedisStateRepository(
                adapters.redis, config.redis_ttl_seconds, metrics_collector=metrics_collector
            )
            if adapters.redis
            else None
        )

        postgres_repo = (
            PostgresStateRepository(adapters.postgres, metrics_collector=metrics_collector)
            if adapters.postgres
            else None
        )

        s3_repo = (
            S3StateRepository(adapters.s3, config.s3_bucket, metrics_collector=metrics_collector)
            if adapters.s3
            else None
        )

        return StateRepositories(
            redis=redis_repo,
            postgres=postgres_repo,
            s3=s3_repo,
        )
