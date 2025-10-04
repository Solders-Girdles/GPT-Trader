"""
Unit tests for RepositoryFactory

Tests repository creation from adapters with proper TTL handling and None cases.
"""

from unittest.mock import Mock, patch

import pytest

from bot_v2.state.adapter_bootstrapper import BootstrappedAdapters
from bot_v2.state.repository_factory import RepositoryFactory
from bot_v2.state.state_manager import StateConfig, StateRepositories


class TestCreateRepositoriesAllAdapters:
    """Test create_repositories with all adapters available."""

    def test_creates_all_repositories(self):
        """Should create all three repositories when all adapters provided."""
        redis_adapter = Mock()
        postgres_adapter = Mock()
        s3_adapter = Mock()

        adapters = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=postgres_adapter,
            s3=s3_adapter,
        )

        config = StateConfig(
            redis_ttl_seconds=7200,
            s3_bucket="test-bucket",
        )

        with (
            patch("bot_v2.state.repository_factory.RedisStateRepository") as mock_redis_repo,
            patch("bot_v2.state.repository_factory.PostgresStateRepository") as mock_postgres_repo,
            patch("bot_v2.state.repository_factory.S3StateRepository") as mock_s3_repo,
        ):

            result = RepositoryFactory.create_repositories(adapters, config)

            # Verify repository constructors called with correct args
            mock_redis_repo.assert_called_once_with(redis_adapter, 7200, metrics_collector=None)
            mock_postgres_repo.assert_called_once_with(postgres_adapter, metrics_collector=None)
            mock_s3_repo.assert_called_once_with(s3_adapter, "test-bucket", metrics_collector=None)

            # Verify result is StateRepositories
            assert isinstance(result, StateRepositories)
            assert result.redis is mock_redis_repo.return_value
            assert result.postgres is mock_postgres_repo.return_value
            assert result.s3 is mock_s3_repo.return_value

    def test_uses_default_redis_ttl(self):
        """Should use default TTL from config."""
        redis_adapter = Mock()
        postgres_adapter = Mock()
        s3_adapter = Mock()

        adapters = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=postgres_adapter,
            s3=s3_adapter,
        )

        config = StateConfig()  # Default TTL = 3600

        with (
            patch("bot_v2.state.repository_factory.RedisStateRepository") as mock_redis_repo,
            patch("bot_v2.state.repository_factory.PostgresStateRepository"),
            patch("bot_v2.state.repository_factory.S3StateRepository"),
        ):

            RepositoryFactory.create_repositories(adapters, config)

            # Verify default TTL used
            mock_redis_repo.assert_called_once_with(redis_adapter, 3600, metrics_collector=None)

    def test_uses_custom_s3_bucket(self):
        """Should use S3 bucket from config."""
        redis_adapter = Mock()
        postgres_adapter = Mock()
        s3_adapter = Mock()

        adapters = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=postgres_adapter,
            s3=s3_adapter,
        )

        config = StateConfig(s3_bucket="custom-bucket-name")

        with (
            patch("bot_v2.state.repository_factory.RedisStateRepository"),
            patch("bot_v2.state.repository_factory.PostgresStateRepository"),
            patch("bot_v2.state.repository_factory.S3StateRepository") as mock_s3_repo,
        ):

            RepositoryFactory.create_repositories(adapters, config)

            # Verify custom bucket used
            mock_s3_repo.assert_called_once_with(
                s3_adapter, "custom-bucket-name", metrics_collector=None
            )


class TestCreateRepositoriesNoneAdapters:
    """Test create_repositories with None adapters."""

    def test_returns_none_repos_when_all_adapters_none(self):
        """Should return None repositories when all adapters are None."""
        adapters = BootstrappedAdapters(
            redis=None,
            postgres=None,
            s3=None,
        )

        config = StateConfig()

        result = RepositoryFactory.create_repositories(adapters, config)

        # Verify result is StateRepositories with all None
        assert isinstance(result, StateRepositories)
        assert result.redis is None
        assert result.postgres is None
        assert result.s3 is None

    def test_returns_none_redis_when_adapter_none(self):
        """Should return None for Redis repo when adapter is None."""
        adapters = BootstrappedAdapters(
            redis=None,
            postgres=Mock(),
            s3=Mock(),
        )

        config = StateConfig()

        with (
            patch("bot_v2.state.repository_factory.PostgresStateRepository"),
            patch("bot_v2.state.repository_factory.S3StateRepository"),
        ):

            result = RepositoryFactory.create_repositories(adapters, config)

            assert result.redis is None
            assert result.postgres is not None
            assert result.s3 is not None

    def test_returns_none_postgres_when_adapter_none(self):
        """Should return None for Postgres repo when adapter is None."""
        adapters = BootstrappedAdapters(
            redis=Mock(),
            postgres=None,
            s3=Mock(),
        )

        config = StateConfig()

        with (
            patch("bot_v2.state.repository_factory.RedisStateRepository"),
            patch("bot_v2.state.repository_factory.S3StateRepository"),
        ):

            result = RepositoryFactory.create_repositories(adapters, config)

            assert result.redis is not None
            assert result.postgres is None
            assert result.s3 is not None

    def test_returns_none_s3_when_adapter_none(self):
        """Should return None for S3 repo when adapter is None."""
        adapters = BootstrappedAdapters(
            redis=Mock(),
            postgres=Mock(),
            s3=None,
        )

        config = StateConfig()

        with (
            patch("bot_v2.state.repository_factory.RedisStateRepository"),
            patch("bot_v2.state.repository_factory.PostgresStateRepository"),
        ):

            result = RepositoryFactory.create_repositories(adapters, config)

            assert result.redis is not None
            assert result.postgres is not None
            assert result.s3 is None


class TestCreateRepositoriesPartialAdapters:
    """Test create_repositories with various combinations."""

    def test_redis_only(self):
        """Should create only Redis repo when only Redis adapter provided."""
        redis_adapter = Mock()

        adapters = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=None,
            s3=None,
        )

        config = StateConfig(redis_ttl_seconds=1800)

        with patch("bot_v2.state.repository_factory.RedisStateRepository") as mock_redis_repo:

            result = RepositoryFactory.create_repositories(adapters, config)

            mock_redis_repo.assert_called_once_with(redis_adapter, 1800, metrics_collector=None)
            assert result.redis is mock_redis_repo.return_value
            assert result.postgres is None
            assert result.s3 is None

    def test_postgres_and_s3_only(self):
        """Should create Postgres and S3 repos when Redis unavailable."""
        postgres_adapter = Mock()
        s3_adapter = Mock()

        adapters = BootstrappedAdapters(
            redis=None,
            postgres=postgres_adapter,
            s3=s3_adapter,
        )

        config = StateConfig(s3_bucket="partial-bucket")

        with (
            patch("bot_v2.state.repository_factory.PostgresStateRepository") as mock_postgres_repo,
            patch("bot_v2.state.repository_factory.S3StateRepository") as mock_s3_repo,
        ):

            result = RepositoryFactory.create_repositories(adapters, config)

            mock_postgres_repo.assert_called_once_with(postgres_adapter, metrics_collector=None)
            mock_s3_repo.assert_called_once_with(
                s3_adapter, "partial-bucket", metrics_collector=None
            )

            assert result.redis is None
            assert result.postgres is mock_postgres_repo.return_value
            assert result.s3 is mock_s3_repo.return_value


class TestTTLPassthrough:
    """Test TTL parameter handling."""

    def test_zero_ttl_passed_through(self):
        """Should pass through zero TTL value."""
        redis_adapter = Mock()

        adapters = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=None,
            s3=None,
        )

        config = StateConfig(redis_ttl_seconds=0)

        with patch("bot_v2.state.repository_factory.RedisStateRepository") as mock_redis_repo:

            RepositoryFactory.create_repositories(adapters, config)

            # Verify 0 TTL passed (not treated as falsy)
            mock_redis_repo.assert_called_once_with(redis_adapter, 0, metrics_collector=None)

    def test_large_ttl_passed_through(self):
        """Should handle large TTL values."""
        redis_adapter = Mock()

        adapters = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=None,
            s3=None,
        )

        config = StateConfig(redis_ttl_seconds=86400 * 30)  # 30 days

        with patch("bot_v2.state.repository_factory.RedisStateRepository") as mock_redis_repo:

            RepositoryFactory.create_repositories(adapters, config)

            mock_redis_repo.assert_called_once_with(
                redis_adapter, 86400 * 30, metrics_collector=None
            )


class TestStaticMethod:
    """Test that create_repositories is a static method."""

    def test_can_call_without_instance(self):
        """Should be callable as static method."""
        adapters = BootstrappedAdapters(
            redis=None,
            postgres=None,
            s3=None,
        )
        config = StateConfig()

        # Should work without creating factory instance
        result = RepositoryFactory.create_repositories(adapters, config)

        assert isinstance(result, StateRepositories)

    def test_multiple_calls_independent(self):
        """Should create independent repositories on each call."""
        redis_adapter = Mock()
        postgres_adapter = Mock()

        adapters = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=postgres_adapter,
            s3=None,
        )

        config = StateConfig()

        with (
            patch("bot_v2.state.repository_factory.RedisStateRepository") as mock_redis_repo,
            patch("bot_v2.state.repository_factory.PostgresStateRepository") as mock_postgres_repo,
        ):

            # Configure mocks to return new instances each time
            mock_redis_repo.side_effect = [Mock(), Mock()]
            mock_postgres_repo.side_effect = [Mock(), Mock()]

            # First call
            result1 = RepositoryFactory.create_repositories(adapters, config)

            # Second call
            result2 = RepositoryFactory.create_repositories(adapters, config)

            # Should be called twice (once per call)
            assert mock_redis_repo.call_count == 2
            assert mock_postgres_repo.call_count == 2

            # Results should be independent instances
            assert result1.redis is not result2.redis
            assert result1.postgres is not result2.postgres
