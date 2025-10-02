"""
Tests for direct repository access pattern via StateRepositories.

Demonstrates how batch operations can use repositories directly
for better performance.
"""

from unittest.mock import Mock

import pytest

from bot_v2.state import StateConfig, StateManager, StateRepositories


@pytest.fixture
def mock_redis():
    """Create mock Redis adapter."""
    adapter = Mock()
    adapter.ping = Mock(return_value=True)
    adapter.get = Mock(return_value=None)
    adapter.setex = Mock(return_value=True)
    adapter.delete = Mock()
    adapter.keys = Mock(return_value=[])
    adapter.dbsize = Mock(return_value=0)
    adapter.close = Mock()
    return adapter


@pytest.fixture
def mock_postgres():
    """Create mock PostgreSQL adapter."""
    adapter = Mock()
    adapter.execute = Mock(return_value=[])
    adapter.commit = Mock()
    adapter.rollback = Mock()
    adapter.close = Mock()
    return adapter


@pytest.fixture
def mock_s3():
    """Create mock S3 adapter."""
    adapter = Mock()
    adapter.head_bucket = Mock()
    adapter.get_object = Mock()
    adapter.put_object = Mock()
    adapter.delete_object = Mock()
    adapter.list_objects_v2 = Mock(return_value={})
    return adapter


@pytest.fixture
def state_manager(mock_redis, mock_postgres, mock_s3):
    """Create StateManager with mock adapters."""
    config = StateConfig()
    return StateManager(
        config=config,
        redis_adapter=mock_redis,
        postgres_adapter=mock_postgres,
        s3_adapter=mock_s3,
    )


class TestRepositoryAccessPattern:
    """Test direct repository access for batch operations."""

    def test_get_repositories_returns_bundle(self, state_manager):
        """Should return StateRepositories bundle."""
        repos = state_manager.get_repositories()

        assert isinstance(repos, StateRepositories)
        assert repos.redis is not None
        assert repos.postgres is not None
        assert repos.s3 is not None

    def test_get_repositories_preserves_none_repos(self):
        """Should preserve None repositories when backends unavailable."""
        config = StateConfig()
        manager = StateManager(
            config=config,
            redis_adapter=None,
            postgres_adapter=None,
            s3_adapter=None,
        )

        repos = manager.get_repositories()

        assert repos.redis is None
        assert repos.postgres is None
        assert repos.s3 is None

    def test_repositories_are_same_instances(self, state_manager):
        """Returned repositories should be same instances as internal repos."""
        repos1 = state_manager.get_repositories()
        repos2 = state_manager.get_repositories()

        # Same instances returned each time
        assert repos1.redis is repos2.redis
        assert repos1.postgres is repos2.postgres
        assert repos1.s3 is repos2.s3

        # Same as internal repositories
        assert repos1.redis is state_manager._redis_repo
        assert repos1.postgres is state_manager._postgres_repo
        assert repos1.s3 is state_manager._s3_repo

    def test_repository_types_correct(self, state_manager):
        """Repositories should have correct types."""
        from bot_v2.state.repositories import (
            PostgresStateRepository,
            RedisStateRepository,
            S3StateRepository,
        )

        repos = state_manager.get_repositories()

        assert isinstance(repos.redis, RedisStateRepository)
        assert isinstance(repos.postgres, PostgresStateRepository)
        assert isinstance(repos.s3, S3StateRepository)
