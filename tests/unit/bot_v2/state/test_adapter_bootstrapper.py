"""
Unit tests for AdapterBootstrapper

Tests adapter creation, validation, and fallback logic for StateManager initialization.
"""

from unittest.mock import Mock, patch

import pytest

from bot_v2.state.adapter_bootstrapper import (
    AdapterBootstrapper,
    BootstrappedAdapters,
)
from bot_v2.state.state_manager import StateConfig
from bot_v2.state.storage_adapter_factory import StorageAdapterFactory


class TestBootstrappedAdapters:
    """Test BootstrappedAdapters dataclass."""

    def test_creation_with_all_adapters(self):
        """Should create bundle with all adapters."""
        redis_adapter = Mock()
        postgres_adapter = Mock()
        s3_adapter = Mock()

        bundle = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=postgres_adapter,
            s3=s3_adapter,
        )

        assert bundle.redis is redis_adapter
        assert bundle.postgres is postgres_adapter
        assert bundle.s3 is s3_adapter

    def test_creation_with_none_adapters(self):
        """Should allow None adapters."""
        bundle = BootstrappedAdapters(
            redis=None,
            postgres=None,
            s3=None,
        )

        assert bundle.redis is None
        assert bundle.postgres is None
        assert bundle.s3 is None

    def test_creation_with_partial_adapters(self):
        """Should allow partial adapter sets."""
        redis_adapter = Mock()

        bundle = BootstrappedAdapters(
            redis=redis_adapter,
            postgres=None,
            s3=None,
        )

        assert bundle.redis is redis_adapter
        assert bundle.postgres is None
        assert bundle.s3 is None


class TestAdapterBootstrapperInit:
    """Test AdapterBootstrapper initialization."""

    def test_init_stores_factory(self):
        """Should store provided factory."""
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        assert bootstrapper.factory is factory


class TestBootstrapWithDefaults:
    """Test bootstrap() with default (None) adapters."""

    def test_bootstrap_creates_all_adapters_from_config(self):
        """Should create all adapters when none provided."""
        config = StateConfig(
            redis_host="redis.local",
            redis_port=6380,
            redis_db=1,
            postgres_host="pg.local",
            postgres_port=5433,
            postgres_database="testdb",
            postgres_user="testuser",
            postgres_password="testpass",
            s3_region="us-west-2",
            s3_bucket="test-bucket",
        )

        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        with (
            patch.object(factory, "create_redis_adapter") as mock_redis,
            patch.object(factory, "create_postgres_adapter") as mock_postgres,
            patch.object(factory, "create_s3_adapter") as mock_s3,
        ):

            mock_redis.return_value = Mock()
            mock_postgres.return_value = Mock()
            mock_s3.return_value = Mock()

            result = bootstrapper.bootstrap(config)

            # Verify factory methods called with config values
            mock_redis.assert_called_once_with(
                host="redis.local",
                port=6380,
                db=1,
            )
            mock_postgres.assert_called_once_with(
                host="pg.local",
                port=5433,
                database="testdb",
                user="testuser",
                password="testpass",
            )
            mock_s3.assert_called_once_with(
                region="us-west-2",
                bucket="test-bucket",
            )

            # Verify result contains created adapters
            assert result.redis is mock_redis.return_value
            assert result.postgres is mock_postgres.return_value
            assert result.s3 is mock_s3.return_value

    def test_bootstrap_with_default_config(self):
        """Should use default config values when not specified."""
        config = StateConfig()  # All defaults
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        with (
            patch.object(factory, "create_redis_adapter") as mock_redis,
            patch.object(factory, "create_postgres_adapter") as mock_postgres,
            patch.object(factory, "create_s3_adapter") as mock_s3,
        ):

            mock_redis.return_value = Mock()
            mock_postgres.return_value = Mock()
            mock_s3.return_value = Mock()

            result = bootstrapper.bootstrap(config)

            # Verify default values passed
            mock_redis.assert_called_once_with(
                host="localhost",
                port=6379,
                db=0,
            )
            mock_postgres.assert_called_once_with(
                host="localhost",
                port=5432,
                database="trading_bot",
                user="trader",
                password="trader123",
            )
            mock_s3.assert_called_once_with(
                region="us-east-1",
                bucket="trading-bot-cold-storage",
            )

            assert isinstance(result, BootstrappedAdapters)


class TestBootstrapWithProvidedAdapters:
    """Test bootstrap() with pre-configured adapters."""

    def test_bootstrap_uses_provided_redis_adapter(self):
        """Should use provided Redis adapter without creating new one."""
        config = StateConfig()
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided_redis = Mock()

        with (
            patch.object(factory, "create_redis_adapter") as mock_create_redis,
            patch.object(factory, "create_postgres_adapter") as mock_postgres,
            patch.object(factory, "create_s3_adapter") as mock_s3,
        ):

            mock_postgres.return_value = Mock()
            mock_s3.return_value = Mock()

            result = bootstrapper.bootstrap(
                config,
                redis_adapter=provided_redis,
            )

            # Should NOT create Redis adapter
            mock_create_redis.assert_not_called()

            # Should use provided adapter
            assert result.redis is provided_redis

            # Should still create other adapters
            mock_postgres.assert_called_once()
            mock_s3.assert_called_once()

    def test_bootstrap_validates_provided_postgres_adapter(self):
        """Should validate provided Postgres adapter."""
        config = StateConfig()
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided_postgres = Mock()
        validated_postgres = Mock()

        with (
            patch.object(factory, "create_redis_adapter") as mock_redis,
            patch.object(factory, "create_postgres_adapter") as mock_create_postgres,
            patch.object(factory, "validate_postgres_adapter") as mock_validate_postgres,
            patch.object(factory, "create_s3_adapter") as mock_s3,
        ):

            mock_redis.return_value = Mock()
            mock_s3.return_value = Mock()
            mock_validate_postgres.return_value = validated_postgres

            result = bootstrapper.bootstrap(
                config,
                postgres_adapter=provided_postgres,
            )

            # Should NOT create Postgres adapter
            mock_create_postgres.assert_not_called()

            # Should validate provided adapter
            mock_validate_postgres.assert_called_once_with(provided_postgres)

            # Should use validated adapter
            assert result.postgres is validated_postgres

            # Should still create other adapters
            mock_redis.assert_called_once()
            mock_s3.assert_called_once()

    def test_bootstrap_validates_provided_s3_adapter(self):
        """Should validate provided S3 adapter."""
        config = StateConfig(s3_bucket="my-bucket")
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided_s3 = Mock()
        validated_s3 = Mock()

        with (
            patch.object(factory, "create_redis_adapter") as mock_redis,
            patch.object(factory, "create_postgres_adapter") as mock_postgres,
            patch.object(factory, "create_s3_adapter") as mock_create_s3,
            patch.object(factory, "validate_s3_adapter") as mock_validate_s3,
        ):

            mock_redis.return_value = Mock()
            mock_postgres.return_value = Mock()
            mock_validate_s3.return_value = validated_s3

            result = bootstrapper.bootstrap(
                config,
                s3_adapter=provided_s3,
            )

            # Should NOT create S3 adapter
            mock_create_s3.assert_not_called()

            # Should validate with bucket from config
            mock_validate_s3.assert_called_once_with(provided_s3, "my-bucket")

            # Should use validated adapter
            assert result.s3 is validated_s3

            # Should still create other adapters
            mock_redis.assert_called_once()
            mock_postgres.assert_called_once()

    def test_bootstrap_with_all_provided_adapters(self):
        """Should use all provided adapters without creation."""
        config = StateConfig()
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided_redis = Mock()
        provided_postgres = Mock()
        validated_postgres = Mock()
        provided_s3 = Mock()
        validated_s3 = Mock()

        with (
            patch.object(factory, "create_redis_adapter") as mock_create_redis,
            patch.object(factory, "create_postgres_adapter") as mock_create_postgres,
            patch.object(factory, "validate_postgres_adapter") as mock_validate_postgres,
            patch.object(factory, "create_s3_adapter") as mock_create_s3,
            patch.object(factory, "validate_s3_adapter") as mock_validate_s3,
        ):

            mock_validate_postgres.return_value = validated_postgres
            mock_validate_s3.return_value = validated_s3

            result = bootstrapper.bootstrap(
                config,
                redis_adapter=provided_redis,
                postgres_adapter=provided_postgres,
                s3_adapter=provided_s3,
            )

            # Should NOT create any adapters
            mock_create_redis.assert_not_called()
            mock_create_postgres.assert_not_called()
            mock_create_s3.assert_not_called()

            # Should validate Postgres and S3
            mock_validate_postgres.assert_called_once_with(provided_postgres)
            mock_validate_s3.assert_called_once_with(provided_s3, config.s3_bucket)

            # Should use provided/validated adapters
            assert result.redis is provided_redis
            assert result.postgres is validated_postgres
            assert result.s3 is validated_s3


class TestErrorPropagation:
    """Test error handling and propagation."""

    def test_bootstrap_propagates_redis_creation_error(self):
        """Should propagate errors from Redis adapter creation."""
        config = StateConfig()
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        with patch.object(factory, "create_redis_adapter") as mock_redis:
            mock_redis.side_effect = ConnectionError("Redis unavailable")

            with pytest.raises(ConnectionError, match="Redis unavailable"):
                bootstrapper.bootstrap(config)

    def test_bootstrap_propagates_postgres_validation_error(self):
        """Should propagate errors from Postgres validation."""
        config = StateConfig()
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided_postgres = Mock()

        with (
            patch.object(factory, "create_redis_adapter"),
            patch.object(factory, "validate_postgres_adapter") as mock_validate,
        ):

            mock_validate.side_effect = RuntimeError("Table creation failed")

            with pytest.raises(RuntimeError, match="Table creation failed"):
                bootstrapper.bootstrap(
                    config,
                    postgres_adapter=provided_postgres,
                )

    def test_bootstrap_propagates_s3_validation_error(self):
        """Should propagate errors from S3 validation."""
        config = StateConfig()
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided_s3 = Mock()

        with (
            patch.object(factory, "create_redis_adapter"),
            patch.object(factory, "create_postgres_adapter"),
            patch.object(factory, "validate_s3_adapter") as mock_validate,
        ):

            mock_validate.side_effect = ValueError("Bucket does not exist")

            with pytest.raises(ValueError, match="Bucket does not exist"):
                bootstrapper.bootstrap(
                    config,
                    s3_adapter=provided_s3,
                )


class TestBootstrapHelperMethods:
    """Test individual bootstrap helper methods."""

    def test_bootstrap_redis_creates_when_none(self):
        """_bootstrap_redis should create adapter when None."""
        config = StateConfig(redis_host="test.local", redis_port=6380, redis_db=2)
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        with patch.object(factory, "create_redis_adapter") as mock_create:
            mock_create.return_value = Mock()

            result = bootstrapper._bootstrap_redis(config, None)

            mock_create.assert_called_once_with(
                host="test.local",
                port=6380,
                db=2,
            )
            assert result is mock_create.return_value

    def test_bootstrap_redis_returns_provided_adapter(self):
        """_bootstrap_redis should return provided adapter unchanged."""
        config = StateConfig()
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided = Mock()

        with patch.object(factory, "create_redis_adapter") as mock_create:
            result = bootstrapper._bootstrap_redis(config, provided)

            mock_create.assert_not_called()
            assert result is provided

    def test_bootstrap_postgres_creates_when_none(self):
        """_bootstrap_postgres should create adapter when None."""
        config = StateConfig(
            postgres_host="pg.test",
            postgres_port=5433,
            postgres_database="testdb",
            postgres_user="testuser",
            postgres_password="testpass",
        )
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        with patch.object(factory, "create_postgres_adapter") as mock_create:
            mock_create.return_value = Mock()

            result = bootstrapper._bootstrap_postgres(config, None)

            mock_create.assert_called_once_with(
                host="pg.test",
                port=5433,
                database="testdb",
                user="testuser",
                password="testpass",
            )
            assert result is mock_create.return_value

    def test_bootstrap_postgres_validates_provided_adapter(self):
        """_bootstrap_postgres should validate provided adapter."""
        config = StateConfig()
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided = Mock()
        validated = Mock()

        with patch.object(factory, "validate_postgres_adapter") as mock_validate:
            mock_validate.return_value = validated

            result = bootstrapper._bootstrap_postgres(config, provided)

            mock_validate.assert_called_once_with(provided)
            assert result is validated

    def test_bootstrap_s3_creates_when_none(self):
        """_bootstrap_s3 should create adapter when None."""
        config = StateConfig(s3_region="us-west-1", s3_bucket="test-bucket")
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        with patch.object(factory, "create_s3_adapter") as mock_create:
            mock_create.return_value = Mock()

            result = bootstrapper._bootstrap_s3(config, None)

            mock_create.assert_called_once_with(
                region="us-west-1",
                bucket="test-bucket",
            )
            assert result is mock_create.return_value

    def test_bootstrap_s3_validates_provided_adapter(self):
        """_bootstrap_s3 should validate provided adapter."""
        config = StateConfig(s3_bucket="validation-bucket")
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)

        provided = Mock()
        validated = Mock()

        with patch.object(factory, "validate_s3_adapter") as mock_validate:
            mock_validate.return_value = validated

            result = bootstrapper._bootstrap_s3(config, provided)

            mock_validate.assert_called_once_with(provided, "validation-bucket")
            assert result is validated
