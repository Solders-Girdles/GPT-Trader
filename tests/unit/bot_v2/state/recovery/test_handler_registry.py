"""
Unit tests for RecoveryHandlerRegistry

Tests handler registration, lookup, batch operations, and edge cases.
"""

from unittest.mock import Mock

import pytest

from bot_v2.state.recovery.handler_registry import RecoveryHandlerRegistry
from bot_v2.state.recovery.models import FailureType


class TestRegistryInit:
    """Test registry initialization."""

    def test_init_creates_empty_registry(self):
        """Should create empty registry."""
        registry = RecoveryHandlerRegistry()

        assert len(registry) == 0
        assert registry.get_registered_failure_types() == []


class TestHandlerRegistration:
    """Test handler registration."""

    def test_register_single_handler(self):
        """Should register handler for failure type."""
        registry = RecoveryHandlerRegistry()
        handler = Mock()

        registry.register(FailureType.REDIS_DOWN, handler)

        assert registry.has_handler(FailureType.REDIS_DOWN)
        assert registry.get_handler(FailureType.REDIS_DOWN) is handler
        assert len(registry) == 1

    def test_register_multiple_handlers(self):
        """Should register multiple different handlers."""
        registry = RecoveryHandlerRegistry()
        redis_handler = Mock()
        postgres_handler = Mock()

        registry.register(FailureType.REDIS_DOWN, redis_handler)
        registry.register(FailureType.POSTGRES_DOWN, postgres_handler)

        assert len(registry) == 2
        assert registry.get_handler(FailureType.REDIS_DOWN) is redis_handler
        assert registry.get_handler(FailureType.POSTGRES_DOWN) is postgres_handler

    def test_register_duplicate_replaces_handler(self, caplog):
        """Should replace handler when registering duplicate."""
        registry = RecoveryHandlerRegistry()
        handler1 = Mock(__name__="handler1")
        handler2 = Mock(__name__="handler2")

        registry.register(FailureType.REDIS_DOWN, handler1)
        registry.register(FailureType.REDIS_DOWN, handler2)

        assert registry.get_handler(FailureType.REDIS_DOWN) is handler2
        assert len(registry) == 1
        assert "already registered" in caplog.text

    def test_register_batch_handlers(self):
        """Should register multiple handlers at once."""
        registry = RecoveryHandlerRegistry()
        handlers = {
            FailureType.REDIS_DOWN: Mock(),
            FailureType.POSTGRES_DOWN: Mock(),
            FailureType.S3_UNAVAILABLE: Mock(),
        }

        registry.register_batch(handlers)

        assert len(registry) == 3
        for failure_type, handler in handlers.items():
            assert registry.get_handler(failure_type) is handler


class TestHandlerLookup:
    """Test handler lookup operations."""

    def test_get_handler_returns_registered_handler(self):
        """Should return handler for registered failure type."""
        registry = RecoveryHandlerRegistry()
        handler = Mock()
        registry.register(FailureType.REDIS_DOWN, handler)

        result = registry.get_handler(FailureType.REDIS_DOWN)

        assert result is handler

    def test_get_handler_returns_none_for_unregistered(self):
        """Should return None for unregistered failure type."""
        registry = RecoveryHandlerRegistry()

        result = registry.get_handler(FailureType.REDIS_DOWN)

        assert result is None

    def test_has_handler_returns_true_for_registered(self):
        """Should return True for registered handler."""
        registry = RecoveryHandlerRegistry()
        registry.register(FailureType.REDIS_DOWN, Mock())

        assert registry.has_handler(FailureType.REDIS_DOWN)

    def test_has_handler_returns_false_for_unregistered(self):
        """Should return False for unregistered handler."""
        registry = RecoveryHandlerRegistry()

        assert not registry.has_handler(FailureType.REDIS_DOWN)

    def test_contains_operator(self):
        """Should support 'in' operator."""
        registry = RecoveryHandlerRegistry()
        registry.register(FailureType.REDIS_DOWN, Mock())

        assert FailureType.REDIS_DOWN in registry
        assert FailureType.POSTGRES_DOWN not in registry


class TestHandlerRetrieval:
    """Test bulk handler retrieval."""

    def test_get_all_handlers_returns_copy(self):
        """Should return copy of handlers dict."""
        registry = RecoveryHandlerRegistry()
        handler1 = Mock()
        handler2 = Mock()
        registry.register(FailureType.REDIS_DOWN, handler1)
        registry.register(FailureType.POSTGRES_DOWN, handler2)

        result = registry.get_all_handlers()

        assert result == {
            FailureType.REDIS_DOWN: handler1,
            FailureType.POSTGRES_DOWN: handler2,
        }
        # Verify it's a copy
        result[FailureType.S3_UNAVAILABLE] = Mock()
        assert FailureType.S3_UNAVAILABLE not in registry

    def test_get_registered_failure_types(self):
        """Should return list of registered failure types."""
        registry = RecoveryHandlerRegistry()
        registry.register(FailureType.REDIS_DOWN, Mock())
        registry.register(FailureType.POSTGRES_DOWN, Mock())

        result = registry.get_registered_failure_types()

        assert set(result) == {FailureType.REDIS_DOWN, FailureType.POSTGRES_DOWN}


class TestHandlerUnregistration:
    """Test handler unregistration."""

    def test_unregister_removes_handler(self):
        """Should remove registered handler."""
        registry = RecoveryHandlerRegistry()
        registry.register(FailureType.REDIS_DOWN, Mock())

        success = registry.unregister(FailureType.REDIS_DOWN)

        assert success
        assert not registry.has_handler(FailureType.REDIS_DOWN)
        assert len(registry) == 0

    def test_unregister_unregistered_returns_false(self):
        """Should return False when unregistering unregistered handler."""
        registry = RecoveryHandlerRegistry()

        success = registry.unregister(FailureType.REDIS_DOWN)

        assert not success

    def test_clear_removes_all_handlers(self):
        """Should remove all registered handlers."""
        registry = RecoveryHandlerRegistry()
        registry.register(FailureType.REDIS_DOWN, Mock())
        registry.register(FailureType.POSTGRES_DOWN, Mock())

        registry.clear()

        assert len(registry) == 0
        assert registry.get_registered_failure_types() == []


class TestRegistrySize:
    """Test registry size operations."""

    def test_len_returns_handler_count(self):
        """Should return number of registered handlers."""
        registry = RecoveryHandlerRegistry()

        assert len(registry) == 0

        registry.register(FailureType.REDIS_DOWN, Mock())
        assert len(registry) == 1

        registry.register(FailureType.POSTGRES_DOWN, Mock())
        assert len(registry) == 2

        registry.unregister(FailureType.REDIS_DOWN)
        assert len(registry) == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_register_with_callable_object(self):
        """Should accept any callable as handler."""

        class CallableHandler:
            def __call__(self, operation):
                return True

        registry = RecoveryHandlerRegistry()
        handler = CallableHandler()

        registry.register(FailureType.REDIS_DOWN, handler)

        assert registry.get_handler(FailureType.REDIS_DOWN) is handler

    def test_register_with_lambda(self):
        """Should accept lambda as handler."""
        registry = RecoveryHandlerRegistry()
        handler = lambda op: True  # noqa: E731

        registry.register(FailureType.REDIS_DOWN, handler)

        assert registry.get_handler(FailureType.REDIS_DOWN) is handler

    def test_multiple_registrations_same_handler(self):
        """Should allow same handler for multiple failure types."""
        registry = RecoveryHandlerRegistry()
        shared_handler = Mock()

        registry.register(FailureType.REDIS_DOWN, shared_handler)
        registry.register(FailureType.POSTGRES_DOWN, shared_handler)

        assert registry.get_handler(FailureType.REDIS_DOWN) is shared_handler
        assert registry.get_handler(FailureType.POSTGRES_DOWN) is shared_handler
        assert len(registry) == 2


class TestAllFailureTypes:
    """Test registry with all failure types."""

    def test_register_all_failure_types(self):
        """Should handle all defined failure types."""
        registry = RecoveryHandlerRegistry()

        # Register handler for each failure type
        for failure_type in FailureType:
            registry.register(failure_type, Mock())

        assert len(registry) == len(FailureType)
        for failure_type in FailureType:
            assert registry.has_handler(failure_type)
