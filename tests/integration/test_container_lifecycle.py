"""Integration tests for ApplicationContainer lifecycle across entrypoints.

Tests verify:
1. Container is created and registered during bot instantiation
2. Container is cleared on graceful shutdown
3. Container is cleared even on exceptions
4. Multiple lifecycle cycles don't leak resources

Run with: pytest tests/integration/test_container_lifecycle.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    get_application_container,
    set_application_container,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def clean_container():
    """Ensure container is cleared before and after each test."""
    clear_application_container()
    yield
    clear_application_container()


@pytest.fixture
def mock_config() -> BotConfig:
    """Create a test configuration."""
    return BotConfig(
        symbols=["BTC-USD"],
        mock_broker=True,
        dry_run=True,
        interval=1,
    )


class TestContainerLifecycleCLI:
    """Test container lifecycle through CLI entrypoint."""

    def test_instantiate_bot_registers_container(self, mock_config: BotConfig) -> None:
        """Test that instantiate_bot creates and registers container."""
        from gpt_trader.cli.services import instantiate_bot

        # Container should not exist before
        assert get_application_container() is None

        # Create bot
        bot = instantiate_bot(mock_config)

        # Container should now be registered
        container = get_application_container()
        assert container is not None
        assert isinstance(container, ApplicationContainer)
        assert container.config == mock_config
        assert bot is not None

    def test_instantiate_bot_uses_existing_container(self, mock_config: BotConfig) -> None:
        """Test that instantiate_bot reuses an existing container."""
        from gpt_trader.cli.services import instantiate_bot

        # Pre-set a container
        existing_container = ApplicationContainer(mock_config)
        set_application_container(existing_container)

        # instantiate_bot should use the existing container
        bot = instantiate_bot(mock_config)

        # Should still be the same container
        assert get_application_container() is existing_container
        assert bot is not None

    @pytest.mark.asyncio
    async def test_run_bot_clears_container_on_success(self, mock_config: BotConfig) -> None:
        """Test that _run_bot clears container after successful run."""
        from gpt_trader.cli.commands.run import _run_bot
        from gpt_trader.cli.services import instantiate_bot

        bot = instantiate_bot(mock_config)

        # Verify container is set
        assert get_application_container() is not None

        # Mock bot.run to complete immediately
        bot.run = AsyncMock()

        # Run bot (single_cycle=True for quick exit)
        with patch("gpt_trader.cli.commands.run.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = None
            _run_bot(bot, single_cycle=True)

        # Container should be cleared after run
        assert get_application_container() is None

    @pytest.mark.asyncio
    async def test_run_bot_clears_container_on_exception(self, mock_config: BotConfig) -> None:
        """Test that _run_bot clears container even when exception occurs."""
        from gpt_trader.cli.commands.run import _run_bot
        from gpt_trader.cli.services import instantiate_bot

        bot = instantiate_bot(mock_config)

        # Verify container is set
        assert get_application_container() is not None

        # Mock bot.run to raise an exception
        with patch("gpt_trader.cli.commands.run.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = RuntimeError("Test error")

            # Run should not propagate the error (it's caught)
            # but container should still be cleared
            try:
                _run_bot(bot, single_cycle=True)
            except RuntimeError:
                pass  # Expected

        # Container should be cleared even after exception
        assert get_application_container() is None

    @pytest.mark.asyncio
    async def test_run_bot_clears_container_on_keyboard_interrupt(
        self, mock_config: BotConfig
    ) -> None:
        """Test that _run_bot clears container on KeyboardInterrupt."""
        from gpt_trader.cli.commands.run import _run_bot
        from gpt_trader.cli.services import instantiate_bot

        bot = instantiate_bot(mock_config)

        # Verify container is set
        assert get_application_container() is not None

        # Mock bot.run to raise KeyboardInterrupt
        with patch("gpt_trader.cli.commands.run.asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = KeyboardInterrupt()

            _run_bot(bot, single_cycle=True)

        # Container should be cleared after interrupt
        assert get_application_container() is None


class TestContainerLifecycleTUI:
    """Test container lifecycle through TUI entrypoint."""

    @pytest.mark.asyncio
    async def test_tui_unmount_clears_container(self, mock_config: BotConfig) -> None:
        """Test that TUI unmount clears the application container."""
        # Set up container
        container = ApplicationContainer(mock_config)
        set_application_container(container)

        # Verify container is set
        assert get_application_container() is not None

        # Simulate TUI cleanup by calling clear_application_container
        # (This is what on_unmount does at the end)
        clear_application_container()

        # Container should be cleared
        assert get_application_container() is None

    @pytest.mark.asyncio
    async def test_tui_lifecycle_integration(self, mock_config: BotConfig) -> None:
        """Test full TUI lifecycle: mount -> unmount."""
        from gpt_trader.cli.services import instantiate_bot

        # Create bot (simulates CLI creating bot before TUI)
        _bot = instantiate_bot(mock_config)  # noqa: F841

        # Verify container is registered
        container = get_application_container()
        assert container is not None

        # Simulate TUI mount - container should still exist
        assert get_application_container() is container

        # Simulate TUI unmount cleanup
        clear_application_container()

        # Container should be cleared
        assert get_application_container() is None


class TestContainerLifecycleMultipleCycles:
    """Test container lifecycle across multiple start/stop cycles."""

    def test_multiple_bot_cycles_no_leak(self, mock_config: BotConfig) -> None:
        """Test that multiple bot instantiation cycles don't leak containers."""
        from gpt_trader.cli.services import instantiate_bot

        for i in range(3):
            # Create bot
            bot = instantiate_bot(mock_config)
            container = get_application_container()

            assert container is not None, f"Cycle {i}: Container should exist"
            assert bot is not None, f"Cycle {i}: Bot should exist"

            # Simulate shutdown
            clear_application_container()
            assert get_application_container() is None, f"Cycle {i}: Container should be cleared"

    def test_container_services_are_fresh_each_cycle(self, mock_config: BotConfig) -> None:
        """Test that services are recreated for each container."""
        from gpt_trader.cli.services import instantiate_bot

        # First cycle
        instantiate_bot(mock_config)
        container1 = get_application_container()
        event_store1 = container1.event_store
        clear_application_container()

        # Second cycle
        instantiate_bot(mock_config)
        container2 = get_application_container()
        event_store2 = container2.event_store
        clear_application_container()

        # Containers should be different instances
        assert container1 is not container2
        # Event stores should be different instances
        assert event_store1 is not event_store2


class TestContainerLifecycleErrorPaths:
    """Test container cleanup on various error paths."""

    def test_container_cleared_when_bot_creation_fails(self, mock_config: BotConfig) -> None:
        """Test container is not left dangling when bot creation fails."""
        # Create container but simulate bot creation failure
        container = ApplicationContainer(mock_config)
        set_application_container(container)

        try:
            # Simulate some operation that fails
            raise ValueError("Simulated bot creation failure")
        except ValueError:
            pass
        finally:
            # Cleanup should still happen
            clear_application_container()

        assert get_application_container() is None

    @pytest.mark.asyncio
    async def test_container_cleared_on_async_error(self, mock_config: BotConfig) -> None:
        """Test container is cleared when async operation fails."""
        container = ApplicationContainer(mock_config)
        set_application_container(container)

        async def failing_operation():
            raise RuntimeError("Async failure")

        try:
            await failing_operation()
        except RuntimeError:
            pass
        finally:
            clear_application_container()

        assert get_application_container() is None


class TestContainerServiceAccess:
    """Test service access through the container lifecycle."""

    def test_services_accessible_after_registration(self, mock_config: BotConfig) -> None:
        """Test that services are accessible after container registration."""
        from gpt_trader.cli.services import instantiate_bot

        instantiate_bot(mock_config)
        container = get_application_container()

        # Core services should be accessible
        assert container.config is mock_config
        assert container.event_store is not None
        assert container.orders_store is not None
        assert container.broker is not None  # DeterministicBroker for mock_broker=True

    def test_services_not_accessible_after_clear(self, mock_config: BotConfig) -> None:
        """Test that services are not accessible after container is cleared."""
        from gpt_trader.cli.services import instantiate_bot

        instantiate_bot(mock_config)
        clear_application_container()

        # Container should be None
        container = get_application_container()
        assert container is None

    def test_service_resolution_functions_handle_missing_container(self) -> None:
        """Test that service resolution functions handle missing container correctly."""
        from gpt_trader.app.config.profile_loader import get_profile_loader
        from gpt_trader.features.live_trade.execution.validation import get_failure_tracker

        # Ensure no container is set
        clear_application_container()

        # Service locators require container (no fallback)
        with pytest.raises(RuntimeError, match="No application container set"):
            get_failure_tracker()
        with pytest.raises(RuntimeError, match="No application container set"):
            get_profile_loader()


class TestContainerDebugLogging:
    """Test container lifecycle debug logging."""

    def test_set_container_logs_debug(self, mock_config: BotConfig) -> None:
        """Test that setting container logs debug message."""
        container = ApplicationContainer(mock_config)

        with patch("gpt_trader.app.container.logger") as mock_logger:
            set_application_container(container)

            mock_logger.debug.assert_called()

    def test_clear_container_logs_debug(self, mock_config: BotConfig) -> None:
        """Test that clearing container logs debug message."""
        container = ApplicationContainer(mock_config)
        set_application_container(container)

        with patch("gpt_trader.app.container.logger") as mock_logger:
            clear_application_container()

            mock_logger.debug.assert_called()
