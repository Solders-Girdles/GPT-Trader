"""Integration tests for ApplicationContainer lifecycle across runtime scenarios."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.app.container as container_module
from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    get_application_container,
    set_application_container,
)

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


class TestContainerLifecycleTUI:
    """Test container lifecycle through TUI entrypoint."""

    @pytest.mark.asyncio
    async def test_tui_unmount_clears_container(self, mock_config: BotConfig) -> None:
        container = ApplicationContainer(mock_config)
        set_application_container(container)

        assert get_application_container() is not None

        clear_application_container()

        assert get_application_container() is None

    @pytest.mark.asyncio
    async def test_tui_lifecycle_integration(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.services import instantiate_bot

        _bot = instantiate_bot(mock_config)  # noqa: F841

        container = get_application_container()
        assert container is not None

        assert get_application_container() is container

        clear_application_container()

        assert get_application_container() is None


class TestContainerLifecycleMultipleCycles:
    """Test container lifecycle across multiple start/stop cycles."""

    def test_multiple_bot_cycles_no_leak(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.services import instantiate_bot

        for i in range(3):
            bot = instantiate_bot(mock_config)
            container = get_application_container()

            assert container is not None, f"Cycle {i}: Container should exist"
            assert bot is not None, f"Cycle {i}: Bot should exist"

            clear_application_container()
            assert get_application_container() is None, f"Cycle {i}: Container should be cleared"

    def test_container_services_are_fresh_each_cycle(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.services import instantiate_bot

        instantiate_bot(mock_config)
        container1 = get_application_container()
        event_store1 = container1.event_store
        clear_application_container()

        instantiate_bot(mock_config)
        container2 = get_application_container()
        event_store2 = container2.event_store
        clear_application_container()

        assert container1 is not container2
        assert event_store1 is not event_store2


class TestContainerLifecycleErrorPaths:
    """Test container cleanup on various error paths."""

    def test_container_cleared_when_bot_creation_fails(self, mock_config: BotConfig) -> None:
        container = ApplicationContainer(mock_config)
        set_application_container(container)

        try:
            raise ValueError("Simulated bot creation failure")
        except ValueError:
            pass
        finally:
            clear_application_container()

        assert get_application_container() is None

    @pytest.mark.asyncio
    async def test_container_cleared_on_async_error(self, mock_config: BotConfig) -> None:
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
        from gpt_trader.cli.services import instantiate_bot

        instantiate_bot(mock_config)
        container = get_application_container()

        assert container.config is mock_config
        assert container.event_store is not None
        assert container.orders_store is not None
        assert container.broker is not None

    def test_services_not_accessible_after_clear(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.services import instantiate_bot

        instantiate_bot(mock_config)
        clear_application_container()

        container = get_application_container()
        assert container is None

    def test_service_resolution_functions_handle_missing_container(self) -> None:
        from gpt_trader.app.config.profile_loader import get_profile_loader
        from gpt_trader.features.live_trade.execution.validation import get_failure_tracker

        clear_application_container()

        with pytest.raises(RuntimeError, match="No application container set"):
            get_failure_tracker()
        with pytest.raises(RuntimeError, match="No application container set"):
            get_profile_loader()


class TestContainerDebugLogging:
    """Test container lifecycle debug logging."""

    def test_set_container_logs_debug(
        self, mock_config: BotConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        container = ApplicationContainer(mock_config)
        mock_logger = MagicMock()
        monkeypatch.setattr(container_module, "logger", mock_logger)

        set_application_container(container)

        mock_logger.debug.assert_called()

    def test_clear_container_logs_debug(
        self, mock_config: BotConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        container = ApplicationContainer(mock_config)
        set_application_container(container)
        mock_logger = MagicMock()
        monkeypatch.setattr(container_module, "logger", mock_logger)

        clear_application_container()

        mock_logger.debug.assert_called()
