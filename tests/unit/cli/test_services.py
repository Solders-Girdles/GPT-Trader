"""Unit tests for CLI services module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    get_application_container,
    set_application_container,
)
from gpt_trader.cli.commands.run import _run_bot
from gpt_trader.cli.services import instantiate_bot
from gpt_trader.orchestration.configuration import BotConfig


@pytest.fixture
def mock_config() -> BotConfig:
    """Create a mock BotConfig for testing."""
    return BotConfig(
        symbols=["BTC-USD"],
        mock_broker=True,  # Use mock broker to avoid credential lookup
    )


@pytest.fixture(autouse=True)
def clean_container():
    """Ensure clean container state for each test."""
    clear_application_container()
    yield
    clear_application_container()


class TestInstantiateBot:
    """Test cases for instantiate_bot function."""

    @patch("gpt_trader.cli.services.create_application_container")
    def test_sets_container_when_none_present(
        self,
        mock_create_container: MagicMock,
        mock_config: BotConfig,
    ) -> None:
        """Test that instantiate_bot sets the container when none is present."""
        # Setup mock container
        mock_container = MagicMock(spec=ApplicationContainer)
        mock_bot = MagicMock()
        mock_container.create_bot.return_value = mock_bot
        mock_create_container.return_value = mock_container

        # Verify no container initially
        assert get_application_container() is None

        # Call instantiate_bot
        bot = instantiate_bot(mock_config)

        # Verify container was created and set
        mock_create_container.assert_called_once_with(mock_config)
        assert get_application_container() is mock_container

        # Verify bot was created from container
        mock_container.create_bot.assert_called_once()
        assert bot is mock_bot

    def test_uses_existing_container(self, mock_config: BotConfig) -> None:
        """Test that instantiate_bot uses existing container if already set."""
        # Setup existing container
        existing_container = MagicMock(spec=ApplicationContainer)
        mock_bot = MagicMock()
        existing_container.create_bot.return_value = mock_bot
        set_application_container(existing_container)

        # Call instantiate_bot
        with patch("gpt_trader.cli.services.create_application_container") as mock_create:
            bot = instantiate_bot(mock_config)

            # Verify no new container was created
            mock_create.assert_not_called()

        # Verify existing container was used
        existing_container.create_bot.assert_called_once()
        assert bot is mock_bot
        assert get_application_container() is existing_container


class TestRunBotCleanup:
    """Test cases for _run_bot cleanup behavior."""

    def test_clears_container_on_shutdown(self) -> None:
        """Test that _run_bot clears the container in finally block."""
        # Setup: Create a mock bot and set a container
        mock_bot = MagicMock()
        mock_bot.running = False  # Prevent signal handler issues

        mock_container = MagicMock(spec=ApplicationContainer)
        set_application_container(mock_container)

        # Verify container is set
        assert get_application_container() is mock_container

        # Run bot (will exit immediately since running=False)
        with patch("gpt_trader.cli.commands.run.asyncio.run"):
            _run_bot(mock_bot, single_cycle=True)

        # Verify container was cleared in finally block
        assert get_application_container() is None

    def test_clears_container_even_on_exception(self) -> None:
        """Test that container is cleared even if bot.run() raises."""
        mock_bot = MagicMock()
        mock_container = MagicMock(spec=ApplicationContainer)
        set_application_container(mock_container)

        # Make asyncio.run raise an exception
        with patch("gpt_trader.cli.commands.run.asyncio.run", side_effect=RuntimeError("test")):
            with pytest.raises(RuntimeError, match="test"):
                _run_bot(mock_bot, single_cycle=True)

        # Verify container was still cleared
        assert get_application_container() is None
