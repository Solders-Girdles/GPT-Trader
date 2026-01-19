"""Integration tests for ApplicationContainer lifecycle through CLI entrypoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

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


class TestContainerLifecycleCLI:
    """Test container lifecycle through CLI entrypoint."""

    def test_instantiate_bot_registers_container(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.services import instantiate_bot

        assert get_application_container() is None

        bot = instantiate_bot(mock_config)

        container = get_application_container()
        assert container is not None
        assert isinstance(container, ApplicationContainer)
        assert container.config == mock_config
        assert bot is not None

    def test_instantiate_bot_uses_existing_container(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.services import instantiate_bot

        existing_container = ApplicationContainer(mock_config)
        set_application_container(existing_container)

        bot = instantiate_bot(mock_config)

        assert get_application_container() is existing_container
        assert bot is not None

    def test_run_bot_clears_container_on_success(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.commands.run import _run_bot
        from gpt_trader.cli.services import instantiate_bot

        bot = instantiate_bot(mock_config)

        assert get_application_container() is not None

        bot.run = AsyncMock(return_value=None)

        _run_bot(bot, single_cycle=True)
        bot.run.assert_awaited_once_with(single_cycle=True)

        assert get_application_container() is None

    def test_run_bot_clears_container_on_exception(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.commands.run import _run_bot
        from gpt_trader.cli.services import instantiate_bot

        bot = instantiate_bot(mock_config)

        assert get_application_container() is not None

        bot.run = AsyncMock(side_effect=RuntimeError("Test error"))
        with pytest.raises(RuntimeError, match="Test error"):
            _run_bot(bot, single_cycle=True)

        assert get_application_container() is None

    def test_run_bot_clears_container_on_keyboard_interrupt(self, mock_config: BotConfig) -> None:
        from gpt_trader.cli.commands.run import _run_bot
        from gpt_trader.cli.services import instantiate_bot

        bot = instantiate_bot(mock_config)

        assert get_application_container() is not None

        bot.run = AsyncMock(side_effect=KeyboardInterrupt())
        _run_bot(bot, single_cycle=True)

        assert get_application_container() is None
