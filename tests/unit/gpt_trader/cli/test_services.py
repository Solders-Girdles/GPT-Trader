"""Unit tests for CLI services module."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.cli.commands.run as run_module
import gpt_trader.cli.services as services_module
from gpt_trader.app.config import BotConfig
from gpt_trader.app.config.validation import ConfigValidationError
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    get_application_container,
    set_application_container,
)
from gpt_trader.cli.commands.run import _run_bot
from gpt_trader.cli.services import instantiate_bot, load_profile_config
from gpt_trader.config.types import Profile


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

    def test_sets_container_when_none_present(
        self,
        mock_config: BotConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that instantiate_bot sets the container when none is present."""
        # Setup mock container
        mock_container = MagicMock(spec=ApplicationContainer)
        mock_bot = MagicMock()
        mock_container.create_bot.return_value = mock_bot
        mock_create_container = MagicMock(return_value=mock_container)
        monkeypatch.setattr(services_module, "create_application_container", mock_create_container)

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

    def test_uses_existing_container(
        self,
        mock_config: BotConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that instantiate_bot uses existing container if already set."""
        # Setup existing container
        existing_container = MagicMock(spec=ApplicationContainer)
        mock_bot = MagicMock()
        existing_container.create_bot.return_value = mock_bot
        set_application_container(existing_container)

        # Patch to verify no new container is created
        mock_create = MagicMock()
        monkeypatch.setattr(services_module, "create_application_container", mock_create)

        bot = instantiate_bot(mock_config)

        # Verify no new container was created
        mock_create.assert_not_called()

        # Verify existing container was used
        existing_container.create_bot.assert_called_once()
        assert bot is mock_bot
        assert get_application_container() is existing_container

    def test_rejects_invalid_config_before_container_creation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that bot creation fails closed on invalid runtime config."""
        invalid_config = BotConfig(cfm_enabled=True, trading_modes=["spot"])
        mock_create = MagicMock()
        monkeypatch.setattr(services_module, "create_application_container", mock_create)

        with pytest.raises(
            ConfigValidationError,
            match="cfm_enabled requires trading_modes to include 'cfm'",
        ):
            instantiate_bot(invalid_config)

        mock_create.assert_not_called()
        assert get_application_container() is None

    def test_allows_observe_read_only_zero_position_fraction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that read-only observe config can create a bot with zero position size."""
        observe_config = load_profile_config(Profile.OBSERVE)
        mock_container = MagicMock(spec=ApplicationContainer)
        mock_bot = MagicMock()
        mock_container.create_bot.return_value = mock_bot
        mock_create = MagicMock(return_value=mock_container)
        monkeypatch.setattr(services_module, "create_application_container", mock_create)

        bot = instantiate_bot(observe_config)

        mock_create.assert_called_once_with(observe_config)
        mock_container.create_bot.assert_called_once()
        assert bot is mock_bot

    def test_rejects_non_observe_zero_position_fraction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that executable profiles still reject zero position fraction."""
        invalid_config = BotConfig()
        invalid_config.risk.position_fraction = Decimal("0")
        mock_create = MagicMock()
        monkeypatch.setattr(services_module, "create_application_container", mock_create)

        with pytest.raises(
            ConfigValidationError,
            match="risk.position_fraction must be between 0 and 1",
        ):
            instantiate_bot(invalid_config)

        mock_create.assert_not_called()

    def test_observe_profile_still_rejects_cfm_mismatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that observe startup only exempts the read-only sizing sentinel."""
        observe_config = load_profile_config(Profile.OBSERVE)
        observe_config.cfm_enabled = True
        observe_config.trading_modes = ["spot"]
        mock_create = MagicMock()
        monkeypatch.setattr(services_module, "create_application_container", mock_create)

        with pytest.raises(
            ConfigValidationError,
            match="cfm_enabled requires trading_modes to include 'cfm'",
        ):
            instantiate_bot(observe_config)

        mock_create.assert_not_called()


class TestRunBotCleanup:
    """Test cases for _run_bot cleanup behavior."""

    def test_clears_container_on_shutdown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that _run_bot clears the container in finally block."""
        # Setup: Create a mock bot and set a container
        mock_bot = MagicMock()
        mock_bot.running = False  # Prevent signal handler issues

        mock_container = MagicMock(spec=ApplicationContainer)
        set_application_container(mock_container)

        # Verify container is set
        assert get_application_container() is mock_container

        # Run bot (will exit immediately since running=False)
        monkeypatch.setattr(run_module, "asyncio", MagicMock())
        _run_bot(mock_bot, single_cycle=True)

        # Verify container was cleared in finally block
        assert get_application_container() is None

    def test_clears_container_even_on_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that container is cleared even if bot.run() raises."""
        mock_bot = MagicMock()
        mock_container = MagicMock(spec=ApplicationContainer)
        set_application_container(mock_container)

        # Make asyncio.run raise an exception
        mock_asyncio = MagicMock()
        mock_asyncio.run.side_effect = RuntimeError("test")
        monkeypatch.setattr(run_module, "asyncio", mock_asyncio)

        with pytest.raises(RuntimeError, match="test"):
            _run_bot(mock_bot, single_cycle=True)

        # Verify container was still cleared
        assert get_application_container() is None
