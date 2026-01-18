"""Unit tests for the application container registry API."""

from __future__ import annotations

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    create_application_container,
    get_application_container,
    set_application_container,
)


class TestCreateApplicationContainer:
    """Test cases for create_application_container function."""

    def test_create_application_container(self, mock_config: BotConfig) -> None:
        """Test that application container is created correctly."""
        container = create_application_container(mock_config)

        assert isinstance(container, ApplicationContainer)
        assert container.config == mock_config


class TestContainerRegistry:
    """Test cases for container registry functions."""

    def test_get_returns_none_when_not_set(self) -> None:
        """Test that get_application_container returns None when not set."""
        clear_application_container()
        assert get_application_container() is None

    def test_set_and_get_container(self, mock_config: BotConfig) -> None:
        """Test that set and get work correctly."""
        clear_application_container()

        container = ApplicationContainer(mock_config)
        set_application_container(container)

        assert get_application_container() is container

        clear_application_container()
        assert get_application_container() is None

    def test_clear_container(self, mock_config: BotConfig) -> None:
        """Test that clear_application_container clears the container."""
        container = ApplicationContainer(mock_config)
        set_application_container(container)

        clear_application_container()

        assert get_application_container() is None

    def test_service_resolution_via_registry(self, mock_config: BotConfig) -> None:
        """Test that services can be resolved via registered container."""
        from gpt_trader.app.config.profile_loader import get_profile_loader
        from gpt_trader.features.live_trade.execution.validation import get_failure_tracker

        clear_application_container()

        with pytest.raises(RuntimeError, match="No application container set"):
            get_failure_tracker()
        with pytest.raises(RuntimeError, match="No application container set"):
            get_profile_loader()

        container = ApplicationContainer(mock_config)
        set_application_container(container)

        tracker_container = get_failure_tracker()
        loader_container = get_profile_loader()

        assert tracker_container is container.validation_failure_tracker
        assert loader_container is container.profile_loader

        clear_application_container()
