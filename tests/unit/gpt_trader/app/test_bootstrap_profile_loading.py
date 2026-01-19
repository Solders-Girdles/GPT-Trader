"""Tests for profile-driven configuration behavior during bootstrap."""

from __future__ import annotations

import pytest

from gpt_trader.app.bootstrap import build_bot
from gpt_trader.app.config import BotConfig
from gpt_trader.config.types import Profile


@pytest.fixture(autouse=True)
def clear_container():
    """Clear the application container before and after each test."""
    from gpt_trader.app.container import clear_application_container

    clear_application_container()
    yield
    clear_application_container()


@pytest.fixture
def mock_config() -> BotConfig:
    """Create a mock BotConfig for testing."""
    return BotConfig(
        profile=Profile.TEST,
        symbols=["BTC-USD", "ETH-USD"],
        mock_broker=True,
        derivatives_enabled=False,
    )


class TestProfileLoading:
    """Tests for profile-based configuration loading."""

    def test_from_profile_loads_yaml_config(self) -> None:
        config = BotConfig.from_profile(profile=Profile.DEV, mock_broker=True)

        assert config.profile == Profile.DEV

    def test_from_profile_with_overrides(self) -> None:
        config = BotConfig.from_profile(
            profile=Profile.DEV,
            mock_broker=True,
            dry_run=True,
        )

        assert config.mock_broker is True
        assert config.dry_run is True

    @pytest.mark.parametrize("profile", list(Profile))
    def test_all_profiles_load_successfully(self, profile: Profile) -> None:
        config = BotConfig.from_profile(profile=profile, mock_broker=True)

        assert config.profile == profile


class TestContainerInitialization:
    """Tests for ApplicationContainer initialization during bootstrap."""

    def test_build_bot_creates_container(self, mock_config: BotConfig) -> None:
        """Test that build_bot uses ApplicationContainer internally."""
        bot = build_bot(mock_config)

        # Bot should have a container reference
        assert bot.container is not None

    def test_container_has_config(self, mock_config: BotConfig) -> None:
        """Test that container created during build_bot has config."""
        bot = build_bot(mock_config)

        assert bot.container.config == mock_config
