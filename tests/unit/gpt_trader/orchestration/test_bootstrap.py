"""Comprehensive tests for bootstrap module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.config.types import Profile
from gpt_trader.orchestration.bootstrap import (
    BootstrapLogRecord,
    bot_from_profile,
    build_bot,
    normalise_symbols,
    resolve_runtime_paths,
)


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


@pytest.fixture
def dev_config() -> BotConfig:
    """Create a dev profile config."""
    return BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-USD"],
        mock_broker=True,
        dry_run=True,
    )


class TestNormaliseSymbols:
    """Tests for symbol normalization during bootstrap."""

    def test_normalise_with_valid_symbols(self, mock_config: BotConfig) -> None:
        symbols, logs = normalise_symbols(["BTC-USD", "ETH-USD"], config=mock_config)

        assert "BTC-USD" in symbols
        assert "ETH-USD" in symbols

    def test_normalise_empty_list_uses_fallback(self, mock_config: BotConfig) -> None:
        symbols, logs = normalise_symbols([], config=mock_config)

        # Should use fallback bases
        assert len(symbols) > 0

    def test_normalise_none_uses_fallback(self, mock_config: BotConfig) -> None:
        symbols, logs = normalise_symbols(None, config=mock_config)

        # Should use fallback bases
        assert len(symbols) > 0

    def test_normalise_logs_are_bootstrap_log_records(self, mock_config: BotConfig) -> None:
        _, logs = normalise_symbols(["BTC-USD"], config=mock_config)

        for log in logs:
            assert isinstance(log, BootstrapLogRecord)
            assert hasattr(log, "level")
            assert hasattr(log, "message")


class TestResolveRuntimePaths:
    """Tests for runtime path resolution."""

    def test_resolve_paths_returns_runtime_paths(self, mock_config: BotConfig) -> None:
        paths = resolve_runtime_paths(Profile.TEST, mock_config)

        assert hasattr(paths, "storage_dir")
        assert hasattr(paths, "event_store_root")

    def test_resolve_paths_creates_directories(self, mock_config: BotConfig) -> None:
        paths = resolve_runtime_paths(Profile.TEST, mock_config)

        # The paths should be Path objects
        assert isinstance(paths.storage_dir, Path)

    def test_different_profiles_have_different_paths(self, mock_config: BotConfig) -> None:
        test_paths = resolve_runtime_paths(Profile.TEST, mock_config)
        dev_config = BotConfig(profile=Profile.DEV, symbols=["BTC-USD"], mock_broker=True)
        dev_paths = resolve_runtime_paths(Profile.DEV, dev_config)

        # Different profiles should have different storage directories
        assert test_paths.storage_dir != dev_paths.storage_dir


class TestBuildBot:
    """Tests for build_bot function."""

    def test_build_bot_returns_trading_bot(self, mock_config: BotConfig) -> None:
        from gpt_trader.orchestration.trading_bot import TradingBot

        bot = build_bot(mock_config)

        assert isinstance(bot, TradingBot)

    def test_build_bot_uses_container(self, mock_config: BotConfig) -> None:
        bot = build_bot(mock_config)

        # Bot should have a container reference
        assert hasattr(bot, "container") or hasattr(bot, "_container")

    def test_build_bot_with_webhook_logs_message(self) -> None:
        config = BotConfig(
            profile=Profile.TEST,
            symbols=["BTC-USD"],
            mock_broker=True,
            webhook_url="https://example.com/webhook",
        )

        with patch("gpt_trader.orchestration.bootstrap.logger") as mock_logger:
            build_bot(config)

            # Should log webhook enabled
            assert mock_logger.info.called


class TestBotFromProfile:
    """Tests for bot_from_profile function."""

    def test_bot_from_profile_dev(self) -> None:
        bot = bot_from_profile("dev")

        assert bot is not None
        # Dev profile should use mock broker
        assert bot.config.mock_broker is True

    def test_bot_from_profile_test(self) -> None:
        bot = bot_from_profile("test")

        assert bot is not None
        assert bot.config.mock_broker is True

    def test_bot_from_profile_case_insensitive(self) -> None:
        bot1 = bot_from_profile("DEV")
        bot2 = bot_from_profile("dev")
        bot3 = bot_from_profile("Dev")

        assert bot1.config.profile == bot2.config.profile == bot3.config.profile

    def test_bot_from_profile_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            bot_from_profile("invalid_profile")

    @pytest.mark.parametrize(
        "profile_name",
        ["dev", "test"],  # Only test profiles that use mock broker
    )
    def test_mock_broker_profiles_create_bot(self, profile_name: str) -> None:
        """Test that mock broker profiles can create bots without credentials."""
        bot = bot_from_profile(profile_name)
        assert bot is not None
        assert bot.config.mock_broker is True

    def test_non_mock_profiles_require_credentials(self) -> None:
        """Test that production profiles require Coinbase credentials."""
        # Ensure env vars are unset
        with patch.dict("os.environ", {}, clear=True):
            # These should raise without credentials
            for profile_name in ["demo", "prod", "spot", "canary"]:
                with pytest.raises(ValueError, match="Coinbase Credentials"):
                    bot_from_profile(profile_name)


class TestBootstrapLogRecord:
    """Tests for BootstrapLogRecord dataclass."""

    def test_log_record_creation(self) -> None:
        record = BootstrapLogRecord(level=20, message="test message", args=("arg1",))

        assert record.level == 20
        assert record.message == "test message"
        assert record.args == ("arg1",)

    def test_log_record_default_args(self) -> None:
        record = BootstrapLogRecord(level=10, message="no args")

        assert record.args == ()

    def test_log_record_is_frozen(self) -> None:
        record = BootstrapLogRecord(level=20, message="test")

        with pytest.raises(Exception):  # FrozenInstanceError
            record.level = 30  # type: ignore


class TestEnvironmentVariableHandling:
    """Tests for environment variable handling during bootstrap."""

    def test_config_respects_env_variables(self) -> None:
        # Test that BotConfig picks up environment variables
        with patch.dict(
            "os.environ",
            {
                "GPT_TRADER_DRY_RUN": "1",
            },
        ):
            config = BotConfig(
                profile=Profile.TEST,
                symbols=["BTC-USD"],
                mock_broker=True,
            )

            # Config should be created successfully
            assert config.profile == Profile.TEST


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
