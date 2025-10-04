"""Tests for BotConfig builder."""

import argparse
from unittest.mock import Mock

from bot_v2.cli.bot_config_builder import BotConfigBuilder


class TestBotConfigBuilder:
    """Test BotConfigBuilder."""

    def test_build_with_config_overrides(self):
        """Test builder passes valid config overrides to factory."""
        args = argparse.Namespace(
            profile="dev",
            dry_run=True,
            symbols=["BTC-PERP", "ETH-PERP"],
            interval=60,
            target_leverage=3,
        )

        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(config_factory=mock_factory)
        result = builder.build(args)

        mock_factory.from_profile.assert_called_once_with(
            "dev",
            dry_run=True,
            symbols=["BTC-PERP", "ETH-PERP"],
            interval=60,
            target_leverage=3,
        )
        assert result == "config_instance"

    def test_build_filters_skip_keys(self):
        """Test builder excludes skip-list keys from config overrides."""
        args = argparse.Namespace(
            profile="prod",
            dry_run=False,
            symbols=["SOL-PERP"],
            # Skip keys that should be filtered out
            account_snapshot=True,
            convert="USD:USDC:100",
            move_funds="DEFAULT:PERP:50",
            preview_order=True,
            order_symbol="BTC-PERP",
            order_side="buy",
            order_quantity="0.5",
        )

        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(config_factory=mock_factory)
        builder.build(args)

        # Verify only valid config keys passed, skip keys excluded
        mock_factory.from_profile.assert_called_once_with(
            "prod",
            dry_run=False,
            symbols=["SOL-PERP"],
        )

    def test_build_filters_none_values(self):
        """Test builder excludes None values from config overrides."""
        args = argparse.Namespace(
            profile="dev",
            dry_run=True,
            symbols=None,  # Should be filtered out
            interval=None,  # Should be filtered out
            target_leverage=3,
        )

        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(config_factory=mock_factory)
        builder.build(args)

        # Verify None values not passed to factory
        mock_factory.from_profile.assert_called_once_with(
            "dev",
            dry_run=True,
            target_leverage=3,
        )

    def test_build_loads_symbols_from_env_when_not_provided(self):
        """Test builder loads symbols from TRADING_SYMBOLS env when not in CLI."""
        args = argparse.Namespace(
            profile="dev",
            dry_run=True,
            symbols=None,
        )

        mock_env = Mock(return_value="BTC-PERP,ETH-PERP,SOL-PERP")
        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(env_reader=mock_env, config_factory=mock_factory)
        builder.build(args)

        mock_env.assert_called_once_with("TRADING_SYMBOLS", "")
        mock_factory.from_profile.assert_called_once_with(
            "dev",
            dry_run=True,
            symbols=["BTC-PERP", "ETH-PERP", "SOL-PERP"],
        )

    def test_build_cli_symbols_override_env_symbols(self):
        """Test CLI symbols take precedence over TRADING_SYMBOLS env."""
        args = argparse.Namespace(
            profile="dev",
            dry_run=True,
            symbols=["BTC-PERP"],  # CLI symbols provided
        )

        mock_env = Mock(return_value="ETH-PERP,SOL-PERP")  # Should be ignored
        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(env_reader=mock_env, config_factory=mock_factory)
        builder.build(args)

        # Env should not be called since CLI symbols provided
        mock_env.assert_not_called()
        mock_factory.from_profile.assert_called_once_with(
            "dev",
            dry_run=True,
            symbols=["BTC-PERP"],
        )

    def test_build_parses_trading_symbols_with_semicolon_delimiter(self):
        """Test TRADING_SYMBOLS parsing with semicolon delimiter."""
        args = argparse.Namespace(profile="dev", symbols=None)

        mock_env = Mock(return_value="BTC-PERP;ETH-PERP;SOL-PERP")
        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(env_reader=mock_env, config_factory=mock_factory)
        builder.build(args)

        mock_factory.from_profile.assert_called_once_with(
            "dev",
            symbols=["BTC-PERP", "ETH-PERP", "SOL-PERP"],
        )

    def test_build_strips_whitespace_from_trading_symbols(self):
        """Test TRADING_SYMBOLS parsing strips whitespace."""
        args = argparse.Namespace(profile="dev", symbols=None)

        mock_env = Mock(return_value="  BTC-PERP  ,  ETH-PERP  ,  SOL-PERP  ")
        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(env_reader=mock_env, config_factory=mock_factory)
        builder.build(args)

        mock_factory.from_profile.assert_called_once_with(
            "dev",
            symbols=["BTC-PERP", "ETH-PERP", "SOL-PERP"],
        )

    def test_build_skips_empty_trading_symbols(self):
        """Test builder skips empty TRADING_SYMBOLS env var."""
        args = argparse.Namespace(profile="dev", symbols=None)

        mock_env = Mock(return_value="")  # Empty env var
        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(env_reader=mock_env, config_factory=mock_factory)
        builder.build(args)

        # No symbols key passed when env is empty
        mock_factory.from_profile.assert_called_once_with("dev")

    def test_build_with_empty_symbols_list_loads_from_env(self):
        """Test builder loads from env when symbols is empty list."""
        args = argparse.Namespace(profile="dev", symbols=[])  # Empty list, not None

        mock_env = Mock(return_value="BTC-PERP,ETH-PERP")
        mock_factory = Mock()
        mock_factory.from_profile = Mock(return_value="config_instance")

        builder = BotConfigBuilder(env_reader=mock_env, config_factory=mock_factory)
        builder.build(args)

        # Empty list should trigger env fallback
        mock_env.assert_called_once_with("TRADING_SYMBOLS", "")
        mock_factory.from_profile.assert_called_once_with(
            "dev",
            symbols=["BTC-PERP", "ETH-PERP"],
        )
