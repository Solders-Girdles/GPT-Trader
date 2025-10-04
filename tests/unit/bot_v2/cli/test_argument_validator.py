"""Tests for CLI argument validator."""

import argparse
import logging
from unittest.mock import Mock

import pytest

from bot_v2.cli.argument_validator import ArgumentValidator


class TestArgumentValidator:
    """Test ArgumentValidator."""

    def test_validate_with_valid_symbols(self):
        """Test validation passes with valid symbols."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(symbols=["BTC-PERP", "ETH-PERP"], profile="dev", dry_run=False)

        validator = ArgumentValidator()
        result = validator.validate(args, parser)

        assert result is args

    def test_validate_with_empty_symbols_raises_error(self):
        """Test validation fails with empty symbols."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            symbols=["BTC-PERP", "", "ETH-PERP"], profile="dev", dry_run=False
        )

        validator = ArgumentValidator()

        with pytest.raises(SystemExit):
            validator.validate(args, parser)

    def test_validate_with_whitespace_symbols_raises_error(self):
        """Test validation fails with whitespace-only symbols."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            symbols=["BTC-PERP", "  ", "ETH-PERP"], profile="dev", dry_run=False
        )

        validator = ArgumentValidator()

        with pytest.raises(SystemExit):
            validator.validate(args, parser)

    def test_validate_with_no_symbols(self):
        """Test validation passes when symbols is None."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(symbols=None, profile="dev", dry_run=False)

        validator = ArgumentValidator()
        result = validator.validate(args, parser)

        assert result is args

    def test_validate_enables_debug_logging_when_perps_debug_set(self):
        """Test PERPS_DEBUG=1 enables debug logging."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(symbols=None, profile="dev", dry_run=False)

        mock_env = Mock(return_value="1")
        mock_log_config = Mock()

        validator = ArgumentValidator(env_reader=mock_env, log_config=mock_log_config)
        validator.validate(args, parser)

        mock_env.assert_called_once_with("PERPS_DEBUG")
        assert mock_log_config.call_count == 2
        mock_log_config.assert_any_call("bot_v2.features.brokerages.coinbase", logging.DEBUG)
        mock_log_config.assert_any_call("bot_v2.orchestration", logging.DEBUG)

    def test_validate_skips_debug_logging_when_perps_debug_not_set(self):
        """Test debug logging is not enabled when PERPS_DEBUG is not set."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(symbols=None, profile="dev", dry_run=False)

        mock_env = Mock(return_value=None)
        mock_log_config = Mock()

        validator = ArgumentValidator(env_reader=mock_env, log_config=mock_log_config)
        validator.validate(args, parser)

        mock_env.assert_called_once_with("PERPS_DEBUG")
        mock_log_config.assert_not_called()

    def test_validate_skips_debug_logging_when_perps_debug_is_zero(self):
        """Test debug logging is not enabled when PERPS_DEBUG=0."""
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(symbols=None, profile="dev", dry_run=False)

        mock_env = Mock(return_value="0")
        mock_log_config = Mock()

        validator = ArgumentValidator(env_reader=mock_env, log_config=mock_log_config)
        validator.validate(args, parser)

        mock_env.assert_called_once_with("PERPS_DEBUG")
        mock_log_config.assert_not_called()

    def test_default_log_config_configures_logger(self):
        """Test default log config sets logger level."""
        validator = ArgumentValidator()

        # Test that default log config works
        validator._default_log_config("bot_v2.cli.test_logger", logging.WARNING)

        assert logging.getLogger("bot_v2.cli.test_logger").level == logging.WARNING
