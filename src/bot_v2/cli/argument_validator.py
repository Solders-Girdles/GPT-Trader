"""Argument validation and environment handling for CLI parser.

Provides ArgumentValidator to validate parsed arguments and configure
logging based on environment variables.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ArgumentValidator:
    """Validates CLI arguments and configures logging based on environment."""

    def __init__(
        self,
        *,
        env_reader: Callable[[str], str | None] | None = None,
        log_config: Callable[[str, int], None] | None = None,
    ) -> None:
        """
        Initialize argument validator.

        Args:
            env_reader: Function to read environment variables (default: os.getenv)
            log_config: Function to configure logger levels (default: logging.getLogger().setLevel)
        """
        import os

        self._env_reader = env_reader or os.getenv
        self._log_config = log_config or self._default_log_config

    @staticmethod
    def _default_log_config(logger_name: str, level: int) -> None:
        """Default logger configuration implementation."""
        logging.getLogger(logger_name).setLevel(level)

    def validate(
        self, args: argparse.Namespace, parser: argparse.ArgumentParser
    ) -> argparse.Namespace:
        """
        Validate parsed CLI arguments.

        Args:
            args: Parsed argument namespace
            parser: ArgumentParser instance (for error reporting)

        Returns:
            Validated argument namespace

        Raises:
            SystemExit: On argument validation errors
        """
        # Validate symbols if provided
        if args.symbols:
            empty = [sym for sym in args.symbols if not str(sym).strip()]
            if empty:
                parser.error("Symbols must be non-empty strings")

        # Enable debug logging if requested
        if self._env_reader("PERPS_DEBUG") == "1":
            logger.info("Debug mode enabled via PERPS_DEBUG=1")
            self._log_config("bot_v2.features.brokerages.coinbase", logging.DEBUG)
            self._log_config("bot_v2.orchestration", logging.DEBUG)

        logger.debug("Parsed CLI arguments: profile=%s, dry_run=%s", args.profile, args.dry_run)

        return args
