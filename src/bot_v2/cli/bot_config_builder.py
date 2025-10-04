"""BotConfig builder for CLI arguments.

Provides BotConfigBuilder to construct BotConfig instances from
parsed CLI arguments with environment fallbacks.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class BotConfigBuilder:
    """Builds BotConfig from CLI arguments with environment fallbacks."""

    # Arguments that should not be passed to BotConfig
    SKIP_KEYS = {
        "profile",
        "account_snapshot",
        "convert",
        "move_funds",
        "preview_order",
        "edit_order_preview",
        "apply_order_edit",
        "order_side",
        "order_type",
        "order_quantity",
        "order_price",
        "order_stop",
        "order_tif",
        "order_client_id",
        "order_reduce_only",
        "order_leverage",
        "order_symbol",
    }

    def __init__(
        self,
        *,
        env_reader: Callable[[str, str], str | None] | None = None,
        config_factory: Any = None,
    ) -> None:
        """
        Initialize BotConfig builder.

        Args:
            env_reader: Function to read environment variables (default: os.getenv)
            config_factory: BotConfig factory class (default: BotConfig from orchestration)
        """
        import os

        self._env_reader = env_reader or os.getenv
        self._config_factory = config_factory or self._default_config_factory()

    @staticmethod
    def _default_config_factory() -> Any:
        """Load default BotConfig factory."""
        from bot_v2.orchestration.configuration import BotConfig

        return BotConfig

    def build(self, args: argparse.Namespace) -> Any:
        """
        Build BotConfig from parsed CLI arguments.

        Args:
            args: Parsed argument namespace

        Returns:
            BotConfig instance configured from arguments
        """
        # Filter arguments to config overrides (skip command-specific args)
        config_overrides = {
            key: value
            for key, value in vars(args).items()
            if value is not None and key not in self.SKIP_KEYS
        }

        # Handle symbols from environment if not provided via CLI
        if "symbols" not in config_overrides or not config_overrides.get("symbols"):
            env_symbols = self._env_reader("TRADING_SYMBOLS", "")
            if env_symbols:
                tokens = [
                    tok.strip() for tok in env_symbols.replace(";", ",").split(",") if tok.strip()
                ]
                if tokens:
                    config_overrides["symbols"] = tokens
                    logger.info("Loaded %d symbols from TRADING_SYMBOLS env var", len(tokens))

        logger.info(
            "Building bot config with profile=%s, overrides=%s", args.profile, config_overrides
        )

        return self._config_factory.from_profile(args.profile, **config_overrides)
