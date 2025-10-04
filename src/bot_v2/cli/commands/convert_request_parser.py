"""Convert request parsing for CLI commands.

Provides ConvertRequest dataclass and ConvertRequestParser to parse
FROM:TO:AMOUNT format conversion arguments.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvertRequest:
    """Parsed conversion request."""

    from_asset: str
    to_asset: str
    amount: str


class ConvertRequestParser:
    """Parses conversion request strings."""

    @staticmethod
    def parse(convert_arg: str, parser: argparse.ArgumentParser) -> ConvertRequest:
        """
        Parse FROM:TO:AMOUNT format into ConvertRequest.

        Args:
            convert_arg: Conversion argument in format "FROM:TO:AMOUNT"
            parser: ArgumentParser for error reporting

        Returns:
            ConvertRequest with parsed values

        Raises:
            SystemExit: If format is invalid (via parser.error)
        """
        logger.info("Parsing convert argument: %s", convert_arg)

        try:
            from_asset, to_asset, amount = (part.strip() for part in convert_arg.split(":", 2))
        except ValueError:
            logger.error("Invalid convert argument format: %s", convert_arg)
            parser.error("--convert requires format FROM:TO:AMOUNT")

        logger.info("Parsed conversion: %s from %s to %s", amount, from_asset, to_asset)

        return ConvertRequest(from_asset=from_asset, to_asset=to_asset, amount=amount)
