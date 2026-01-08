"""
DEPRECATED: This module has moved to gpt_trader.app.config.validation

This shim exists for backwards compatibility. Update imports to use:
    from gpt_trader.app.config.validation import validate_config, ConfigValidationError
    # or
    from gpt_trader.app.config import validate_config, ConfigValidationError
"""

from __future__ import annotations

import warnings

from gpt_trader.app.config.validation import ConfigValidationError, validate_config

__all__ = ["ConfigValidationError", "validate_config"]

warnings.warn(
    "gpt_trader.orchestration.configuration.bot_config.validator is deprecated. "
    "Import from gpt_trader.app.config.validation instead.",
    DeprecationWarning,
    stacklevel=2,
)
