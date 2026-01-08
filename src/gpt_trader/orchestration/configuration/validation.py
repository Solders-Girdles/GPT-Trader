"""
DEPRECATED: This module has moved to gpt_trader.app.config.validation

This shim exists for backwards compatibility. Update imports to use:
    from gpt_trader.app.config.validation import ConfigValidationError, ConfigValidationResult
    # or
    from gpt_trader.app.config import ConfigValidationError, ConfigValidationResult
"""

from __future__ import annotations

import warnings

from gpt_trader.app.config.validation import (
    ConfigValidationError,
    ConfigValidationResult,
    format_validation_errors,
)

__all__ = [
    "ConfigValidationError",
    "ConfigValidationResult",
    "format_validation_errors",
]

warnings.warn(
    "gpt_trader.orchestration.configuration.validation is deprecated. "
    "Import from gpt_trader.app.config.validation instead.",
    DeprecationWarning,
    stacklevel=2,
)
