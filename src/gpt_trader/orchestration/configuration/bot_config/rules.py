"""
DEPRECATED: This module has moved to gpt_trader.app.config.validation_rules

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.configuration.bot_config.rules import apply_rule

    # New (preferred)
    from gpt_trader.app.config.validation_rules import apply_rule
"""

from __future__ import annotations

import warnings

# Re-export all symbols from canonical location
from gpt_trader.app.config.validation_rules import (
    DECIMAL_RULE,
    FLOAT_RULE,
    INT_RULE,
    STRING_RULE,
    SYMBOL_LIST_RULE,
    SYMBOL_RULE,
    apply_rule,
    ensure_condition,
)

__all__ = [
    "apply_rule",
    "ensure_condition",
    "INT_RULE",
    "DECIMAL_RULE",
    "FLOAT_RULE",
    "STRING_RULE",
    "SYMBOL_RULE",
    "SYMBOL_LIST_RULE",
]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.configuration.bot_config.rules is deprecated. "
    "Import from gpt_trader.app.config.validation_rules instead.",
    DeprecationWarning,
    stacklevel=2,
)
