"""
Backward compatibility re-export for RiskConfig.

The canonical location is now:
    gpt_trader.features.live_trade.risk.config

This module re-exports RiskConfig for backward compatibility with existing
imports from orchestration.configuration.risk.model.

.. deprecated::
    Import from gpt_trader.features.live_trade.risk.config instead.

    Removal target: v3.0
    Tracker: docs/DEPRECATIONS.md
"""

from __future__ import annotations

import warnings

# Single-shot deprecation warning
_deprecation_warned = False


def _emit_deprecation_warning() -> None:
    """Emit deprecation warning once per process."""
    global _deprecation_warned
    if not _deprecation_warned:
        warnings.warn(
            "Importing RiskConfig from "
            "gpt_trader.orchestration.configuration.risk.model is deprecated. "
            "Use gpt_trader.features.live_trade.risk.config instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        _deprecation_warned = True


_emit_deprecation_warning()

# Re-export from canonical location
from gpt_trader.features.live_trade.risk.config import (  # noqa: E402
    RISK_CONFIG_ENV_ALIASES,
    RISK_CONFIG_ENV_KEYS,
    RiskConfig,
)

__all__ = ["RiskConfig", "RISK_CONFIG_ENV_KEYS", "RISK_CONFIG_ENV_ALIASES"]
