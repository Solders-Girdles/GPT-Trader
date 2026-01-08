"""
DEPRECATED: This module has moved to gpt_trader.app.config.controller

This shim exists for backwards compatibility. Update imports to use:
    from gpt_trader.app.config.controller import ConfigController
    # or
    from gpt_trader.app.config import ConfigController
"""

from __future__ import annotations

import warnings

from gpt_trader.app.config.controller import ConfigController

__all__ = ["ConfigController"]

warnings.warn(
    "gpt_trader.orchestration.config_controller is deprecated. "
    "Import from gpt_trader.app.config.controller instead.",
    DeprecationWarning,
    stacklevel=2,
)
