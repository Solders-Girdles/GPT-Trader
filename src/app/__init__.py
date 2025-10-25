"""Legacy compatibility package for ``app`` namespace."""

import sys

from bot_v2.app.container import (  # noqa: F401
    ApplicationContainer,
    create_application_container,
)

sys.modules[__name__ + ".container"] = sys.modules["bot_v2.app.container"]

__all__ = ["ApplicationContainer", "create_application_container"]
