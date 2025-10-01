"""Deprecated compatibility facade for the legacy live trade module.

The implementation now lives in ``archived.legacy_live_trade_facade.live_trade``.
This shim exists so existing imports continue to work while downstream code
migrates to the orchestration stack.
"""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType
from typing import Any, Iterable

_DELEGATE_MODULE = "archived.legacy_live_trade_facade.live_trade"

_delegate: ModuleType = importlib.import_module(_DELEGATE_MODULE)

__doc__ = _delegate.__doc__

_exposed: Iterable[str] | None = getattr(_delegate, "__all__", None)
if _exposed is None:
    _exposed = [name for name in vars(_delegate) if not name.startswith("_")]

__all__ = tuple(_exposed)

globals().update({name: getattr(_delegate, name) for name in __all__})

warnings.warn(
    "bot_v2.features.live_trade.live_trade has moved to archived.legacy_live_trade_"
    "facade.live_trade and will be removed in a future release. Migrate to the"
    " orchestration entrypoints instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough helper
    return getattr(_delegate, name)


def __dir__() -> list[str]:  # pragma: no cover - passthrough helper
    return sorted(set(__all__) | set(dir(_delegate)))
