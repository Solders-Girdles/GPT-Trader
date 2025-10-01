"""Compatibility shim for the legacy ExecutionEngine."""

from __future__ import annotations

import importlib
import warnings
from types import ModuleType
from typing import Any
from collections.abc import Iterable

_DELEGATE_MODULE = "archived.legacy_live_trade_facade.execution"

_delegate: ModuleType = importlib.import_module(_DELEGATE_MODULE)

__doc__ = _delegate.__doc__

_exposed: Iterable[str] | None = getattr(_delegate, "__all__", None)
if _exposed is None:
    _exposed = [name for name in vars(_delegate) if not name.startswith("_")]

__all__ = tuple(_exposed)

globals().update({name: getattr(_delegate, name) for name in __all__})

warnings.warn(
    "bot_v2.features.live_trade.execution is deprecated."
    " Use bot_v2.orchestration.live_execution.LiveExecutionEngine instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough helper
    return getattr(_delegate, name)


def __dir__() -> list[str]:  # pragma: no cover - passthrough helper
    return sorted(set(__all__) | set(dir(_delegate)))
