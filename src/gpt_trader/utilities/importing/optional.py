"""Helpers for optional imports that degrade gracefully."""

from __future__ import annotations

import importlib
from typing import Any

from .lazy import LazyImport, lazy_import


class OptionalImport:
    """Optional import that handles missing dependencies gracefully."""

    def __init__(self, module_path: str, attribute: str | None = None) -> None:
        self.module_path = module_path
        self.attribute = attribute
        self._module: Any = None
        self._available = False
        self._attempted = False

    def _try_load(self) -> Any:
        if not self._attempted:
            try:
                module = importlib.import_module(self.module_path)
                if self.attribute:
                    module = getattr(module, self.attribute)
                self._module = module
                self._available = True
            except ImportError:
                self._available = False
            self._attempted = True
        return self._module if self._available else None

    def is_available(self) -> bool:
        self._try_load()
        return self._available

    def get(self, default: Any = None) -> Any:
        result = self._try_load()
        return result if result is not None else default

    def require(self, error_message: str | None = None) -> Any:
        if not self.is_available():
            msg = error_message or f"Optional dependency {self.module_path} is required but missing"
            raise ImportError(msg)
        return self._module  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        module = self._try_load()
        if module is None:
            raise AttributeError(f"Optional import {self.module_path} is not available")
        return getattr(module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        module = self.require()
        return module(*args, **kwargs)

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "unavailable"
        attr_part = f".{self.attribute}" if self.attribute else ""
        return f"<OptionalImport {self.module_path}{attr_part} [{status}]>"


def optional_import(module_path: str, attribute: str | None = None) -> OptionalImport:
    """Create an optional import wrapper."""
    return OptionalImport(module_path, attribute)


def conditional_import(
    condition: bool,
    module_path: str,
    attribute: str | None = None,
) -> LazyImport | OptionalImport:
    """Create a lazy import when the condition is truthy, otherwise optional."""
    return (
        lazy_import(module_path, attribute)
        if condition
        else optional_import(module_path, attribute)
    )


__all__ = ["OptionalImport", "optional_import", "conditional_import"]
