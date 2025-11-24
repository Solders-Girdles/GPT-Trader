"""Helpers for lazily importing heavy dependencies."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="utilities")

T = TypeVar("T")


class LazyImport:
    """Lazy import wrapper that defers import until first access."""

    def __init__(self, module_path: str, attribute: str | None = None) -> None:
        self.module_path = module_path
        self.attribute = attribute
        self._module: Any = None
        self._loaded = False

    def _load(self) -> Any:
        """Load the module or attribute on first access."""
        if not self._loaded:
            start_time = time.time()
            module = importlib.import_module(self.module_path)

            if self.attribute:
                module = getattr(module, self.attribute)

            self._module = module
            self._loaded = True

            load_time = time.time() - start_time
            if load_time > 0.1:
                logging.getLogger(__name__).debug(
                    "Slow import: %s took %.3fs", self.module_path, load_time
                )
                logger.debug("Slow import: %s took %.3fs", self.module_path, load_time)
        return self._module

    def __getattr__(self, name: str) -> Any:
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        module = self._load()
        return module(*args, **kwargs)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "lazy"
        attr_part = f".{self.attribute}" if self.attribute else ""
        return f"<LazyImport {self.module_path}{attr_part} [{status}]>"


def lazy_import(module_path: str, attribute: str | None = None) -> LazyImport:
    """Create a lazy import wrapper."""
    return LazyImport(module_path, attribute)


def with_lazy_imports(**imports: LazyImport) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that injects resolved lazy imports into keyword arguments."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            resolved = {name: lazy_obj._load() for name, lazy_obj in imports.items()}
            merged_kwargs = {**kwargs, **resolved}
            return func(*args, **merged_kwargs)

        return wrapper

    return decorator


__all__ = ["LazyImport", "lazy_import", "with_lazy_imports"]
