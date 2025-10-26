"""Core validation rule abstractions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class BaseValidationRule:
    """Base helper to make rules callable."""

    def __call__(self, value: Any, field_name: str = "value") -> Any:
        return self.apply(value, field_name=field_name)

    def apply(self, value: Any, *, field_name: str = "value") -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


class RuleError(ValueError):
    """ValueError subclass that carries the offending value."""

    def __init__(self, message: str, *, value: Any | None = None) -> None:
        super().__init__(message)
        self.value = value


class FunctionRule(BaseValidationRule):
    """Adapter that converts a callable into a rule."""

    def __init__(self, func: Callable[[Any, str], Any]) -> None:
        self._func = func

    def apply(self, value: Any, *, field_name: str = "value") -> Any:
        return self._func(value, field_name)


def normalize_rule(rule: BaseValidationRule | Callable[[Any, str], Any]) -> BaseValidationRule:
    """Normalize arbitrary callables into rule instances."""
    if isinstance(rule, BaseValidationRule):
        return rule
    if callable(rule):
        return FunctionRule(rule)
    raise TypeError(f"Unsupported rule type: {type(rule)!r}")


class RuleChain(BaseValidationRule):
    """Compose multiple rules executed sequentially."""

    def __init__(self, *rules: BaseValidationRule | Callable[[Any, str], Any]) -> None:
        if not rules:
            raise RuleError("RuleChain requires at least one rule")
        self._rules = tuple(normalize_rule(rule) for rule in rules)

    def apply(self, value: Any, *, field_name: str = "value") -> Any:
        for rule in self._rules:
            value = rule(value, field_name)
        return value


__all__ = [
    "BaseValidationRule",
    "FunctionRule",
    "RuleChain",
    "RuleError",
    "normalize_rule",
]
