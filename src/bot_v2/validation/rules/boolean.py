"""Boolean parsing rule."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bot_v2.utilities.parsing import (
    FALSE_BOOLEAN_TOKENS,
    TRUE_BOOLEAN_TOKENS,
    interpret_tristate_bool,
)

from .base import BaseValidationRule, RuleError


class BooleanRule(BaseValidationRule):
    """Parse boolean-like values using common token conventions."""

    def __init__(
        self,
        *,
        default: bool | None = None,
        true_tokens: Iterable[str] | None = None,
        false_tokens: Iterable[str] | None = None,
    ) -> None:
        self._default = default
        self._true_tokens = {token.lower() for token in TRUE_BOOLEAN_TOKENS}
        self._false_tokens = {token.lower() for token in FALSE_BOOLEAN_TOKENS}
        if true_tokens:
            self._true_tokens.update(token.lower() for token in true_tokens)
        if false_tokens:
            self._false_tokens.update(token.lower() for token in false_tokens)
        self._token_display = ", ".join(sorted(self._true_tokens | self._false_tokens))

    def apply(self, value: Any, *, field_name: str = "value") -> bool:
        if value is None:
            if self._default is not None:
                return self._default
            raise RuleError(f"{field_name} expected a boolean but received None", value=value)

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(int(value))

        if isinstance(value, str):
            candidate = interpret_tristate_bool(value.strip())
            if candidate is not None:
                return candidate
            lowered = value.strip().lower()
            if lowered in self._true_tokens:
                return True
            if lowered in self._false_tokens:
                return False

        raise RuleError(
            f"{field_name} expected a boolean value ({self._token_display}) but received {value!r}",
            value=value,
        )


__all__ = ["BooleanRule"]
