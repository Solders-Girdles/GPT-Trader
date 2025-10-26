"""Validation rule helpers shared across BotConfig components."""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticCustomError

from bot_v2.validation import (
    BaseValidationRule,
    DecimalRule,
    FloatRule,
    IntegerRule,
    ListRule,
    RuleError,
    StripStringRule,
    SymbolRule,
)

# Instantiate shared validation rules once to avoid repeated construction.
INT_RULE = IntegerRule()
DECIMAL_RULE = DecimalRule()
FLOAT_RULE = FloatRule()
STRING_RULE = StripStringRule()
SYMBOL_RULE = SymbolRule()
SYMBOL_LIST_RULE = ListRule(item_rule=SYMBOL_RULE, allow_blank_items=False)


def apply_rule(
    rule: BaseValidationRule,
    value: Any,
    *,
    field_label: str,
    error_code: str,
    error_template: str,
) -> Any:
    """Apply a configured validation rule and convert RuleError to Pydantic error."""
    try:
        return rule(value, field_label)
    except RuleError as exc:
        raise PydanticCustomError(
            error_code,
            error_template,
            {"value": value, "error": str(exc)},
        ) from exc


def ensure_condition(
    condition: bool,
    *,
    error_code: str,
    error_template: str,
    context: dict[str, Any],
) -> None:
    """Raise a Pydantic error when an invalid condition is met."""
    if condition:
        raise PydanticCustomError(error_code, error_template, context)


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
