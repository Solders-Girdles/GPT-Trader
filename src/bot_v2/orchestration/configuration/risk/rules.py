"""Validation rule helpers for risk configuration."""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticCustomError

from bot_v2.validation import (
    BaseValidationRule,
    BooleanRule,
    DecimalRule,
    FloatRule,
    IntegerRule,
    MappingRule,
    PercentageRule,
    RuleError,
    StripStringRule,
    TimeOfDayRule,
)


def apply_rule(
    rule: BaseValidationRule,
    value: Any,
    *,
    field_label: str,
    error_code: str,
    error_template: str,
) -> Any:
    """Apply a validation rule with proper error handling."""
    try:
        return rule(value, field_label)
    except RuleError as exc:
        if isinstance(exc, PydanticCustomError):
            raise
        raise PydanticCustomError(
            error_code,
            error_template,
            {"value": value, "error": str(exc)},
        ) from exc
    except PydanticCustomError:
        raise


INT_RULE = IntegerRule()
DECIMAL_RULE = DecimalRule()
FLOAT_RULE = FloatRule()
STRING_RULE = StripStringRule()
BOOL_RULE = BooleanRule()
MAPPING_RULE = MappingRule()
PCT_RULE = PercentageRule()
TIME_RULE = TimeOfDayRule()

__all__ = [
    "apply_rule",
    "INT_RULE",
    "DECIMAL_RULE",
    "FLOAT_RULE",
    "STRING_RULE",
    "BOOL_RULE",
    "MAPPING_RULE",
    "PCT_RULE",
    "TIME_RULE",
]
