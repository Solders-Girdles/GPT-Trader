"""Modular validation rule collection."""

from __future__ import annotations

from .base import BaseValidationRule, FunctionRule, RuleChain, RuleError, normalize_rule
from .boolean import BooleanRule
from .collections import ListRule, MappingRule
from .numbers import DecimalRule, FloatRule, IntegerRule, PercentageRule
from .strings import StripStringRule, SymbolRule, TimeOfDayRule

__all__ = [
    "BaseValidationRule",
    "FunctionRule",
    "RuleChain",
    "RuleError",
    "normalize_rule",
    "BooleanRule",
    "MappingRule",
    "ListRule",
    "IntegerRule",
    "DecimalRule",
    "FloatRule",
    "PercentageRule",
    "StripStringRule",
    "SymbolRule",
    "TimeOfDayRule",
]
