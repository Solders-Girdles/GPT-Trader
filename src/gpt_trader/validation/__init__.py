"""
Comprehensive validation framework for GPT-Trader.

Provides input validation, data validation, and configuration validation
with detailed error messages and type checking.

This module acts as a facade, re-exporting all validators from specialized modules.
"""

# Import all validators from specialized modules
from .base_validators import Validator
from .composite_validators import CompositeValidator
from .config_validators import validate_config
from .data_validators import DataFrameValidator, OHLCDataValidator, SeriesValidator
from .decorators import validate_inputs
from .numeric_validators import PercentageValidator, PositiveNumberValidator
from .pattern_validators import StrategyNameValidator, SymbolValidator
from .rules import (
    BaseValidationRule,
    BooleanRule,
    DecimalRule,
    FloatRule,
    IntegerRule,
    ListRule,
    MappingRule,
    PercentageRule,
    RuleChain,
    RuleError,
    StripStringRule,
    SymbolRule,
    TimeOfDayRule,
)
from .temporal_validators import DateValidator
from .type_validators import ChoiceValidator, RangeValidator, RegexValidator, TypeValidator

# Export main components
__all__ = [
    # Base
    "Validator",
    # Type validators
    "TypeValidator",
    "RangeValidator",
    "ChoiceValidator",
    "RegexValidator",
    # Pattern validators
    "SymbolValidator",
    "StrategyNameValidator",
    # Numeric validators
    "PositiveNumberValidator",
    "PercentageValidator",
    # Temporal validators
    "DateValidator",
    # Data validators
    "DataFrameValidator",
    "SeriesValidator",
    "OHLCDataValidator",
    # Composite
    "CompositeValidator",
    # Functions
    "validate_inputs",
    "validate_config",
    # Rules (from rules module)
    "BaseValidationRule",
    "BooleanRule",
    "DecimalRule",
    "FloatRule",
    "IntegerRule",
    "ListRule",
    "MappingRule",
    "PercentageRule",
    "RuleChain",
    "RuleError",
    "StripStringRule",
    "SymbolRule",
    "TimeOfDayRule",
]
