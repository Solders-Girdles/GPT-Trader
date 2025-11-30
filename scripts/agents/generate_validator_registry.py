#!/usr/bin/env python3
"""Generate a validator registry for AI agent consumption.

This script documents all available validators in the validation framework,
including:
- Validator types and their purposes
- Input/output types
- Example usage
- Validation rules

Usage:
    python scripts/agents/generate_validator_registry.py [--output-dir DIR]

Output:
    var/agents/validation/
    - validator_registry.json (all validators with documentation)
    - rules_registry.json (validation rules documentation)
    - index.json (discovery file)
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def get_validator_info(cls: type) -> dict[str, Any]:
    """Extract information about a validator class."""
    info: dict[str, Any] = {
        "name": cls.__name__,
        "module": cls.__module__,
        "docstring": inspect.getdoc(cls) or "",
    }

    # Get __init__ signature
    try:
        sig = inspect.signature(cls.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            param_info: dict[str, Any] = {
                "required": param.default is inspect.Parameter.empty,
            }
            if param.default is not inspect.Parameter.empty:
                # Handle default values
                default = param.default
                if default is None or isinstance(default, (str, int, float, bool)):
                    param_info["default"] = default
                else:
                    param_info["default"] = str(default)
            if param.annotation is not inspect.Parameter.empty:
                param_info["type"] = str(param.annotation)
            params[name] = param_info
        info["parameters"] = params
    except (ValueError, TypeError):
        info["parameters"] = {}

    # Get validate method signature
    if hasattr(cls, "validate"):
        try:
            validate_sig = inspect.signature(cls.validate)
            info["validate_params"] = [p for p in validate_sig.parameters.keys() if p != "self"]
        except (ValueError, TypeError):
            pass

    return info


def generate_validator_registry() -> dict[str, Any]:
    """Generate documentation for all validators."""
    from gpt_trader.validation import (
        ChoiceValidator,
        CompositeValidator,
        DataFrameValidator,
        DateValidator,
        OHLCDataValidator,
        PercentageValidator,
        PositiveNumberValidator,
        RangeValidator,
        RegexValidator,
        SeriesValidator,
        StrategyNameValidator,
        SymbolValidator,
        TypeValidator,
        Validator,
    )

    validators = {
        "base": {
            "Validator": {
                **get_validator_info(Validator),
                "description": "Base validator class with optional predicate support",
                "usage": "Validator(predicate=lambda x: x > 0, error_message='Must be positive')",
            },
        },
        "type": {
            "TypeValidator": {
                **get_validator_info(TypeValidator),
                "description": "Validates value is of expected type",
                "usage": "TypeValidator(expected_type=int)",
            },
            "RangeValidator": {
                **get_validator_info(RangeValidator),
                "description": "Validates numeric value is within range",
                "usage": "RangeValidator(min_value=0, max_value=100)",
            },
            "ChoiceValidator": {
                **get_validator_info(ChoiceValidator),
                "description": "Validates value is one of allowed choices",
                "usage": "ChoiceValidator(choices=['BUY', 'SELL'])",
            },
            "RegexValidator": {
                **get_validator_info(RegexValidator),
                "description": "Validates string matches regex pattern",
                "usage": "RegexValidator(pattern=r'^[A-Z]+-[A-Z]+$')",
            },
        },
        "numeric": {
            "PositiveNumberValidator": {
                **get_validator_info(PositiveNumberValidator),
                "description": "Validates value is a positive number",
                "usage": "PositiveNumberValidator()",
            },
            "PercentageValidator": {
                **get_validator_info(PercentageValidator),
                "description": "Validates value is a valid percentage (0-1 or 0-100)",
                "usage": "PercentageValidator(as_decimal=True)",
            },
        },
        "pattern": {
            "SymbolValidator": {
                **get_validator_info(SymbolValidator),
                "description": "Validates trading symbol format (e.g., BTC-USD)",
                "usage": "SymbolValidator()",
            },
            "StrategyNameValidator": {
                **get_validator_info(StrategyNameValidator),
                "description": "Validates strategy name format",
                "usage": "StrategyNameValidator()",
            },
        },
        "temporal": {
            "DateValidator": {
                **get_validator_info(DateValidator),
                "description": "Validates and parses date strings",
                "usage": "DateValidator(format='%Y-%m-%d')",
            },
        },
        "data": {
            "DataFrameValidator": {
                **get_validator_info(DataFrameValidator),
                "description": "Validates pandas DataFrame structure",
                "usage": "DataFrameValidator(required_columns=['open', 'high', 'low', 'close'])",
            },
            "SeriesValidator": {
                **get_validator_info(SeriesValidator),
                "description": "Validates pandas Series",
                "usage": "SeriesValidator()",
            },
            "OHLCDataValidator": {
                **get_validator_info(OHLCDataValidator),
                "description": "Validates OHLC candlestick data integrity",
                "usage": "OHLCDataValidator()",
            },
        },
        "composite": {
            "CompositeValidator": {
                **get_validator_info(CompositeValidator),
                "description": "Chains multiple validators together",
                "usage": "CompositeValidator([TypeValidator(int), RangeValidator(0, 100)])",
            },
        },
    }

    return {
        "version": "1.0",
        "description": "Validator registry for GPT-Trader validation framework",
        "validators": validators,
    }


def generate_rules_registry() -> dict[str, Any]:
    """Generate documentation for validation rules."""
    from gpt_trader.validation.rules import (
        BaseValidationRule,
        BooleanRule,
        DecimalRule,
        FloatRule,
        IntegerRule,
        ListRule,
        MappingRule,
        PercentageRule,
        RuleChain,
        StripStringRule,
        SymbolRule,
        TimeOfDayRule,
    )

    rules = {
        "base": {
            "BaseValidationRule": {
                **get_validator_info(BaseValidationRule),
                "description": "Base class for validation rules",
            },
            "RuleChain": {
                **get_validator_info(RuleChain),
                "description": "Chains multiple rules for sequential validation",
                "usage": "RuleChain([StripStringRule(), SymbolRule()])",
            },
        },
        "type_rules": {
            "BooleanRule": {
                **get_validator_info(BooleanRule),
                "description": "Validates and coerces to boolean",
                "accepts": ["true", "false", "yes", "no", "1", "0", bool],
                "usage": "BooleanRule()",
            },
            "IntegerRule": {
                **get_validator_info(IntegerRule),
                "description": "Validates and coerces to integer",
                "usage": "IntegerRule(min_value=0, max_value=100)",
            },
            "FloatRule": {
                **get_validator_info(FloatRule),
                "description": "Validates and coerces to float",
                "usage": "FloatRule(min_value=0.0)",
            },
            "DecimalRule": {
                **get_validator_info(DecimalRule),
                "description": "Validates and coerces to Decimal for precision",
                "usage": "DecimalRule(min_value=Decimal('0'), precision=8)",
            },
        },
        "string_rules": {
            "StripStringRule": {
                **get_validator_info(StripStringRule),
                "description": "Strips whitespace from strings",
                "usage": "StripStringRule()",
            },
        },
        "domain_rules": {
            "SymbolRule": {
                **get_validator_info(SymbolRule),
                "description": "Validates trading symbol format",
                "pattern": "BASE-QUOTE (e.g., BTC-USD)",
                "usage": "SymbolRule()",
            },
            "PercentageRule": {
                **get_validator_info(PercentageRule),
                "description": "Validates percentage values",
                "accepts": "0-1 as decimal or 0-100 as percentage",
                "usage": "PercentageRule(as_decimal=True)",
            },
            "TimeOfDayRule": {
                **get_validator_info(TimeOfDayRule),
                "description": "Validates time of day format",
                "format": "HH:MM",
                "usage": "TimeOfDayRule()",
            },
        },
        "collection_rules": {
            "ListRule": {
                **get_validator_info(ListRule),
                "description": "Validates list with optional item rule",
                "usage": "ListRule(item_rule=SymbolRule(), min_length=1)",
            },
            "MappingRule": {
                **get_validator_info(MappingRule),
                "description": "Validates dictionary/mapping with key/value rules",
                "usage": "MappingRule(key_rule=StripStringRule(), value_rule=DecimalRule())",
            },
        },
    }

    return {
        "version": "1.0",
        "description": "Validation rules registry for GPT-Trader",
        "rules": rules,
        "error_handling": {
            "exception": "RuleError",
            "fields": ["message", "field", "value", "rule_name"],
        },
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate validator registry for AI agents")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("var/agents/validation"),
        help="Output directory for registry files",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output to stdout instead of files",
    )

    args = parser.parse_args()

    print("Generating validator registry...")
    validator_registry = generate_validator_registry()
    rules_registry = generate_rules_registry()

    if args.stdout:
        output = {
            "validators": validator_registry,
            "rules": rules_registry,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write validator registry
    validators_path = output_dir / "validator_registry.json"
    with open(validators_path, "w") as f:
        json.dump(validator_registry, f, indent=2, default=str)
    print(f"Validator registry written to: {validators_path}")

    # Write rules registry
    rules_path = output_dir / "rules_registry.json"
    with open(rules_path, "w") as f:
        json.dump(rules_registry, f, indent=2, default=str)
    print(f"Rules registry written to: {rules_path}")

    # Write index
    total_validators = sum(len(cat) for cat in validator_registry["validators"].values())
    total_rules = sum(len(cat) for cat in rules_registry["rules"].values())

    index = {
        "version": "1.0",
        "description": "Validation framework registry for AI agent consumption",
        "files": {
            "validator_registry": "validator_registry.json",
            "rules_registry": "rules_registry.json",
        },
        "summary": {
            "total_validators": total_validators,
            "total_rules": total_rules,
            "categories": {
                "validators": list(validator_registry["validators"].keys()),
                "rules": list(rules_registry["rules"].keys()),
            },
        },
        "usage": {
            "import": "from gpt_trader.validation import SymbolValidator, RangeValidator",
            "basic": "validator = SymbolValidator(); result = validator.validate('BTC-USD')",
            "composite": "chain = CompositeValidator([TypeValidator(str), SymbolValidator()])",
            "decorator": "@validate_inputs(symbol=SymbolValidator(), quantity=PositiveNumberValidator())",
        },
    }
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index written to: {index_path}")

    print(f"\nDocumented {total_validators} validators and {total_rules} rules")

    return 0


if __name__ == "__main__":
    sys.exit(main())
