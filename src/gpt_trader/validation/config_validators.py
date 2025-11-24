"""
Configuration validators for validating config dictionaries.
"""

from typing import Any

from gpt_trader.errors import ValidationError

from .base_validators import Validator


def _get_logger():
    """Lazy load logger to avoid circular imports"""
    from gpt_trader.utilities.logging_patterns import get_logger

    return get_logger(__name__, component="validation")


def validate_config(config: dict[str, Any], schema: dict[str, Validator]) -> dict[str, Any]:
    """Validate configuration dictionary against schema"""
    validated = {}

    for key, validator in schema.items():
        if key not in config:
            raise ValidationError(f"Missing required config key: {key}", field=key)

        validated[key] = validator(config[key], key)

    # Check for extra keys
    extra_keys = set(config.keys()) - set(schema.keys())
    if extra_keys:
        _get_logger().warning(f"Unknown config keys will be ignored: {extra_keys}")

    return validated
