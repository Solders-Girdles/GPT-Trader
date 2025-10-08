"""Configuration parsing utilities for consistent environment variable handling."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Callable, TypeVar

from bot_v2.errors import ValidationError

T = TypeVar("T")


def parse_mapping_env(
    var_name: str,
    cast: Callable[[str], T],
    default: dict[str, T] | None = None,
    separator: str = ":",
) -> dict[str, T]:
    """Parse a colon-separated environment variable into a mapping.
    
    Args:
        var_name: Environment variable name
        cast: Function to cast individual values
        default: Default mapping if variable not found
        separator: Separator character (default: ':')
        
    Returns:
        Dictionary mapping keys to cast values
        
    Raises:
        ValidationError: If parsing fails
    """
    env_value = os.getenv(var_name)
    if not env_value:
        return default or {}
    
    try:
        result = {}
        for pair in env_value.split(separator):
            if not pair.strip():
                continue
            if "=" not in pair:
                raise ValueError(f"Invalid key=value pair: {pair}")
            key, value = pair.split("=", 1)
            result[key.strip()] = cast(value.strip())
        return result
    except Exception as exc:
        raise ValidationError(
            f"Failed to parse {var_name}: {exc}",
            field=var_name,
            value=env_value
        ) from exc


def parse_list_env(
    var_name: str,
    cast: Callable[[str], T],
    default: list[T] | None = None,
    separator: str = ",",
) -> list[T]:
    """Parse a comma-separated environment variable into a list.
    
    Args:
        var_name: Environment variable name
        cast: Function to cast individual values
        default: Default list if variable not found
        separator: Separator character (default: ',')
        
    Returns:
        List of cast values
        
    Raises:
        ValidationError: If parsing fails
    """
    env_value = os.getenv(var_name)
    if not env_value:
        return default or []
    
    try:
        result = []
        for item in env_value.split(separator):
            item = item.strip()
            if item:
                result.append(cast(item))
        return result
    except Exception as exc:
        raise ValidationError(
            f"Failed to parse {var_name}: {exc}",
            field=var_name,
            value=env_value
        ) from exc


def parse_bool_env(var_name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable.
    
    Args:
        var_name: Environment variable name
        default: Default value if variable not found
        
    Returns:
        Boolean value
    """
    env_value = os.getenv(var_name)
    if not env_value:
        return default
    
    return env_value.strip().lower() in ("1", "true", "yes", "on", "enabled")


def parse_decimal_env(var_name: str, default: Decimal | None = None) -> Decimal | None:
    """Parse a decimal environment variable.
    
    Args:
        var_name: Environment variable name
        default: Default value if variable not found
        
    Returns:
        Decimal value or None
        
    Raises:
        ValidationError: If parsing fails
    """
    env_value = os.getenv(var_name)
    if not env_value:
        return default
    
    try:
        return Decimal(env_value.strip())
    except Exception as exc:
        raise ValidationError(
            f"Failed to parse {var_name} as decimal: {exc}",
            field=var_name,
            value=env_value
        ) from exc


def parse_int_env(var_name: str, default: int | None = None) -> int | None:
    """Parse an integer environment variable.
    
    Args:
        var_name: Environment variable name
        default: Default value if variable not found
        
    Returns:
        Integer value or None
        
    Raises:
        ValidationError: If parsing fails
    """
    env_value = os.getenv(var_name)
    if not env_value:
        return default
    
    try:
        return int(env_value.strip())
    except Exception as exc:
        raise ValidationError(
            f"Failed to parse {var_name} as integer: {exc}",
            field=var_name,
            value=env_value
        ) from exc


def parse_float_env(var_name: str, default: float | None = None) -> float | None:
    """Parse a float environment variable.
    
    Args:
        var_name: Environment variable name
        default: Default value if variable not found
        
    Returns:
        Float value or None
        
    Raises:
        ValidationError: If parsing fails
    """
    env_value = os.getenv(var_name)
    if not env_value:
        return default
    
    try:
        return float(env_value.strip())
    except Exception as exc:
        raise ValidationError(
            f"Failed to parse {var_name} as float: {exc}",
            field=var_name,
            value=env_value
        ) from exc


def validate_required_env(var_names: list[str]) -> None:
    """Validate that required environment variables are present.
    
    Args:
        var_names: List of required environment variable names
        
    Raises:
        ValidationError: If any required variable is missing
    """
    missing = [name for name in var_names if not os.getenv(name)]
    if missing:
        raise ValidationError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
