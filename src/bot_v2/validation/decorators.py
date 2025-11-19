"""
Validation decorators for function input validation.
"""

from collections.abc import Callable
from typing import Any

from .base_validators import Validator


def validate_inputs(**validators: Validator | Callable[[Any, str], Any]) -> Callable:
    """Decorator to validate function inputs"""

    normalized: dict[str, Validator] = {}
    for name, validator in validators.items():
        if isinstance(validator, Validator):
            normalized[name] = validator
        else:

            def _predicate(
                value: Any, field_name: str = name, func: Callable[[Any, str], Any] = validator
            ) -> tuple[bool, Any]:
                return True, func(value, field_name)

            normalized[name] = Validator(predicate=_predicate)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each parameter
            for param_name, validator in normalized.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    bound.arguments[param_name] = validator(value, param_name)

            # Call function with validated arguments
            return func(**bound.arguments)

        return wrapper

    return decorator
