"""
Pattern-based validators for domain-specific validation (symbols, strategies, etc).
"""

import re
from typing import Any

from gpt_trader.errors import ValidationError

from .base_validators import Validator


class SymbolValidator(Validator):
    """Validate trading symbol (equities or crypto).

    Accepts common formats such as:
    - Equities: "AAPL", "MSFT"
    - Crypto spot: "BTC-USD", "ETH-USD"
    - Perpetuals: "BTC-PERP", "ETH-PERP"
    """

    def validate(self, value: Any, field_name: str = "symbol") -> str:
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name, value=value)

        # Normalize to uppercase
        value = value.upper()

        # Allow simple tickers or hyphenated pairs/suffixes (e.g., BTC-USD, BTC-PERP)
        if not re.match(r"^[A-Z0-9]{1,10}(-[A-Z0-9]{2,10})?$", value):
            raise ValidationError(
                f"{field_name} must be a valid symbol (e.g., AAPL, BTC-USD, BTC-PERP)",
                field=field_name,
                value=value,
            )

        return value


class StrategyNameValidator(Validator):
    """Validate strategy name identifier.

    Accepts names like 'SimpleMAStrategy', 'Momentum', 'mean_reversion'.
    Must start with a letter; allows letters, numbers, underscores and dashes.
    """

    def validate(self, value: Any, field_name: str = "strategy") -> str:
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name, value=value)
        if not re.match(r"^[A-Za-z][A-Za-z0-9_\-]*$", value):
            raise ValidationError(
                f"{field_name} must start with a letter and contain only letters, numbers, '_' or '-'",
                field=field_name,
                value=value,
            )
        return value
