"""Symbol validation for trading symbols."""

from gpt_trader.validation import RuleError, SymbolRule

from .input_sanitizer import ValidationResult


class SymbolValidator:
    """Validate trading symbols."""

    _SYMBOL_RULE = SymbolRule()

    @classmethod
    def validate_symbol(cls, symbol: str) -> ValidationResult:
        """Validate trading symbol"""
        errors = []

        try:
            normalised = cls._SYMBOL_RULE(symbol, "symbol")
        except RuleError:
            errors.append("Invalid symbol format")
            normalised = None
        except Exception:
            errors.append("Invalid symbol format")
            normalised = None

        # Check against blocklist (simplified)
        blocked_symbols = {"TEST", "DEBUG", "HACK"}
        candidate = normalised or (symbol.upper() if isinstance(symbol, str) else "")
        if candidate in blocked_symbols:
            errors.append("Symbol is blocked")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_value=candidate if not errors else None,
        )
