"""Input sanitization for preventing injection attacks."""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    """Result of validation check"""

    is_valid: bool
    errors: list[str]
    sanitized_value: Any = None


class InputSanitizer:
    """Sanitize string inputs to prevent injection attacks."""

    # Regex patterns for validation
    PATTERNS = {
        "sql_injection": r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE|EVAL|OR)\b)|((?<![</])\b(SCRIPT|JAVASCRIPT)\b)",
        "xss_tags": r"<[^>]*>",
        "path_traversal": r"(\.\.[\\/])|(%2[fF])|(%5[cC])|(%2[eE]%2[eE])|(file://)",
    }

    @classmethod
    def sanitize_string(cls, input_str: str, max_length: int = 255) -> ValidationResult:
        """
        Sanitize string input and check for injection attempts.

        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length

        Returns:
            ValidationResult with sanitized value
        """
        errors = []

        if not input_str:
            return ValidationResult(False, ["Input cannot be empty"])

        # Check length
        if len(input_str) > max_length:
            errors.append(f"Input exceeds maximum length of {max_length}")
            input_str = input_str[:max_length]

        # Check for path traversal
        if re.search(cls.PATTERNS["path_traversal"], input_str, re.IGNORECASE):
            errors.append("Path traversal attempt detected")
            return ValidationResult(False, errors)

        # Check for SQL injection patterns
        if re.search(cls.PATTERNS["sql_injection"], input_str, re.IGNORECASE):
            errors.append("Potential SQL injection detected")
            return ValidationResult(False, errors)

        # Check for XSS attempts
        if re.search(cls.PATTERNS["xss_tags"], input_str):
            errors.append("HTML tags not allowed")
            input_str = re.sub(cls.PATTERNS["xss_tags"], "", input_str)

        # Escape special characters
        sanitized = input_str.replace("'", "''").replace('"', '""').strip()

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, sanitized_value=sanitized)
