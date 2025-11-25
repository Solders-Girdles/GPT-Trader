"""General request validation."""

from typing import Any

from .input_sanitizer import InputSanitizer, ValidationResult


class RequestValidator:
    """Validate general request structure and content."""

    @classmethod
    def validate_request(cls, request: dict[str, Any]) -> ValidationResult:
        """
        Comprehensive request validation.

        Args:
            request: Request data

        Returns:
            ValidationResult
        """
        errors = []

        # Check required fields
        required_fields = ["action", "timestamp"]
        for field in required_fields:
            if field not in request:
                errors.append(f"Missing required field: {field}")

        # Validate each field type
        if "action" in request:
            action_result = InputSanitizer.sanitize_string(request["action"], max_length=50)
            if not action_result.is_valid:
                errors.extend(action_result.errors)

        # Check request size using JSON serialization for accurate payload size
        import json

        try:
            request_size = len(json.dumps(request, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            request_size = 0  # If serialization fails, skip size check

        if request_size > 1048576:  # 1MB
            errors.append("Request size exceeds 1MB limit")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_value=request if not errors else None,
        )
