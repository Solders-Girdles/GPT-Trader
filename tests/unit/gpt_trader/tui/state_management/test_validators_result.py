"""Tests for ValidationResult."""

from __future__ import annotations

from gpt_trader.tui.state_management.validators import ValidationResult


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_initial_state_is_valid(self):
        """Test new ValidationResult is valid with empty lists."""
        result = ValidationResult()

        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_sets_invalid(self):
        """Test adding error marks result as invalid."""
        result = ValidationResult()
        result.add_error("field", "error message", "value")

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "field"
        assert result.errors[0].message == "error message"
        assert result.errors[0].severity == "error"

    def test_add_warning_keeps_valid(self):
        """Test adding warning does not invalidate result."""
        result = ValidationResult()
        result.add_warning("field", "warning message")

        assert result.valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].severity == "warning"

    def test_merge_combines_results(self):
        """Test merge combines errors and warnings from both results."""
        result1 = ValidationResult()
        result1.add_error("field1", "error1")

        result2 = ValidationResult()
        result2.add_warning("field2", "warning2")

        result1.merge(result2)

        assert result1.valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1

    def test_merge_propagates_invalid_state(self):
        """Test merging invalid result makes target invalid."""
        result1 = ValidationResult()  # valid
        result2 = ValidationResult()
        result2.add_error("field", "error")  # invalid

        result1.merge(result2)

        assert result1.valid is False
