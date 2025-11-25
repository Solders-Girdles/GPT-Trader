"""Tests for security/input_sanitizer.py."""

from __future__ import annotations

import pytest

from gpt_trader.security.input_sanitizer import InputSanitizer, ValidationResult


# ============================================================
# Test: ValidationResult dataclass
# ============================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_fields(self) -> None:
        """Test ValidationResult stores fields correctly."""
        result = ValidationResult(is_valid=True, errors=[], sanitized_value="test")

        assert result.is_valid is True
        assert result.errors == []
        assert result.sanitized_value == "test"

    def test_validation_result_default_sanitized_value(self) -> None:
        """Test ValidationResult default sanitized_value is None."""
        result = ValidationResult(is_valid=False, errors=["error"])

        assert result.sanitized_value is None


# ============================================================
# Test: sanitize_string - Success cases
# ============================================================


class TestSanitizeStringSuccess:
    """Tests for successful string sanitization."""

    def test_sanitize_string_valid_input(self) -> None:
        """Test valid string input passes sanitization."""
        result = InputSanitizer.sanitize_string("hello world")

        assert result.is_valid is True
        assert result.errors == []
        assert result.sanitized_value == "hello world"

    def test_sanitize_string_strips_whitespace(self) -> None:
        """Test that output is stripped of leading/trailing whitespace."""
        result = InputSanitizer.sanitize_string("  hello world  ")

        assert result.is_valid is True
        assert result.sanitized_value == "hello world"

    def test_sanitize_string_escapes_quotes(self) -> None:
        """Test that single and double quotes are escaped."""
        result = InputSanitizer.sanitize_string("it's a \"test\"")

        assert result.is_valid is True
        assert result.sanitized_value == "it''s a \"\"test\"\""


# ============================================================
# Test: sanitize_string - Empty/blank input
# ============================================================


class TestSanitizeStringEmptyBlank:
    """Tests for empty and blank input handling."""

    def test_sanitize_string_empty_string(self) -> None:
        """Test that empty string is rejected."""
        result = InputSanitizer.sanitize_string("")

        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()

    def test_sanitize_string_none_input(self) -> None:
        """Test that None input is rejected."""
        result = InputSanitizer.sanitize_string(None)  # type: ignore

        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()

    def test_sanitize_string_whitespace_only_default(self) -> None:
        """Test that whitespace-only strings pass by default."""
        result = InputSanitizer.sanitize_string("   ")

        # By default, whitespace-only is allowed (stripped to empty in output)
        assert result.is_valid is True
        assert result.sanitized_value == ""

    def test_sanitize_string_whitespace_only_reject_blank(self) -> None:
        """Test that whitespace-only strings are rejected when reject_blank=True."""
        result = InputSanitizer.sanitize_string("   ", reject_blank=True)

        assert result.is_valid is False
        assert "blank" in result.errors[0].lower() or "whitespace" in result.errors[0].lower()

    def test_sanitize_string_tabs_newlines_reject_blank(self) -> None:
        """Test that tabs and newlines are rejected when reject_blank=True."""
        result = InputSanitizer.sanitize_string("\t\n\r", reject_blank=True)

        assert result.is_valid is False

    def test_sanitize_string_valid_with_whitespace_reject_blank(self) -> None:
        """Test valid input with surrounding whitespace passes with reject_blank=True."""
        result = InputSanitizer.sanitize_string("  valid  ", reject_blank=True)

        assert result.is_valid is True
        assert result.sanitized_value == "valid"


# ============================================================
# Test: sanitize_string - Length validation
# ============================================================


class TestSanitizeStringLength:
    """Tests for length validation."""

    def test_sanitize_string_within_max_length(self) -> None:
        """Test string within max length passes."""
        result = InputSanitizer.sanitize_string("short", max_length=10)

        assert result.is_valid is True

    def test_sanitize_string_exceeds_max_length(self) -> None:
        """Test string exceeding max length is truncated with error."""
        result = InputSanitizer.sanitize_string("this is too long", max_length=5)

        assert result.is_valid is False
        assert any("length" in error.lower() for error in result.errors)
        # String is truncated
        assert len(result.sanitized_value) <= 5

    def test_sanitize_string_exact_max_length(self) -> None:
        """Test string at exact max length passes."""
        result = InputSanitizer.sanitize_string("12345", max_length=5)

        assert result.is_valid is True


# ============================================================
# Test: sanitize_string - Injection detection
# ============================================================


class TestSanitizeStringInjection:
    """Tests for injection attack detection."""

    @pytest.mark.parametrize(
        "input_str",
        [
            "SELECT * FROM users",
            "DROP TABLE users",
            "1; DELETE FROM orders",
            "UNION SELECT password",
            "INSERT INTO users",
            "UPDATE users SET",
            "EXEC sp_executesql",
        ],
    )
    def test_sanitize_string_sql_injection(self, input_str: str) -> None:
        """Test SQL injection patterns are detected."""
        result = InputSanitizer.sanitize_string(input_str)

        assert result.is_valid is False
        assert any("sql" in error.lower() for error in result.errors)

    @pytest.mark.parametrize(
        "input_str",
        [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert(1)>",
            "<div onclick='malicious()'>",
        ],
    )
    def test_sanitize_string_xss_tags(self, input_str: str) -> None:
        """Test XSS HTML tags are detected and removed."""
        result = InputSanitizer.sanitize_string(input_str)

        assert result.is_valid is False
        assert any("html" in error.lower() or "tag" in error.lower() for error in result.errors)

    def test_sanitize_string_javascript_protocol(self) -> None:
        """Test javascript: protocol in href is detected as SQL injection pattern."""
        # Note: The JAVASCRIPT keyword is in the SQL injection pattern for XSS prevention
        result = InputSanitizer.sanitize_string("<a href='javascript:evil()'>")

        assert result.is_valid is False
        # Detected as SQL injection because JAVASCRIPT is in that pattern
        assert any("sql" in error.lower() for error in result.errors)

    @pytest.mark.parametrize(
        "input_str",
        [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2f%2e%2e%2f",
            "file:///etc/passwd",
        ],
    )
    def test_sanitize_string_path_traversal(self, input_str: str) -> None:
        """Test path traversal patterns are detected."""
        result = InputSanitizer.sanitize_string(input_str)

        assert result.is_valid is False
        assert any("path" in error.lower() or "traversal" in error.lower() for error in result.errors)


# ============================================================
# Test: sanitize_string - Safe edge cases
# ============================================================


class TestSanitizeStringSafeEdgeCases:
    """Tests for strings that look suspicious but are safe."""

    def test_sanitize_string_legitimate_select(self) -> None:
        """Test that 'select' in normal context might be flagged (security-first)."""
        # Note: The sanitizer is aggressive - it flags SQL keywords even in normal text
        result = InputSanitizer.sanitize_string("Please select an option")

        # This is expected behavior - security-first approach
        assert result.is_valid is False

    def test_sanitize_string_url_with_slashes(self) -> None:
        """Test normal URLs are allowed."""
        result = InputSanitizer.sanitize_string("https://example.com/path/to/resource")

        assert result.is_valid is True

    def test_sanitize_string_numbers_and_symbols(self) -> None:
        """Test strings with numbers and safe symbols."""
        result = InputSanitizer.sanitize_string("order-12345_test")

        assert result.is_valid is True
