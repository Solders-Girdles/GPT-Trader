"""Tests for input sanitisation in SecurityValidator."""

from __future__ import annotations

from typing import Any

import pytest


class TestInputSanitisation:
    """Test input sanitisation scenarios."""

    @pytest.mark.parametrize(
        "input_str",
        [
            "SELECT * FROM users",
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT password FROM users",
            "INSERT INTO users VALUES ('hacker', 'password')",
            "UPDATE users SET password='hacked' WHERE id=1",
            "DELETE FROM users WHERE id=1",
            "EXEC xp_cmdshell('dir')",
            "SCRIPT alert('xss')",
        ],
    )
    def test_sql_injection_detection(self, security_validator: Any, input_str: str) -> None:
        """Test SQL injection pattern detection."""
        result = security_validator.sanitize_string(input_str)

        assert not result.is_valid
        assert any("SQL injection" in error for error in result.errors)

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("<script>alert('xss')</script>", "alert('xss')"),
            ("<b onclick='alert(1)'>bold</b>", "bold"),
            ("<img src='x' onerror='alert(1)'>", ""),
            ("<div>content</div>", "content"),
            ("<p>paragraph <span>text</span></p>", "paragraph text"),
            ("normal text", "normal text"),
        ],
    )
    def test_xss_sanitisation(self, security_validator: Any, input_str: str, expected: str) -> None:
        """Test XSS tag removal and sanitisation."""
        result = security_validator.sanitize_string(input_str)

        if "<script>" in input_str or "onclick" in input_str or "onerror" in input_str:
            assert not result.is_valid
            assert "HTML tags not allowed" in result.errors[0]
        else:
            assert result.is_valid
            assert result.sanitized_value == expected

    @pytest.mark.parametrize(
        "input_str",
        [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "file:///etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "../../config/database.yml",
            "..\\..\\config\\settings.ini",
        ],
    )
    def test_path_traversal_detection(self, security_validator: Any, input_str: str) -> None:
        """Test path traversal pattern detection."""
        result = security_validator.sanitize_string(input_str)

        assert not result.is_valid
        assert any("Path traversal" in error for error in result.errors)

    @pytest.mark.parametrize(
        "input_str",
        [
            "normal_string",
            "user123",
            "BTC-USD",
            "Valid input string",
            "simple text without special chars",
            "UPPERCASE_TEXT",
            "mixedCaseText",
        ],
    )
    def test_valid_input_passes(self, security_validator: Any, input_str: str) -> None:
        """Test valid input passes sanitisation."""
        result = security_validator.sanitize_string(input_str)

        assert result.is_valid
        assert result.sanitized_value == input_str

    def test_empty_input_rejection(self, security_validator: Any) -> None:
        """Test empty input is rejected."""
        result = security_validator.sanitize_string("")

        assert not result.is_valid
        assert "Input cannot be empty" in result.errors[0]

    def test_none_input_handling(self, security_validator: Any) -> None:
        """Test None input handling."""
        result = security_validator.sanitize_string(None)  # type: ignore

        assert not result.is_valid
        assert "Input cannot be empty" in result.errors[0]

    def test_input_length_truncation(self, security_validator: Any) -> None:
        """Test input length truncation."""
        long_input = "a" * 300  # Longer than default max_length
        max_length = 255

        result = security_validator.sanitize_string(long_input, max_length=max_length)

        assert len(result.errors) == 1  # Should have length error
        assert "exceeds maximum length" in result.errors[0]
        assert len(result.sanitized_value) == max_length

    def test_special_character_escaping(self, security_validator: Any) -> None:
        """Test special character escaping."""
        input_str = "test'with\"quotes"

        result = security_validator.sanitize_string(input_str)

        assert result.is_valid
        assert result.sanitized_value == "test''with\"\"quotes"

    def test_whitespace_handling(self, security_validator: Any) -> None:
        """Test whitespace handling."""
        input_str = "  text with spaces  "

        result = security_validator.sanitize_string(input_str)

        assert result.is_valid
        assert result.sanitized_value == "text with spaces"

    def test_multiple_attack_vectors(self, security_validator: Any) -> None:
        """Test input with multiple attack vectors."""
        combined_attack = "SELECT * FROM users'; <script>alert('xss')</script>; ../../../etc/passwd"

        result = security_validator.sanitize_string(combined_attack)

        assert not result.is_valid
        # Should detect SQL injection first (most severe)
        assert any("SQL injection" in error for error in result.errors)

    def test_unicode_handling(self, security_validator: Any) -> None:
        """Test Unicode character handling."""
        unicode_input = "测试中文字符 ñáéíóú"

        result = security_validator.sanitize_string(unicode_input)

        assert result.is_valid
        assert result.sanitized_value == unicode_input

    def test_case_insensitive_pattern_matching(self, security_validator: Any) -> None:
        """Test case insensitive pattern matching."""
        case_variants = [
            "select * from users",
            "SELECT * FROM USERS",
            "Select * From Users",
            "sElEcT * FrOm UsErS",
        ]

        for variant in case_variants:
            result = security_validator.sanitize_string(variant)
            assert not result.is_valid
            assert any("SQL injection" in error for error in result.errors)

    def test_partial_pattern_matching(self, security_validator: Any) -> None:
        """Test partial pattern matching."""
        partial_attacks = [
            "partial select",
            "script tag",
            "path traversal attempt",
        ]

        for attack in partial_attacks:
            result = security_validator.sanitize_string(attack)
            # Should pass if not complete attack pattern
            assert result.is_valid

    def test_custom_max_length(self, security_validator: Any) -> None:
        """Test custom maximum length parameter."""
        input_str = "test"
        custom_max = 2

        result = security_validator.sanitize_string(input_str, max_length=custom_max)

        assert not result.is_valid
        assert "exceeds maximum length" in result.errors[0]
        assert len(result.sanitized_value) == custom_max

    def test_sanitisation_error_accumulation(self, security_validator: Any) -> None:
        """Test that multiple sanitisation errors are accumulated."""
        # Input that triggers multiple violations
        multi_violation = "<script>'../../../etc/passwd'; SELECT * FROM users</script>"

        result = security_validator.sanitize_string(multi_violation)

        # Should have multiple errors
        assert len(result.errors) >= 1
        assert any("SQL injection" in error for error in result.errors)

    def test_sanitisation_idempotency(self, security_validator: Any) -> None:
        """Test sanitisation is idempotent."""
        input_str = "normal text"

        result1 = security_validator.sanitize_string(input_str)
        result2 = security_validator.sanitize_string(result1.sanitized_value)

        assert result1.is_valid
        assert result2.is_valid
        assert result1.sanitized_value == result2.sanitized_value

    def test_sanitisation_with_newlines_and_tabs(self, security_validator: Any) -> None:
        """Test sanitisation with newlines and tabs."""
        input_str = "text\nwith\tnewlines\nand\ttabs"

        result = security_validator.sanitize_string(input_str)

        assert result.is_valid
        assert "\n" in result.sanitized_value
        assert "\t" in result.sanitized_value
