from __future__ import annotations

from enum import Enum

import pytest

from gpt_trader.utilities.parsing import (
    FALSE_BOOLEAN_TOKENS,
    TRUE_BOOLEAN_TOKENS,
    coerce_enum,
    interpret_tristate_bool,
)


class SampleEnum(Enum):
    """Sample enum for testing coerce_enum."""

    FOO = "FOO"
    BAR = "BAR"
    BAZ = "BAZ"


@pytest.mark.parametrize("token", sorted(TRUE_BOOLEAN_TOKENS))
def test_interpret_tristate_bool_truthy_tokens(token: str) -> None:
    assert interpret_tristate_bool(token) is True
    assert interpret_tristate_bool(token.upper()) is True


@pytest.mark.parametrize("token", sorted(FALSE_BOOLEAN_TOKENS))
def test_interpret_tristate_bool_falsy_tokens(token: str) -> None:
    assert interpret_tristate_bool(token) is False
    assert interpret_tristate_bool(token.upper()) is False


def test_interpret_tristate_bool_handles_whitespace() -> None:
    assert interpret_tristate_bool("  yes  ") is True
    assert interpret_tristate_bool("  off  ") is False


def test_interpret_tristate_bool_returns_none_for_unknown_values() -> None:
    assert interpret_tristate_bool(None) is None
    assert interpret_tristate_bool("") is None
    assert interpret_tristate_bool("    ") is None
    assert interpret_tristate_bool("maybe") is None


# Tests for coerce_enum


class TestCoerceEnum:
    """Tests for the coerce_enum utility function."""

    def test_enum_value_returns_unchanged(self) -> None:
        """Passing an enum value should return it unchanged."""
        result, string = coerce_enum(SampleEnum.FOO, SampleEnum)
        assert result is SampleEnum.FOO
        assert string == "FOO"

    def test_string_value_uppercase_matches(self) -> None:
        """Uppercase string should match enum value."""
        result, string = coerce_enum("FOO", SampleEnum)
        assert result is SampleEnum.FOO
        assert string == "FOO"

    def test_string_value_lowercase_matches_case_insensitive(self) -> None:
        """Lowercase string should match (default case insensitive)."""
        result, string = coerce_enum("foo", SampleEnum)
        assert result is SampleEnum.FOO
        assert string == "FOO"

    def test_string_value_mixed_case_matches(self) -> None:
        """Mixed case string should match (default case insensitive)."""
        result, string = coerce_enum("FoO", SampleEnum)
        assert result is SampleEnum.FOO
        assert string == "FOO"

    def test_invalid_string_returns_none_and_normalized(self) -> None:
        """Invalid string should return None with normalized string."""
        result, string = coerce_enum("invalid", SampleEnum)
        assert result is None
        assert string == "INVALID"

    def test_case_sensitive_mode_lowercase_fails(self) -> None:
        """In case-sensitive mode, lowercase should not match."""
        result, string = coerce_enum("foo", SampleEnum, case_sensitive=True)
        assert result is None
        assert string == "foo"

    def test_case_sensitive_mode_exact_matches(self) -> None:
        """In case-sensitive mode, exact match should work."""
        result, string = coerce_enum("FOO", SampleEnum, case_sensitive=True)
        assert result is SampleEnum.FOO
        assert string == "FOO"

    def test_alias_mapping_basic(self) -> None:
        """Alias should map to different enum value."""
        aliases = {"ALIAS": SampleEnum.BAR}
        result, string = coerce_enum("alias", SampleEnum, aliases=aliases)
        assert result is SampleEnum.BAR
        assert string == "ALIAS"

    def test_alias_takes_precedence_over_enum_match(self) -> None:
        """Alias should take precedence even if string matches enum."""
        aliases = {"FOO": SampleEnum.BAZ}
        result, string = coerce_enum("foo", SampleEnum, aliases=aliases)
        assert result is SampleEnum.BAZ
        assert string == "FOO"

    def test_enum_value_bypasses_aliases(self) -> None:
        """When passing an actual enum, aliases should not apply."""
        aliases = {"FOO": SampleEnum.BAZ}
        result, string = coerce_enum(SampleEnum.FOO, SampleEnum, aliases=aliases)
        assert result is SampleEnum.FOO
        assert string == "FOO"

    def test_empty_aliases_dict(self) -> None:
        """Empty aliases dict should not affect behavior."""
        result, string = coerce_enum("foo", SampleEnum, aliases={})
        assert result is SampleEnum.FOO
        assert string == "FOO"

    def test_none_aliases(self) -> None:
        """None aliases should not affect behavior."""
        result, string = coerce_enum("foo", SampleEnum, aliases=None)
        assert result is SampleEnum.FOO
        assert string == "FOO"
