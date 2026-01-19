import pytest

from gpt_trader.validation import (
    RuleError,
    StripStringRule,
    SymbolRule,
    TimeOfDayRule,
)


class TestTimeOfDayRule:
    def test_valid_time_passes(self) -> None:
        rule = TimeOfDayRule()
        assert rule("09:30") == "09:30"

    def test_invalid_format_raises(self) -> None:
        rule = TimeOfDayRule()
        with pytest.raises(RuleError):
            rule("9:3")


class TestTimeOfDayRuleExtended:
    """Extended tests for TimeOfDayRule edge cases."""

    def test_none_allowed(self) -> None:
        rule = TimeOfDayRule(allow_none=True)
        assert rule(None) is None

    def test_none_not_allowed(self) -> None:
        rule = TimeOfDayRule(allow_none=False)
        with pytest.raises(RuleError):
            rule(None)

    def test_non_string_raises(self) -> None:
        rule = TimeOfDayRule()
        with pytest.raises(RuleError):
            rule(1230)

    def test_empty_string_allowed(self) -> None:
        rule = TimeOfDayRule(allow_none=True)
        assert rule("  ") is None

    def test_empty_string_not_allowed(self) -> None:
        rule = TimeOfDayRule(allow_none=False)
        with pytest.raises(RuleError):
            rule("")

    def test_strips_whitespace(self) -> None:
        rule = TimeOfDayRule()
        assert rule("  23:59  ") == "23:59"

    def test_boundary_times(self) -> None:
        rule = TimeOfDayRule()
        assert rule("00:00") == "00:00"
        assert rule("23:59") == "23:59"


class TestStripStringRuleExtended:
    """Extended tests for StripStringRule."""

    def test_none_with_no_default_raises(self) -> None:
        rule = StripStringRule()
        with pytest.raises(RuleError):
            rule(None)

    def test_none_with_default(self) -> None:
        rule = StripStringRule(default="fallback")
        assert rule(None) == "fallback"

    def test_empty_with_default(self) -> None:
        rule = StripStringRule(default="fallback")
        assert rule("") == "fallback"

    def test_whitespace_with_default(self) -> None:
        rule = StripStringRule(default="fallback")
        assert rule("   ") == "fallback"

    def test_empty_without_default_raises(self) -> None:
        rule = StripStringRule()
        with pytest.raises(RuleError):
            rule("")

    def test_converts_non_string_to_string(self) -> None:
        rule = StripStringRule()
        assert rule(123) == "123"


class TestSymbolRule:
    def test_normalizes_and_validates(self) -> None:
        rule = SymbolRule()
        assert rule(" btc-usd ") == "BTC-USD"

    def test_invalid_symbol_raises(self) -> None:
        rule = SymbolRule()
        with pytest.raises(RuleError):
            rule("BTC/USD")

    def test_empty_string_raises(self) -> None:
        rule = SymbolRule()
        with pytest.raises(RuleError):
            rule("   ")

    def test_non_string_raises(self) -> None:
        rule = SymbolRule()
        with pytest.raises(RuleError):
            rule(123)

    def test_lowercase_option(self) -> None:
        rule = SymbolRule(uppercase=False)
        assert rule("BTC-USD") == "BTC-USD"
