from decimal import Decimal

import pytest

from gpt_trader.validation import (
    BooleanRule,
    DecimalRule,
    FloatRule,
    IntegerRule,
    ListRule,
    MappingRule,
    PercentageRule,
    RuleChain,
    RuleError,
    StripStringRule,
    SymbolRule,
    TimeOfDayRule,
)


class TestBooleanRule:
    def test_parses_tokens_and_numbers(self) -> None:
        rule = BooleanRule(default=False)
        assert rule(True) is True
        assert rule("yes") is True
        assert rule("NO") is False
        assert rule(1) is True
        assert rule(0) is False

    def test_invalid_boolean_raises(self) -> None:
        rule = BooleanRule()
        with pytest.raises(RuleError):
            rule("maybe")


class TestMappingRule:
    def test_parses_mapping_from_string(self) -> None:
        rule = MappingRule(value_converter=int)
        result = rule("btc:1,eth:2")
        assert result == {"btc": 1, "eth": 2}

    def test_invalid_entry_raises(self) -> None:
        rule = MappingRule(value_converter=float)
        with pytest.raises(RuleError):
            rule("btc=1")

    def test_disallows_blank_entries_when_requested(self) -> None:
        rule = MappingRule(value_converter=int, allow_blank_items=False)
        with pytest.raises(RuleError):
            rule("alpha:1,,beta:2", "my_map")


class TestDecimalRule:
    def test_coerces_str_and_numbers(self) -> None:
        rule = DecimalRule()
        assert rule("1.20") == Decimal("1.20")
        assert rule(5) == Decimal("5")

    def test_empty_string_returns_default(self) -> None:
        rule = DecimalRule(default=Decimal("3.14"), allow_none=True)
        assert rule("") == Decimal("3.14")


class TestFloatRule:
    def test_uses_default_for_none(self) -> None:
        rule = FloatRule(default=2.5, allow_none=True)
        assert rule(None) == 2.5

    def test_blank_string_errors_when_no_default(self) -> None:
        rule = FloatRule()
        with pytest.raises(RuleError):
            rule("   ")


class TestPercentageRule:
    def test_accepts_fractional_range(self) -> None:
        rule = PercentageRule()
        assert rule(0.5) == pytest.approx(0.5)

    def test_out_of_range_raises(self) -> None:
        rule = PercentageRule()
        with pytest.raises(RuleError):
            rule(1.5)


class TestTimeOfDayRule:
    def test_valid_time_passes(self) -> None:
        rule = TimeOfDayRule()
        assert rule("09:30") == "09:30"

    def test_invalid_format_raises(self) -> None:
        rule = TimeOfDayRule()
        with pytest.raises(RuleError):
            rule("9:3")


class TestRuleChain:
    def test_composes_rules(self) -> None:
        chain = RuleChain(StripStringRule(), FloatRule())
        assert chain(" 42 ") == 42.0


class TestListRule:
    def test_parses_string_list(self) -> None:
        rule = ListRule(item_converter=str)
        assert rule("alpha, beta ,gamma") == ["alpha", "beta", "gamma"]

    def test_disallows_empty_entries(self) -> None:
        rule = ListRule(item_converter=str, allow_blank_items=False)
        with pytest.raises(RuleError):
            rule("one,,two", "numbers")

    def test_cast_failure_surfaces(self) -> None:
        rule = ListRule(item_converter=int)
        with pytest.raises(RuleError):
            rule("1, two", "numbers")


class TestIntegerRule:
    def test_parses_integers(self) -> None:
        rule = IntegerRule()
        assert rule("10") == 10
        assert rule(7) == 7

    def test_invalid_integer_raises(self) -> None:
        rule = IntegerRule()
        with pytest.raises(RuleError):
            rule("not-int", "field")


class TestSymbolRule:
    def test_normalizes_and_validates(self) -> None:
        rule = SymbolRule()
        assert rule(" btc-usd ") == "BTC-USD"

    def test_invalid_symbol_raises(self) -> None:
        rule = SymbolRule()
        with pytest.raises(RuleError):
            rule("BTC/USD")
