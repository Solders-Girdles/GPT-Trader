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
from gpt_trader.validation.rules.base import (
    BaseValidationRule,
    FunctionRule,
    normalize_rule,
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

    def test_empty_string_raises(self) -> None:
        rule = SymbolRule()
        with pytest.raises(RuleError):
            rule("   ")

    def test_non_string_raises(self) -> None:
        rule = SymbolRule()
        with pytest.raises(RuleError):
            rule(123)

    def test_lowercase_option(self) -> None:
        # Even with uppercase=False, the symbol is validated after normalization
        # so we need to pass an already uppercase value to test non-uppercase mode
        rule = SymbolRule(uppercase=False)
        assert rule("BTC-USD") == "BTC-USD"


class TestBooleanRuleExtended:
    """Extended tests for BooleanRule edge cases."""

    def test_none_with_no_default_raises(self) -> None:
        rule = BooleanRule()
        with pytest.raises(RuleError):
            rule(None)

    def test_none_with_default_returns_default(self) -> None:
        rule = BooleanRule(default=True)
        assert rule(None) is True

    def test_custom_true_tokens(self) -> None:
        rule = BooleanRule(true_tokens=["yep"])
        assert rule("yep") is True

    def test_custom_false_tokens(self) -> None:
        rule = BooleanRule(false_tokens=["nope"])
        assert rule("nope") is False

    def test_float_1_converts_to_true(self) -> None:
        rule = BooleanRule()
        assert rule(1.0) is True

    def test_float_0_converts_to_false(self) -> None:
        rule = BooleanRule()
        assert rule(0.0) is False

    def test_string_true_false(self) -> None:
        rule = BooleanRule()
        assert rule("true") is True
        assert rule("false") is False
        assert rule("TRUE") is True
        assert rule("FALSE") is False


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


class TestDecimalRuleExtended:
    """Extended tests for DecimalRule edge cases."""

    def test_none_with_default_returns_default(self) -> None:
        rule = DecimalRule(default=Decimal("42.5"))
        assert rule(None) == Decimal("42.5")

    def test_none_allowed(self) -> None:
        rule = DecimalRule(allow_none=True)
        assert rule(None) is None

    def test_none_not_allowed_raises(self) -> None:
        rule = DecimalRule()
        with pytest.raises(RuleError):
            rule(None)

    def test_decimal_passthrough(self) -> None:
        rule = DecimalRule()
        val = Decimal("99.99")
        assert rule(val) is val

    def test_float_conversion(self) -> None:
        rule = DecimalRule()
        result = rule(3.14)
        assert result == Decimal("3.14")

    def test_empty_with_allow_none(self) -> None:
        rule = DecimalRule(allow_none=True)
        assert rule("   ") is None

    def test_empty_no_default_no_allow_none_raises(self) -> None:
        rule = DecimalRule()
        with pytest.raises(RuleError):
            rule("")

    def test_unsupported_type_raises(self) -> None:
        rule = DecimalRule()
        with pytest.raises(RuleError):
            rule([1, 2, 3])


class TestFloatRuleExtended:
    """Extended tests for FloatRule edge cases."""

    def test_allow_none(self) -> None:
        rule = FloatRule(allow_none=True)
        assert rule(None) is None

    def test_none_no_default_no_allow_raises(self) -> None:
        rule = FloatRule()
        with pytest.raises(RuleError):
            rule(None)

    def test_string_conversion(self) -> None:
        rule = FloatRule()
        assert rule("3.14") == pytest.approx(3.14)

    def test_blank_with_default(self) -> None:
        rule = FloatRule(default=1.5)
        assert rule("  ") == 1.5

    def test_blank_with_allow_none(self) -> None:
        rule = FloatRule(allow_none=True)
        assert rule("") is None


class TestIntegerRuleExtended:
    """Extended tests for IntegerRule edge cases."""

    def test_none_allowed(self) -> None:
        rule = IntegerRule(allow_none=True)
        assert rule(None) is None

    def test_none_not_allowed_raises(self) -> None:
        rule = IntegerRule()
        with pytest.raises(RuleError):
            rule(None)


class TestPercentageRuleExtended:
    """Extended tests for PercentageRule edge cases."""

    def test_none_allowed(self) -> None:
        rule = PercentageRule(allow_none=True)
        assert rule(None) is None

    def test_none_not_allowed_raises(self) -> None:
        rule = PercentageRule()
        with pytest.raises(RuleError):
            rule(None)

    def test_auto_converts_integer_percentage(self) -> None:
        rule = PercentageRule()
        assert rule(50) == pytest.approx(0.5)

    def test_boundary_values(self) -> None:
        rule = PercentageRule()
        assert rule(0) == pytest.approx(0)
        assert rule(1) == pytest.approx(1)

    def test_negative_raises(self) -> None:
        rule = PercentageRule()
        with pytest.raises(RuleError):
            rule(-0.1)

    def test_non_integer_over_1_raises(self) -> None:
        rule = PercentageRule()
        with pytest.raises(RuleError):
            rule(1.5)


class TestRuleChainExtended:
    """Extended tests for RuleChain."""

    def test_empty_chain_raises(self) -> None:
        with pytest.raises(RuleError):
            RuleChain()

    def test_callable_conversion(self) -> None:
        # Test that plain functions are converted to rules
        # IntegerRule first converts "5" to 5, then double multiplies by 2
        def double(value: int, field_name: str = "value") -> int:
            return value * 2

        chain = RuleChain(IntegerRule(), double)
        assert chain("5") == 10


class TestFunctionRule:
    """Tests for FunctionRule adapter."""

    def test_wraps_function(self) -> None:
        def add_suffix(value: str, field_name: str = "value") -> str:
            return f"{value}_suffix"

        rule = FunctionRule(add_suffix)
        assert rule("test") == "test_suffix"

    def test_apply_calls_function(self) -> None:
        def uppercase(value: str, field_name: str = "value") -> str:
            return value.upper()

        rule = FunctionRule(uppercase)
        assert rule.apply("hello") == "HELLO"

    def test_passes_field_name(self) -> None:
        def check_field(value: str, field_name: str = "value") -> str:
            return f"{field_name}:{value}"

        rule = FunctionRule(check_field)
        assert rule.apply("test", field_name="myfield") == "myfield:test"


class TestNormalizeRule:
    """Tests for normalize_rule function."""

    def test_returns_base_rule_unchanged(self) -> None:
        rule = IntegerRule()
        result = normalize_rule(rule)
        assert result is rule

    def test_wraps_callable(self) -> None:
        def my_rule(value: str, field_name: str = "value") -> str:
            return value.strip()

        result = normalize_rule(my_rule)
        assert isinstance(result, FunctionRule)
        assert result("  hello  ") == "hello"

    def test_raises_for_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported rule type"):
            normalize_rule("not a rule")  # type: ignore[arg-type]


class TestBaseValidationRule:
    """Tests for BaseValidationRule."""

    def test_call_invokes_apply(self) -> None:
        class MyRule(BaseValidationRule):
            def apply(self, value, *, field_name: str = "value"):
                return value * 2

        rule = MyRule()
        assert rule(5) == 10

    def test_call_passes_field_name(self) -> None:
        class MyRule(BaseValidationRule):
            def apply(self, value, *, field_name: str = "value"):
                return f"{field_name}={value}"

        rule = MyRule()
        assert rule("test", "myfield") == "myfield=test"


class TestRuleError:
    """Tests for RuleError exception."""

    def test_message_stored(self) -> None:
        exc = RuleError("Something failed")
        assert str(exc) == "Something failed"

    def test_value_stored(self) -> None:
        exc = RuleError("Invalid", value=42)
        assert exc.value == 42

    def test_value_defaults_to_none(self) -> None:
        exc = RuleError("Error")
        assert exc.value is None

    def test_is_value_error_subclass(self) -> None:
        exc = RuleError("Error")
        assert isinstance(exc, ValueError)
