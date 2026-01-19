from decimal import Decimal

import pytest

from gpt_trader.validation import (
    DecimalRule,
    FloatRule,
    IntegerRule,
    PercentageRule,
    RuleError,
)


class TestDecimalRule:
    def test_coerces_str_and_numbers(self) -> None:
        rule = DecimalRule()
        assert rule("1.20") == Decimal("1.20")
        assert rule(5) == Decimal("5")

    def test_empty_string_returns_default(self) -> None:
        rule = DecimalRule(default=Decimal("3.14"), allow_none=True)
        assert rule("") == Decimal("3.14")


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


class TestFloatRule:
    def test_uses_default_for_none(self) -> None:
        rule = FloatRule(default=2.5, allow_none=True)
        assert rule(None) == 2.5

    def test_blank_string_errors_when_no_default(self) -> None:
        rule = FloatRule()
        with pytest.raises(RuleError):
            rule("   ")


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


class TestIntegerRule:
    def test_parses_integers(self) -> None:
        rule = IntegerRule()
        assert rule("10") == 10
        assert rule(7) == 7

    def test_invalid_integer_raises(self) -> None:
        rule = IntegerRule()
        with pytest.raises(RuleError):
            rule("not-int", "field")


class TestIntegerRuleExtended:
    """Extended tests for IntegerRule edge cases."""

    def test_none_allowed(self) -> None:
        rule = IntegerRule(allow_none=True)
        assert rule(None) is None

    def test_none_not_allowed_raises(self) -> None:
        rule = IntegerRule()
        with pytest.raises(RuleError):
            rule(None)


class TestPercentageRule:
    def test_accepts_fractional_range(self) -> None:
        rule = PercentageRule()
        assert rule(0.5) == pytest.approx(0.5)

    def test_out_of_range_raises(self) -> None:
        rule = PercentageRule()
        with pytest.raises(RuleError):
            rule(1.5)


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
