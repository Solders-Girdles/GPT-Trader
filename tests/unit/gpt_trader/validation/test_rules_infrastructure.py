import pytest

from gpt_trader.validation import (
    FloatRule,
    IntegerRule,
    RuleChain,
    RuleError,
    StripStringRule,
)
from gpt_trader.validation.rules.base import (
    BaseValidationRule,
    FunctionRule,
    normalize_rule,
)


class TestRuleChain:
    def test_composes_rules(self) -> None:
        chain = RuleChain(StripStringRule(), FloatRule())
        assert chain(" 42 ") == 42.0


class TestRuleChainExtended:
    """Extended tests for RuleChain."""

    def test_empty_chain_raises(self) -> None:
        with pytest.raises(RuleError):
            RuleChain()

    def test_callable_conversion(self) -> None:
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
