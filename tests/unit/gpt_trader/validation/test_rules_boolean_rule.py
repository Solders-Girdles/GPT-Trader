import pytest

from gpt_trader.validation import BooleanRule, RuleError


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
