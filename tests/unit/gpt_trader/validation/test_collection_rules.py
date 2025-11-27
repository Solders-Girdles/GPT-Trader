"""Tests for collection and mapping validation rules."""

from __future__ import annotations

import pytest

from gpt_trader.validation.rules.base import RuleError
from gpt_trader.validation.rules.collections import ListRule, MappingRule


class TestMappingRuleInit:
    """Tests for MappingRule initialization."""

    def test_default_allow_none(self) -> None:
        rule = MappingRule()
        assert rule._allow_none is True

    def test_allow_none_false(self) -> None:
        rule = MappingRule(allow_none=False)
        assert rule._allow_none is False

    def test_custom_separators(self) -> None:
        rule = MappingRule(item_separator=";", kv_separator="=")
        assert rule._item_separator == ";"
        assert rule._kv_separator == "="


class TestMappingRuleApply:
    """Tests for MappingRule.apply method."""

    def test_none_when_allowed(self) -> None:
        rule = MappingRule(allow_none=True)
        result = rule.apply(None)
        assert result == {}

    def test_none_when_not_allowed(self) -> None:
        rule = MappingRule(allow_none=False)
        with pytest.raises(RuleError, match="requires a mapping but received None"):
            rule.apply(None)

    def test_dict_input(self) -> None:
        rule = MappingRule()
        result = rule.apply({"key1": "value1", "key2": "value2"})
        assert result == {"key1": "value1", "key2": "value2"}

    def test_string_input(self) -> None:
        rule = MappingRule()
        result = rule.apply("key1:value1,key2:value2")
        assert result == {"key1": "value1", "key2": "value2"}

    def test_invalid_type_raises(self) -> None:
        rule = MappingRule()
        with pytest.raises(RuleError, match="expected a mapping or a string"):
            rule.apply(12345)

    def test_empty_key_raises(self) -> None:
        rule = MappingRule()
        with pytest.raises(RuleError, match="includes an entry with an empty key"):
            rule.apply({"": "value"})

    def test_none_value_raises(self) -> None:
        rule = MappingRule()
        with pytest.raises(RuleError, match="with an empty value"):
            rule.apply({"key": None})

    def test_empty_string_value_raises(self) -> None:
        rule = MappingRule()
        with pytest.raises(RuleError, match="with an empty value"):
            rule.apply({"key": "  "})

    def test_value_converter_applied(self) -> None:
        rule = MappingRule(value_converter=int)
        result = rule.apply({"key": "42"})
        assert result == {"key": 42}

    def test_value_rule_applied(self) -> None:
        def double(value: int, field_name: str = "") -> int:
            return value * 2

        rule = MappingRule(value_converter=int, value_rule=double)
        result = rule.apply({"key": "5"})
        assert result == {"key": 10}


class TestMappingRuleParseString:
    """Tests for MappingRule string parsing."""

    def test_blank_items_allowed(self) -> None:
        rule = MappingRule(allow_blank_items=True)
        result = rule.apply("key1:val1,,key2:val2")
        assert result == {"key1": "val1", "key2": "val2"}

    def test_blank_items_not_allowed(self) -> None:
        rule = MappingRule(allow_blank_items=False)
        with pytest.raises(RuleError, match="contains an empty mapping entry"):
            rule.apply("key1:val1,,key2:val2")

    def test_missing_separator_raises(self) -> None:
        rule = MappingRule()
        with pytest.raises(RuleError, match="has an invalid entry"):
            rule.apply("key1_without_separator")

    def test_custom_separators(self) -> None:
        rule = MappingRule(item_separator=";", kv_separator="=")
        result = rule.apply("key1=val1;key2=val2")
        assert result == {"key1": "val1", "key2": "val2"}


class TestListRuleInit:
    """Tests for ListRule initialization."""

    def test_default_allow_none(self) -> None:
        rule = ListRule()
        assert rule._allow_none is True

    def test_allow_none_false(self) -> None:
        rule = ListRule(allow_none=False)
        assert rule._allow_none is False

    def test_custom_separator(self) -> None:
        rule = ListRule(separator=";")
        assert rule._separator == ";"


class TestListRuleApply:
    """Tests for ListRule.apply method."""

    def test_none_when_allowed(self) -> None:
        rule = ListRule(allow_none=True)
        result = rule.apply(None)
        assert result == []

    def test_none_when_not_allowed(self) -> None:
        rule = ListRule(allow_none=False)
        with pytest.raises(RuleError, match="requires a list but received None"):
            rule.apply(None)

    def test_string_input(self) -> None:
        rule = ListRule()
        result = rule.apply("a,b,c")
        assert result == ["a", "b", "c"]

    def test_list_input(self) -> None:
        rule = ListRule()
        result = rule.apply(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_iterable_input(self) -> None:
        rule = ListRule()
        result = rule.apply({"a", "b"})
        assert set(result) == {"a", "b"}

    def test_invalid_type_raises(self) -> None:
        rule = ListRule()
        with pytest.raises(RuleError, match="expected a delimited string or iterable"):
            rule.apply(12345)

    def test_blank_items_allowed(self) -> None:
        rule = ListRule(allow_blank_items=True)
        result = rule.apply("a,,b")
        assert result == ["a", "b"]

    def test_blank_items_not_allowed(self) -> None:
        rule = ListRule(allow_blank_items=False)
        with pytest.raises(RuleError, match="contains an empty list entry"):
            rule.apply("a,,b")

    def test_none_item_in_list_when_blank_not_allowed(self) -> None:
        rule = ListRule(allow_blank_items=False)
        with pytest.raises(RuleError, match="contains an empty list entry"):
            rule.apply([1, None, 2])

    def test_item_converter_applied(self) -> None:
        rule = ListRule(item_converter=int)
        result = rule.apply("1,2,3")
        assert result == [1, 2, 3]

    def test_item_rule_applied(self) -> None:
        def double(value: int, field_name: str = "") -> int:
            return value * 2

        rule = ListRule(item_converter=int, item_rule=double)
        result = rule.apply("1,2,3")
        assert result == [2, 4, 6]

    def test_custom_separator(self) -> None:
        rule = ListRule(separator=";")
        result = rule.apply("a;b;c")
        assert result == ["a", "b", "c"]

    def test_strips_whitespace(self) -> None:
        rule = ListRule()
        result = rule.apply(" a , b , c ")
        assert result == ["a", "b", "c"]
